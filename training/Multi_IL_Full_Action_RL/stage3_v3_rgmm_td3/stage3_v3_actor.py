"""Exact robomimic BC-RNN-GMM loading and recurrent execution helpers."""
from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils


CANONICAL_KEYS = (
    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
    "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos", "object",
)
CANONICAL_SHAPES = OrderedDict((
    ("robot0_eef_pos", (3,)), ("robot0_eef_quat", (4,)),
    ("robot0_gripper_qpos", (2,)), ("robot1_eef_pos", (3,)),
    ("robot1_eef_quat", (4,)), ("robot1_gripper_qpos", (2,)),
    ("object", (41,)),
))


def module_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def flat_to_obs(flat):
    """Convert canonical [..., 59] vectors back to the checkpoint obs dictionary."""
    if flat.shape[-1] != 59:
        raise ValueError(f"Expected final observation dimension 59, got {flat.shape}")
    result = OrderedDict()
    cursor = 0
    for key, shape in CANONICAL_SHAPES.items():
        width = int(np.prod(shape))
        result[key] = flat[..., cursor:cursor + width].reshape(*flat.shape[:-1], *shape)
        cursor += width
    return result


def obs_to_flat(observation):
    missing = set(CANONICAL_KEYS) - set(observation)
    if missing:
        raise KeyError(f"Observation misses canonical keys: {sorted(missing)}")
    return np.concatenate([
        np.asarray(observation[key], np.float32).reshape(-1)
        for key in CANONICAL_KEYS
    ]).astype(np.float32, copy=False)


def _normalization_tensors(rollout, device):
    stats = rollout.action_normalization_stats
    if stats is None or set(stats) != {"actions"}:
        raise RuntimeError("BC checkpoint must contain one vector action normalization entry")
    scale = torch.as_tensor(stats["actions"]["scale"], dtype=torch.float32, device=device).reshape(1, 1, 1, -1)
    offset = torch.as_tensor(stats["actions"]["offset"], dtype=torch.float32, device=device).reshape(1, 1, 1, -1)
    if scale.shape[-1] != 14 or offset.shape[-1] != 14:
        raise RuntimeError("BC checkpoint action normalization is not 14-dimensional")
    return scale, offset


def load_exact_actor(checkpoint, device):
    """Load the authoritative checkpoint and make an exact trainable policy copy."""
    rollout, payload = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint), device=device, verbose=False,
    )
    algo = rollout.policy
    source = algo.nets["policy"]
    expected_shapes = {key: tuple(value) for key, value in CANONICAL_SHAPES.items()}
    actual_shapes = {
        key: tuple(value) for key, value in source.nets["encoder"].nets["obs"].obs_shapes.items()
    }
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"Unexpected BC observation contract: {actual_shapes}")
    expected = {
        "class": "RNNGMMActorNetwork", "ac_dim": 14, "num_modes": 5,
        "min_std": 0.0001, "std_activation": "softplus",
        "low_noise_eval": True, "use_tanh": False,
        "horizon": 10, "open_loop": False,
    }
    actual = {
        "class": type(source).__name__, "ac_dim": int(source.ac_dim),
        "num_modes": int(source.num_modes), "min_std": float(source.min_std),
        "std_activation": str(source.std_activation),
        "low_noise_eval": bool(source.low_noise_eval), "use_tanh": bool(source.use_tanh),
        "horizon": int(algo._rnn_horizon), "open_loop": bool(algo._rnn_is_open_loop),
    }
    if actual != expected:
        raise RuntimeError(f"Authoritative BC-RNN-GMM contract changed: {actual}")
    actor = copy.deepcopy(source).to(device)
    incompatible = actor.load_state_dict(source.state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict Actor transfer failed: {incompatible}")
    scale, offset = _normalization_tensors(rollout, device)
    metadata = {
        "checkpoint_keys": sorted(payload), "network_class": type(source).__name__,
        "parameter_count": sum(parameter.numel() for parameter in actor.parameters()),
        "state_dict_entries": len(actor.state_dict()), "missing_keys": [],
        "unexpected_keys": [], "source_hash": module_hash(source),
        "actor_hash": module_hash(actor), "rnn_horizon": actual["horizon"],
        "open_loop": actual["open_loop"], "num_modes": actual["num_modes"],
        "low_noise_eval": actual["low_noise_eval"], "use_tanh": actual["use_tanh"],
        "observation_shapes": {key: list(value) for key, value in actual_shapes.items()},
        "action_scale": scale.reshape(-1).detach().cpu().tolist(),
        "action_offset": offset.reshape(-1).detach().cpu().tolist(),
    }
    return actor, rollout, metadata


def distribution_tensors(distribution):
    base = distribution.component_distribution.base_dist
    return {
        "means_normalized": base.loc,
        "scales": base.scale,
        "logits": distribution.mixture_distribution.logits,
        "probs": distribution.mixture_distribution.probs,
    }


def environment_means(distribution, action_scale, action_offset):
    means = distribution.component_distribution.base_dist.loc
    shape = (1,) * (means.ndim - 1) + (means.shape[-1],)
    return means * action_scale.reshape(shape) + action_offset.reshape(shape)


def normalize_actions(actions, action_scale, action_offset):
    scale = action_scale.reshape(1, 1, -1)
    offset = action_offset.reshape(1, 1, -1)
    return (actions - offset) / scale


def _zero_rows(state, reset):
    if state is None:
        return state
    if isinstance(state, tuple):
        return tuple(_zero_rows(item, reset) for item in state)
    # Keep the reset mask on the device. Converting reset.any() to a Python
    # bool forced an NPU/CPU synchronization at every recurrent timestep;
    # the target Actor runs this path for every Critic update.
    return torch.where(reset.reshape(1, -1, 1), torch.zeros_like(state), state)


def recurrent_distributions(actor, observations, episode_steps, horizon=10,
                            initial_state=None, no_grad_prefix=0):
    """Run exact horizon-reset semantics and return one distribution per step.

    `no_grad_prefix` is the burn-in length. Hidden state is reconstructed without
    an Actor graph, detached, and then BPTT starts over the learning segment.
    """
    if observations.ndim != 3 or observations.shape[-1] != 59:
        raise ValueError("Recurrent observations must have shape [B,T,59]")
    if episode_steps.shape != observations.shape[:2]:
        raise ValueError("episode_steps must have shape [B,T]")
    state = initial_state
    outputs = []
    for index in range(observations.shape[1]):
        reset = episode_steps[:, index].remainder(int(horizon)).eq(0)
        state = _zero_rows(state, reset)
        context = torch.no_grad() if index < int(no_grad_prefix) else torch.enable_grad()
        with context:
            dist, state = actor.forward_train_step(
                flat_to_obs(observations[:, index]), rnn_state=state,
            )
        if index + 1 == int(no_grad_prefix):
            if isinstance(state, tuple):
                state = tuple(item.detach() for item in state)
            elif state is not None:
                state = state.detach()
        outputs.append(dist)
    return outputs, state


@torch.no_grad()
def target_final_distribution(actor, observations, episode_steps, horizon=10):
    """Reconstruct target RNN state without decoding unused burn-in GMM heads.

    The Critic target consumes only the final distribution. Keep the native
    ``forward_train_step`` for that final step and use the very same encoder
    and LSTM modules for the preceding steps, including horizon resets.
    """
    if observations.ndim != 3 or observations.shape[-1] != 59:
        raise ValueError("Recurrent observations must have shape [B,T,59]")
    if episode_steps.shape != observations.shape[:2]:
        raise ValueError("episode_steps must have shape [B,T]")
    if observations.shape[1] < 1:
        raise ValueError("Target sequence must contain at least one observation")
    state = None
    for index in range(observations.shape[1] - 1):
        reset = episode_steps[:, index].remainder(int(horizon)).eq(0)
        state = _zero_rows(state, reset)
        encoded = actor.nets["encoder"](obs=flat_to_obs(observations[:, index]))
        if state is None:
            state = actor.get_rnn_init_state(encoded.shape[0], encoded.device)
        _, state = actor.nets["rnn"].nets(encoded.unsqueeze(1), state)
    last = observations.shape[1] - 1
    state = _zero_rows(state, episode_steps[:, last].remainder(int(horizon)).eq(0))
    return actor.forward_train_step(flat_to_obs(observations[:, last]), rnn_state=state)


class BatchedGMMExecutor:
    """Batched recurrent execution with per-environment hidden slots.

    The previous implementation called ``forward_train_step`` once per
    environment and copied every action from the NPU separately.  That made a
    16-environment rollout launch the recurrent network 16 times per vector
    step and forced 16 synchronization points.  Hidden states are now packed
    into one batch, the actor is evaluated once, and the complete action batch
    is copied back to the host once.
    """
    def __init__(self, actor, action_scale, action_offset, num_envs, horizon=10):
        self.actor = actor
        self.device = next(actor.parameters()).device
        self.scale = action_scale.reshape(1, -1).to(self.device)
        self.offset = action_offset.reshape(1, -1).to(self.device)
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.hidden = [None] * self.num_envs
        self.counters = [0] * self.num_envs
        self._low = None
        self._high = None

    @staticmethod
    def _pack_hidden(states, reset):
        """Pack [layer, 1, hidden] per-env states into one batch state."""
        template = next((state for state in states if state is not None), None)
        if template is None:
            return None

        def pack_tensor(component, component_index=None):
            pieces = []
            for state in states:
                if state is None:
                    pieces.append(torch.zeros_like(component[:, :1, :]))
                else:
                    value = (state[component_index]
                             if component_index is not None else state)
                    pieces.append(value[:, :1, :].detach())
            return torch.cat(pieces, dim=1)

        if isinstance(template, tuple):
            packed = tuple(
                pack_tensor(template[index], index) for index in range(len(template))
            )
        else:
            packed = pack_tensor(template)
        return _zero_rows(packed, reset)

    @staticmethod
    def _unpack_hidden(state, count):
        """Split a batched recurrent state back into one slot per env."""
        if state is None:
            return [None] * count

        def split_tensor(component, index):
            return component[:, index:index + 1, :].detach()

        if isinstance(state, tuple):
            return [tuple(split_tensor(component, index) for component in state)
                    for index in range(count)]
        return [split_tensor(state, index) for index in range(count)]

    def reset_indices(self, indices):
        for index in indices:
            self.hidden[int(index)] = None
            self.counters[int(index)] = 0

    def actions_for(self, indices, observations, external_noise_std=0.0,
                    action_low=None, action_high=None):
        indices = [int(index) for index in indices]
        if len(indices) != len(observations):
            raise ValueError("indices and observations must have equal length")
        if not indices:
            return []
        if len(set(indices)) != len(indices):
            raise ValueError("indices contains duplicates")

        was_training = self.actor.training
        self.actor.eval()
        try:
            reset = torch.as_tensor(
                [self.counters[env_id] % self.horizon == 0 for env_id in indices],
                dtype=torch.bool, device=self.device,
            )
            state = self._pack_hidden([self.hidden[env_id] for env_id in indices], reset)
            flat = torch.as_tensor(
                np.stack([obs_to_flat(observation) for observation in observations], axis=0),
                dtype=torch.float32, device=self.device,
            )
            with torch.no_grad():
                distribution, state = self.actor.forward_train_step(
                    flat_to_obs(flat), rnn_state=state,
                )
                normalized = distribution.sample()
                action = normalized * self.scale + self.offset
                if float(external_noise_std) > 0:
                    action = action + torch.randn_like(action) * float(external_noise_std)
                if action_low is not None:
                    if self._low is None or self._low.shape[-1] != len(action_low):
                        self._low = torch.as_tensor(
                            action_low, dtype=torch.float32, device=self.device
                        ).reshape(1, -1)
                        self._high = torch.as_tensor(
                            action_high, dtype=torch.float32, device=self.device
                        ).reshape(1, -1)
                    action = torch.maximum(torch.minimum(action, self._high), self._low)

            slots = self._unpack_hidden(state, len(indices))
            for position, env_id in enumerate(indices):
                self.hidden[env_id] = slots[position]
                self.counters[env_id] += 1
            # One synchronization/copy for the whole vector action batch.
            host_actions = action.detach().cpu().numpy().astype(np.float32, copy=False)
            result = [host_actions[position] for position in range(len(indices))]
        finally:
            self.actor.train(was_training)
        return result
