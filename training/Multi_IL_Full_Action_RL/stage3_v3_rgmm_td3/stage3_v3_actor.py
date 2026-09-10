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
    if state is None or not bool(reset.any()):
        return state
    if isinstance(state, tuple):
        return tuple(_zero_rows(item, reset) for item in state)
    state = state.clone()
    state[:, reset, :] = 0
    return state


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


class BatchedGMMExecutor:
    """Per-environment hidden slots with the checkpoint's sampled GMM semantics."""
    def __init__(self, actor, action_scale, action_offset, num_envs, horizon=10):
        self.actor = actor
        self.scale = action_scale.reshape(1, -1)
        self.offset = action_offset.reshape(1, -1)
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.hidden = [None] * self.num_envs
        self.counters = [0] * self.num_envs

    def reset_indices(self, indices):
        for index in indices:
            self.hidden[int(index)] = None
            self.counters[int(index)] = 0

    def actions_for(self, indices, observations, external_noise_std=0.0,
                    action_low=None, action_high=None):
        result = []
        was_training = self.actor.training
        self.actor.eval()
        try:
            for env_id, observation in zip(indices, observations):
                env_id = int(env_id)
                if self.counters[env_id] % self.horizon == 0:
                    self.hidden[env_id] = None
                flat = torch.as_tensor(
                    obs_to_flat(observation)[None], dtype=torch.float32,
                    device=next(self.actor.parameters()).device,
                )
                with torch.no_grad():
                    dist, hidden = self.actor.forward_train_step(
                        flat_to_obs(flat), rnn_state=self.hidden[env_id],
                    )
                    normalized = dist.sample()
                    action = normalized * self.scale + self.offset
                    if float(external_noise_std) > 0:
                        action = action + torch.randn_like(action) * float(external_noise_std)
                    if action_low is not None:
                        low = torch.as_tensor(action_low, device=action.device)
                        high = torch.as_tensor(action_high, device=action.device)
                        action = torch.maximum(torch.minimum(action, high), low)
                self.hidden[env_id] = hidden
                self.counters[env_id] += 1
                result.append(action[0].cpu().numpy().astype(np.float32, copy=False))
        finally:
            self.actor.train(was_training)
        return result
