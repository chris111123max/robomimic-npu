"""Official checkpoint loading with Experiment 00 frame-stack semantics."""

import gc
from collections import deque

import numpy as np


class UnifiedRolloutPolicy:
    def __init__(self, rollout_policy, frame_stack, observation_keys):
        self.rollout_policy = rollout_policy
        self.frame_stack = int(frame_stack)
        self.observation_keys = tuple(observation_keys)
        self._history = None

    @property
    def policy_class_name(self):
        return self.rollout_policy.policy.__class__.__name__

    @property
    def native_algo(self):
        return self.rollout_policy.policy

    def start_episode(self):
        self.rollout_policy.start_episode()
        self._history = None

    def _stack(self, observation):
        missing = [key for key in self.observation_keys if key not in observation]
        if missing:
            raise RuntimeError(f"Environment missing policy observations: {missing}")
        values = {key: np.asarray(observation[key]) for key in self.observation_keys}
        if self.frame_stack <= 1:
            return values
        if self._history is None:
            self._history = {key: deque([value.copy() for _ in range(self.frame_stack)],
                                        maxlen=self.frame_stack) for key, value in values.items()}
        else:
            for key, value in values.items():
                self._history[key].append(value.copy())
        return {key: np.stack(tuple(history), axis=0) for key, history in self._history.items()}

    def __call__(self, ob):
        return self.rollout_policy(ob=self._stack(ob), batched_ob=False)

    def release(self):
        self._history = None
        if self.rollout_policy is not None:
            self.rollout_policy.policy = None
        self.rollout_policy = None


def select_device():
    import robomimic.utils.torch_utils as TorchUtils
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    if device is None:
        raise RuntimeError("No Ascend NPU device is available")
    return device


def checkpoint_metadata(policy_name, checkpoint_path):
    import robomimic.utils.file_utils as FileUtils
    checkpoint = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(checkpoint_path))
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint)
    shape = checkpoint["shape_metadata"]
    transformer = bool(config.algo.transformer.enabled)
    rnn = bool(config.algo.rnn.enabled)
    return {
        "policy_name": policy_name, "checkpoint_path": str(checkpoint_path),
        "algo_name": checkpoint["algo_name"], "environment_metadata": checkpoint["env_metadata"],
        "observation_keys": list(shape["all_shapes"]),
        "observation_shapes": {k: list(v) for k, v in shape["all_shapes"].items()},
        "action_dimension": int(shape["ac_dim"]),
        "checkpoint_horizon": int(config.experiment.rollout.horizon),
        "frame_stack": int(config.train.frame_stack) if "frame_stack" in config.train else 1,
        "temporal_type": "transformer" if transformer else "rnn" if rnn else "feedforward",
        "context_length": int(config.algo.transformer.context_length) if transformer else None,
        "rnn_horizon": int(config.algo.rnn.horizon) if rnn else None,
        "low_noise_eval": bool(config.algo.gmm.low_noise_eval),
        "uses_action_history": False,
    }


def load_policy(policy_name, checkpoint_path, device=None):
    import robomimic.utils.file_utils as FileUtils
    device = select_device() if device is None else device
    native, checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint_path), device=device, verbose=False)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint)
    frame_stack = int(config.train.frame_stack) if "frame_stack" in config.train else 1
    if config.algo.transformer.enabled and frame_stack != int(config.algo.transformer.context_length):
        raise RuntimeError("Transformer frame_stack and context_length differ")
    policy = UnifiedRolloutPolicy(native, frame_stack, checkpoint["shape_metadata"]["all_shapes"].keys())
    return policy, checkpoint, device


def warm_history(policy, observations, branch_step):
    """Restore policy state before a_t by consuming obs_0 ... obs_{t-1}."""
    if branch_step < 0 or branch_step > len(observations):
        raise ValueError(f"Invalid branch step {branch_step} for {len(observations)} observations")
    policy.start_episode()
    for index in range(branch_step):
        policy(ob=observations[index])  # output intentionally ignored and never executed
    return branch_step


def release_policy(policy):
    import torch
    if policy is not None:
        policy.release()
    gc.collect()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
