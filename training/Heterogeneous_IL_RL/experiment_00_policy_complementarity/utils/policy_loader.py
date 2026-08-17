"""Unified loading and validation for all four native robomimic policies."""

import gc
import hashlib
import random
from collections import deque

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


class UnifiedRolloutPolicy:
    """Native RolloutPolicy plus checkpoint-defined frame-stack semantics.

    robomimic normally applies FrameStackWrapper in env_from_checkpoint. This
    experiment keeps one unwrapped dataset-metadata environment so simulator
    state verification is identical for every policy. Applying the same padded
    sliding window here is equivalent at the policy input boundary and keeps
    persisted trajectories as raw low-dimensional observations.
    """

    def __init__(self, rollout_policy, frame_stack):
        self.rollout_policy = rollout_policy
        self.frame_stack = int(frame_stack)
        self._history = None

    @property
    def policy_class_name(self):
        return self.rollout_policy.policy.__class__.__name__

    def start_episode(self):
        self.rollout_policy.start_episode()
        self._history = None

    def _stack_observation(self, observation):
        if self.frame_stack <= 1:
            return observation
        if self._history is None:
            self._history = {
                key: deque(
                    [np.asarray(value).copy() for _ in range(self.frame_stack)],
                    maxlen=self.frame_stack,
                )
                for key, value in observation.items()
            }
        else:
            if set(observation) != set(self._history):
                raise RuntimeError("Environment observation keys changed within an episode")
            for key, value in observation.items():
                self._history[key].append(np.asarray(value).copy())
        return {
            key: np.stack(list(values), axis=0)
            for key, values in self._history.items()
        }

    def __call__(self, ob, goal=None, batched_ob=False):
        if batched_ob:
            raise ValueError("UnifiedRolloutPolicy expects one unbatched environment observation")
        prepared = self._stack_observation(ob)
        return self.rollout_policy(ob=prepared, goal=goal, batched_ob=False)

    def release(self):
        self._history = None
        if self.rollout_policy is not None and hasattr(self.rollout_policy, "policy"):
            self.rollout_policy.policy = None
        self.rollout_policy = None


def select_device():
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    if device is None:
        raise RuntimeError("No CUDA or Ascend NPU device is available; this experiment requires the server accelerator runtime")
    return device


def checkpoint_metadata(policy_name, checkpoint_path):
    checkpoint = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(checkpoint_path))
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint)
    shape = checkpoint["shape_metadata"]
    return {
        "policy_name": policy_name,
        "checkpoint_path": str(checkpoint_path),
        "algo_name": checkpoint["algo_name"],
        "environment_name": checkpoint["env_metadata"]["env_name"],
        "environment_metadata": checkpoint["env_metadata"],
        "observation_keys": list(shape["all_shapes"].keys()),
        "observation_shapes": {key: list(value) for key, value in shape["all_shapes"].items()},
        "action_dimension": int(shape["ac_dim"]),
        "checkpoint_horizon": int(config.experiment.rollout.horizon),
    }


def load_policy(policy_name, checkpoint_path, device=None):
    if device is None:
        device = select_device()
    native_policy, checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint_path), device=device, verbose=False,
    )
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint)
    frame_stack = int(config.train.frame_stack) if "frame_stack" in config.train else 1
    if config.algo.transformer.enabled:
        context_length = int(config.algo.transformer.context_length)
        if frame_stack != context_length:
            raise RuntimeError(
                f"{policy_name} transformer context_length={context_length} but "
                f"train.frame_stack={frame_stack}"
            )
    policy = UnifiedRolloutPolicy(native_policy, frame_stack=frame_stack)
    return policy, checkpoint, device


def set_policy_sampling_seed(meta_seed, policy_name, initial_state_id):
    # Python / NumPy are shared with robosuite, so give every policy the same
    # per-initial-condition environment stream. Torch drives native GMM sampling
    # and receives a separate, order-independent per-policy stream.
    environment_material = f"{int(meta_seed)}:environment:{int(initial_state_id)}".encode("utf-8")
    environment_seed = int.from_bytes(hashlib.sha256(environment_material).digest()[:4], "little") & 0x7FFFFFFF
    policy_material = f"{int(meta_seed)}:{policy_name}:{int(initial_state_id)}".encode("utf-8")
    policy_seed = int.from_bytes(hashlib.sha256(policy_material).digest()[:4], "little") & 0x7FFFFFFF
    random.seed(environment_seed)
    np.random.seed(environment_seed)
    torch.manual_seed(policy_seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(policy_seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy_seed)
    return policy_seed


def release_policy(policy):
    # Break the wrapper -> model reference before emptying the accelerator cache.
    if hasattr(policy, "release"):
        policy.release()
    elif hasattr(policy, "policy"):
        policy.policy = None
    del policy
    gc.collect()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
