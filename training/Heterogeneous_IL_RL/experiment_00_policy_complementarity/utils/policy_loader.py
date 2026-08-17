"""Unified loading and validation for all four native robomimic policies."""

import gc
import hashlib
import random

import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils


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
    policy, checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint_path), device=device, verbose=False,
    )
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
    if hasattr(policy, "policy"):
        policy.policy = None
    del policy
    gc.collect()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
