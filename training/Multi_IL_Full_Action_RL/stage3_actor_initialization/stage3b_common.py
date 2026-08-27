"""Shared Stage 3B DAgger rollout utilities."""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
STAGE1_DIR = THIS_DIR.parent / "stage1_rollout_collection"
for path in (THIS_DIR, STAGE1_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from actor_network import load_actor_checkpoint  # noqa: E402
from collect_multi_il_rollouts import progress_observation_schema  # noqa: E402
from common import atomic_json, extract_canonical_observation  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402


ACTION_BOUND_TOLERANCE = 1e-3


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def select_device(name):
    if name.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU requested but torch_npu cannot be imported") from exc
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but unavailable")
        torch.npu.set_device(torch.device(name))
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_scalar(value.item())
    return value


def load_initial_states(path, seeds):
    requested = set(int(seed) for seed in seeds)
    result = {}
    with h5py.File(path, "r") as handle:
        for group in handle["seeds"].values():
            seed = int(group.attrs["initial_seed"])
            if seed not in requested:
                continue
            state = {
                "states": np.asarray(group["states"]).copy(),
                "model": decode_scalar(group["model"][()]),
            }
            ep_meta = decode_scalar(group["ep_meta"][()])
            if ep_meta:
                state["ep_meta"] = ep_meta
            result[seed] = state
    missing = sorted(requested - set(result))
    if missing:
        raise RuntimeError(f"Initial-state file is missing seeds: {missing}")
    return result


def load_canonical_schema(dataset_path):
    with h5py.File(dataset_path, "r") as handle:
        keys = json.loads(decode_scalar(handle.attrs["canonical_observation_keys"]))
        shapes = json.loads(decode_scalar(handle.attrs["canonical_observation_shapes"]))
        policy_id = decode_scalar(handle.attrs.get("policy_id", ""))
    if policy_id != "bc_rnn":
        raise RuntimeError(f"Expected bc_rnn Stage 1 dataset, got {policy_id!r}")
    if sum(int(np.prod(shapes[key])) for key in keys) != 59:
        raise RuntimeError(f"Canonical schema is not 59D: keys={keys}, shapes={shapes}")
    return list(keys), shapes


def flatten_canonical(observation, keys, shapes):
    canonical = extract_canonical_observation(observation, keys, shapes)
    state = np.concatenate([canonical[key].reshape(-1) for key in keys]).astype(np.float32)
    if state.shape != (59,) or not np.isfinite(state).all():
        raise RuntimeError(f"Invalid canonical student state: shape={state.shape}")
    return canonical, state


def extract_progress(canonical, schema):
    result = {}
    for name in ("trash_in_trash_bin", "payload_in_target_bin"):
        descriptor = schema["fields"][name]
        value = float(
            np.asarray(canonical[descriptor["canonical_key"]]).reshape(-1)[int(descriptor["flat_index"])]
        )
        if value not in (0.0, 1.0):
            raise RuntimeError(f"Non-boolean progress observable: {name}={value}")
        result[name] = bool(value)
    return result


def env_success(env):
    value = env.is_success()
    return bool(value["task"] if isinstance(value, dict) else value)


def close_env(env):
    raw = getattr(env, "unwrapped", env)
    suite_env = getattr(raw, "env", None)
    close = getattr(suite_env if suite_env is not None else raw, "close", None)
    if callable(close):
        close()


def validate_action(action, source):
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (14,) or not np.isfinite(action).all():
        raise RuntimeError(f"Invalid {source} action: shape={action.shape}")
    minimum, maximum = float(action.min()), float(action.max())
    if minimum < -1.0 - ACTION_BOUND_TOLERANCE or maximum > 1.0 + ACTION_BOUND_TOLERANCE:
        raise RuntimeError(
            f"{source} action exceeds tolerance around [-1, 1]: "
            f"min={minimum}, max={maximum}, tolerance={ACTION_BOUND_TOLERANCE}"
        )
    return action


def validate_frozen_log_std(actor, expected=-3.0):
    layer = actor.last_fc_log_std
    weight = layer.weight.detach().cpu().numpy()
    bias = layer.bias.detach().cpu().numpy()
    if any(parameter.requires_grad for parameter in layer.parameters()):
        raise RuntimeError("Stage 3B student log_std must be frozen")
    if not np.allclose(weight, 0.0, rtol=0.0, atol=0.0):
        raise RuntimeError("Stage 3B log_std weight is not exactly zero")
    if not np.allclose(bias, float(expected), rtol=0.0, atol=1e-7):
        raise RuntimeError(
            f"Stage 3B log_std bias differs from {expected}: "
            f"range=[{float(bias.min())}, {float(bias.max())}]"
        )


def load_teacher_and_env(checkpoint_path, device):
    checkpoint = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(checkpoint_path))
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint)
    if not bool(config.algo.rnn.enabled):
        raise RuntimeError("Stage 3B teacher checkpoint is not recurrent")
    teacher, loaded = FileUtils.policy_from_checkpoint(
        ckpt_dict=checkpoint, device=device, verbose=False
    )
    model = teacher.policy
    model.set_eval()
    model.nets.eval()
    parameter_count = 0
    for parameter in model.nets.parameters():
        parameter.requires_grad_(False)
        parameter_count += int(parameter.numel())
    if not parameter_count or any(parameter.requires_grad for parameter in model.nets.parameters()):
        raise RuntimeError("Failed to freeze BC-RNN teacher")
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=loaded, render=False, render_offscreen=False, verbose=False
    )
    return teacher, env, parameter_count


def load_student(checkpoint_path, device):
    actor, payload = load_actor_checkpoint(checkpoint_path, device=device, freeze_log_std=True)
    actor.eval()
    if payload["architecture"] != {"state_dim": 59, "action_dim": 14, "hidden_dims": [256, 256]}:
        raise RuntimeError(f"Unexpected Stage 3B actor architecture: {payload['architecture']}")
    validate_frozen_log_std(actor, expected=-3.0)
    return actor, payload


def student_action(actor, state, device):
    with torch.no_grad():
        tensor = torch.as_tensor(state[None], dtype=torch.float32, device=device)
        action = actor(tensor, deterministic=True, return_log_prob=False)[0][0].cpu().numpy()
    return validate_action(action, "student")


def teacher_action(teacher, observation):
    with torch.no_grad():
        action = np.asarray(teacher(ob=observation), dtype=np.float32)
    return validate_action(action, "teacher")


def prepare_rollout(config, student_checkpoint, seeds, device_name):
    device = select_device(device_name)
    keys, shapes = load_canonical_schema(config["stage1_rnn_dataset"])
    student, student_payload = load_student(student_checkpoint, device)
    teacher, env, teacher_parameter_count = load_teacher_and_env(config["teacher_checkpoint"], device)
    states = load_initial_states(config["initial_states"], seeds)
    progress_schema = progress_observation_schema(env, keys, shapes)
    return {
        "device": device,
        "keys": keys,
        "shapes": shapes,
        "student": student,
        "student_payload": student_payload,
        "teacher": teacher,
        "env": env,
        "teacher_parameter_count": teacher_parameter_count,
        "initial_states": states,
        "progress_schema": progress_schema,
    }
