"""Read-only Stage 1 BC-GMM transition dataset for Stage 3C."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


EXPECTED_KEYS = [
    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
    "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos", "object",
]
EXPECTED_SHAPES = {
    "robot0_eef_pos": [3], "robot0_eef_quat": [4], "robot0_gripper_qpos": [2],
    "robot1_eef_pos": [3], "robot1_eef_quat": [4], "robot1_gripper_qpos": [2],
    "object": [41],
}
ACTION_TOLERANCE = 1e-3


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return decode(value.item())
    return value


def json_attr(handle, name):
    if name not in handle.attrs:
        raise RuntimeError(f"Missing root attribute {name!r}: {handle.filename}")
    value = decode(handle.attrs[name])
    return json.loads(value) if isinstance(value, str) else value


def episode_seed(group):
    if "initial_seed" in group.attrs:
        return int(group.attrs["initial_seed"])
    return int(group["initial_seed"][0])


def episode_success(group):
    if "success" in group.attrs:
        return bool(group.attrs["success"])
    return bool(group["episode_success"][0])


def inspect_dataset(path):
    path = Path(path).resolve()
    with h5py.File(path, "r") as handle:
        policy_id = decode(handle.attrs.get("policy_id", ""))
        keys = list(json_attr(handle, "canonical_observation_keys"))
        shapes = {key: list(value) for key, value in json_attr(handle, "canonical_observation_shapes").items()}
        action_shape = list(json_attr(handle, "action_shape"))
        checkpoint = decode(handle.attrs.get("checkpoint", ""))
        seeds = [episode_seed(group) for group in handle["episodes"].values()]
    if policy_id != "bc_gmm":
        raise RuntimeError(f"Stage 3C requires policy_id=bc_gmm, got {policy_id!r}")
    if keys != EXPECTED_KEYS or shapes != EXPECTED_SHAPES:
        raise RuntimeError(f"Canonical schema mismatch: keys={keys}, shapes={shapes}")
    if action_shape != [14] or len(set(seeds)) != len(seeds):
        raise RuntimeError(f"Invalid action shape or duplicate seeds: action={action_shape}")
    return {
        "path": str(path), "policy_id": policy_id, "checkpoint": checkpoint,
        "canonical_keys": keys, "canonical_shapes": shapes,
        "state_dim": 59, "action_dim": 14, "seeds": sorted(seeds),
    }


class Stage3CDataset:
    def __init__(self, path, seeds):
        self.metadata = inspect_dataset(path)
        self.state_dim, self.action_dim = 59, 14
        self.seeds = [int(seed) for seed in seeds]
        states, actions = [], []
        success_count = 0
        lengths = {}
        with h5py.File(self.metadata["path"], "r") as handle:
            lookup = {episode_seed(group): group for group in handle["episodes"].values()}
            missing = sorted(set(self.seeds) - set(lookup))
            if missing:
                raise RuntimeError(f"Dataset missing seeds: {missing}")
            for seed in self.seeds:
                group = lookup[seed]
                parts = []
                for key in self.metadata["canonical_keys"]:
                    value = np.asarray(group["obs"][key], dtype=np.float32)
                    parts.append(value.reshape(value.shape[0], -1))
                state = np.concatenate(parts, axis=1)
                action = np.asarray(group["actions"], dtype=np.float32)
                if state.shape != (len(action), 59) or action.shape[1:] != (14,):
                    raise RuntimeError(f"Bad shapes seed={seed}: state={state.shape}, action={action.shape}")
                if not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise RuntimeError(f"Non-finite transition seed={seed}")
                states.append(state)
                actions.append(action)
                lengths[str(seed)] = len(action)
                success_count += int(episode_success(group))
        self.states = np.concatenate(states).astype(np.float32, copy=False)
        self.actions = np.concatenate(actions).astype(np.float32, copy=False)
        action_min, action_max = float(self.actions.min()), float(self.actions.max())
        if action_min < -1.0 - ACTION_TOLERANCE or action_max > 1.0 + ACTION_TOLERANCE:
            raise RuntimeError(f"Saved action outside tolerance: [{action_min}, {action_max}]")
        self.statistics = {
            "total_episodes": len(self.seeds), "success_episodes": success_count,
            "failure_episodes": len(self.seeds) - success_count,
            "transition_count": len(self.states), "episode_lengths": lengths,
            "state_shape": [59], "target_action_shape": [14],
            "action_range": [action_min, action_max],
            "canonical_keys_in_flatten_order": self.metadata["canonical_keys"],
            "target_source": "saved_stage1_bc_gmm_behavior_action",
        }

    def __len__(self):
        return len(self.states)
