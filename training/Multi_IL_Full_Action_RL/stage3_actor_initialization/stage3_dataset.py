"""Read-only transition dataset for BC-RNN actor distillation."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


EXPECTED_KEYS = {
    "object": (41,),
    "robot0_eef_pos": (3,),
    "robot0_eef_quat": (4,),
    "robot0_gripper_qpos": (2,),
    "robot1_eef_pos": (3,),
    "robot1_eef_quat": (4,),
    "robot1_gripper_qpos": (2,),
}


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return decode(value.item())
    return value


def root_json_attribute(handle, name):
    if name not in handle.attrs:
        raise RuntimeError(f"missing HDF5 root attribute {name!r}: {handle.filename}")
    value = decode(handle.attrs[name])
    return json.loads(value) if isinstance(value, str) else value


def episode_seed(group):
    if "initial_seed" in group.attrs:
        return int(group.attrs["initial_seed"])
    if "initial_seed" in group:
        return int(group["initial_seed"][0])
    raise RuntimeError(f"missing initial_seed: {group.name}")


def inspect_dataset(dataset_path):
    path = Path(dataset_path).resolve()
    with h5py.File(path, "r") as handle:
        if decode(handle.attrs.get("policy_id", "")) != "bc_rnn":
            raise RuntimeError(f"Stage 3 requires bc_rnn dataset: {path}")
        keys = list(root_json_attribute(handle, "canonical_observation_keys"))
        shapes = {
            key: tuple(value) for key, value in
            root_json_attribute(handle, "canonical_observation_shapes").items()
        }
        action_shape = tuple(root_json_attribute(handle, "action_shape"))
        if set(keys) != set(EXPECTED_KEYS) or any(shapes[key] != EXPECTED_KEYS[key] for key in keys):
            raise RuntimeError(f"Unexpected canonical observation schema: keys={keys}, shapes={shapes}")
        if sum(int(np.prod(shapes[key])) for key in keys) != 59:
            raise RuntimeError("Canonical state dimension is not 59")
        if action_shape != (14,):
            raise RuntimeError(f"Action shape is not [14]: {action_shape}")
        checkpoint = decode(handle.attrs.get("checkpoint"))
        seeds = [episode_seed(group) for group in handle["episodes"].values()]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("BC-RNN dataset contains duplicate seeds")
    return {
        "path": str(path),
        "canonical_keys": keys,
        "canonical_shapes": {key: list(shapes[key]) for key in keys},
        "state_dim": 59,
        "action_dim": 14,
        "checkpoint": checkpoint,
        "seeds": sorted(seeds),
    }


class Stage3TransitionDataset:
    def __init__(self, dataset_path, seeds):
        self.metadata = inspect_dataset(dataset_path)
        self.state_dim = self.metadata["state_dim"]
        self.action_dim = self.metadata["action_dim"]
        self.seeds = [int(seed) for seed in seeds]
        states, actions, episode_lengths = [], [], {}
        action_min = np.inf
        action_max = -np.inf
        with h5py.File(self.metadata["path"], "r") as handle:
            lookup = {episode_seed(group): group for group in handle["episodes"].values()}
            missing = sorted(set(self.seeds) - set(lookup))
            if missing:
                raise RuntimeError(f"BC-RNN dataset missing seeds: {missing}")
            for seed in self.seeds:
                group = lookup[seed]
                observation_group = group["obs"]
                components = []
                for key in self.metadata["canonical_keys"]:
                    value = np.asarray(observation_group[key], dtype=np.float32)
                    components.append(value.reshape(value.shape[0], -1))
                state = np.concatenate(components, axis=1)
                action = np.asarray(group["actions"], dtype=np.float32)
                if state.shape[0] != action.shape[0]:
                    raise RuntimeError(f"State/action length mismatch for seed={seed}")
                if state.shape[1:] != (59,) or action.shape[1:] != (14,):
                    raise RuntimeError(
                        f"Unexpected transition shapes seed={seed}: {state.shape}, {action.shape}"
                    )
                if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
                    raise RuntimeError(f"NaN/Inf in Stage 3 data for seed={seed}")
                states.append(state)
                actions.append(action)
                episode_lengths[seed] = int(action.shape[0])
                action_min = min(action_min, float(action.min()))
                action_max = max(action_max, float(action.max()))
        self.states = np.concatenate(states, axis=0).astype(np.float32, copy=False)
        self.actions = np.concatenate(actions, axis=0).astype(np.float32, copy=False)
        if action_min < -1.0001 or action_max > 1.0001:
            raise RuntimeError(
                f"Stored env action is outside expected [-1, 1]: min={action_min}, max={action_max}"
            )
        self.statistics = {
            "seeds": self.seeds,
            "episode_count": len(self.seeds),
            "transition_count": len(self.states),
            "episode_lengths": episode_lengths,
            "state_shape": [59],
            "action_shape": [14],
            "action_min": action_min,
            "action_max": action_max,
            "canonical_keys_in_flatten_order": self.metadata["canonical_keys"],
        }

    def __len__(self):
        return len(self.states)

    def batch(self, indices):
        return self.states[indices], self.actions[indices]
