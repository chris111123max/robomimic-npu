"""Read-only success-only BC-RNN transition dataset for Stage 3A-v2."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
V1_DIR = THIS_DIR.parent / "stage3_actor_initialization"
if str(V1_DIR) not in sys.path:
    sys.path.insert(0, str(V1_DIR))

from stage3_dataset import (  # noqa: E402
    STORED_ACTION_BOUND_TOLERANCE,
    episode_seed,
    inspect_dataset,
)


def episode_success(group):
    """Read and cross-check the Stage 1 episode-level success label."""
    attribute = group.attrs.get("success")
    dataset_value = None
    if "episode_success" in group:
        values = np.asarray(group["episode_success"], dtype=np.bool_).reshape(-1)
        if values.size == 0 or not np.all(values == values[0]):
            raise RuntimeError(f"Inconsistent episode_success values: {group.name}")
        dataset_value = bool(values[0])
    if attribute is None and dataset_value is None:
        raise RuntimeError(f"Missing episode success label: {group.name}")
    if attribute is not None and dataset_value is not None and bool(attribute) != dataset_value:
        raise RuntimeError(f"Episode success attribute/dataset mismatch: {group.name}")
    return bool(attribute) if attribute is not None else dataset_value


class SuccessOnlyTransitionDataset:
    """Load only successful episodes within an already separated seed split."""

    def __init__(self, dataset_path, requested_seeds):
        self.metadata = inspect_dataset(dataset_path)
        self.requested_seeds = [int(seed) for seed in requested_seeds]
        states, actions = [], []
        successful_seeds, failed_seeds = [], []
        successful_lengths = {}
        action_min, action_max = np.inf, -np.inf

        with h5py.File(self.metadata["path"], "r") as handle:
            lookup = {episode_seed(group): group for group in handle["episodes"].values()}
            missing = sorted(set(self.requested_seeds) - set(lookup))
            if missing:
                raise RuntimeError(f"BC-RNN dataset is missing split seeds: {missing}")
            for seed in self.requested_seeds:
                group = lookup[seed]
                if not episode_success(group):
                    failed_seeds.append(seed)
                    continue
                successful_seeds.append(seed)
                observation_group = group["obs"]
                components = []
                for key in self.metadata["canonical_keys"]:
                    value = np.asarray(observation_group[key], dtype=np.float32)
                    components.append(value.reshape(value.shape[0], -1))
                state = np.concatenate(components, axis=1)
                action = np.asarray(group["actions"], dtype=np.float32)
                if state.shape != (action.shape[0], 59) or action.shape[1:] != (14,):
                    raise RuntimeError(
                        f"Unexpected successful transition shapes seed={seed}: {state.shape}, {action.shape}"
                    )
                if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
                    raise RuntimeError(f"NaN/Inf in successful episode seed={seed}")
                states.append(state)
                actions.append(action)
                successful_lengths[seed] = int(action.shape[0])
                action_min = min(action_min, float(action.min()))
                action_max = max(action_max, float(action.max()))

        if not successful_seeds:
            raise RuntimeError("The requested split contains no successful BC-RNN episodes")
        self.states = np.concatenate(states, axis=0).astype(np.float32, copy=False)
        self.actions = np.concatenate(actions, axis=0).astype(np.float32, copy=False)
        if (action_min < -1.0 - STORED_ACTION_BOUND_TOLERANCE or
                action_max > 1.0 + STORED_ACTION_BOUND_TOLERANCE):
            raise RuntimeError(
                "Stored successful action exceeds numerical tolerance around [-1, 1]: "
                f"min={action_min}, max={action_max}"
            )
        self.state_dim = 59
        self.action_dim = 14
        self.successful_seeds = successful_seeds
        self.failed_seeds = failed_seeds
        self.statistics = {
            "requested_seed_start": min(self.requested_seeds),
            "requested_seed_end": max(self.requested_seeds),
            "total_episodes": len(self.requested_seeds),
            "successful_episodes": len(successful_seeds),
            "failed_episodes": len(failed_seeds),
            "successful_transition_count": int(self.states.shape[0]),
            "successful_seeds": successful_seeds,
            "excluded_failed_seeds": failed_seeds,
            "successful_episode_lengths": successful_lengths,
            "failed_episodes_loaded_into_student_dataset": 0,
            "state_shape": [59],
            "action_shape": [14],
            "canonical_keys_in_flatten_order": self.metadata["canonical_keys"],
            "action_min": action_min,
            "action_max": action_max,
            "targets_clipped_or_rescaled": False,
        }

    def __len__(self):
        return int(self.states.shape[0])
