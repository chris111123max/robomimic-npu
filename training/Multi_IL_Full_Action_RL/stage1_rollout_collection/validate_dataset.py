#!/usr/bin/env python3
"""Validate Stage 1 HDF5 datasets without loading robomimic or policy checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # allow --help outside the rollout environment
    h5py = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SCHEMA_VERSION, VALID_POLICY_IDS, atomic_json, read_json, sha256_array


REQUIRED_TRANSITION_DATASETS = (
    "actions", "rewards", "dones", "terminated", "truncated", "policy_id",
    "episode_id", "initial_seed", "timestep", "episode_success",
    "episode_return", "episode_length",
)


def decode_strings(values):
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in values]


def require_finite(name, values):
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{name} contains NaN or Inf")


def validate_policy(policy_dir, policy_id, seeds, shared_schema, require_both):
    dataset_path = policy_dir / "transitions.hdf5"
    episodes_path = policy_dir / "episodes.json"
    metadata_path = policy_dir / "metadata.json"
    for path in (dataset_path, episodes_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    episode_json = read_json(episodes_path)
    metadata = read_json(metadata_path)
    if metadata.get("success_filter_applied") is not False:
        raise RuntimeError(f"{policy_id}: success_filter_applied must be false")
    successes, failures, transitions = 0, 0, 0
    seen_seeds, state_hashes = [], {}
    with h5py.File(dataset_path, "r") as handle:
        if handle.attrs.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"{policy_id}: schema version mismatch")
        if handle.attrs.get("policy_id") != policy_id:
            raise RuntimeError(f"{policy_id}: root policy_id mismatch")
        keys = json.loads(handle.attrs["canonical_observation_keys"])
        shapes = json.loads(handle.attrs["canonical_observation_shapes"])
        action_shape = json.loads(handle.attrs["action_shape"])
        schema = {"keys": keys, "shapes": shapes, "action_shape": action_shape}
        if shared_schema and schema != shared_schema:
            raise RuntimeError(f"{policy_id}: canonical schema differs from other policies")
        groups = sorted(handle["episodes"].keys())
        if len(groups) != len(seeds):
            raise RuntimeError(f"{policy_id}: {len(groups)} episodes != {len(seeds)} seeds")
        if len(episode_json) != len(groups):
            raise RuntimeError(f"{policy_id}: episodes.json count mismatch")
        for expected_index, group_name in enumerate(groups):
            group = handle[f"episodes/{group_name}"]
            for name in REQUIRED_TRANSITION_DATASETS:
                if name not in group:
                    raise RuntimeError(f"{policy_id}/{group_name}: missing {name}")
            length = int(group["actions"].shape[0])
            if length <= 0:
                raise RuntimeError(f"{policy_id}/{group_name}: empty episode")
            for name in REQUIRED_TRANSITION_DATASETS:
                if int(group[name].shape[0]) != length:
                    raise RuntimeError(f"{policy_id}/{group_name}: {name} length mismatch")
            if tuple(group["actions"].shape[1:]) != tuple(action_shape):
                raise RuntimeError(f"{policy_id}/{group_name}: action shape mismatch")
            require_finite(f"{policy_id}/{group_name}/actions", group["actions"][:])
            require_finite(f"{policy_id}/{group_name}/rewards", group["rewards"][:])
            if set(group["obs"].keys()) != set(keys) or set(group["next_obs"].keys()) != set(keys):
                raise RuntimeError(f"{policy_id}/{group_name}: observation keys mismatch")
            for key in keys:
                expected_shape = (length, *shapes[key])
                if group[f"obs/{key}"].shape != expected_shape:
                    raise RuntimeError(f"{policy_id}/{group_name}: obs/{key} shape mismatch")
                if group[f"next_obs/{key}"].shape != expected_shape:
                    raise RuntimeError(f"{policy_id}/{group_name}: next_obs/{key} shape mismatch")
                require_finite(f"{policy_id}/{group_name}/obs/{key}", group[f"obs/{key}"][:])
                require_finite(f"{policy_id}/{group_name}/next_obs/{key}", group[f"next_obs/{key}"][:])
            timesteps = group["timestep"][:]
            if not np.array_equal(timesteps, np.arange(length)):
                raise RuntimeError(f"{policy_id}/{group_name}: non-contiguous timestep")
            episode_ids = group["episode_id"][:]
            if not np.all(episode_ids == expected_index):
                raise RuntimeError(f"{policy_id}/{group_name}: episode_id mismatch")
            seed_values = group["initial_seed"][:]
            if not np.all(seed_values == seeds[expected_index]):
                raise RuntimeError(f"{policy_id}/{group_name}: initial_seed mismatch")
            policy_values = decode_strings(group["policy_id"][:])
            if any(value != policy_id for value in policy_values):
                raise RuntimeError(f"{policy_id}/{group_name}: transition policy_id mismatch")
            success_values = group["episode_success"][:].astype(bool)
            if not np.all(success_values == success_values[0]):
                raise RuntimeError(f"{policy_id}/{group_name}: inconsistent episode_success")
            return_values = group["episode_return"][:]
            actual_return = float(np.sum(group["rewards"][:]))
            if not np.allclose(return_values, actual_return, rtol=0.0, atol=1e-8):
                raise RuntimeError(f"{policy_id}/{group_name}: episode_return mismatch")
            if not np.all(group["episode_length"][:] == length):
                raise RuntimeError(f"{policy_id}/{group_name}: episode_length mismatch")
            dones = group["dones"][:].astype(bool)
            terminated = group["terminated"][:].astype(bool)
            truncated = group["truncated"][:].astype(bool)
            if not np.array_equal(dones, terminated | truncated):
                raise RuntimeError(f"{policy_id}/{group_name}: done != terminated OR truncated")
            if np.any(terminated & truncated):
                raise RuntimeError(f"{policy_id}/{group_name}: transition both terminated and truncated")
            if not dones[-1] or np.any(dones[:-1]):
                raise RuntimeError(f"{policy_id}/{group_name}: episode boundary flags are invalid")
            if "mc_return" in group:
                if group["mc_return"].shape != (length,):
                    raise RuntimeError(f"{policy_id}/{group_name}: mc_return shape mismatch")
                require_finite(f"{policy_id}/{group_name}/mc_return", group["mc_return"][:])
            json_row = episode_json[expected_index]
            if (int(json_row["episode_id"]) != expected_index or
                    int(json_row["initial_seed"]) != seeds[expected_index] or
                    bool(json_row["success"]) != bool(success_values[0]) or
                    int(json_row["episode_length"]) != length):
                raise RuntimeError(f"{policy_id}/{group_name}: episodes.json metadata mismatch")
            state_hash = str(group.attrs["initial_state_vector_hash"])
            state_values = group["initial_state/states"][:]
            if sha256_array(state_values) != state_hash:
                raise RuntimeError(f"{policy_id}/{group_name}: initial state vector hash mismatch")
            state_hashes[seeds[expected_index]] = state_hash
            seen_seeds.append(seeds[expected_index])
            successes += int(success_values[0])
            failures += int(not success_values[0])
            transitions += length
    if seen_seeds != seeds:
        raise RuntimeError(f"{policy_id}: seed order mismatch")
    if require_both and (successes == 0 or failures == 0):
        raise RuntimeError(f"{policy_id}: require-both-outcomes requested, got success={successes}, failure={failures}")
    if metadata["total_transitions"] != transitions:
        raise RuntimeError(f"{policy_id}: metadata transition count mismatch")
    return {
        "policy_id": policy_id,
        "episodes": len(seeds),
        "success_episodes": successes,
        "failure_episodes": failures,
        "success_rate": successes / len(seeds),
        "total_transitions": transitions,
        "state_vector_hashes": state_hashes,
        "schema": schema,
    }


def validate(dataset_root, require_both=False):
    dataset_root = Path(dataset_root)
    seed_payload = read_json(dataset_root / "seed_list.json")
    seeds = [int(value) for value in seed_payload["seeds"]]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise RuntimeError("seed_list.json is empty or contains duplicates")
    initial_states_path = dataset_root / "initial_states.hdf5"
    if not initial_states_path.is_file():
        raise FileNotFoundError(initial_states_path)
    initial_hashes = {}
    with h5py.File(initial_states_path, "r") as initial_handle:
        groups = sorted(initial_handle["seeds"].keys())
        if len(groups) != len(seeds):
            raise RuntimeError("initial_states.hdf5 seed count mismatch")
        for episode_index, group_name in enumerate(groups):
            group = initial_handle[f"seeds/{group_name}"]
            if int(group.attrs["initial_seed"]) != seeds[episode_index]:
                raise RuntimeError(f"{group_name}: persisted initial seed mismatch")
            digest = sha256_array(group["states"][:])
            if digest != str(group.attrs["state_vector_hash"]):
                raise RuntimeError(f"{group_name}: persisted initial state hash mismatch")
            initial_hashes[seeds[episode_index]] = digest
    summary_path = dataset_root / "collection_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = read_json(summary_path)
    reports, shared_schema = {}, None
    shared_hashes = None
    for policy_id in VALID_POLICY_IDS:
        report = validate_policy(dataset_root / policy_id, policy_id, seeds,
                                 shared_schema, require_both)
        if shared_schema is None:
            shared_schema = report["schema"]
            shared_hashes = report["state_vector_hashes"]
            if shared_hashes != initial_hashes:
                raise RuntimeError("Policy dataset initial states differ from initial_states.hdf5")
        elif report["state_vector_hashes"] != shared_hashes:
            raise RuntimeError(f"{policy_id}: same-seed initial simulator states differ")
        reports[policy_id] = report
    outcomes = read_json(dataset_root / "same_seed_outcomes.json")
    if len(outcomes) != len(seeds):
        raise RuntimeError("same_seed_outcomes.json count mismatch")
    patterns = Counter(str(row["success_pattern"]) for row in outcomes)
    expected_patterns = summary["same_seed_success_patterns"]
    for index in range(8):
        pattern = format(index, "03b")
        if patterns.get(pattern, 0) != int(expected_patterns[pattern]):
            raise RuntimeError(f"Same-seed pattern count mismatch for {pattern}")
    total_transitions = sum(row["total_transitions"] for row in reports.values())
    if total_transitions != int(summary["total_transitions"]):
        raise RuntimeError("Collection total transition count mismatch")
    public_reports = {
        key: {k: v for k, v in value.items() if k not in ("state_vector_hashes", "schema")}
        for key, value in reports.items()
    }
    return {
        "valid": True,
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "num_seeds": len(seeds),
        "total_episodes": len(seeds) * len(VALID_POLICY_IDS),
        "total_transitions": total_transitions,
        "canonical_schema": shared_schema,
        "same_seed_state_vector_match": True,
        "same_seed_success_patterns": expected_patterns,
        "policies": public_reports,
        "success_and_failure_filter_check": "No success filter was applied; natural outcome counts are reported.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--require-both-outcomes", action="store_true")
    parser.add_argument("--report")
    args = parser.parse_args()
    if h5py is None:
        raise RuntimeError("h5py is required for dataset validation; activate robosuite_npu")
    report = validate(args.dataset_root, args.require_both_outcomes)
    report_path = Path(args.report) if args.report else Path(args.dataset_root) / "validation_report.json"
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"DATASET VALIDATION PASSED: {report_path}")


if __name__ == "__main__":
    main()
