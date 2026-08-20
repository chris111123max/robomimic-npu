#!/usr/bin/env python3
"""Analyze selected Stage 1 datasets without modifying training_runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR))

from dataset_utils import (HDF5RolloutReader, POLICIES, atomic_json,
                           final_progress, format_schema_text,
                           inspect_candidate, iter_dataset_chunks,
                           json_scalar, numeric_counts, observation_datasets,
                           observation_schema, read_json,
                           require_output_outside_readonly_root,
                           scalar_from_episode, utc_timestamp, write_csv)
from discover_stage1_data import discover


DISPLAY_NAMES = {
    "bc_rnn": "RNN",
    "bc_transformer": "Transformer",
    "bc_gmm": "GMM",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-runs-root",
        default="/data/home/3220251075/lerobot_workspace/training_runs",
    )
    parser.add_argument(
        "--selected-manifest",
        default=str(PROJECT_ROOT / "analysis/stage1_5_selected_datasets.json"),
    )
    parser.add_argument(
        "--inventory",
        default=str(PROJECT_ROOT / "analysis/stage1_5_dataset_inventory.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "analysis"),
    )
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--rediscover", action="store_true")
    return parser.parse_args()


def finite_counts_for_observations(group, fields):
    result = {"obs": {"nan": 0, "inf": 0}, "next_obs": {"nan": 0, "inf": 0}}
    for concept in ("obs", "next_obs"):
        for dataset in observation_datasets(group, fields[concept]).values():
            nan_count, inf_count = numeric_counts(dataset)
            result[concept]["nan"] += nan_count
            result[concept]["inf"] += inf_count
    return result


def reward_statistics(dataset, unique_values, unique_limit=10000):
    zero = positive = negative = nonzero = nan_count = inf_count = 0
    nonzero_timesteps = []
    offset = 0
    for values in iter_dataset_chunks(dataset):
        flat = np.asarray(values).reshape(-1)
        nan_count += int(np.isnan(flat).sum())
        inf_count += int(np.isinf(flat).sum())
        finite = flat[np.isfinite(flat)]
        zero += int(np.count_nonzero(finite == 0))
        positive += int(np.count_nonzero(finite > 0))
        negative += int(np.count_nonzero(finite < 0))
        nonzero_indices = np.flatnonzero(np.isfinite(flat) & (flat != 0))
        nonzero += int(nonzero_indices.size)
        nonzero_timesteps.extend((nonzero_indices + offset).astype(int).tolist())
        if len(unique_values) <= unique_limit:
            unique_values.update(float(value) for value in np.unique(finite))
        offset += flat.size
    return {
        "zero": zero,
        "positive": positive,
        "negative": negative,
        "nonzero": nonzero,
        "nan": nan_count,
        "inf": inf_count,
        "nonzero_timesteps": nonzero_timesteps,
    }


def analyze_policy(path, expected_policy):
    episode_rows = []
    lengths, returns = [], []
    success_count = failure_count = 0
    missing_success = missing_seed = 0
    seeds = []
    reward_unique = set()
    reward_zero = reward_positive = reward_negative = reward_nonzero = 0
    success_reward_timesteps = []
    failure_all_zero = failure_with_nonzero = 0
    action_dims = set()
    schema = None
    nan_counts = defaultdict(int)
    inf_counts = defaultdict(int)
    return_consistency_anomalies = []
    progress_counts = Counter()
    progress_anomalies = []

    with HDF5RolloutReader(path) as reader:
        if reader.policy != expected_policy:
            raise RuntimeError(
                f"Selected policy mismatch: manifest={expected_policy}, "
                f"content={reader.policy}, path={path}"
            )
        progress_mapping = reader.progress_mapping()
        for episode_index, (group_path, group, fields) in enumerate(reader.episodes()):
            current_schema = observation_schema(group, fields)
            if schema is None:
                schema = current_schema
            elif current_schema != schema:
                raise RuntimeError(f"Observation schema changed at {path}:{group_path}")
            action_dataset = group[fields["actions"]]
            reward_dataset = group[fields["rewards"]]
            length = int(action_dataset.shape[0])
            if reward_dataset.shape[0] != length:
                raise RuntimeError(f"Reward/action length mismatch at {path}:{group_path}")
            action_dim = int(np.prod(action_dataset.shape[1:])) if len(action_dataset.shape) > 1 else 1
            action_dims.add(action_dim)
            action_nan, action_inf = numeric_counts(action_dataset)
            nan_counts["actions"] += action_nan
            inf_counts["actions"] += action_inf
            observation_finite = finite_counts_for_observations(group, fields)
            for concept in ("obs", "next_obs"):
                nan_counts[concept] += observation_finite[concept]["nan"]
                inf_counts[concept] += observation_finite[concept]["inf"]
            reward_stats = reward_statistics(reward_dataset, reward_unique)
            nan_counts["rewards"] += reward_stats["nan"]
            inf_counts["rewards"] += reward_stats["inf"]
            reward_zero += reward_stats["zero"]
            reward_positive += reward_stats["positive"]
            reward_negative += reward_stats["negative"]
            reward_nonzero += reward_stats["nonzero"]
            episode_return = 0.0
            for values in iter_dataset_chunks(reward_dataset):
                episode_return += float(np.sum(values, dtype=np.float64))
            if "episode_return" in group:
                stored_return = float(np.asarray(group["episode_return"][0]))
                if not np.isclose(stored_return, episode_return, rtol=0.0, atol=1e-8):
                    return_consistency_anomalies.append({
                        "episode": group_path,
                        "stored": stored_return,
                        "computed": episode_return,
                    })
            seed_value = scalar_from_episode(group, fields, "seed")
            success_value = scalar_from_episode(group, fields, "success")
            seed = None if seed_value is None else int(seed_value)
            success = None if success_value is None else bool(success_value)
            if seed is None:
                missing_seed += 1
            else:
                seeds.append(seed)
            if success is None:
                missing_success += 1
            elif success:
                success_count += 1
                success_reward_timesteps.append({
                    "seed": seed,
                    "episode": group_path,
                    "timesteps": reward_stats["nonzero_timesteps"],
                })
            else:
                failure_count += 1
                if reward_stats["nonzero"] == 0:
                    failure_all_zero += 1
                else:
                    failure_with_nonzero += 1
            progress = final_progress(group, fields, progress_mapping)
            progress_class = None
            if progress is not None:
                payload = int(progress["payload_in_target_bin"])
                trash = int(progress["trash_in_trash_bin"])
                progress_class = f"payload={payload},trash={trash}"
                if success is False:
                    progress_counts[progress_class] += 1
                    if payload == 1 and trash == 1:
                        progress_anomalies.append({
                            "seed": seed,
                            "episode": group_path,
                            "episode_success": False,
                            "payload_in_target_bin": True,
                            "trash_in_trash_bin": True,
                        })
            lengths.append(length)
            returns.append(episode_return)
            episode_rows.append({
                "episode_index": episode_index,
                "episode_path": group_path,
                "seed": seed,
                "success": success,
                "length": length,
                "return": episode_return,
                "reward_nonzero_count": reward_stats["nonzero"],
                "reward_nonzero_timesteps": reward_stats["nonzero_timesteps"],
                "final_progress": progress,
                "failure_progress_class": progress_class if success is False else None,
                "initial_state_vector_hash": (
                    None if "initial_state_vector_hash" not in group.attrs
                    else str(json_scalar(group.attrs["initial_state_vector_hash"]))
                ),
                "initial_state_hash": (
                    None if "initial_state_hash" not in group.attrs
                    else str(json_scalar(group.attrs["initial_state_hash"]))
                ),
                "initial_observation_hash": (
                    None if "initial_observation_hash" not in group.attrs
                    else str(json_scalar(group.attrs["initial_observation_hash"]))
                ),
            })

        root_attributes = {
            key: json_scalar(value)
            for key, value in reader.handle.attrs.items()
        }

    if len(action_dims) != 1:
        raise RuntimeError(f"Action dimension changed across episodes: {sorted(action_dims)}")
    seed_counter = Counter(seeds)
    observation_keys = sorted(schema["obs"] if schema else [])
    observation_dimension = sum(
        int(np.prod(value["shape"])) for value in (schema or {"obs": {}})["obs"].values()
    )
    unique_values = sorted(reward_unique)
    unique_overflow = len(unique_values) > 10000
    if unique_overflow:
        unique_values = unique_values[:10000]
    summary = {
        "policy": expected_policy,
        "display_name": DISPLAY_NAMES[expected_policy],
        "path": str(Path(path).resolve()),
        "episodes": len(episode_rows),
        "unique_seeds": len(seed_counter),
        "seeds": sorted(seed_counter),
        "duplicate_seeds": sorted(seed for seed, count in seed_counter.items() if count > 1),
        "missing_seed_episodes": missing_seed,
        "success_episodes": success_count,
        "failure_episodes": failure_count,
        "missing_success_episodes": missing_success,
        "success_rate": None if missing_success else success_count / len(episode_rows),
        "total_transitions": int(sum(lengths)),
        "mean_episode_length": float(np.mean(lengths)),
        "median_episode_length": float(np.median(lengths)),
        "min_episode_length": int(min(lengths)),
        "max_episode_length": int(max(lengths)),
        "mean_episode_return": float(np.mean(returns)),
        "min_episode_return": float(min(returns)),
        "max_episode_return": float(max(returns)),
        "reward_unique_values": unique_values,
        "reward_unique_values_truncated": unique_overflow,
        "reward_zero_transition_count": reward_zero,
        "reward_positive_transition_count": reward_positive,
        "reward_negative_transition_count": reward_negative,
        "reward_nonzero_transition_count": reward_nonzero,
        "success_episode_nonzero_reward_timesteps": success_reward_timesteps,
        "failure_episodes_all_reward_zero": failure_all_zero,
        "failure_episodes_with_nonzero_reward": failure_with_nonzero,
        "action_dimension": next(iter(action_dims)),
        "observation_dimension": observation_dimension,
        "observation_keys": observation_keys,
        "observation_schema": schema,
        "nan_count": dict(nan_counts),
        "inf_count": dict(inf_counts),
        "nan_total": int(sum(nan_counts.values())),
        "inf_total": int(sum(inf_counts.values())),
        "progress_mapping": progress_mapping,
        "failure_final_progress_counts": dict(sorted(progress_counts.items())),
        "progress_consistency_anomalies": progress_anomalies,
        "episode_return_consistency_anomalies": return_consistency_anomalies,
        "root_attributes": root_attributes,
        "missing_fields": [
            name for name, missing in (
                ("seed", missing_seed > 0),
                ("episode_success", missing_success > 0),
                ("payload_in_target_bin", progress_mapping is None),
                ("trash_in_trash_bin", progress_mapping is None),
            ) if missing
        ],
    }
    return summary, episode_rows


def same_seed_analysis(policy_summaries, episode_rows, expected_episodes):
    seed_counters = {
        policy: Counter(row["seed"] for row in rows if row["seed"] is not None)
        for policy, rows in episode_rows.items()
    }
    seed_sets = {policy: set(counter) for policy, counter in seed_counters.items()}
    union = set().union(*seed_sets.values())
    intersection = set.intersection(*seed_sets.values())
    duplicates = {
        policy: sorted(seed for seed, count in counter.items() if count > 1)
        for policy, counter in seed_counters.items()
    }
    missing = {policy: sorted(union - seeds) for policy, seeds in seed_sets.items()}
    exclusive = {
        policy: sorted(seeds - set().union(*(seed_sets[other] for other in POLICIES if other != policy)))
        for policy, seeds in seed_sets.items()
    }
    set_equality = all(seed_sets[policy] == seed_sets[POLICIES[0]] for policy in POLICIES[1:])
    complete_counts = all(len(seed_sets[policy]) == expected_episodes for policy in POLICIES)
    no_duplicates = all(not values for values in duplicates.values())
    no_missing_metadata = all(policy_summaries[policy]["missing_seed_episodes"] == 0 for policy in POLICIES)
    passed = set_equality and complete_counts and no_duplicates and no_missing_metadata
    return {
        "status": "PASS" if passed else "FAIL",
        "expected_seed_count": expected_episodes,
        "seed_counts": {policy: len(values) for policy, values in seed_sets.items()},
        "intersection_seed_count": len(intersection),
        "intersection_seeds": sorted(intersection),
        "union_seed_count": len(union),
        "rnn_only_seeds": exclusive["bc_rnn"],
        "transformer_only_seeds": exclusive["bc_transformer"],
        "gmm_only_seeds": exclusive["bc_gmm"],
        "missing_seeds": missing,
        "duplicate_seeds": duplicates,
        "set_equality": set_equality,
    }


def outcome_rows(episodes, same_seed_report):
    by_policy = {
        policy: {
            row["seed"]: row for row in rows
            if row["seed"] is not None
            and sum(int(other["seed"] == row["seed"]) for other in rows) == 1
        }
        for policy, rows in episodes.items()
    }
    rows, patterns = [], Counter()
    for seed in same_seed_report["intersection_seeds"]:
        selected = {policy: by_policy[policy].get(seed) for policy in POLICIES}
        if any(value is None or value["success"] is None for value in selected.values()):
            continue
        pattern = "".join(str(int(selected[policy]["success"])) for policy in POLICIES)
        patterns[pattern] += 1
        rows.append({
            "seed": seed,
            "rnn_success": int(selected["bc_rnn"]["success"]),
            "transformer_success": int(selected["bc_transformer"]["success"]),
            "gmm_success": int(selected["bc_gmm"]["success"]),
            "rnn_length": selected["bc_rnn"]["length"],
            "transformer_length": selected["bc_transformer"]["length"],
            "gmm_length": selected["bc_gmm"]["length"],
            "rnn_return": selected["bc_rnn"]["return"],
            "transformer_return": selected["bc_transformer"]["return"],
            "gmm_return": selected["bc_gmm"]["return"],
            "pattern": pattern,
        })
    pattern_rows = [
        {"pattern": format(index, "03b"), "count": patterns.get(format(index, "03b"), 0)}
        for index in range(8)
    ]
    return rows, pattern_rows


def same_initial_state_analysis(episode_rows, expected_episodes, seed_start=10000):
    """Compare persisted simulator-state hashes; vector hash is authoritative."""
    by_policy = {}
    duplicate_seeds = {}
    for policy in POLICIES:
        grouped = defaultdict(list)
        for row in episode_rows[policy]:
            if row["seed"] is not None:
                grouped[row["seed"]].append(row)
        duplicate_seeds[policy] = sorted(
            seed for seed, rows in grouped.items() if len(rows) != 1
        )
        by_policy[policy] = {
            seed: rows[0] for seed, rows in grouped.items() if len(rows) == 1
        }

    common_seeds = sorted(set.intersection(*(
        set(by_policy[policy]) for policy in POLICIES
    )))
    expected_seeds = set(range(seed_start, seed_start + expected_episodes))
    mismatch_details = []
    auxiliary_mismatches = {
        "initial_state_hash": [],
        "initial_observation_hash": [],
    }
    for seed in common_seeds:
        rows = {policy: by_policy[policy][seed] for policy in POLICIES}
        vector_hashes = {
            policy: rows[policy]["initial_state_vector_hash"] for policy in POLICIES
        }
        values = list(vector_hashes.values())
        if any(value in (None, "") for value in values) or len(set(values)) != 1:
            mismatch_details.append({
                "seed": seed,
                "hashes": vector_hashes,
                "initial_state_hashes": {
                    policy: rows[policy]["initial_state_hash"] for policy in POLICIES
                },
                "initial_observation_hashes": {
                    policy: rows[policy]["initial_observation_hash"] for policy in POLICIES
                },
            })
        for field in auxiliary_mismatches:
            hashes = {policy: rows[policy][field] for policy in POLICIES}
            if any(value in (None, "") for value in hashes.values()) or len(set(hashes.values())) != 1:
                auxiliary_mismatches[field].append({"seed": seed, "hashes": hashes})

    mismatched_seeds = [row["seed"] for row in mismatch_details]
    missing_expected_seeds = {
        policy: sorted(expected_seeds - set(by_policy[policy])) for policy in POLICIES
    }
    unexpected_common_seeds = sorted(set(common_seeds) - expected_seeds)
    passed = (
        set(common_seeds) == expected_seeds
        and len(common_seeds) == expected_episodes
        and not mismatched_seeds
        and all(not seeds for seeds in duplicate_seeds.values())
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "primary_hash_field": "initial_state_vector_hash",
        "expected_seed_start": seed_start,
        "expected_seed_end": seed_start + expected_episodes - 1,
        "expected_seed_count": expected_episodes,
        "common_seed_count": len(common_seeds),
        "common_seeds": common_seeds,
        "matched_seed_count": len(common_seeds) - len(mismatched_seeds),
        "mismatched_seed_count": len(mismatched_seeds),
        "mismatched_seeds": mismatched_seeds,
        "mismatch_hashes": mismatch_details,
        "missing_expected_seeds": missing_expected_seeds,
        "unexpected_common_seeds": unexpected_common_seeds,
        "duplicate_seeds": duplicate_seeds,
        "auxiliary_hash_mismatches": auxiliary_mismatches,
    }


def summary_csv_rows(summaries):
    return [
        {
            "policy": DISPLAY_NAMES[policy],
            "policy_id": policy,
            "episodes": row["episodes"],
            "unique_seeds": row["unique_seeds"],
            "success_episodes": row["success_episodes"],
            "failure_episodes": row["failure_episodes"],
            "success_rate": row["success_rate"],
            "total_transitions": row["total_transitions"],
            "mean_episode_length": row["mean_episode_length"],
            "median_episode_length": row["median_episode_length"],
            "min_episode_length": row["min_episode_length"],
            "max_episode_length": row["max_episode_length"],
            "mean_episode_return": row["mean_episode_return"],
            "min_episode_return": row["min_episode_return"],
            "max_episode_return": row["max_episode_return"],
            "action_dimension": row["action_dimension"],
            "observation_dimension": row["observation_dimension"],
            "reward_zero_transitions": row["reward_zero_transition_count"],
            "reward_positive_transitions": row["reward_positive_transition_count"],
            "reward_negative_transitions": row["reward_negative_transition_count"],
            "reward_nonzero_transitions": row["reward_nonzero_transition_count"],
            "nan_total": row["nan_total"],
            "inf_total": row["inf_total"],
        }
        for policy, row in ((policy, summaries[policy]) for policy in POLICIES)
    ]


def ensure_selected_manifest(args):
    manifest_path = require_output_outside_readonly_root(
        args.selected_manifest, args.training_runs_root
    )
    inventory_path = require_output_outside_readonly_root(
        args.inventory, args.training_runs_root
    )
    if args.rediscover or not manifest_path.is_file():
        inventory, manifest = discover(
            args.training_runs_root,
            expected_episodes=args.expected_episodes,
            include_schema=True,
        )
        atomic_json(inventory_path, inventory)
        atomic_json(manifest_path, manifest)
    manifest = read_json(manifest_path)
    if manifest.get("status") != "selected":
        raise RuntimeError(
            f"Dataset selection status is {manifest.get('status')!r}: {manifest.get('reason')}. "
            f"Inspect {manifest_path}; analysis will not guess."
        )
    root = Path(args.training_runs_root).resolve()
    for policy in POLICIES:
        if policy not in manifest.get("datasets", {}):
            raise RuntimeError(f"Selected manifest missing_field dataset for {policy}")
        selected = Path(manifest["datasets"][policy]).resolve()
        if os.path.commonpath((str(root), str(selected))) != str(root):
            raise RuntimeError(f"Selected path is outside training_runs root: {selected}")
    return manifest


def print_terminal_summary(summaries, same_seed, initial_state_check, pattern_rows):
    print("=" * 88)
    print("Stage 1.5 Dataset Audit")
    print("=" * 88)
    print(f"{'Policy':16s} {'Episodes':>8s} {'Seeds':>7s} {'Success':>8s} {'Failure':>8s} {'SR':>8s} {'Transitions':>12s} {'MeanLen':>9s}")
    for policy in POLICIES:
        row = summaries[policy]
        success_rate = "missing" if row["success_rate"] is None else f"{row['success_rate']:.3f}"
        print(
            f"{DISPLAY_NAMES[policy]:16s} {row['episodes']:8d} {row['unique_seeds']:7d} "
            f"{row['success_episodes']:8d} {row['failure_episodes']:8d} {success_rate:>8s} "
            f"{row['total_transitions']:12d} {row['mean_episode_length']:9.1f}"
        )
    print("\nSame-seed check:")
    print("SAME-SEED CHECK:", same_seed["status"])
    print("Intersection:", same_seed["intersection_seed_count"])
    print("\nOutcome patterns (RNN Transformer GMM):")
    pattern_map = {row["pattern"]: row["count"] for row in pattern_rows}
    for pattern in (format(index, "03b") for index in range(8)):
        print(f"{pattern}: {pattern_map[pattern]}")
    print("\nKey heterogeneous patterns:")
    print("100 (RNN success, Transformer failure, GMM failure):", pattern_map["100"])
    print("110 (RNN success, Transformer success, GMM failure):", pattern_map["110"])
    print("101 (RNN success, Transformer failure, GMM success):", pattern_map["101"])
    print("\nSparse reward:")
    for policy in POLICIES:
        row = summaries[policy]
        print(
            f"{DISPLAY_NAMES[policy]}: zero transitions={row['reward_zero_transition_count']} "
            f"positive transitions={row['reward_positive_transition_count']} "
            f"nonzero transitions={row['reward_nonzero_transition_count']}"
        )
    print("\nNaN / Inf:")
    for policy in POLICIES:
        row = summaries[policy]
        print(f"{DISPLAY_NAMES[policy]}: NaN={row['nan_total']} Inf={row['inf_total']}")
    print("\nSame initial simulator state:")
    print("Common seeds   :", initial_state_check["common_seed_count"])
    print("Matched seeds  :", initial_state_check["matched_seed_count"])
    print("Mismatched     :", initial_state_check["mismatched_seed_count"])
    print("Mismatch seeds :", initial_state_check["mismatched_seeds"])
    for mismatch in initial_state_check["mismatch_hashes"]:
        print(f"seed={mismatch['seed']} initial_state_vector_hash={mismatch['hashes']}")
    print("=" * 88)


def main():
    args = parse_args()
    output_dir = require_output_outside_readonly_root(
        args.output_dir, args.training_runs_root
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ensure_selected_manifest(args)
    summaries, episodes = {}, {}
    schema_candidates = []
    for policy in POLICIES:
        path = manifest["datasets"][policy]
        summary, rows = analyze_policy(path, policy)
        summaries[policy] = summary
        episodes[policy] = rows
        schema_candidates.append(inspect_candidate(path, include_schema=True))
    same_seed = same_seed_analysis(summaries, episodes, args.expected_episodes)
    initial_state_check = same_initial_state_analysis(
        episodes, args.expected_episodes, seed_start=10000
    )
    same_seed_rows, pattern_rows = outcome_rows(episodes, same_seed)
    report = {
        "generated_at": utc_timestamp(),
        "stage": "1.5_dataset_audit",
        "training_runs_read_only": True,
        "selected_manifest": str(Path(args.selected_manifest).resolve()),
        "selected_family": manifest.get("selected_family"),
        "policies": summaries,
        "same_seed_check": same_seed,
        "same_initial_state_check": initial_state_check,
        "same_seed_pattern_summary": {row["pattern"]: row["count"] for row in pattern_rows},
        "key_patterns": {
            "100_rnn_only_success": next(row["count"] for row in pattern_rows if row["pattern"] == "100"),
            "110_rnn_transformer_success": next(row["count"] for row in pattern_rows if row["pattern"] == "110"),
            "101_rnn_gmm_success": next(row["count"] for row in pattern_rows if row["pattern"] == "101"),
        },
        "stage2_started": False,
    }
    atomic_json(output_dir / "stage1_5_dataset_report.json", report)
    atomic_json(output_dir / "stage1_5_initial_state_check.json", initial_state_check)
    write_csv(
        output_dir / "stage1_5_policy_summary.csv",
        list(summary_csv_rows(summaries)[0].keys()),
        summary_csv_rows(summaries),
    )
    write_csv(
        output_dir / "same_seed_outcomes.csv",
        [
            "seed", "rnn_success", "transformer_success", "gmm_success",
            "rnn_length", "transformer_length", "gmm_length",
            "rnn_return", "transformer_return", "gmm_return", "pattern",
        ],
        same_seed_rows,
    )
    write_csv(
        output_dir / "same_seed_pattern_summary.csv",
        ["pattern", "count"],
        pattern_rows,
    )
    (output_dir / "stage1_5_schema_report.txt").write_text(
        format_schema_text(schema_candidates), encoding="utf-8"
    )
    print_terminal_summary(summaries, same_seed, initial_state_check, pattern_rows)
    print("Report directory:", output_dir.resolve())
    print(f"SAME INITIAL STATE VECTOR CHECK: {initial_state_check['status']}")


if __name__ == "__main__":
    main()
