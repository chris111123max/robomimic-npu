#!/usr/bin/env python3
"""Discover and content-validate Stage 1 rollout datasets under training_runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR))

from dataset_utils import (atomic_json, require_output_outside_readonly_root,
                           scan_candidates, select_formal_datasets, utc_timestamp)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-runs-root",
        default="/data/home/3220251075/lerobot_workspace/training_runs",
    )
    parser.add_argument(
        "--inventory",
        default=str(PROJECT_ROOT / "analysis/stage1_5_dataset_inventory.json"),
    )
    parser.add_argument(
        "--selected-manifest",
        default=str(PROJECT_ROOT / "analysis/stage1_5_selected_datasets.json"),
    )
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument(
        "--compact-schema",
        action="store_true",
        help="Do not embed the complete recursive HDF5 schema in inventory JSON.",
    )
    return parser.parse_args()


def discover(training_runs_root, expected_episodes=100, include_schema=True):
    candidates = scan_candidates(training_runs_root, include_schema=include_schema)
    selection = select_formal_datasets(candidates, expected_episodes=expected_episodes)
    inventory = {
        "generated_at": utc_timestamp(),
        "training_runs_root": str(Path(training_runs_root).resolve()),
        "read_only": True,
        "candidate_file_count": len(candidates),
        "validated_rollout_dataset_count": sum(
            int(row.get("is_rollout_dataset") is True) for row in candidates
        ),
        "datasets": candidates,
        "selection_status": selection["status"],
        "selection_reason": selection["reason"],
    }
    manifest = {
        "generated_at": inventory["generated_at"],
        "training_runs_root": inventory["training_runs_root"],
        "read_only": True,
        **selection,
    }
    return inventory, manifest


def main():
    args = parse_args()
    inventory_path = require_output_outside_readonly_root(
        args.inventory, args.training_runs_root
    )
    manifest_path = require_output_outside_readonly_root(
        args.selected_manifest, args.training_runs_root
    )
    inventory, manifest = discover(
        args.training_runs_root,
        expected_episodes=args.expected_episodes,
        include_schema=not args.compact_schema,
    )
    atomic_json(inventory_path, inventory)
    atomic_json(manifest_path, manifest)
    print("=" * 88)
    print("Stage 1.5 Runtime Dataset Discovery")
    print("=" * 88)
    print("Training runs root :", inventory["training_runs_root"])
    print("Candidate files    :", inventory["candidate_file_count"])
    print("Validated rollouts :", inventory["validated_rollout_dataset_count"])
    print("Selection status   :", manifest["status"])
    print("Selection reason   :", manifest["reason"])
    print("Inventory          :", inventory_path)
    print("Selected manifest  :", manifest_path)
    if manifest["status"] == "selected":
        for policy, path in sorted(manifest["datasets"].items()):
            print(f"  {policy:16s}: {path}")
    else:
        print("Candidate families requiring review:")
        for row in manifest["candidate_families"]:
            print(
                f"  score={row['score']:4d} family={row['family']} "
                f"episodes={row['exact_episode_count']} same_seeds={row['same_seed_sets']}"
            )
        print("No dataset was silently selected. Resolve ambiguity before analysis.")
    print(json.dumps({"status": manifest["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
