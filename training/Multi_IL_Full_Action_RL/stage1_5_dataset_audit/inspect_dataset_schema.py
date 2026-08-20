#!/usr/bin/env python3
"""Write a human-readable schema report for discovered or selected datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
sys.path.insert(0, str(MODULE_DIR))

from dataset_utils import (format_schema_text, inspect_candidate, read_json,
                           require_output_outside_readonly_root)


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
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "analysis/stage1_5_schema_report.txt"),
    )
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--all-candidates", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = require_output_outside_readonly_root(args.output, args.training_runs_root)
    if args.path:
        candidates = [inspect_candidate(path, include_schema=True) for path in args.path]
    elif args.all_candidates:
        inventory = read_json(args.inventory)
        candidates = [
            inspect_candidate(row["path"], include_schema=True)
            for row in inventory["datasets"]
        ]
    else:
        manifest = read_json(args.selected_manifest)
        if manifest.get("status") != "selected":
            raise RuntimeError(
                f"Selected manifest status is {manifest.get('status')!r}; "
                "use --path or resolve discovery ambiguity"
            )
        candidates = [
            inspect_candidate(path, include_schema=True)
            for _, path in sorted(manifest["datasets"].items())
        ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_schema_text(candidates), encoding="utf-8")
    print(f"SCHEMA REPORT WRITTEN: {output.resolve()}")


if __name__ == "__main__":
    main()
