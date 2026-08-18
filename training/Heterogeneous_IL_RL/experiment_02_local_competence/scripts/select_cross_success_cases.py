#!/usr/bin/env python3
"""Select every Experiment 00 RNN/Transformer cross-success case."""

import argparse
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from utils.exp00_reader import RNN, TRANSFORMER, validate_source
from utils.result_utils import atomic_csv, read_json


FIELDS = ["initial_state_id", "environment_seed", "direction", "source_policy", "target_policy",
          "source_success_exp00", "target_success_exp00", "initial_state_path"]


def select(config_path, run_dir, source_run, max_cases=None):
    config = read_json(config_path)
    rows, states, _, _ = validate_source(config, source_run, Path(run_dir) / "source_run_manifest.json")
    state_by_id = {int(x["initial_state_id"]): x for x in states}
    directions = [
        ("RNN_FAIL_TRANSFORMER_SUCCESS", RNN, TRANSFORMER),
        ("TRANSFORMER_FAIL_RNN_SUCCESS", TRANSFORMER, RNN),
    ]
    selected = []
    counts = {}
    for direction, source, target in directions:
        candidates = [row for row in rows if row[source] == 0 and row[target] == 1]
        counts[direction] = len(candidates)
        if max_cases is not None:
            candidates = candidates[:int(max_cases)]  # deterministic smoke-only limit per direction
        for row in candidates:
            sid = int(row["initial_state_id"])
            entry = state_by_id[sid]
            selected.append({
                "initial_state_id": sid, "environment_seed": int(row["seed"]),
                "direction": direction, "source_policy": source, "target_policy": target,
                "source_success_exp00": 0, "target_success_exp00": 1,
                "initial_state_path": str(Path(source_run) / entry["state_file"]),
            })
    atomic_csv(Path(run_dir) / "selected_cases.csv", FIELDS, selected)
    print(f"[select] RNN fail / Transformer success = {counts['RNN_FAIL_TRANSFORMER_SUCCESS']}")
    print(f"[select] Transformer fail / RNN success = {counts['TRANSFORMER_FAIL_RNN_SUCCESS']}")
    print(f"[select] selected for this run = {len(selected)}" +
          (f" (smoke limit {max_cases} per direction)" if max_cases is not None else ""))
    if max_cases is None and len(selected) != 58:
        raise RuntimeError(f"Formal selection must contain all 58 cases, got {len(selected)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    select(args.config, args.run_dir, args.source_run, args.max_cases)


if __name__ == "__main__":
    main()
