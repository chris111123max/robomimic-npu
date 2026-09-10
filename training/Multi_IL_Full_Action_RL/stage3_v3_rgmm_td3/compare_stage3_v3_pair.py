#!/usr/bin/env python3
"""Read-only paired-run fairness and milestone comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run-dir", required=True)
    args = parser.parse_args()
    pair = Path(args.pair_run_dir).resolve()
    fairness = read(pair / "shared" / "pair_fairness.json")
    if not all(fairness[key] for key in (
        "actor_hashes_identical", "environment_seeds_identical",
        "evaluation_seeds_identical", "offline_dataset_identical",
        "replay_settings_identical", "training_schedules_identical")):
        raise RuntimeError("PAIR_FAIRNESS_FAIL")
    result = {"pair_run_dir": str(pair), "fairness": "PASS", "milestones": {}}
    names = set()
    for branch in ("rnn_q", "multi_q"):
        directory = pair / branch / "evaluations"
        names.update(path.name for path in directory.glob("step_*.json"))
    for name in sorted(names):
        row = {}
        for branch in ("rnn_q", "multi_q"):
            path = pair / branch / "evaluations" / name
            row[branch] = ({key: read(path).get(key) for key in
                            ("success_count", "success_rate", "mean_return", "mean_length")}
                           if path.is_file() else None)
        result["milestones"][name] = row
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
