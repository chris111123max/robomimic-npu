#!/usr/bin/env python3
"""Summarize measured frozen/active throughput without adding train-time sync."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise RuntimeError(f"No throughput rows in {path}")
    result = {}
    for phase in ("actor_frozen", "actor_active"):
        selected = [row for row in rows if row.get("phase", (
            "actor_active" if int(row["env_steps"]) >= 10000 else "actor_frozen")) == phase]
        result[phase] = {
            "samples": len(selected),
            **{key: (sum(float(row[key]) for row in selected if key in row)
                      / sum(key in row for row in selected)
                      if any(key in row for row in selected) else None)
               for key in ("aggregate_env_steps_per_sec", "critic_updates_per_sec",
                           "actor_updates_per_sec")},
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run-dir", required=True)
    parser.add_argument("--group", choices=("rnn_q", "multi_q"), required=True)
    parser.add_argument("--v3-throughput-jsonl", help="optional matched v3 log for speed ratio")
    args = parser.parse_args()
    path = Path(args.pair_run_dir) / args.group / "throughput_metrics.jsonl"
    report = {"stage": "stage3-v4", "group": args.group,
              "throughput": summarize(path)}
    if args.v3_throughput_jsonl:
        baseline = summarize(args.v3_throughput_jsonl)
        report["v3_baseline"] = baseline
        numerator = report["throughput"]["actor_active"]["aggregate_env_steps_per_sec"]
        denominator = baseline["actor_active"]["aggregate_env_steps_per_sec"]
        report["actor_active_speedup_vs_v3"] = (
            numerator / denominator if numerator is not None and denominator else None)
    output = path.with_name("throughput_benchmark.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **report}, indent=2))


if __name__ == "__main__":
    main()
