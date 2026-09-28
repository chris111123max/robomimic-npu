#!/usr/bin/env python3
"""Compare prior 2Q moving-mean 10K diagnostics with 2Q random-one 10K.

This script never trains. It only reads two diagnostic JSON files produced by
test_stage3_step0_bootstrap_feedback_causal.py and writes a compact comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mean-json", required=True)
    parser.add_argument("--random-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--milestone", type=int, default=10000)
    return parser.parse_args()


def load(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    if payload.get("status") != "PASS":
        raise RuntimeError(f"{path}: diagnostic status is not PASS")
    return path, payload


def row_at(result, update):
    rows = {int(row["update"]): row for row in result["trajectory"]}
    if int(update) not in rows:
        raise RuntimeError(
            f"{result['branch']}: missing requested milestone {update}"
        )
    return rows[int(update)]


def metric_snapshot(row, dataset):
    m = row["datasets"][dataset]
    return {
        "spearman_qmin_mc": float(m["spearman_qmin_mc"]),
        "qmin_mae": float(m["qmin_minus_mc"]["mae"]),
        "qmin_bias": float(m["qmin_minus_mc"]["mean"]),
        "future_qmin_bias": float(
            m["behavior_future_qmin_minus_mc_next"]["mean"]
        ),
        "training_target_bias": float(
            m["training_target_minus_mc"]["mean"]
        ),
    }


def better(metric, left, right):
    if metric == "spearman_qmin_mc":
        return "random_one" if right > left else "moving_mean" if left > right else "tie"
    if metric == "qmin_mae":
        return "random_one" if right < left else "moving_mean" if left < right else "tie"
    if metric in ("qmin_bias", "future_qmin_bias", "training_target_bias"):
        la, ra = abs(left), abs(right)
        return "random_one" if ra < la else "moving_mean" if la < ra else "tie"
    raise KeyError(metric)


def main():
    args = arguments()
    mean_path, mean = load(args.mean_json)
    random_path, random = load(args.random_json)

    if mean.get("stage3_run_dir") != random.get("stage3_run_dir"):
        raise RuntimeError("mean/random diagnostics use different Stage3 runs")

    report = {
        "status": "RUNNING",
        "mean_json": str(mean_path),
        "random_json": str(random_path),
        "stage3_run_dir": mean["stage3_run_dir"],
        "milestone": int(args.milestone),
        "groups": {},
        "selection_rule": {
            "primary_dataset": "late",
            "metrics": {
                "spearman_qmin_mc": "higher_is_better",
                "qmin_mae": "lower_is_better",
                "qmin_bias": "smaller_absolute_bias_is_better",
                "future_qmin_bias": "smaller_absolute_bias_is_better",
                "training_target_bias": "smaller_absolute_bias_is_better",
            },
            "note": (
                "No automatic production winner is forced here. "
                "The report exposes metric wins and old-data guardrails "
                "for final human selection."
            ),
        },
    }

    csv_rows = []
    for group in ("multi_q", "rnn_q"):
        mean_branch = mean["groups"][group]["branches"].get("moving_mean")
        random_branch = random["groups"][group]["branches"].get("moving_random_one")
        if mean_branch is None or random_branch is None:
            raise RuntimeError(f"{group}: required branch missing")

        if (
            mean_branch["initial_hashes"]["online_hash"]
            != random_branch["initial_hashes"]["online_hash"]
        ):
            raise RuntimeError(f"{group}: online step0 hash mismatch")
        if (
            mean_branch["initial_hashes"]["target_hash"]
            != random_branch["initial_hashes"]["target_hash"]
        ):
            raise RuntimeError(f"{group}: target step0 hash mismatch")
        if (
            mean_branch["initial_hashes"]["actor_hash"]
            != random_branch["initial_hashes"]["actor_hash"]
        ):
            raise RuntimeError(f"{group}: actor step0 hash mismatch")

        mean_row = row_at(mean_branch, args.milestone)
        random_row = row_at(random_branch, args.milestone)

        group_report = {
            "same_step0_online_hash": True,
            "same_step0_target_hash": True,
            "same_step0_actor_hash": True,
            "random_one_selector": random_branch.get("random_one_selector"),
            "datasets": {},
        }

        for dataset in ("late", "old"):
            left = metric_snapshot(mean_row, dataset)
            right = metric_snapshot(random_row, dataset)
            comparison = {
                key: {
                    "moving_mean": left[key],
                    "random_one": right[key],
                    "delta_random_minus_mean": right[key] - left[key],
                    "better": better(key, left[key], right[key]),
                }
                for key in left
            }
            wins = {"moving_mean": 0, "random_one": 0, "tie": 0}
            for item in comparison.values():
                wins[item["better"]] += 1
            group_report["datasets"][dataset] = {
                "metrics": comparison,
                "metric_wins": wins,
            }
            for key, item in comparison.items():
                csv_rows.append({
                    "group": group,
                    "dataset": dataset,
                    "milestone": int(args.milestone),
                    "metric": key,
                    "moving_mean": item["moving_mean"],
                    "random_one": item["random_one"],
                    "delta_random_minus_mean": item["delta_random_minus_mean"],
                    "better": item["better"],
                })

        report["groups"][group] = group_report

    report["status"] = "PASS"
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "mean_vs_random_one_10k.json"
    csv_path = out / "mean_vs_random_one_10k.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[STATUS] {report['status']}")
    print(f"[JSON] {json_path}")
    print(f"[CSV] {csv_path}")


if __name__ == "__main__":
    main()
