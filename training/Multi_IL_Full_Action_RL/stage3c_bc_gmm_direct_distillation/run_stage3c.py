#!/usr/bin/env python3
"""Run Stage 3C audit, training, held-out screening, and actor selection."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
EVALUATOR = THIS_DIR.parent / "stage3_actor_initialization" / "evaluate_stage3_actor.py"
TRAINER = THIS_DIR / "train_stage3c.py"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(THIS_DIR / "stage3c_config.json"))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def run(command):
    print("Command:", " ".join(str(item) for item in command), flush=True)
    subprocess.run([str(item) for item in command], cwd=REPO_ROOT, check=True)


def run_audit(config, device, run_dir):
    audit_script = REPO_ROOT / config["audit_script"]
    repository_report = REPO_ROOT / config["audit_report"]
    run_report = run_dir / "bc_gmm_sac_compatibility_audit.json"
    run([
        sys.executable, "-u", audit_script,
        "--checkpoint", config["teacher_checkpoint"],
        "--dataset", config["dataset"], "--device", device,
        "--output", run_report,
    ])
    report = read_json(run_report)
    shutil.copy2(run_report, repository_report)
    ordering = report["state_ordering"]
    if not ordering.get("saved_action_direct_distillation_compatible", False):
        raise RuntimeError(f"Audit saved-action distillation compatibility failed: {ordering}")
    if report["bc_gmm"]["action_dim"] != 14 or not report["forward_sanity"]["pass"]:
        raise RuntimeError(f"Audit action mapping sanity failed: {report['forward_sanity']}")
    return report


def checkpoint_validation_mse(path):
    payload = torch.load(path, map_location="cpu")
    return float(payload["validation"]["val_mse"]), int(payload["epoch"])


def seed_shards(seed_start, num_seeds, num_workers):
    worker_count = min(int(num_workers), int(num_seeds))
    quotient, remainder = divmod(int(num_seeds), worker_count)
    shards = []
    cursor = int(seed_start)
    for worker_id in range(worker_count):
        size = quotient + int(worker_id < remainder)
        shards.append((worker_id, cursor, size))
        cursor += size
    return shards


def aggregate_worker_reports(checkpoint, output, worker_reports):
    reports = [read_json(path) for path in worker_reports]
    episodes = sorted(
        [episode for report in reports for episode in report["episodes"]],
        key=lambda row: int(row["initial_seed"]),
    )
    if len(episodes) != 20 or [int(row["initial_seed"]) for row in episodes] != list(range(10080, 10100)):
        raise RuntimeError("Parallel candidate evaluation did not produce exactly seeds 10080..10099")
    successes = sum(int(row["success"]) for row in episodes)

    def count(field):
        return sum(int(row[field]) for row in episodes)

    trash = count("ever_trash_in_bin")
    payload = count("ever_payload_in_bin")
    both = sum(int(row["ever_trash_in_bin"] and row["ever_payload_in_bin"]) for row in episodes)
    combined = {
        "stage": "stage3c_parallel_candidate_evaluation",
        "checkpoint": str(checkpoint),
        "parallel_environment_workers": len(reports),
        "worker_reports": [str(path) for path in worker_reports],
        "requested_episodes": 20,
        "completed_episodes": len(episodes),
        "evaluation_complete": len(episodes) == 20,
        "successes": successes,
        "success_rate": successes / len(episodes),
        "subtask_completion": {
            "trash": {"ever_completed_count": trash, "ever_completed_rate": trash / len(episodes)},
            "payload": {"ever_completed_count": payload, "ever_completed_rate": payload / len(episodes)},
            "both": {"ever_completed_count": both, "ever_completed_rate": both / len(episodes)},
        },
        "mean_partial_progress_score": float(
            sum(float(row["partial_progress_score"]) for row in episodes) / len(episodes)
        ),
        "episodes": episodes,
    }
    write_json(output, combined)
    return combined


def evaluate_candidate_parallel(path, output, evaluation_dir, device, num_workers):
    worker_dir = evaluation_dir / "workers" / path.stem
    worker_dir.mkdir(parents=True, exist_ok=False)
    processes = []
    report_paths = []
    log_streams = []
    try:
        for worker_id, seed_start, seed_count in seed_shards(10080, 20, num_workers):
            report_path = worker_dir / f"worker_{worker_id:02d}.json"
            log_path = worker_dir / f"worker_{worker_id:02d}.log"
            stream = open(log_path, "w", encoding="utf-8")
            command = [
                sys.executable, "-u", str(EVALUATOR), "--checkpoint", str(path),
                "--seed-start", str(seed_start), "--num-seeds", str(seed_count),
                "--device", device, "--deterministic", "--output", str(report_path),
            ]
            print(f"Starting evaluation worker {worker_id:02d}: seeds {seed_start}..{seed_start + seed_count - 1}")
            processes.append((worker_id, command, subprocess.Popen(
                command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT,
            )))
            report_paths.append(report_path)
            log_streams.append(stream)
        failures = []
        for worker_id, command, process in processes:
            return_code = process.wait()
            if return_code:
                failures.append((worker_id, return_code, command))
        if failures:
            raise RuntimeError(f"Parallel candidate evaluation worker failures: {failures}")
    finally:
        for stream in log_streams:
            stream.close()
    return aggregate_worker_reports(path, output, report_paths)


def candidate_paths(checkpoint_dir):
    paths = sorted(checkpoint_dir.glob("bc_gmm_distill_epoch_*.pth"), key=lambda path: checkpoint_validation_mse(path)[1])
    best = checkpoint_dir / "bc_gmm_distill_best_val_mse.pth"
    if best.is_file():
        paths.append(best)
    return paths


def screen_candidates(run_dir, device, num_workers):
    evaluation_dir = run_dir / "candidate_evaluations"
    evaluation_dir.mkdir(exist_ok=False)
    rows = []
    for path in candidate_paths(run_dir / "checkpoints"):
        output = evaluation_dir / f"{path.stem}.json"
        report = evaluate_candidate_parallel(path, output, evaluation_dir, device, num_workers)
        if not report["evaluation_complete"]:
            raise RuntimeError(f"Incomplete candidate evaluation: {output}")
        subtasks = report["subtask_completion"]
        val_mse, epoch = checkpoint_validation_mse(path)
        rows.append({
            "checkpoint": str(path), "checkpoint_name": path.name, "epoch": epoch,
            "successes": int(report["successes"]), "success_rate": float(report["success_rate"]),
            "trash_ever_count": int(subtasks["trash"]["ever_completed_count"]),
            "trash_ever_rate": float(subtasks["trash"]["ever_completed_rate"]),
            "payload_ever_count": int(subtasks["payload"]["ever_completed_count"]),
            "payload_ever_rate": float(subtasks["payload"]["ever_completed_rate"]),
            "both_ever_count": int(subtasks["both"]["ever_completed_count"]),
            "both_ever_rate": float(subtasks["both"]["ever_completed_rate"]),
            "mean_progress_score": float(report["mean_partial_progress_score"]),
            "val_mse": val_mse, "evaluation_json": str(output),
        })
    if not rows:
        raise RuntimeError("No Stage3C candidate checkpoints were saved")
    selected = max(rows, key=lambda row: (
        row["success_rate"], row["payload_ever_rate"], row["both_ever_rate"],
        row["mean_progress_score"], row["trash_ever_rate"], -row["val_mse"],
    ))
    final_path = run_dir / "checkpoints" / "stage3c_bc_gmm_shared_actor_best.pth"
    shutil.copy2(selected["checkpoint"], final_path)
    result = {
        "selection_priority": [
            "closed_loop_success_rate", "payload_ever", "both_ever",
            "mean_progress_score", "trash_ever", "lowest_validation_mse",
        ],
        "heldout_seeds": list(range(10080, 10100)), "candidates": rows,
        "selected_checkpoint": selected["checkpoint"], "selected_epoch": selected["epoch"],
        "selected_metrics": selected,
        "final_actor_checkpoint": str(final_path), "stage4_started": False,
    }
    write_json(run_dir / "stage3c_actor_selection.json", result)
    return result


def main():
    args = parse_args()
    config = read_json(args.config)
    run_id = args.run_id or (("smoke_" if args.smoke_test else "") + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = Path(config["output_root"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"Stage3C run directory: {run_dir}")
    audit = run_audit(config, args.device, run_dir)
    run([
        sys.executable, "-u", TRAINER, "--config", Path(args.config).resolve(),
        "--run-dir", run_dir, "--device", args.device,
        *(("--smoke-test",) if args.smoke_test else ()),
    ])
    if args.smoke_test:
        print("STAGE3C SMOKE TEST PASSED")
        print(f"Audit forward sanity: {audit['forward_sanity']}")
        print(f"Run directory: {run_dir}")
        print("Stopped before closed-loop screening and Stage 4.")
        return
    selection = screen_candidates(
        run_dir, args.device, int(config.get("evaluation_num_envs", 16))
    )
    summary = read_json(run_dir / "training_summary.json")
    summary["status"] = "STAGE3C_COMPLETE"
    summary["audit_forward_sanity"] = audit["forward_sanity"]
    summary["actor_selection"] = selection
    summary["stage4_started"] = False
    write_json(run_dir / "training_summary.json", summary)
    print("=" * 80)
    print("STAGE3C COMPLETE")
    print(f"Selected: {selection['selected_checkpoint']}")
    print(f"Final actor: {selection['final_actor_checkpoint']}")
    print("Stage 4 was not started.")
    print("=" * 80)


if __name__ == "__main__":
    main()
