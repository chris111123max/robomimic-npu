#!/usr/bin/env python3
"""Orchestrate Stage 3B smoke checks or the explicitly requested full pipeline."""

from __future__ import annotations

import argparse
import copy
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from stage3b_common import atomic_json, read_json


THIS_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(THIS_DIR / "stage3b_config.json"))
    parser.add_argument(
        "--stage", choices=("smoke", "bootstrap", "run-all", "select"), required=True
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()


def run(command, log_path=None):
    print("Command:", " ".join(str(item) for item in command))
    if log_path is None:
        subprocess.run([str(item) for item in command], check=True)
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            [str(item) for item in command], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_stream.write(line)
            log_stream.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, [str(item) for item in command])


def evaluator_command(checkpoint, seed_start, num_seeds, output, device):
    return [
        sys.executable, "-u", THIS_DIR / "evaluate_stage3_actor.py",
        "--checkpoint", checkpoint, "--seed-start", seed_start,
        "--num-seeds", num_seeds, "--deterministic", "--device", device,
        "--output", output,
    ]


def evaluation_metrics(path):
    report = read_json(path)
    subtasks = report["subtask_completion"]
    return {
        "path": str(path),
        "success_rate": report["success_rate"],
        "trash_ever_rate": subtasks["trash"]["ever_completed_rate"],
        "trash_final_rate": subtasks["trash"]["completed_at_end_rate"],
        "payload_ever_rate": subtasks["payload"]["ever_completed_rate"],
        "payload_final_rate": subtasks["payload"]["completed_at_end_rate"],
        "both_ever_rate": subtasks["both"]["ever_completed_rate"],
        "both_final_rate": subtasks["both"]["completed_at_end_rate"],
        "progress_histogram": report["progress_histogram"],
        "mean_progress_score": report["mean_partial_progress_score"],
        "failure_progress": report["failure_progress"],
    }


def select_actor(config, run_dir, candidates):
    rows = []
    for round_id, checkpoint, evaluation, val_mse in candidates:
        metrics = evaluation_metrics(evaluation)
        rows.append({
            "round": round_id, "source_checkpoint": str(checkpoint),
            "imitation_val_mse": val_mse, **metrics,
        })
    selected = max(rows, key=lambda row: (
        row["success_rate"], row["payload_ever_rate"], row["both_ever_rate"],
        row["mean_progress_score"], row["trash_ever_rate"], -row["imitation_val_mse"],
    ))
    final_checkpoint = run_dir / "checkpoints" / "stage3b_shared_actor_best.pth"
    shutil.copy2(selected["source_checkpoint"], final_checkpoint)
    selection = {
        "selection_priority": [
            "success_rate", "payload_ever_rate", "both_ever_rate",
            "mean_progress_score", "trash_ever_rate", "lowest_imitation_val_mse",
        ],
        "selected_round": selected["round"],
        "success_rate_heldout20": selected["success_rate"],
        "trash_ever_rate": selected["trash_ever_rate"],
        "payload_ever_rate": selected["payload_ever_rate"],
        "both_ever_rate": selected["both_ever_rate"],
        "mean_progress_score": selected["mean_progress_score"],
        "imitation_val_mse": selected["imitation_val_mse"],
        "source_checkpoint": selected["source_checkpoint"],
        "final_checkpoint": str(final_checkpoint),
        "all_candidates": rows,
        "stage4_started": False,
    }
    atomic_json(run_dir / "stage3b_shared_actor_selection.json", selection)
    return selection, final_checkpoint, rows


def round_candidates(config, run_dir):
    initial_checkpoint = run_dir / "checkpoints" / "round0_best.pth"
    if not initial_checkpoint.is_file():
        raise RuntimeError(f"Stage 3B Round-0 checkpoint is missing: {initial_checkpoint}")
    initial_payload = torch.load(initial_checkpoint, map_location="cpu")
    initial_val_mse = initial_payload.get("best_val_mse")
    if initial_val_mse is None:
        validation = initial_payload.get("validation", {})
        initial_val_mse = validation.get("action_mse", validation.get("val_mse"))
    if initial_val_mse is None:
        raise RuntimeError("Stage 3B Round-0 checkpoint has no validation MSE")
    candidates = [(
        0,
        initial_checkpoint,
        run_dir / "evaluations" / "round0_heldout20.json",
        float(initial_val_mse),
    )]
    for round_id in range(1, len(config["rounds"]) + 1):
        summary = read_json(run_dir / f"round_{round_id}" / "training_summary.json")
        candidates.append((
            round_id,
            run_dir / "checkpoints" / f"round{round_id}_best.pth",
            run_dir / "evaluations" / f"round{round_id}_heldout20.json",
            float(summary["best_val_mse"]),
        ))
    return candidates


def run_smoke(config_path, config, run_dir, device):
    collector = THIS_DIR / "stage3b_collect_dagger.py"
    trainer = THIS_DIR / "stage3b_train_dagger.py"
    run([
        sys.executable, "-u", THIS_DIR / "stage3b_train_round0.py",
        "--config", config_path, "--run-dir", run_dir,
        "--epochs", 2, "--device", device,
    ])
    initial = run_dir / "checkpoints" / "round0_best.pth"
    run([
        sys.executable, "-u", collector, "--config", config_path,
        "--run-dir", run_dir, "--student-checkpoint", initial,
        "--mode", "teacher-sanity", "--device", device,
    ])
    smoke_dataset = run_dir / "datasets" / "round1_smoke_2seeds_corrective.hdf5"
    run([
        sys.executable, "-u", collector, "--config", config_path,
        "--run-dir", run_dir, "--student-checkpoint", initial,
        "--mode", "collect", "--round-id", 1, "--beta", config["rounds"][0]["beta"],
        "--seed-start", config["train_seeds"][0], "--num-seeds", 2,
        "--device", device, "--output", smoke_dataset, "--num-workers", 1,
    ])
    run([
        sys.executable, "-u", trainer, "--config", config_path,
        "--run-dir", run_dir, "--round-id", 1,
        "--input-checkpoint", initial, "--corrective-dataset", smoke_dataset,
        "--epochs", 2, "--device", device,
    ])
    summary = read_json(run_dir / "round_1" / "training_summary.json")
    atomic_json(run_dir / "stage3b_smoke_summary.json", {
        "status": "PASS", "teacher_sanity": read_json(run_dir / "teacher_sanity.json"),
        "corrective_dataset": str(smoke_dataset), "training": summary,
        "formal_collection_started": False, "stage4_started": False,
    })
    print(f"Stage 3B smoke test: PASS | {run_dir}")


def run_bootstrap(config_path, config, run_dir, device):
    run([
        sys.executable, "-u", THIS_DIR / "stage3b_train_round0.py",
        "--config", config_path, "--run-dir", run_dir, "--device", device,
    ], run_dir / "logs" / "round0_training.log")
    round0 = run_dir / "checkpoints" / "round0_best.pth"
    run([
        sys.executable, "-u", THIS_DIR / "stage3b_collect_dagger.py",
        "--config", config_path, "--run-dir", run_dir,
        "--student-checkpoint", round0, "--mode", "teacher-sanity", "--device", device,
    ], run_dir / "logs" / "teacher_sanity.log")
    print(f"Stage 3B fresh Round 0 and teacher sanity complete | {run_dir}")


def run_all(config_path, config, run_dir, device):
    sanity = run_dir / "teacher_sanity.json"
    if not sanity.is_file() or read_json(sanity).get("status") != "PASS":
        raise RuntimeError("Run teacher sanity first; full DAgger is blocked")
    evaluations = run_dir / "evaluations"
    evaluations.mkdir(exist_ok=True)
    initial = run_dir / "checkpoints" / "round0_best.pth"
    if not initial.is_file():
        raise RuntimeError("Train Stage 3B Round 0 before run-all")
    run(evaluator_command(
        initial, config["heldout_seeds"][0], 20,
        evaluations / "round0_heldout20.json", device,
    ), run_dir / "logs" / "round0_heldout20.log")
    corrective_paths = []
    input_checkpoint = initial
    for round_spec in config["rounds"]:
        round_id, beta = round_spec["round_id"], round_spec["beta"]
        dataset = run_dir / "datasets" / f"round{round_id}_corrective.hdf5"
        run([
            sys.executable, "-u", THIS_DIR / "stage3b_collect_dagger.py",
            "--config", config_path, "--run-dir", run_dir,
            "--student-checkpoint", input_checkpoint, "--mode", "collect",
            "--round-id", round_id, "--beta", beta, "--device", device,
            "--output", dataset,
        ], run_dir / "logs" / f"round{round_id}_collection.log")
        corrective_paths.append(dataset)
        command = [
            sys.executable, "-u", THIS_DIR / "stage3b_train_dagger.py",
            "--config", config_path, "--run-dir", run_dir,
            "--round-id", round_id, "--input-checkpoint", input_checkpoint,
            "--device", device,
        ]
        for path in corrective_paths:
            command.extend(("--corrective-dataset", path))
        run(command, run_dir / "logs" / f"round{round_id}_training.log")
        input_checkpoint = run_dir / "checkpoints" / f"round{round_id}_best.pth"
        run(evaluator_command(
            input_checkpoint, config["heldout_seeds"][0], 20,
            evaluations / f"round{round_id}_heldout20.json", device,
        ), run_dir / "logs" / f"round{round_id}_heldout20.log")
    selection, final_checkpoint, rows = select_actor(
        config, run_dir, round_candidates(config, run_dir)
    )
    final100 = evaluations / "stage3b_shared_actor_best_100seeds.json"
    run(
        evaluator_command(final_checkpoint, 10000, 100, final100, device),
        evaluations / "stage3b_shared_actor_best_100seeds.log",
    )
    final_metrics = evaluation_metrics(final100)
    summary = {
        "experiment": {"stage": config["stage"], "name": config["name"], "goal": config["goal"]},
        "initial_actor": {
            "source": "Stage3B fresh random success-only BC-RNN distillation",
            "checkpoint": str(run_dir / "checkpoints" / "round0_best.pth"),
            "historical_actor_checkpoint_loaded": False,
        },
        "teacher": {"type": "BC-RNN LSTM", "checkpoint": config["teacher_checkpoint"], "frozen": True},
        "actor": {"state_dim": 59, "action_dim": 14, "hidden_dims": [256, 256], "log_std": -3.0},
        "dagger": {
            "rounds": 3, "betas": [item["beta"] for item in config["rounds"]],
            "train_seeds": config["train_seeds"], "heldout_seeds": config["heldout_seeds"],
            "expert_fraction": config["expert_fraction"],
        },
        "round_results": rows, "selected_round": selection["selected_round"],
        "final_actor_checkpoint": str(final_checkpoint),
        "heldout20_result": selection, "final100_result": final_metrics,
        "stage4_started": False,
    }
    atomic_json(run_dir / "stage3b_summary.json", summary)
    print(f"Stage 3B complete. Stage 4 was not started. Run directory: {run_dir}")


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        prefix = "smoke_" if args.stage == "smoke" else ""
        run_dir = Path(config["output_root"]) / (prefix + datetime.now().strftime("%Y%m%d_%H%M%S"))
    for name in ("round_0", "datasets", "checkpoints", "logs", "evaluations"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "config.json", config)
    if args.stage == "smoke":
        run_smoke(config_path, config, run_dir, args.device)
    elif args.stage == "bootstrap":
        run_bootstrap(config_path, config, run_dir, args.device)
    elif args.stage == "run-all":
        run_all(config_path, config, run_dir, args.device)
    else:
        selection, _, _ = select_actor(config, run_dir, round_candidates(config, run_dir))
        print(f"Selected round: {selection['selected_round']}")


if __name__ == "__main__":
    main()
