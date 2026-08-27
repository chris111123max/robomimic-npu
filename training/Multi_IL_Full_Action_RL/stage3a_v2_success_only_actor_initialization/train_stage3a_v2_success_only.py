#!/usr/bin/env python3
"""Stage 3A-v2: distill only successful BC-RNN trajectories."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
V1_DIR = PROJECT_DIR / "stage3_actor_initialization"
for path in (THIS_DIR, V1_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from actor_network import (  # noqa: E402
    build_actor,
    load_actor_checkpoint,
    stochastic_action_and_log_prob,
)
from success_only_dataset import SuccessOnlyTransitionDataset, inspect_dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(THIS_DIR / "stage3a_v2_success_only_config.json"))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_device(name):
    if name.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU requested but torch_npu cannot be imported") from exc
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but unavailable")
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def validate_single_variable_design(config, baseline):
    pairs = {
        "architecture": (config["architecture"], baseline["architecture"]),
        "batch_size": (config["batch_size"], baseline["batch_size"]),
        "epochs": (config["epochs"], baseline["epochs"]),
        "learning_rate": (config["learning_rate"], baseline["learning_rate"]),
        "weight_decay": (config["weight_decay"], baseline["weight_decay"]),
        "random_seed": (config["random_seed"], baseline["random_seed"]),
        "candidate_epochs": (config["candidate_epochs"], baseline["candidate_epochs"]),
        "evaluation_horizon": (config["evaluation_horizon"], baseline["evaluation_horizon"]),
        "terminate_on_success": (config["terminate_on_success"], baseline["terminate_on_success"]),
        "distribution": (config["distribution"], baseline["distribution"]),
        "distillation_loss": (config["distillation_loss"], baseline["distillation_loss"]),
        "action_convention": (config["action_convention"], baseline["action_convention"]),
    }
    differences = [name for name, (current, old) in pairs.items() if current != old]
    if differences:
        raise RuntimeError(f"Stage 3A-v2 differs from v1 outside dataset filtering: {differences}")


def offline_metrics(actor, dataset, batch_size, device):
    actor.eval()
    total = 0
    squared = absolute = l2_sum = 0.0
    max_absolute = 0.0
    per_dim_squared = np.zeros(dataset.action_dim, dtype=np.float64)
    student_min, student_max = np.inf, -np.inf
    nonfinite = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            states = torch.as_tensor(dataset.states[start:start + batch_size], device=device)
            target = dataset.actions[start:start + batch_size].astype(np.float64)
            student = actor(states, deterministic=True, return_log_prob=False)[0].cpu().numpy().astype(np.float64)
            difference = student - target
            count = int(target.shape[0])
            total += count
            squared += float(np.square(difference).sum())
            absolute += float(np.abs(difference).sum())
            l2_sum += float(np.linalg.norm(difference, axis=1).sum())
            max_absolute = max(max_absolute, float(np.abs(difference).max()))
            per_dim_squared += np.square(difference).sum(axis=0)
            student_min = min(student_min, float(student.min()))
            student_max = max(student_max, float(student.max()))
            nonfinite += int((~np.isfinite(student)).sum())
    return {
        "num_transitions": total,
        "action_mse": squared / (total * dataset.action_dim),
        "action_mae": absolute / (total * dataset.action_dim),
        "max_absolute_error": max_absolute,
        "per_dimension_mse": (per_dim_squared / total).tolist(),
        "mean_l2_action_error": l2_sum / total,
        "student_action_min": student_min,
        "student_action_max": student_max,
        "nan_inf_count": nonfinite,
    }


def checkpoint_payload(actor, optimizer, epoch, config, dataset_info, train_data,
                       validation_data, validation, best_val_mse, total_updates):
    return {
        "format_version": "multi_il_full_action_rl.stage3.actor.v1",
        "stage": "stage3a_v2_success_only_actor_initialization",
        "experiment_id": config["experiment_id"],
        "epoch": int(epoch),
        "actor_state_dict": copy.deepcopy(actor.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "architecture": copy.deepcopy(config["architecture"]),
        "action_distribution": copy.deepcopy(config["distribution"]),
        "action_convention": copy.deepcopy(config["action_convention"]),
        "observation_keys": list(dataset_info["canonical_keys"]),
        "observation_shapes": copy.deepcopy(dataset_info["canonical_shapes"]),
        "teacher_checkpoint": dataset_info["checkpoint"],
        "source_dataset": dataset_info["path"],
        "train_seeds": list(train_data.successful_seeds),
        "validation_seeds": list(validation_data.successful_seeds),
        "success_only": True,
        "validation": copy.deepcopy(validation),
        "best_val_mse": float(best_val_mse),
        "total_gradient_updates": int(total_updates),
        "evaluation_horizon": int(config["evaluation_horizon"]),
        "terminate_on_success": bool(config["terminate_on_success"]),
    }


def main():
    args = parse_args()
    config = read_json(args.config)
    baseline = read_json(config["baseline_config"])
    validate_single_variable_design(config, baseline)
    split = read_json(config["seed_split"])
    train_seeds = [int(seed) for seed in split["train_seeds"]]
    validation_seeds = [int(seed) for seed in split["validation_seeds"]]
    if train_seeds != list(range(10000, 10080)):
        raise RuntimeError("Train split must be seeds 10000..10079")
    if validation_seeds != list(range(10080, 10100)):
        raise RuntimeError("Validation split must be seeds 10080..10099")
    if args.output_root:
        config["output_root"] = args.output_root
    config["device"] = args.device
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        config["epochs"] = int(args.epochs)
        if int(args.epochs) not in config["candidate_epochs"]:
            config["candidate_epochs"] = [*config["candidate_epochs"], int(args.epochs)]
    if args.smoke_test:
        config["epochs"] = 2
        config["candidate_epochs"] = [1, 2]

    dataset_info = inspect_dataset(config["source_dataset"])
    if Path(dataset_info["checkpoint"]).resolve() != Path(config["teacher"]["checkpoint"]).resolve():
        raise RuntimeError("Selected dataset was not generated by the fixed BC-RNN teacher")
    train_data = SuccessOnlyTransitionDataset(config["source_dataset"], train_seeds)
    validation_data = SuccessOnlyTransitionDataset(config["source_dataset"], validation_seeds)
    if set(train_data.successful_seeds) & set(validation_data.successful_seeds):
        raise RuntimeError("Successful train and validation seeds overlap")
    if train_data.statistics["failed_episodes_loaded_into_student_dataset"] != 0:
        raise RuntimeError("A failed trajectory entered the training dataset")
    if validation_data.statistics["failed_episodes_loaded_into_student_dataset"] != 0:
        raise RuntimeError("A failed trajectory entered the validation dataset")

    run_id = args.run_id or (("smoke_test_" if args.smoke_test else "") + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = Path(config["output_root"]) / run_id
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "stage3a_v2_success_only_config.json", config)
    dataset_statistics = {
        "source": dataset_info,
        "filter": "episode success == True within each fixed seed split",
        "train": train_data.statistics,
        "validation": validation_data.statistics,
    }
    write_json(run_dir / "dataset_statistics.json", dataset_statistics)

    device = select_device(args.device)
    seed_everything(int(config["random_seed"]))
    actor = build_actor(
        state_dim=config["architecture"]["state_dim"],
        action_dim=config["architecture"]["action_dim"],
        hidden_dims=config["architecture"]["hidden_dims"],
        initial_log_std=config["distribution"]["initial_log_std"],
        freeze_log_std=config["distribution"]["freeze_log_std_during_distillation"],
        device=device,
    )
    if any(parameter.requires_grad for parameter in actor.last_fc_log_std.parameters()):
        raise RuntimeError("log_std head must remain frozen during distillation")
    optimizer = torch.optim.Adam(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    rng = np.random.default_rng(int(config["random_seed"]))

    print("=" * 80)
    print("Stage 3A-v2: Success-Only RNN Distillation")
    print("=" * 80)
    print("Baseline: Stage 3A-v1 All-RNN Distillation")
    print("Single changed variable: failed RNN trajectories excluded")
    print(f"Train total episodes: {train_data.statistics['total_episodes']}")
    print(f"Train successful episodes: {train_data.statistics['successful_episodes']}")
    print(f"Train failed episodes excluded: {train_data.statistics['failed_episodes']}")
    print(f"Train successful transitions: {len(train_data)}")
    print(f"Train tensors: state={list(train_data.states.shape)} action={list(train_data.actions.shape)}")
    print(f"Validation total episodes: {validation_data.statistics['total_episodes']}")
    print(f"Validation successful episodes: {validation_data.statistics['successful_episodes']}")
    print(f"Validation failed episodes excluded: {validation_data.statistics['failed_episodes']}")
    print(f"Validation successful transitions: {len(validation_data)}")
    print(
        f"Validation tensors: state={list(validation_data.states.shape)} "
        f"action={list(validation_data.actions.shape)}"
    )
    print("Student: MLP SAC Actor | 59 -> 256 -> 256 -> Gaussian(14)")
    print("Action target: stored post-tanh RNN env action")
    print("log_std: -3.0 frozen")
    print(f"Device: {device}")
    print("=" * 80)

    train_rows, validation_rows = [], []
    best_val_mse, best_epoch, best_validation = math.inf, None, None
    total_updates = 0
    candidates = set(int(epoch) for epoch in config["candidate_epochs"])
    for epoch in range(1, int(config["epochs"]) + 1):
        actor.train()
        order = rng.permutation(len(train_data))
        train_squared, train_elements = 0.0, 0
        epoch_updates = 0
        for start in range(0, len(order), int(config["batch_size"])):
            indices = order[start:start + int(config["batch_size"])]
            state = torch.as_tensor(train_data.states[indices], device=device)
            target = torch.as_tensor(train_data.actions[indices], device=device)
            prediction = actor(state, deterministic=True, return_log_prob=False)[0]
            loss = torch.mean(torch.square(prediction - target))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_squared += float(torch.square(prediction.detach() - target).sum().cpu())
            train_elements += int(target.numel())
            epoch_updates += 1
            total_updates += 1
        train_mse = train_squared / train_elements
        validation = offline_metrics(actor, validation_data, int(config["batch_size"]), device)
        if validation["nan_inf_count"]:
            raise RuntimeError(f"Non-finite validation actor output at epoch {epoch}")
        train_rows.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "epoch_gradient_updates": epoch_updates,
            "total_gradient_updates": total_updates,
        })
        validation_rows.append({
            "epoch": epoch,
            "val_mse": validation["action_mse"],
            "val_mae": validation["action_mae"],
            "val_max_abs_error": validation["max_absolute_error"],
            "val_mean_l2_error": validation["mean_l2_action_error"],
        })
        print(
            f"epoch {epoch:03d}/{config['epochs']} | train_mse {train_mse:.8f} | "
            f"val_mse {validation['action_mse']:.8f} | val_mae {validation['action_mae']:.8f}"
        )
        payload = checkpoint_payload(
            actor, optimizer, epoch, config, dataset_info, train_data,
            validation_data, validation, min(best_val_mse, validation["action_mse"]), total_updates,
        )
        if validation["action_mse"] < best_val_mse:
            best_val_mse = validation["action_mse"]
            best_epoch = epoch
            best_validation = copy.deepcopy(validation)
            payload["best_val_mse"] = best_val_mse
            torch.save(payload, checkpoint_dir / "success_only_best_val_mse.pth")
        if epoch in candidates:
            torch.save(payload, checkpoint_dir / f"success_only_epoch_{epoch}.pth")
        if epoch == int(config["epochs"]):
            torch.save(payload, checkpoint_dir / "success_only_last.pth")

    write_csv(
        run_dir / "train_metrics.csv", train_rows,
        ["epoch", "train_mse", "epoch_gradient_updates", "total_gradient_updates"],
    )
    write_csv(
        run_dir / "validation_metrics.csv", validation_rows,
        ["epoch", "val_mse", "val_mae", "val_max_abs_error", "val_mean_l2_error"],
    )

    reloaded, reloaded_payload = load_actor_checkpoint(
        checkpoint_dir / "success_only_best_val_mse.pth", device=device, freeze_log_std=True
    )
    sample_size = min(8, len(validation_data))
    sample = torch.as_tensor(validation_data.states[:sample_size], device=device)
    with torch.no_grad():
        output = reloaded(sample, deterministic=True, return_log_prob=False)
        deterministic, reloaded_log_std = output[0], output[2]
        stochastic, log_prob = stochastic_action_and_log_prob(reloaded, sample)
    reload_sanity = {
        "checkpoint_reloaded": True,
        "format_version": reloaded_payload["format_version"],
        "state_batch_shape": list(sample.shape),
        "deterministic_action_shape": list(deterministic.shape),
        "stochastic_action_shape": list(stochastic.shape),
        "log_prob_shape": list(log_prob.shape),
        "all_outputs_finite": bool(
            torch.isfinite(deterministic).all()
            and torch.isfinite(stochastic).all()
            and torch.isfinite(log_prob).all()
        ),
        "log_std_is_minus_three": bool(torch.allclose(
            reloaded_log_std, torch.full_like(reloaded_log_std, -3.0)
        )),
        "log_std_head_frozen": not any(
            parameter.requires_grad for parameter in reloaded.last_fc_log_std.parameters()
        ),
    }
    expected_shapes = (
        reload_sanity["state_batch_shape"] == [sample_size, 59]
        and reload_sanity["deterministic_action_shape"] == [sample_size, 14]
        and reload_sanity["stochastic_action_shape"] == [sample_size, 14]
        and reload_sanity["log_prob_shape"] == [sample_size, 1]
    )
    if not expected_shapes or not reload_sanity["all_outputs_finite"]:
        raise RuntimeError(f"Reloaded actor API sanity failed: {reload_sanity}")
    if not reload_sanity["log_std_is_minus_three"] or not reload_sanity["log_std_head_frozen"]:
        raise RuntimeError(f"Reloaded log_std sanity failed: {reload_sanity}")
    smoke_loss_decreased = train_rows[-1]["train_mse"] < train_rows[0]["train_mse"]
    if args.smoke_test and not smoke_loss_decreased:
        raise RuntimeError(
            "Stage 3A-v2 smoke loss did not decrease over two epochs: "
            f"{train_rows[0]['train_mse']} -> {train_rows[-1]['train_mse']}"
        )
    write_json(run_dir / "checkpoint_reload_sanity.json", reload_sanity)

    summary = {
        "status": "SMOKE_TEST_PASSED" if args.smoke_test else "TRAINING_COMPLETE",
        "experiment": {
            "stage": config["stage"],
            "name": config["name"],
            "baseline": config["baseline"],
            "single_changed_variable": config["single_changed_variable"],
        },
        "teacher": copy.deepcopy(config["teacher"]),
        "student": {
            "type": "MLP-SAC-Actor",
            **copy.deepcopy(config["architecture"]),
        },
        "dataset": {
            "source": dataset_info["path"],
            "train_seed_range": [10000, 10079],
            "val_seed_range": [10080, 10099],
            "train_total_episodes": train_data.statistics["total_episodes"],
            "train_success_episodes": train_data.statistics["successful_episodes"],
            "train_failed_episodes": train_data.statistics["failed_episodes"],
            "train_success_transitions": len(train_data),
            "val_total_episodes": validation_data.statistics["total_episodes"],
            "val_success_episodes": validation_data.statistics["successful_episodes"],
            "val_failed_episodes": validation_data.statistics["failed_episodes"],
            "val_success_transitions": len(validation_data),
            "failed_trajectories_loaded": 0,
        },
        "training": {
            "epochs": int(config["epochs"]),
            "batch_size": int(config["batch_size"]),
            "learning_rate": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
            "total_gradient_updates": total_updates,
        },
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "best_val_mae": best_validation["action_mae"],
        "best_checkpoint": str(checkpoint_dir / "success_only_best_val_mse.pth"),
        "action_convention": {
            "target": "stored post-tanh env action",
            "student_deterministic_action": "tanh(mu)",
            "extra_scaling": False,
        },
        "log_std": {
            "initial_value": -3.0,
            "frozen_during_distillation": True,
        },
        "baseline_reference": copy.deepcopy(config["baseline_reference"]),
        "checkpoint_reload_sanity": reload_sanity,
        "fresh_random_student_initialization": True,
        "stage3a_v1_actor_loaded": False,
        "smoke_train_loss_decreased": smoke_loss_decreased if args.smoke_test else None,
        "run_directory": str(run_dir),
        "automatic_environment_evaluation_run": False,
        "stage4_started": False,
    }
    write_json(run_dir / "stage3a_v2_success_only_training_summary.json", summary)

    print("=" * 80)
    print("Stage 3A-v2 Training Complete")
    print("=" * 80)
    print(f"Best epoch: {best_epoch}")
    print(f"Best val MSE: {best_val_mse:.10f}")
    print(f"Best val MAE: {best_validation['action_mae']:.10f}")
    print(f"Total gradient updates: {total_updates}")
    print(
        "Checkpoint/API sanity: "
        f"state={reload_sanity['state_batch_shape']}, "
        f"deterministic_action={reload_sanity['deterministic_action_shape']}, "
        f"stochastic_action={reload_sanity['stochastic_action_shape']}, "
        f"log_prob={reload_sanity['log_prob_shape']}, "
        f"stochastic/log_prob finite={reload_sanity['all_outputs_finite']}, "
        f"log_std=-3 frozen={reload_sanity['log_std_is_minus_three'] and reload_sanity['log_std_head_frozen']}"
    )
    if args.smoke_test:
        print(
            "Smoke loss decreased: "
            f"{train_rows[0]['train_mse']:.10f} -> {train_rows[-1]['train_mse']:.10f}"
        )
    print(f"Best checkpoint: {checkpoint_dir / 'success_only_best_val_mse.pth'}")
    print(f"Run directory: {run_dir}")
    print("Next: 100-seed deterministic progress evaluation")
    print("Do NOT start Stage 4.")
    print("=" * 80)


if __name__ == "__main__":
    main()
