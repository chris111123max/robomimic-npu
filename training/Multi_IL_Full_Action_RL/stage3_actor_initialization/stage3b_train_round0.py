#!/usr/bin/env python3
"""Train the Stage 3B Round-0 Actor from scratch on successful BC-RNN data."""

from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
V2_DIR = THIS_DIR.parent / "stage3a_v2_success_only_actor_initialization"
for path in (THIS_DIR, V2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from actor_network import (  # noqa: E402
    build_actor,
    load_actor_checkpoint,
    stochastic_action_and_log_prob,
)
from stage3b_common import (  # noqa: E402
    atomic_json,
    read_json,
    seed_everything,
    select_device,
    validate_frozen_log_std,
)
from success_only_dataset import SuccessOnlyTransitionDataset, inspect_dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(THIS_DIR / "stage3b_config.json"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def validation_metrics(actor, dataset, batch_size, device):
    actor.eval()
    squared = absolute = 0.0
    elements = 0
    max_abs = 0.0
    nonfinite = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            state = torch.as_tensor(
                dataset.states[start:start + batch_size], dtype=torch.float32, device=device
            )
            target = torch.as_tensor(
                dataset.actions[start:start + batch_size], dtype=torch.float32, device=device
            )
            prediction = actor(state, deterministic=True, return_log_prob=False)[0]
            difference = prediction - target
            squared += float(torch.square(difference).sum().cpu())
            absolute += float(torch.abs(difference).sum().cpu())
            max_abs = max(max_abs, float(torch.abs(difference).max().cpu()))
            elements += int(difference.numel())
            nonfinite += int((~torch.isfinite(prediction)).sum().cpu())
    return {
        "val_mse": squared / elements,
        "val_mae": absolute / elements,
        "val_max_abs_error": max_abs,
        "validation_transitions": len(dataset),
        "nan_inf_count": nonfinite,
    }


def checkpoint_payload(actor, optimizer, config, dataset_info, train_data,
                       validation_data, epoch, validation, best_val_mse, total_updates):
    return {
        "format_version": "multi_il_full_action_rl.stage3.actor.v1",
        "stage": "stage3b_dagger_actor_distillation_round0",
        "experiment_id": config["experiment_id"],
        "round_id": 0,
        "epoch": int(epoch),
        "actor_state_dict": copy.deepcopy(actor.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "architecture": {
            "state_dim": int(config["state_dim"]),
            "action_dim": int(config["action_dim"]),
            "hidden_dims": list(config["hidden_dims"]),
        },
        "action_distribution": {
            "initial_log_std": float(config["log_std"]),
            "log_std_min": -20.0,
            "log_std_max": 2.0,
            "log_std_mode": "state_dependent_head_initialized_constant",
            "freeze_log_std_during_distillation": True,
        },
        "action_convention": {
            "teacher_target": "Stage 1 action exactly passed to env.step",
            "student_distribution": "squashed Gaussian",
            "deterministic_action": "tanh(mu)",
            "loss_space": "post-tanh environment action space",
            "extra_action_scaling": False,
        },
        "observation_keys": list(dataset_info["canonical_keys"]),
        "observation_shapes": copy.deepcopy(dataset_info["canonical_shapes"]),
        "teacher_checkpoint": config["teacher_checkpoint"],
        "source_dataset": dataset_info["path"],
        "train_seeds": list(train_data.successful_seeds),
        "validation_seeds": list(validation_data.successful_seeds),
        "success_only": True,
        "fresh_random_initialization": True,
        "validation": copy.deepcopy(validation),
        "best_val_mse": float(best_val_mse),
        "total_gradient_updates": int(total_updates),
        "evaluation_horizon": int(config["horizon"]),
        "terminate_on_success": bool(config["terminate_on_success"]),
    }


def main():
    args = parse_args()
    config = read_json(args.config)
    run_dir = Path(args.run_dir)
    round_dir = run_dir / "round_0"
    checkpoint_dir = run_dir / "checkpoints"
    round_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "config.json", config)
    best_path = checkpoint_dir / "round0_best.pth"
    last_path = checkpoint_dir / "round0_last.pth"
    if best_path.exists() or last_path.exists():
        raise RuntimeError(
            "Round 0 checkpoints already exist; refusing to overwrite the independent Actor run"
        )

    train_start, train_end = config["train_seeds"]
    heldout_start, heldout_end = config["heldout_seeds"]
    dataset_info = inspect_dataset(config["stage1_rnn_dataset"])
    if Path(dataset_info["checkpoint"]).resolve() != Path(config["teacher_checkpoint"]).resolve():
        raise RuntimeError("Round 0 dataset does not belong to the configured BC-RNN teacher")
    train_data = SuccessOnlyTransitionDataset(
        config["stage1_rnn_dataset"], range(train_start, train_end + 1)
    )
    validation_data = SuccessOnlyTransitionDataset(
        config["stage1_rnn_dataset"], range(heldout_start, heldout_end + 1)
    )
    if set(train_data.successful_seeds) & set(validation_data.successful_seeds):
        raise RuntimeError("Round 0 train and validation seed leakage detected")

    device = select_device(args.device)
    seed_everything(int(config["random_seed"]))
    actor = build_actor(
        state_dim=int(config["state_dim"]), action_dim=int(config["action_dim"]),
        hidden_dims=config["hidden_dims"], initial_log_std=float(config["log_std"]),
        freeze_log_std=True, device=device,
    )
    validate_frozen_log_std(actor, expected=float(config["log_std"]))
    optimizer = torch.optim.Adam(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    rng = np.random.default_rng(int(config["random_seed"]))
    epochs = int(args.epochs or config["round0_epochs"])
    if epochs <= 0:
        raise ValueError("--epochs must be positive")
    batch_size = int(config["batch_size"])

    print("=" * 80)
    print("Stage 3B Round 0 | Fresh Success-Only BC-RNN Actor Distillation")
    print("Historical Actor checkpoint loaded: NO")
    print(f"Train successful episodes: {len(train_data.successful_seeds)}")
    print(f"Train successful transitions: {len(train_data)}")
    print(f"Held-out successful episodes: {len(validation_data.successful_seeds)}")
    print(f"Held-out validation transitions: {len(validation_data)}")
    print(f"Actor: {config['state_dim']} -> 256 -> 256 -> Gaussian({config['action_dim']})")
    print(f"Epochs: {epochs} | Device: {device}")
    print("=" * 80)

    rows = []
    best_mse, best_epoch, best_validation = math.inf, None, None
    total_updates = 0
    for epoch in range(1, epochs + 1):
        actor.train()
        order = rng.permutation(len(train_data))
        squared = 0.0
        elements = 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            state = torch.as_tensor(train_data.states[indices], dtype=torch.float32, device=device)
            target = torch.as_tensor(train_data.actions[indices], dtype=torch.float32, device=device)
            prediction = actor(state, deterministic=True, return_log_prob=False)[0]
            loss = torch.mean(torch.square(prediction - target))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Round 0 loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            squared += float(torch.square(prediction.detach() - target).sum().cpu())
            elements += int(target.numel())
            total_updates += 1
        train_mse = squared / elements
        validation = validation_metrics(actor, validation_data, batch_size, device)
        if validation["nan_inf_count"]:
            raise RuntimeError(f"Non-finite Round 0 validation output at epoch {epoch}")
        rows.append({
            "epoch": epoch, "train_mse": train_mse,
            "val_mse": validation["val_mse"], "val_mae": validation["val_mae"],
            "val_max_abs_error": validation["val_max_abs_error"],
            "total_gradient_updates": total_updates,
        })
        payload = checkpoint_payload(
            actor, optimizer, config, dataset_info, train_data, validation_data,
            epoch, validation, min(best_mse, validation["val_mse"]), total_updates,
        )
        torch.save(payload, last_path)
        if validation["val_mse"] < best_mse:
            best_mse = validation["val_mse"]
            best_epoch = epoch
            best_validation = copy.deepcopy(validation)
            payload["best_val_mse"] = best_mse
            torch.save(payload, best_path)
        print(
            f"round 0 epoch {epoch:03d}/{epochs} | train_mse {train_mse:.8f} | "
            f"val_mse {validation['val_mse']:.8f} | val_mae {validation['val_mae']:.8f}"
        )

    with (round_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    reloaded, payload = load_actor_checkpoint(best_path, device=device, freeze_log_std=True)
    validate_frozen_log_std(reloaded, expected=float(config["log_std"]))
    sample_size = min(8, len(validation_data))
    sample = torch.as_tensor(validation_data.states[:sample_size], dtype=torch.float32, device=device)
    with torch.no_grad():
        deterministic = reloaded(sample, deterministic=True, return_log_prob=False)[0]
        stochastic, log_prob = stochastic_action_and_log_prob(reloaded, sample)
    reload_ok = bool(
        deterministic.shape == (sample_size, 14)
        and stochastic.shape == (sample_size, 14)
        and log_prob.shape == (sample_size, 1)
        and torch.isfinite(deterministic).all()
        and torch.isfinite(stochastic).all()
        and torch.isfinite(log_prob).all()
        and payload.get("round_id") == 0
    )
    if not reload_ok:
        raise RuntimeError("Round 0 best checkpoint reload sanity failed")

    summary = {
        "status": "TRAINING_COMPLETE",
        "round_id": 0,
        "initialization": "fresh_random",
        "historical_actor_checkpoint_loaded": False,
        "teacher_checkpoint_used_only_as_dataset_provenance": config["teacher_checkpoint"],
        "train_success_episodes": len(train_data.successful_seeds),
        "train_success_transitions": len(train_data),
        "heldout_success_episodes": len(validation_data.successful_seeds),
        "heldout_validation_transitions": len(validation_data),
        "heldout_seeds_used_for_gradient": False,
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "best_val_mse": best_mse,
        "best_val_mae": best_validation["val_mae"],
        "total_gradient_updates": total_updates,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "checkpoint_reload_sanity": "PASS",
        "stage4_started": False,
    }
    atomic_json(round_dir / "training_summary.json", summary)
    print("=" * 80)
    print(f"Round 0 complete | best_epoch={best_epoch} best_val_mse={best_mse:.10f}")
    print(f"Best checkpoint: {best_path}")
    print("Historical Actor checkpoint loaded: NO")
    print("=" * 80)


if __name__ == "__main__":
    main()
