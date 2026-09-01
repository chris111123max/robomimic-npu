#!/usr/bin/env python3
"""Train a fresh SAC-compatible actor on saved Stage 1 BC-GMM actions."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
ACTOR_DIR = PROJECT_DIR / "stage3_actor_initialization"
for path in (THIS_DIR, ACTOR_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from actor_network import build_actor, load_actor_checkpoint, stochastic_action_and_log_prob  # noqa: E402
from stage3c_dataset import Stage3CDataset, inspect_dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(THIS_DIR / "stage3c_config.json"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def select_device(name):
    if name.startswith("npu"):
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but unavailable")
        torch.npu.set_device(torch.device(name))
    return torch.device(name)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def metrics(actor, dataset, batch_size, device):
    actor.eval()
    squared = absolute = 0.0
    maximum = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            states = torch.as_tensor(dataset.states[start:start + batch_size], device=device)
            targets = torch.as_tensor(dataset.actions[start:start + batch_size], device=device)
            outputs = actor(states, deterministic=True, return_log_prob=False)[0]
            difference = outputs - targets
            squared += float(torch.square(difference).sum().cpu())
            absolute += float(torch.abs(difference).sum().cpu())
            maximum = max(maximum, float(torch.abs(difference).max().cpu()))
            count += targets.numel()
    return {"val_mse": squared / count, "val_mae": absolute / count, "val_max_abs_error": maximum}


def checkpoint_payload(actor, optimizer, epoch, config, info, validation):
    return {
        "format_version": "multi_il_full_action_rl.stage3c.actor.v1",
        "stage": "stage3c_bc_gmm_direct_distillation",
        "epoch": int(epoch),
        "actor_state_dict": copy.deepcopy(actor.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "architecture": copy.deepcopy(config["architecture"]),
        "action_distribution": {
            "initial_log_std": float(config["distribution"]["initial_log_std"]),
            "freeze_log_std_during_distillation": True,
        },
        "action_convention": {
            "student_deterministic": "tanh(mu)",
            "target": "saved Stage1 post-processing/environment BC-GMM action",
            "target_clipped_or_rescaled": False,
        },
        "observation_keys": list(info["canonical_keys"]),
        "observation_shapes": copy.deepcopy(info["canonical_shapes"]),
        "teacher_checkpoint": info["checkpoint"],
        "source_dataset": info["path"],
        "train_seeds": list(config["train_seeds"]),
        "validation_seeds": list(config["validation_seeds"]),
        "validation": copy.deepcopy(validation),
        "evaluation_horizon": int(config["evaluation_horizon"]),
        "terminate_on_success": bool(config["terminate_on_success"]),
        "random_initialization": True,
    }


def save_checkpoint(path, actor, optimizer, epoch, config, info, validation):
    torch.save(checkpoint_payload(actor, optimizer, epoch, config, info, validation), path)


def main():
    args = parse_args()
    config = read_json(args.config)
    if config["train_seeds"] != list(range(10000, 10080)) or config["validation_seeds"] != list(range(10080, 10100)):
        raise RuntimeError("Stage 3C requires the exact 80/20 episode-seed split")
    if args.smoke_test:
        config["train_seeds"] = config["train_seeds"][:5]
        config["validation_seeds"] = config["validation_seeds"][:2]
        config["max_epochs"] = 2
        config["candidate_epochs"] = [1, 2]
        config["early_stopping_patience"] = 30
    run_dir = Path(args.run_dir).resolve()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    device = select_device(args.device)
    seed_all(int(config["random_seed"]))
    info = inspect_dataset(config["dataset"])
    if Path(info["checkpoint"]).resolve() != Path(config["teacher_checkpoint"]).resolve():
        raise RuntimeError("Stage1 dataset teacher checkpoint differs from configured BC-GMM")
    train = Stage3CDataset(info["path"], config["train_seeds"])
    validation = Stage3CDataset(info["path"], config["validation_seeds"])
    write_json(run_dir / "resolved_config.json", config)
    write_json(run_dir / "dataset_summary.json", {"dataset": info, "train": train.statistics, "validation": validation.statistics})

    actor = build_actor(59, 14, (256, 256), -3.0, True, device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    if any(parameter.requires_grad for parameter in actor.last_fc_log_std.parameters()):
        raise RuntimeError("log_std head is not frozen")
    rng = np.random.default_rng(int(config["random_seed"]))
    best_mse, best_epoch, stale = math.inf, None, 0
    history = []
    candidates = set(int(value) for value in config["candidate_epochs"])
    max_epochs = int(config["max_epochs"])
    batch_size = int(config["batch_size"])

    print("=" * 80)
    print("Stage 3C BC-GMM -> SAC Direct Distillation")
    print(f"Device: {device} | epochs: {max_epochs} | batch: {batch_size}")
    print(f"State [B,59], saved target [B,14], student tanh(mu) [B,14]")
    print(f"Train: {train.statistics}")
    print(f"Validation: {validation.statistics}")
    print("Teacher is not queried during training; Student starts from random initialization.")
    print("=" * 80)

    for epoch in range(1, max_epochs + 1):
        actor.train()
        order = rng.permutation(len(train))
        squared, elements = 0.0, 0
        gradients_finite = True
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            states = torch.as_tensor(train.states[indices], device=device)
            targets = torch.as_tensor(train.actions[indices], device=device)
            outputs = actor(states, deterministic=True, return_log_prob=False)[0]
            if outputs.shape != targets.shape or outputs.shape[1:] != (14,):
                raise RuntimeError(f"Shape mismatch: state={states.shape}, output={outputs.shape}, target={targets.shape}")
            loss = torch.mean(torch.square(outputs - targets))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients_finite &= all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in actor.parameters()
            )
            if not gradients_finite:
                raise RuntimeError("Non-finite gradient")
            optimizer.step()
            squared += float(torch.square(outputs.detach() - targets).sum().cpu())
            elements += targets.numel()
        values = metrics(actor, validation, batch_size, device)
        row = {"epoch": epoch, "train_mse": squared / elements, **values}
        history.append(row)
        print(
            f"epoch {epoch:03d}/{max_epochs} | train_mse {row['train_mse']:.8f} | "
            f"val_mse {row['val_mse']:.8f} | val_mae {row['val_mae']:.8f} | "
            f"val_max_abs_error {row['val_max_abs_error']:.8f}"
        )
        if values["val_mse"] < best_mse:
            best_mse, best_epoch, stale = values["val_mse"], epoch, 0
            save_checkpoint(checkpoint_dir / "bc_gmm_distill_best_val_mse.pth", actor, optimizer, epoch, config, info, values)
        else:
            stale += 1
        if epoch in candidates:
            save_checkpoint(checkpoint_dir / f"bc_gmm_distill_epoch_{epoch}.pth", actor, optimizer, epoch, config, info, values)
        save_checkpoint(checkpoint_dir / "bc_gmm_distill_last.pth", actor, optimizer, epoch, config, info, values)
        if stale >= int(config["early_stopping_patience"]):
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    with open(run_dir / "training_metrics.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    reloaded, payload = load_actor_checkpoint(checkpoint_dir / "bc_gmm_distill_best_val_mse.pth", device, freeze_log_std=True)
    sample = torch.as_tensor(validation.states[:min(8, len(validation))], device=device)
    with torch.no_grad():
        deterministic = reloaded(sample, deterministic=True)[0]
        stochastic, log_prob = stochastic_action_and_log_prob(reloaded, sample)
    sanity = {
        "state_shape": list(sample.shape), "target_action_shape": [len(sample), 14],
        "deterministic_action_shape": list(deterministic.shape),
        "stochastic_action_shape": list(stochastic.shape), "log_prob_shape": list(log_prob.shape),
        "loss_finite": True, "gradients_finite": gradients_finite,
        "checkpoint_save_reload": True,
        "stochastic_action_finite": bool(torch.isfinite(stochastic).all()),
        "log_prob_finite": bool(torch.isfinite(log_prob).all()),
        "log_std_initial_value": -3.0,
        "log_std_frozen": not any(parameter.requires_grad for parameter in reloaded.last_fc_log_std.parameters()),
    }
    smoke_pass = all([
        sanity["deterministic_action_shape"] == [len(sample), 14],
        sanity["stochastic_action_shape"] == [len(sample), 14], sanity["log_prob_shape"] == [len(sample), 1],
        sanity["stochastic_action_finite"], sanity["log_prob_finite"], sanity["log_std_frozen"],
    ])
    if not smoke_pass:
        raise RuntimeError(f"Stage3C API sanity failed: {sanity}")
    write_json(run_dir / "smoke_sanity.json", {"status": "PASS", **sanity})
    summary = {
        "status": "SMOKE_TEST_PASSED" if args.smoke_test else "TRAINING_COMPLETE",
        "run_directory": str(run_dir), "train": train.statistics, "validation": validation.statistics,
        "best_epoch": best_epoch, "best_val_mse": best_mse,
        "best_val_mae": next(row["val_mae"] for row in history if row["epoch"] == best_epoch),
        "epochs_completed": len(history), "early_stopped": len(history) < max_epochs,
        "candidate_checkpoints": sorted(path.name for path in checkpoint_dir.glob("*.pth")),
        "smoke_sanity": sanity, "stage4_started": False,
    }
    write_json(run_dir / "training_summary.json", summary)
    print(f"STAGE3C_TRAINING_COMPLETE run_dir={run_dir}")
    print(f"best_epoch={best_epoch} best_val_mse={best_mse:.10f}")


if __name__ == "__main__":
    main()
