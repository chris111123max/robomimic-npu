#!/usr/bin/env python3
"""Train one Stage 3B round on balanced expert and aggregated corrective data."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent

from actor_network import load_actor_checkpoint, stochastic_action_and_log_prob  # noqa: E402
from stage3b_common import (  # noqa: E402
    ACTION_BOUND_TOLERANCE,
    atomic_json,
    read_json,
    seed_everything,
    select_device,
    validate_frozen_log_std,
)
from stage3b_success_dataset import (  # noqa: E402
    Stage3BSuccessTransitionDataset,
    inspect_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(THIS_DIR / "stage3b_config.json"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--input-checkpoint", required=True)
    parser.add_argument("--corrective-dataset", action="append", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def load_corrective(paths, train_start, train_end):
    state_parts, action_parts = [], []
    round_ids, episode_seeds = set(), set()
    per_file = []
    for path in paths:
        path = Path(path)
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema_version") != "multi_il_full_action_rl.stage3b.dagger.v1":
                raise RuntimeError(f"Unexpected corrective schema: {path}")
            file_count = 0
            for group in handle["episodes"].values():
                state = np.asarray(group["state_59d"], dtype=np.float32)
                teacher_action = np.asarray(group["teacher_action_14d"], dtype=np.float32)
                seed = int(group.attrs["seed"])
                if seed < train_start or seed > train_end:
                    raise RuntimeError(f"Held-out seed entered corrective dataset: seed={seed} path={path}")
                if state.shape != (teacher_action.shape[0], 59) or teacher_action.shape[1:] != (14,):
                    raise RuntimeError(f"Invalid corrective shapes: {path}:{group.name}")
                if not np.isfinite(state).all() or not np.isfinite(teacher_action).all():
                    raise RuntimeError(f"NaN/Inf corrective data: {path}:{group.name}")
                if (float(teacher_action.min()) < -1.0 - ACTION_BOUND_TOLERANCE or
                        float(teacher_action.max()) > 1.0 + ACTION_BOUND_TOLERANCE):
                    raise RuntimeError(f"Corrective teacher action is out of range: {path}:{group.name}")
                state_parts.append(state)
                action_parts.append(teacher_action)
                episode_seeds.add(seed)
                round_ids.add(int(group.attrs["round_id"]))
                file_count += len(state)
            per_file.append({"path": str(path), "transitions": file_count})
    if not state_parts:
        raise RuntimeError("No corrective transitions were loaded")
    return {
        "states": np.concatenate(state_parts, axis=0),
        "actions": np.concatenate(action_parts, axis=0),
        "round_ids": sorted(round_ids),
        "episode_seeds": sorted(episode_seeds),
        "files": per_file,
    }


def validation_metrics(actor, dataset, batch_size, device):
    actor.eval()
    squared = absolute = 0.0
    elements = transitions = 0
    max_abs = 0.0
    nonfinite = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            state = torch.as_tensor(dataset.states[start:start + batch_size], device=device)
            target = torch.as_tensor(dataset.actions[start:start + batch_size], device=device)
            prediction = actor(state, deterministic=True, return_log_prob=False)[0]
            difference = prediction - target
            squared += float(torch.square(difference).sum().cpu())
            absolute += float(torch.abs(difference).sum().cpu())
            max_abs = max(max_abs, float(torch.abs(difference).max().cpu()))
            elements += int(difference.numel())
            transitions += int(difference.shape[0])
            nonfinite += int((~torch.isfinite(prediction)).sum().cpu())
    return {
        "val_mse": squared / elements,
        "val_mae": absolute / elements,
        "val_max_abs_error": max_abs,
        "validation_transitions": transitions,
        "nan_inf_count": nonfinite,
    }


def sample_balanced(expert, corrective, batch_size, expert_fraction, rng):
    expert_count = int(round(batch_size * expert_fraction))
    corrective_count = batch_size - expert_count
    expert_indices = rng.integers(0, len(expert), size=expert_count)
    corrective_indices = rng.integers(0, len(corrective["states"]), size=corrective_count)
    states = np.concatenate((expert.states[expert_indices], corrective["states"][corrective_indices]), axis=0)
    actions = np.concatenate((expert.actions[expert_indices], corrective["actions"][corrective_indices]), axis=0)
    permutation = rng.permutation(batch_size)
    return states[permutation], actions[permutation]


def save_checkpoint(path, actor, optimizer, source_payload, config, round_id, epoch,
                    validation, total_updates, corrective):
    payload = copy.deepcopy(source_payload)
    payload.update({
        "format_version": "multi_il_full_action_rl.stage3.actor.v1",
        "stage": "stage3b_dagger_actor_distillation",
        "experiment_id": config["experiment_id"],
        "round_id": round_id,
        "epoch": epoch,
        "actor_state_dict": copy.deepcopy(actor.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "validation": copy.deepcopy(validation),
        "best_val_mse": validation["val_mse"],
        "total_gradient_updates": total_updates,
        "dagger_target": "teacher_action_on_actual_mixed_policy_history",
        "expert_fraction": config["expert_fraction"],
        "corrective_datasets": [item["path"] for item in corrective["files"]],
        "corrective_transition_count": len(corrective["states"]),
        "train_seeds": list(range(config["train_seeds"][0], config["train_seeds"][1] + 1)),
        "validation_seeds": list(range(config["heldout_seeds"][0], config["heldout_seeds"][1] + 1)),
    })
    torch.save(payload, path)


def main():
    args = parse_args()
    config = read_json(args.config)
    run_dir = Path(args.run_dir)
    atomic_json(run_dir / "config.json", config)
    round_dir = run_dir / f"round_{args.round_id}"
    round_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_start, train_end = config["train_seeds"]
    heldout_start, heldout_end = config["heldout_seeds"]
    if set(range(train_start, train_end + 1)) & set(range(heldout_start, heldout_end + 1)):
        raise RuntimeError("Stage 3B train and held-out seed ranges overlap")
    dataset_info = inspect_dataset(config["stage1_rnn_dataset"])
    if Path(dataset_info["checkpoint"]).resolve() != Path(config["teacher_checkpoint"]).resolve():
        raise RuntimeError("Base expert dataset teacher checkpoint mismatch")
    expert = Stage3BSuccessTransitionDataset(
        config["stage1_rnn_dataset"], range(train_start, train_end + 1)
    )
    validation = Stage3BSuccessTransitionDataset(
        config["stage1_rnn_dataset"], range(heldout_start, heldout_end + 1)
    )
    corrective = load_corrective(args.corrective_dataset, train_start, train_end)
    expected_corrective_rounds = list(range(1, args.round_id + 1))
    if corrective["round_ids"] != expected_corrective_rounds:
        raise RuntimeError(
            "DAgger training requires all corrective rounds accumulated through the current "
            f"round: expected={expected_corrective_rounds}, got={corrective['round_ids']}"
        )
    if any(seed >= heldout_start for seed in corrective["episode_seeds"]):
        raise RuntimeError("Held-out validation leakage detected in corrective data")

    device = select_device(args.device)
    seed_everything(int(config["random_seed"]) + args.round_id)
    actor, source_payload = load_actor_checkpoint(
        args.input_checkpoint, device=device, freeze_log_std=True
    )
    if source_payload["architecture"] != {
        "state_dim": 59, "action_dim": 14, "hidden_dims": [256, 256]
    }:
        raise RuntimeError("Stage 3B actor architecture changed")
    if args.round_id == 1:
        if int(source_payload.get("round_id", -1)) != 0:
            raise RuntimeError("Round 1 training must initialize from the Stage 3B Round-0 Actor")
    elif int(source_payload.get("round_id", -1)) != args.round_id - 1:
        raise RuntimeError(
            f"Round {args.round_id} training must continue from Round {args.round_id - 1} Actor"
        )
    validate_frozen_log_std(actor, expected=float(config["log_std"]))
    optimizer = torch.optim.Adam(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
    )
    optimizer_state = source_payload.get("optimizer_state_dict")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    epochs = int(args.epochs or config["epochs_per_round"])
    patience = int(config["early_stopping_patience"])
    batch_size = int(config["batch_size"])
    steps_per_epoch = int(math.ceil((len(expert) + len(corrective["states"])) / batch_size))
    rng = np.random.default_rng(int(config["random_seed"]) + 1000 * args.round_id)

    print("=" * 80)
    print(f"Stage 3B Round {args.round_id} DAgger Actor Training")
    print(f"Input checkpoint: {args.input_checkpoint}")
    print(f"Expert successful transitions: {len(expert)}")
    print(f"Aggregated corrective transitions: {len(corrective['states'])}")
    print(f"Corrective rounds: {corrective['round_ids']}")
    print(f"Expert/corrective fraction: {config['expert_fraction']:.2f}/{1-config['expert_fraction']:.2f}")
    print(f"Held-out successful validation transitions: {len(validation)}")
    print("Target action: frozen BC-RNN teacher action (never executed action)")
    print("=" * 80)

    best_mse, best_epoch = math.inf, None
    stale_epochs = total_updates = 0
    rows = []
    best_path = checkpoint_dir / f"round{args.round_id}_best.pth"
    last_path = checkpoint_dir / f"round{args.round_id}_last.pth"
    for epoch in range(1, epochs + 1):
        actor.train()
        squared, elements = 0.0, 0
        for _ in range(steps_per_epoch):
            state_np, action_np = sample_balanced(
                expert, corrective, batch_size, float(config["expert_fraction"]), rng
            )
            state = torch.as_tensor(state_np, dtype=torch.float32, device=device)
            target = torch.as_tensor(action_np, dtype=torch.float32, device=device)
            prediction = actor(state, deterministic=True, return_log_prob=False)[0]
            loss = torch.mean(torch.square(prediction - target))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Stage 3B loss round={args.round_id} epoch={epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            squared += float(torch.square(prediction.detach() - target).sum().cpu())
            elements += int(target.numel())
            total_updates += 1
        train_mse = squared / elements
        metrics = validation_metrics(actor, validation, batch_size, device)
        if metrics["nan_inf_count"]:
            raise RuntimeError("Non-finite held-out imitation prediction")
        row = {"epoch": epoch, "train_mse": train_mse, **metrics, "total_updates": total_updates}
        rows.append(row)
        print(
            f"round {args.round_id} epoch {epoch:03d}/{epochs} | train_mse {train_mse:.8f} | "
            f"val_mse {metrics['val_mse']:.8f} | val_mae {metrics['val_mae']:.8f}"
        )
        save_checkpoint(
            last_path, actor, optimizer, source_payload, config, args.round_id,
            epoch, metrics, total_updates, corrective,
        )
        if metrics["val_mse"] < best_mse:
            best_mse, best_epoch, stale_epochs = metrics["val_mse"], epoch, 0
            save_checkpoint(
                best_path, actor, optimizer, source_payload, config, args.round_id,
                epoch, metrics, total_updates, corrective,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}; patience={patience}")
                break

    with (round_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    reloaded, _ = load_actor_checkpoint(best_path, device=device, freeze_log_std=True)
    validate_frozen_log_std(reloaded, expected=float(config["log_std"]))
    sample = torch.as_tensor(validation.states[:8], device=device)
    with torch.no_grad():
        deterministic = reloaded(sample, deterministic=True, return_log_prob=False)[0]
        stochastic, log_prob = stochastic_action_and_log_prob(reloaded, sample)
    if deterministic.shape != (8, 14) or stochastic.shape != (8, 14) or log_prob.shape != (8, 1):
        raise RuntimeError("Stage 3B checkpoint reload API shape failure")
    if not all(torch.isfinite(value).all() for value in (deterministic, stochastic, log_prob)):
        raise RuntimeError("Stage 3B checkpoint reload produced NaN/Inf")
    summary = {
        "round_id": args.round_id, "input_checkpoint": str(Path(args.input_checkpoint)),
        "expert_success_transitions": len(expert),
        "aggregated_corrective_transitions": len(corrective["states"]),
        "corrective_round_ids": corrective["round_ids"],
        "corrective_seeds": corrective["episode_seeds"],
        "heldout_seeds_used_for_gradient": False,
        "expert_fraction": config["expert_fraction"], "best_epoch": best_epoch,
        "best_val_mse": best_mse, "epochs_completed": len(rows),
        "total_gradient_updates": total_updates, "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path), "target_is_teacher_action": True,
        "checkpoint_reload_sanity": "PASS",
    }
    atomic_json(round_dir / "training_summary.json", summary)
    print(f"Round {args.round_id} training complete | best_epoch={best_epoch} best_val_mse={best_mse:.10f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
