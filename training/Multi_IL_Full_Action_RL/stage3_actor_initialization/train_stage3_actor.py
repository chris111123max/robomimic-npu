#!/usr/bin/env python3
"""Stage 3: distill the frozen BC-RNN behavior into a SAC-compatible actor."""

import argparse
import copy
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from actor_network import (  # noqa: E402
    build_actor,
    load_actor_checkpoint,
    stochastic_action_and_log_prob,
)
from stage3_dataset import Stage3TransitionDataset, inspect_dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(THIS_DIR / "stage3_config.json"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def select_device(name):
    requested = name or "npu:0"
    if requested.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU was requested, but torch_npu cannot be imported") from exc
        if not torch.npu.is_available():
            raise RuntimeError("NPU was requested, but torch.npu.is_available() is false")
    return torch.device(requested)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def resolve_dataset(config, manifest_path):
    manifest = load_json(manifest_path)
    if manifest.get("status") != "selected":
        raise RuntimeError(f"Stage 1.5 manifest status is {manifest.get('status')!r}; refusing to guess data")
    selected = manifest.get("datasets", {})
    entry = selected.get("bc_rnn")
    if entry is None:
        raise RuntimeError("Stage 1.5 manifest does not contain selected bc_rnn data")
    path = entry if isinstance(entry, str) else entry.get("path", entry.get("dataset_path"))
    if path is None:
        raise RuntimeError("Cannot resolve bc_rnn dataset path from Stage 1.5 manifest")
    return Path(path), manifest


def offline_metrics(actor, dataset, batch_size, device):
    actor.eval()
    total = 0
    squared = 0.0
    absolute = 0.0
    max_absolute = 0.0
    l2_sum = 0.0
    per_dim_squared = np.zeros(dataset.action_dim, dtype=np.float64)
    target_sum = np.zeros(dataset.action_dim, dtype=np.float64)
    target_square_sum = np.zeros(dataset.action_dim, dtype=np.float64)
    student_sum = np.zeros(dataset.action_dim, dtype=np.float64)
    student_square_sum = np.zeros(dataset.action_dim, dtype=np.float64)
    target_min = np.full(dataset.action_dim, np.inf)
    target_max = np.full(dataset.action_dim, -np.inf)
    student_min = np.full(dataset.action_dim, np.inf)
    student_max = np.full(dataset.action_dim, -np.inf)
    nonfinite = 0
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            states = torch.as_tensor(dataset.states[start:start + batch_size], device=device)
            target = dataset.actions[start:start + batch_size].astype(np.float64)
            student = actor(states, deterministic=True, return_log_prob=False)[0].cpu().numpy().astype(np.float64)
            diff = student - target
            count = target.shape[0]
            total += count
            squared += float(np.square(diff).sum())
            absolute += float(np.abs(diff).sum())
            max_absolute = max(max_absolute, float(np.abs(diff).max()))
            l2_sum += float(np.linalg.norm(diff, axis=1).sum())
            per_dim_squared += np.square(diff).sum(axis=0)
            target_sum += target.sum(axis=0)
            target_square_sum += np.square(target).sum(axis=0)
            student_sum += student.sum(axis=0)
            student_square_sum += np.square(student).sum(axis=0)
            target_min = np.minimum(target_min, target.min(axis=0))
            target_max = np.maximum(target_max, target.max(axis=0))
            student_min = np.minimum(student_min, student.min(axis=0))
            student_max = np.maximum(student_max, student.max(axis=0))
            nonfinite += int((~np.isfinite(student)).sum())
    target_mean = target_sum / total
    student_mean = student_sum / total
    target_std = np.sqrt(np.maximum(target_square_sum / total - np.square(target_mean), 0.0))
    student_std = np.sqrt(np.maximum(student_square_sum / total - np.square(student_mean), 0.0))
    return {
        "num_transitions": total,
        "action_mse": squared / (total * dataset.action_dim),
        "action_mae": absolute / (total * dataset.action_dim),
        "max_absolute_error": max_absolute,
        "per_dimension_mse": (per_dim_squared / total).tolist(),
        "mean_l2_action_error": l2_sum / total,
        "target_action_mean": target_mean.tolist(),
        "target_action_std": target_std.tolist(),
        "student_action_mean": student_mean.tolist(),
        "student_action_std": student_std.tolist(),
        "target_action_min": target_min.tolist(),
        "target_action_max": target_max.tolist(),
        "student_action_min": student_min.tolist(),
        "student_action_max": student_max.tolist(),
        "student_nonfinite_count": nonfinite,
        "student_within_environment_range": bool(np.all(student_min >= -1.0001) and np.all(student_max <= 1.0001)),
    }


def make_checkpoint(actor, optimizer, epoch, config, dataset_info, validation, best_val_mse):
    return {
        "format_version": "multi_il_full_action_rl.stage3.actor.v1",
        "stage": "stage3_actor_initialization",
        "epoch": int(epoch),
        "actor_state_dict": copy.deepcopy(actor.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "architecture": {
            "state_dim": int(config["architecture"]["state_dim"]),
            "action_dim": int(config["architecture"]["action_dim"]),
            "hidden_dims": list(config["architecture"]["hidden_dims"]),
        },
        "action_distribution": copy.deepcopy(config["distribution"]),
        "action_convention": copy.deepcopy(config["action_convention"]),
        "observation_keys": list(dataset_info["canonical_keys"]),
        "observation_shapes": copy.deepcopy(dataset_info["canonical_shapes"]),
        "teacher_checkpoint": dataset_info.get("checkpoint"),
        "source_dataset": dataset_info["path"],
        "train_seeds": list(config["train_seeds"]),
        "validation_seeds": list(config["validation_seeds"]),
        "validation": copy.deepcopy(validation),
        "best_val_mse": float(best_val_mse),
        "evaluation_horizon": int(config["evaluation_horizon"]),
        "terminate_on_success": bool(config["terminate_on_success"]),
    }


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    declared_split = load_json(Path(config["seed_split"]))
    if config["train_seeds"] != declared_split.get("train_seeds") or config["validation_seeds"] != declared_split.get("validation_seeds"):
        raise RuntimeError("Stage 3 seed lists differ from the canonical Stage 2 seed split")
    if config["train_seeds"] != list(range(10000, 10080)) or config["validation_seeds"] != list(range(10080, 10100)):
        raise RuntimeError("Stage 3 requires train seeds 10000..10079 and validation seeds 10080..10099")
    if args.device:
        config["device"] = args.device
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.output_root:
        config["output_root"] = args.output_root
    if args.smoke_test:
        config["epochs"] = 2
        config["candidate_epochs"] = [1, 2]
        config["train_seeds"] = config["train_seeds"][:5]
        config["validation_seeds"] = config["validation_seeds"][:2]

    device = select_device(config.get("device"))
    seed_everything(int(config["random_seed"]))
    dataset_path, manifest = resolve_dataset(config, Path(config["stage1_5_manifest"]))
    dataset_info = inspect_dataset(dataset_path)
    if Path(dataset_info["checkpoint"]).resolve() != Path(config["teacher_checkpoint"]).resolve():
        raise RuntimeError(
            "Selected bc_rnn dataset checkpoint differs from the fixed Stage 3 teacher: "
            f"{dataset_info['checkpoint']} != {config['teacher_checkpoint']}"
        )
    expected_keys = list(dataset_info["canonical_keys"])
    train_data = Stage3TransitionDataset(dataset_path, config["train_seeds"])
    val_data = Stage3TransitionDataset(dataset_path, config["validation_seeds"])
    if dataset_info["state_dim"] != config["architecture"]["state_dim"] or dataset_info["action_dim"] != config["architecture"]["action_dim"]:
        raise RuntimeError("Configured dimensions disagree with the selected Stage 1 dataset")

    run_id = args.run_id or (("smoke_test_" if args.smoke_test else "") + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir = Path(config["output_root"]) / run_id
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "resolved_config.json", config)
    write_json(run_dir / "dataset_summary.json", {
        "manifest": str(Path(config["stage1_5_manifest"])),
        "manifest_status": manifest.get("status"),
        "dataset": dataset_info,
        "train": train_data.statistics,
        "validation": val_data.statistics,
    })

    actor = build_actor(
        state_dim=config["architecture"]["state_dim"],
        action_dim=config["architecture"]["action_dim"],
        hidden_dims=config["architecture"]["hidden_dims"],
        initial_log_std=config["distribution"]["initial_log_std"],
        freeze_log_std=config["distribution"]["freeze_log_std_during_distillation"],
        device=device,
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    rng = np.random.default_rng(int(config["random_seed"]))
    history = []
    best_val_mse = math.inf
    best_epoch = None
    candidate_epochs = set(int(value) for value in config["candidate_epochs"])

    print("=" * 80)
    print("Stage 3 SAC-compatible Actor Initialization")
    print(f"Device: {device}")
    print(f"Dataset: {dataset_path}")
    print(f"Observation order: {expected_keys}")
    print(f"Train transitions: {len(train_data)} | validation transitions: {len(val_data)}")
    print("Target: stored post-tanh environment action; actor output: tanh(mu); no extra scaling")
    print(f"log_std initialized to {config['distribution']['initial_log_std']} and frozen during distillation")
    print("=" * 80)

    for epoch in range(1, int(config["epochs"]) + 1):
        actor.train()
        order = rng.permutation(len(train_data))
        train_squared = 0.0
        train_elements = 0
        for start in range(0, len(order), int(config["batch_size"])):
            indices = order[start:start + int(config["batch_size"])]
            states = torch.as_tensor(train_data.states[indices], device=device)
            targets = torch.as_tensor(train_data.actions[indices], device=device)
            predictions = actor(states, deterministic=True, return_log_prob=False)[0]
            loss = torch.mean(torch.square(predictions - targets))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_squared += float(torch.square(predictions.detach() - targets).sum().cpu())
            train_elements += int(targets.numel())
        validation = offline_metrics(actor, val_data, int(config["batch_size"]), device)
        train_mse = train_squared / train_elements
        record = {"epoch": epoch, "train_action_mse": train_mse, **validation}
        history.append(record)
        print(f"epoch {epoch:03d}/{config['epochs']} | train_mse {train_mse:.8f} | val_mse {validation['action_mse']:.8f} | val_mae {validation['action_mae']:.8f}")
        payload = make_checkpoint(actor, optimizer, epoch, config, dataset_info, validation, min(best_val_mse, validation["action_mse"]))
        if validation["action_mse"] < best_val_mse:
            best_val_mse = validation["action_mse"]
            best_epoch = epoch
            payload["best_val_mse"] = best_val_mse
            torch.save(payload, checkpoint_dir / "best_val_mse.pth")
        if epoch in candidate_epochs:
            torch.save(payload, checkpoint_dir / f"candidate_epoch_{epoch:03d}.pth")
        if epoch == int(config["epochs"]):
            torch.save(payload, checkpoint_dir / "last.pth")

    write_json(run_dir / "metrics_history.json", history)
    reloaded_actor, reloaded = load_actor_checkpoint(checkpoint_dir / "best_val_mse.pth", device=device)
    sample = torch.as_tensor(val_data.states[:min(8, len(val_data))], device=device)
    with torch.no_grad():
        deterministic = reloaded_actor(sample, deterministic=True, return_log_prob=False)[0]
        stochastic, log_prob = stochastic_action_and_log_prob(reloaded_actor, sample)
    sanity = {
        "checkpoint_reloaded": True,
        "deterministic_shape": list(deterministic.shape),
        "stochastic_shape": list(stochastic.shape),
        "log_prob_shape": list(log_prob.shape),
        "all_finite": bool(torch.isfinite(deterministic).all() and torch.isfinite(stochastic).all() and torch.isfinite(log_prob).all()),
        "action_range_valid": bool((deterministic >= -1.0001).all() and (deterministic <= 1.0001).all() and (stochastic >= -1.0001).all() and (stochastic <= 1.0001).all()),
        "format_version": reloaded["format_version"],
    }
    if sanity["deterministic_shape"] != [len(sample), dataset_info["action_dim"]] or sanity["stochastic_shape"] != [len(sample), dataset_info["action_dim"]] or sanity["log_prob_shape"] != [len(sample), 1] or not sanity["all_finite"] or not sanity["action_range_valid"]:
        raise RuntimeError(f"Actor checkpoint API sanity failed: {sanity}")
    write_json(run_dir / "checkpoint_reload_sanity.json", sanity)
    summary = {
        "status": "SMOKE_TEST_PASSED" if args.smoke_test else "TRAINING_COMPLETE",
        "run_directory": str(run_dir),
        "source_dataset": str(dataset_path),
        "teacher_checkpoint": dataset_info.get("checkpoint"),
        "teacher_loaded_during_training": False,
        "target_source": "stored_bc_rnn_post_tanh_environment_actions",
        "train_seed_count": len(config["train_seeds"]),
        "validation_seed_count": len(config["validation_seeds"]),
        "train_transition_count": len(train_data),
        "validation_transition_count": len(val_data),
        "architecture": config["architecture"],
        "distribution": config["distribution"],
        "action_convention": config["action_convention"],
        "best_epoch": best_epoch,
        "best_validation_action_mse": best_val_mse,
        "last_validation": history[-1],
        "checkpoint_reload_sanity": sanity,
        "candidate_checkpoints": sorted(path.name for path in checkpoint_dir.glob("candidate_epoch_*.pth")),
        "automatic_environment_evaluation_run": False,
        "shared_actor_selected": False,
    }
    write_json(run_dir / "stage3_training_summary.json", summary)
    latest = REPO_ROOT / "training" / "Multi_IL_Full_Action_RL" / "analysis" / "stage3" / "latest_stage3_run.json"
    write_json(latest, {"run_directory": str(run_dir), "summary": str(run_dir / "stage3_training_summary.json"), "status": summary["status"]})
    print("=" * 80)
    print(f"Stage 3 status: {summary['status']}")
    print(f"Best epoch: {best_epoch} | best validation MSE: {best_val_mse:.10f}")
    print(f"Run directory: {run_dir}")
    print("Training stops here. Candidate environment evaluation is a separate manual command.")
    print("=" * 80)


if __name__ == "__main__":
    main()
