#!/usr/bin/env python3
"""Train Stage2.3: five independent recurrent Q critics on finite MC returns.

The history/data contract intentionally matches the historical full-prefix
Stage2.2 source used by stage3v5_stage22_rnn4k_multi6k_20260922.  Only the
ensemble size changes from 2 to 5.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STAGE2_2 = HERE.parent / "stage2_2_history_aware_critic"
for directory in (HERE, STAGE2_2):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from evaluation import evaluate, select_best_checkpoint  # noqa: E402
from fiveq_critic import (  # noqa: E402
    build_critic,
    checkpoint_payload,
    per_q_hashes,
)
from sequence_dataset import load_splits  # noqa: E402
from sequence_sampler import SequenceSampler  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(HERE / "stage2_3_config.json")
    )
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--mode",
        choices=("rnn_q", "multi_q", "both"),
        default="both",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--max-updates", type=int)
    return parser.parse_args()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def select_device(name):
    name = str(name)
    if name.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "NPU requested but torch_npu cannot be imported"
            ) from exc
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("NPU requested but unavailable")
        torch.npu.set_device(name)
    return torch.device(name)


def synchronize(device):
    if device.type == "npu":
        torch.npu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def labels(mode):
    return ("rnn_q", "multi_q") if mode == "both" else (mode,)


def all_finite(parameters):
    return all(torch.isfinite(p).all().item() for p in parameters)


def gradients_finite(model):
    bad = []
    sq_sum = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if not torch.isfinite(grad).all():
            bad.append(name)
        sq_sum += float(torch.sum(grad.float() ** 2).detach().cpu())
    return bad, float(np.sqrt(sq_sum))


def optimizer_finite(optimizer):
    bad = []
    for param_index, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                bad.append(f"{param_index}:{key}")
    return bad


def finite_validation(value, path="validation"):
    if isinstance(value, dict):
        for key, item in value.items():
            finite_validation(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_validation(item, f"{path}[{index}]")
    elif value is not None and isinstance(value, (float, int)):
        if not np.isfinite(value):
            raise FloatingPointError(f"non-finite {path}: {value}")


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fail(out, step, reason, extra=None):
    payload = {
        "status": "FAIL",
        "step": int(step),
        "reason": reason,
        "extra": extra or {},
    }
    write(out / "failure.json", payload)
    raise RuntimeError(f"Stage2.3 failed at step {step}: {reason}")


def main():
    args = arguments()
    config = json.loads(Path(args.config).read_text())
    for key, value in (
        ("dataset_root", args.dataset_root),
        ("output_root", args.output_root),
        ("max_updates", args.max_updates),
    ):
        if value is not None:
            config[key] = value

    if config["stage_version"] != "2.3":
        raise RuntimeError("Stage2.3 config version changed")
    if config["critic_type"] != "history_aware_5q":
        raise RuntimeError("Stage2.3 requires history_aware_5q")
    if int(config["num_qs"]) != 5:
        raise RuntimeError("Stage2.3 requires exactly five Qs")
    if (
        config["history_semantics"]
        != "full_episode_prefix_unroll_learning_mask"
    ):
        raise RuntimeError("Stage2.3 historical history contract changed")
    if int(config["max_updates"]) != 50000 and args.max_updates is None:
        raise RuntimeError("default Stage2.3 training must remain 50K")

    device = select_device(args.device)
    seed = int(config["training_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "npu":
        torch.npu.manual_seed_all(seed)

    train, val = load_splits(
        config["dataset_root"],
        range(config["train_seed_start"], config["train_seed_end"] + 1),
        range(config["val_seed_start"], config["val_seed_end"] + 1),
        config["gamma"],
    )

    run = Path(config["output_root"]) / (
        args.run_id
        or f"stage2_3_5q_{datetime.now():%Y%m%d_%H%M%S}"
    )
    if run.exists():
        raise FileExistsError(run)
    (run / "shared").mkdir(parents=True)

    write(run / "shared" / "config_resolved.json", config)
    write(run / "shared" / "architecture_contract.json", {
        "stage_version": "2.3",
        "critic_type": "history_aware_5q",
        "num_qs": 5,
        "history_semantics": config["history_semantics"],
        "recurrent_initial_state": "episode_step_zero",
        "token_schema": config["token_schema"],
        "current_action_enters_recurrence": False,
        "supervision": "learning_mask_over_full_episode_prefix",
    })
    write(run / "shared" / "dataset_manifest.json", {
        "train_seeds": [config["train_seed_start"], config["train_seed_end"]],
        "validation_seeds": [config["val_seed_start"], config["val_seed_end"]],
        "sources": {policy: train[policy].path for policy in train},
    })

    torch.manual_seed(seed)
    if device.type == "npu":
        torch.npu.manual_seed_all(seed)
    template = build_critic(config, device)
    hashes = per_q_hashes(template)
    if len(set(hashes)) != 5:
        raise RuntimeError("five Q networks are not independently initialized")
    initial_state = copy.deepcopy(template.state_dict())
    torch.save(
        initial_state,
        run / "shared" / "initial_state_history_aware_5q.pth",
    )
    write(run / "shared" / "initialization_audit.json", {
        "same_initial_5q_state_used_for_multi_and_rnn": True,
        "five_q_parameter_hashes": hashes,
        "all_five_initial_qs_distinct": True,
        "full_model_hash": tensor_hash(initial_state),
    })
    del template
    gc.collect()

    for label in labels(args.mode):
        out = run / label
        checkpoints = out / "checkpoints"
        checkpoints.mkdir(parents=True)

        model = build_critic(config, device)
        model.load_state_dict(copy.deepcopy(initial_state), strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["critic_lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        sampler = SequenceSampler(
            train,
            config["legacy_replay_burn_in_length"],
            config["learning_sequence_length"],
            config["horizon"],
            seed,
            label == "multi_q",
        )

        validation_records = []
        max_updates = int(config["max_updates"])
        eval_interval = int(config["eval_interval"])
        checkpoint_interval = int(config["checkpoint_interval"])
        if checkpoint_interval != eval_interval:
            raise RuntimeError(
                "Stage2.3 selection requires checkpoint_interval == eval_interval"
            )

        for step in range(1, max_updates + 1):
            model.train()
            started = time.perf_counter()
            batch = sampler.sample(config["sequence_batch_size"])
            sample_ms = (time.perf_counter() - started) * 1000.0

            tensors = {
                key: torch.as_tensor(batch[key], device=device)
                for key in (
                    "observations",
                    "previous_actions",
                    "progress",
                    "actions",
                    "returns",
                    "learning_mask",
                )
            }
            for key, value in tensors.items():
                if not torch.isfinite(value).all():
                    fail(out, step, f"nonfinite_input:{key}")

            synchronize(device)
            started = time.perf_counter()
            qs = model.forward_sequence(
                tensors["observations"],
                tensors["previous_actions"],
                tensors["progress"],
                tensors["actions"],
            )
            synchronize(device)
            forward_ms = (time.perf_counter() - started) * 1000.0

            if len(qs) != 5:
                fail(out, step, "expected_five_q_outputs")
            if any(not torch.isfinite(q).all() for q in qs):
                fail(out, step, "nonfinite_q_output")

            mask = tensors["learning_mask"].bool()
            target = tensors["returns"]
            effective = int(mask.sum().item())
            losses = [
                torch.nn.functional.mse_loss(q[mask], target[mask])
                for q in qs
            ]
            loss = sum(losses)
            if not torch.isfinite(loss):
                fail(out, step, "nonfinite_loss")

            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            started = time.perf_counter()
            loss.backward()
            synchronize(device)
            backward_ms = (time.perf_counter() - started) * 1000.0

            bad_gradients, grad_norm = gradients_finite(model)
            if bad_gradients:
                fail(
                    out,
                    step,
                    "nonfinite_gradient",
                    {"parameters": bad_gradients},
                )

            synchronize(device)
            started = time.perf_counter()
            optimizer.step()
            synchronize(device)
            optimizer_ms = (time.perf_counter() - started) * 1000.0

            if not all_finite(model.parameters()):
                fail(out, step, "nonfinite_parameter_after_optimizer")
            bad_optimizer = optimizer_finite(optimizer)
            if bad_optimizer:
                fail(
                    out,
                    step,
                    "nonfinite_optimizer_state",
                    {"states": bad_optimizer},
                )

            total_ms = sample_ms + forward_ms + backward_ms + optimizer_ms
            row = {
                "step": int(step),
                "model_training": bool(model.training),
                "q_losses": [float(x.detach().cpu()) for x in losses],
                "loss_sum": float(loss.detach().cpu()),
                "sample_ms": sample_ms,
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "optimizer_ms": optimizer_ms,
                "grad_norm_preclip": grad_norm,
                "effective_timesteps": effective,
                "effective_timesteps_per_second": (
                    effective * 1000.0 / max(total_ms, 1e-12)
                ),
                "sampling": sampler.proportions(),
            }

            if step % eval_interval == 0 or step == max_updates:
                validation = evaluate(
                    model, val, device, config["horizon"]
                )
                finite_validation(validation)
                row["validation"] = validation
                checkpoint_path = (
                    checkpoints / f"step_{step:08d}.pth"
                )
                torch.save(
                    checkpoint_payload(
                        model, optimizer, config, step, validation
                    ),
                    checkpoint_path,
                )
                record = {
                    "step": int(step),
                    "checkpoint": str(checkpoint_path.resolve()),
                    "validation": validation,
                }
                validation_records.append(record)
                append(out / "validation_records.jsonl", record)

            append(out / "train_metrics.jsonl", row)

        torch.save(
            checkpoint_payload(
                model,
                optimizer,
                config,
                max_updates,
                validation_records[-1]["validation"],
            ),
            checkpoints / "last.pth",
        )

        selection = select_best_checkpoint(validation_records, label)
        write(out / "checkpoint_selection.json", selection)
        selected = torch.load(
            selection["selected_checkpoint"], map_location=device
        )
        selected["best_selection"] = {
            "scope": selection["scope"],
            "rule": selection["rule"],
            "selected_step": selection["selected_step"],
            "selected_composite_rank_score": (
                selection["selected_composite_rank_score"]
            ),
        }
        torch.save(selected, checkpoints / "best.pth")
        model.load_state_dict(selected["critic_state_dict"], strict=True)

        final = evaluate(model, val, device, config["horizon"])
        finite_validation(final)
        write(out / "final_validation.json", final)
        write(out / "sampling_audit.json", {
            "counts": sampler.counts,
            "ratios": sampler.proportions(),
        })

        peak = (
            torch.npu.max_memory_allocated(device)
            if device.type == "npu"
            else (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0
            )
        )
        write(out / "performance.json", {
            "per_q_parameters": [
                int(sum(p.numel() for p in q.parameters()))
                for q in model.qs
            ],
            "total_parameters": int(
                sum(p.numel() for p in model.parameters())
            ),
            "peak_device_memory_bytes": int(peak),
            "history_semantics": config["history_semantics"],
            "num_qs": 5,
            "all_five_initial_qs_distinct": True,
            "best_checkpoint_step": selection["selected_step"],
            "best_checkpoint_rule": selection["rule"],
        })

        del model, optimizer, sampler, selected
        gc.collect()
        if device.type == "npu":
            torch.npu.empty_cache()

    print(json.dumps({
        "status": "COMPLETE",
        "run_dir": str(run),
        "max_updates": int(config["max_updates"]),
        "num_qs": 5,
    }, indent=2))


if __name__ == "__main__":
    main()
