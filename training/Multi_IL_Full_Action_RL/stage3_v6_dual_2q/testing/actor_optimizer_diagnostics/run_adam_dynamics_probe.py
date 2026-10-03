#!/usr/bin/env python3
"""Testing-only multi-step Actor optimizer dynamics probe for Stage3-v6.

Purpose
-------
Diagnose whether a mature pretrained Actor, handed to a *fresh* Adam optimizer
at critic_ready, accumulates policy drift materially faster than plain SGD
under the same nominal production LR schedule.

Scope is deliberately narrow:
  * random2q / multi_q only
  * starts from that branch's own critic_ready checkpoint
  * frozen critic at critic_ready
  * production Actor objective unchanged (final-token Q1 component mean)
  * production-shaped 50/50 offline-online aligned Actor batches
  * exact same pre-sampled batch order for both optimizer branches
  * no simulator, no rollout, no production mutation, no checkpoint overwrite

This is a mechanism probe, not an exact replay of online training because the
Critic and replay contents are intentionally frozen at critic_ready.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL_ROOT = HERE.parents[2]
V5 = RL_ROOT / "stage3_v5_rgmm_td3"
V6 = RL_ROOT / "stage3_v6_dual_2q"
for folder in (HERE, V5, V6):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from run_optimizer_step_probe import (  # noqa: E402
    actor_group,
    actor_outputs,
    checkpoint_optimizer_audit,
    policy_drift,
    production_actor_loss,
    resolve_device,
)
from stage3_v5_actor import load_exact_actor, module_hash  # noqa: E402
from stage3_v5_replay import (  # noqa: E402
    BalancedOfflineDemonstrations,
    OnlineSequenceReplay,
    aligned_sequence_batch,
)
from stage3_v6_agent import strict_stage2_load  # noqa: E402


SEED = 20261003
DEFAULT_UPDATES = 4000
DEFAULT_BATCH_BANK = 64
MILESTONES = (0, 1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 2000, 3000, 4000)


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with open(path, "x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def clone_parameters(actor):
    return {
        name: parameter.detach().clone()
        for name, parameter in actor.named_parameters()
    }


def group_numel(actor):
    counts = {}
    for name, parameter in actor.named_parameters():
        group = actor_group(name)
        counts[group] = counts.get(group, 0) + parameter.numel()
    counts["total"] = sum(parameter.numel() for parameter in actor.parameters())
    return counts


def squared_norm_by_group(named_tensors):
    accum = {}
    for name, tensor in named_tensors.items():
        if tensor is None:
            continue
        group = actor_group(name)
        value = tensor.detach().float().square().sum()
        accum[group] = accum.get(group, 0.0) + value
    if accum:
        values = list(accum.values())
        total = values[0]
        for value in values[1:]:
            total = total + value
        accum["total"] = total
    return accum


def parameter_delta(actor, before):
    return {
        name: parameter.detach() - before[name]
        for name, parameter in actor.named_parameters()
    }


def parameter_drift(actor, reference):
    return {
        name: parameter.detach() - reference[name]
        for name, parameter in actor.named_parameters()
    }


def named_gradients(actor):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in actor.named_parameters()
    }


def dot_maps(first, second):
    total = None
    by_group = {}
    for name in first:
        a = first[name]
        b = second.get(name)
        if a is None or b is None:
            continue
        value = (a.detach().float() * b.detach().float()).sum()
        group = actor_group(name)
        by_group[group] = by_group.get(group, 0.0) + value
        total = value if total is None else total + value
    if total is None:
        total = torch.tensor(0.0)
    by_group["total"] = total
    return by_group


def group_stats(named_tensors, counts):
    sq = squared_norm_by_group(named_tensors)
    result = {}
    for group, value in sq.items():
        count = counts[group]
        l2 = scalar(value.sqrt())
        result[group] = {
            "l2": l2,
            "rms": l2 / math.sqrt(count),
            "numel": int(count),
        }
    return result


def cosine_total(first, second):
    dots = dot_maps(first, second)
    first_sq = squared_norm_by_group(first).get("total")
    second_sq = squared_norm_by_group(second).get("total")
    if first_sq is None or second_sq is None:
        return torch.tensor(float("nan"))
    denom = (first_sq * second_sq).sqrt().clamp_min(1e-30)
    return dots["total"] / denom


def optimizer_state_snapshot(optimizer, actor, counts):
    exp_avg_sq = {}
    exp_avg2_sq = {}
    steps = {}
    for name, parameter in actor.named_parameters():
        state = optimizer.state.get(parameter, {})
        if not state:
            continue
        group = actor_group(name)
        exp_avg = state["exp_avg"].detach().float()
        exp_avg_sq_tensor = state["exp_avg_sq"].detach().float()
        exp_avg2_sq[group] = exp_avg2_sq.get(group, 0.0) + exp_avg.square().sum()
        exp_avg_sq[group] = exp_avg_sq.get(group, 0.0) + exp_avg_sq_tensor.sum()
        step = state.get("step", 0)
        step_value = (
            int(step.detach().cpu().item())
            if torch.is_tensor(step)
            else int(step)
        )
        steps.setdefault(group, set()).add(step_value)

    result = {}
    for group in sorted(exp_avg_sq):
        result[group] = {
            "step_values": sorted(steps[group]),
            "exp_avg_rms": scalar(exp_avg2_sq[group].sqrt()) / math.sqrt(counts[group]),
            "sqrt_exp_avg_sq_rms": scalar(exp_avg_sq[group].sqrt()) / math.sqrt(counts[group]),
            "numel": int(counts[group]),
        }
    return result


def lr_at_actor_update(config, ready_step, actor_update_index):
    utd = float(config["utd"])
    policy_delay = int(config["policy_delay"])
    env_per_actor = policy_delay / utd
    if not math.isfinite(env_per_actor) or env_per_actor <= 0:
        raise RuntimeError("Invalid UTD/policy-delay schedule")
    # Current formal contract is exactly 4 / 0.25 = 16 env steps / Actor update.
    env_step = int(round(ready_step + actor_update_index * env_per_actor))
    warmup_steps = min(300000, max(100000, int(ready_step)))
    progress = min(1.0, max(0, env_step - ready_step) / warmup_steps)
    lr = float(config["actor_warmup"]["target_actor_lr"]) * progress
    return lr, env_step, {
        "utd": utd,
        "policy_delay": policy_delay,
        "env_steps_per_actor_update": env_per_actor,
        "critic_ready_step": int(ready_step),
        "actor_warmup_steps": int(warmup_steps),
        "warmup_progress": progress,
    }


def build_batch_bank(config, ready, bank_size):
    offline_paths = [
        config["offline_sources"][key]
        for key in ("bc_rnn", "bc_transformer", "bc_gmm")
    ]
    offline = BalancedOfflineDemonstrations(
        offline_paths,
        seed=int(config["training_seed"]),
    )
    if ready.get("offline_sampler_state") is not None:
        offline.load_state_dict(ready["offline_sampler_state"])

    replay_path = Path(ready["online_sequence_replay"])
    if not replay_path.is_file():
        raise FileNotFoundError(replay_path)
    online = OnlineSequenceReplay.load(replay_path)
    online.current = {}

    batch_size = int(config["recurrent_replay"]["actor_sequence_batch_size"])
    horizon = int(config["actor_source_contract"]["rnn_horizon"])
    bank = [
        aligned_sequence_batch(
            offline,
            online,
            count=batch_size,
            length=horizon,
            horizon=horizon,
        )
        for _ in range(int(bank_size) + 1)
    ]
    return bank[:-1], bank[-1], replay_path


def prepare_branch(reference_actor, ready_optimizer, kind):
    actor = copy.deepcopy(reference_actor)
    actor.requires_grad_(True)
    actor.train()
    if kind == "PRODUCTION_ADAM":
        optimizer = torch.optim.Adam(actor.parameters(), lr=0.0, weight_decay=0.0)
        optimizer.load_state_dict(copy.deepcopy(ready_optimizer))
        if optimizer.state:
            raise RuntimeError("Expected fresh Adam state at critic_ready")
    elif kind == "SGD_SAME_LR":
        optimizer = torch.optim.SGD(actor.parameters(), lr=0.0, weight_decay=0.0)
    else:
        raise ValueError(kind)
    return actor, optimizer


def one_update(
    actor,
    optimizer,
    critic,
    batch,
    scale,
    offset,
    config,
    lr,
    reference_parameters,
    counts,
    previous_gradient,
):
    for group in optimizer.param_groups:
        group["lr"] = float(lr)

    optimizer.zero_grad(set_to_none=True)
    loss, details = production_actor_loss(
        actor,
        critic,
        batch,
        scale,
        offset,
        config,
        scale.device,
    )
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("Non-finite Actor loss")
    loss.backward()

    raw_norm = torch.nn.utils.clip_grad_norm_(
        actor.parameters(),
        float(config["actor_max_grad_norm"]),
    )
    gradients = named_gradients(actor)
    clipped_sq = squared_norm_by_group(gradients)
    clipped_norm = clipped_sq["total"].sqrt()
    grad_cos_prev = (
        None
        if previous_gradient is None
        else cosine_total(gradients, previous_gradient).detach()
    )

    before = clone_parameters(actor)
    optimizer.step()
    delta = parameter_delta(actor, before)
    delta_sq = squared_norm_by_group(delta)
    delta_norm = delta_sq["total"].sqrt()

    negative_grad = {
        name: None if grad is None else -grad
        for name, grad in gradients.items()
    }
    delta_cos_neg_grad = cosine_total(delta, negative_grad).detach()

    numel = counts["total"]
    grad_rms = clipped_norm / math.sqrt(numel)
    delta_rms = delta_norm / math.sqrt(numel)
    amplification = (
        delta_rms / (float(lr) * grad_rms).clamp_min(1e-30)
        if lr > 0
        else torch.tensor(float("nan"), device=delta_norm.device)
    )

    cumulative = parameter_drift(actor, reference_parameters)
    cumulative_sq = squared_norm_by_group(cumulative)
    cumulative_norm = cumulative_sq["total"].sqrt()

    record = {
        "loss": loss.detach(),
        "q1_expected": details["expected_q1"].mean().detach(),
        "raw_grad_norm": raw_norm.detach(),
        "clipped_grad_norm": clipped_norm.detach(),
        "parameter_step_l2": delta_norm.detach(),
        "parameter_step_rms": delta_rms.detach(),
        "effective_amplification": amplification.detach(),
        "delta_cosine_negative_gradient": delta_cos_neg_grad,
        "gradient_cosine_previous": grad_cos_prev,
        "cumulative_parameter_drift_l2": cumulative_norm.detach(),
    }
    return record, gradients, delta


def host_record(record):
    result = {}
    for key, value in record.items():
        if isinstance(value, dict):
            result[key] = value
        elif value is None:
            result[key] = None
        elif torch.is_tensor(value):
            result[key] = scalar(value)
        else:
            result[key] = value
    return result


@torch.no_grad()
def milestone_metrics(actor, optimizer, reference_outputs, diagnostic_observations,
                      reference_parameters, counts, device, kind):
    outputs = actor_outputs(actor, diagnostic_observations, device)
    drift = policy_drift(outputs, reference_outputs)
    cumulative = group_stats(parameter_drift(actor, reference_parameters), counts)
    result = {
        "policy_drift": drift,
        "cumulative_parameter_drift": cumulative,
        "actor_hash": module_hash(actor),
    }
    if kind == "PRODUCTION_ADAM":
        result["adam_state"] = optimizer_state_snapshot(optimizer, actor, counts)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--batch-bank-size", type=int, default=DEFAULT_BATCH_BANK)
    parser.add_argument(
        "--trace-every",
        type=int,
        default=10,
        help="Write compact per-update dynamics every N updates; milestones are always written.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results_adam_dynamics_probe.json",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=HERE / "trace_adam_dynamics.jsonl",
    )
    args = parser.parse_args()

    if not 1 <= int(args.updates) <= 10000:
        raise ValueError("--updates must be in [1,10000]")
    if not 4 <= int(args.batch_bank_size) <= 256:
        raise ValueError("--batch-bank-size must be in [4,256]")
    if not 1 <= int(args.trace_every) <= 1000:
        raise ValueError("--trace-every must be in [1,1000]")

    run = args.run.resolve()
    device = resolve_device(args.device)
    config = read_json(run / "shared" / "config_resolved.json")
    manifest = read_json(run / "shared" / "stage2_source_manifest.json")
    ready_path = run / "random2q" / "multi_q" / "checkpoints" / "critic_ready.pth"
    ready = torch.load(ready_path, map_location="cpu", weights_only=False)

    if ready.get("critic_target_mode") != "random2q":
        raise RuntimeError("random2q checkpoint required")
    if ready.get("group") != "multi_q":
        raise RuntimeError("multi_q checkpoint required")
    if int(ready.get("actor_updates", -1)) != 0:
        raise RuntimeError("critic_ready Actor must have zero Actor updates")

    optimizer_audit = checkpoint_optimizer_audit(ready)
    if not optimizer_audit["state_is_empty"]:
        raise RuntimeError("Expected fresh Adam state at random2q critic_ready")

    reference_actor, rollout, _ = load_exact_actor(
        run / "shared" / "bc_rnn_gmm_source.pth",
        device,
    )
    reference_actor.load_state_dict(ready["actor"], strict=True)
    reference_actor.eval().requires_grad_(False)
    reference_hash = module_hash(reference_actor)

    critic, _ = strict_stage2_load(manifest["multi_q"]["checkpoint"], device)
    critic.load_state_dict(ready["q1_q2"], strict=True)
    critic.eval().requires_grad_(False)
    critic_hash = module_hash(critic)

    scale = torch.as_tensor(
        rollout.action_normalization_stats["actions"]["scale"],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 1, 14)
    offset = torch.as_tensor(
        rollout.action_normalization_stats["actions"]["offset"],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 1, 14)

    batch_bank, diagnostic_batch, replay_path = build_batch_bank(
        config,
        ready,
        args.batch_bank_size,
    )
    diagnostic_observations = torch.as_tensor(
        diagnostic_batch["observations"],
        dtype=torch.float32,
        device=device,
    )
    reference_outputs = actor_outputs(
        reference_actor,
        diagnostic_observations,
        device,
    )
    reference_parameters = clone_parameters(reference_actor)
    counts = group_numel(reference_actor)

    branches = {}
    for kind in ("PRODUCTION_ADAM", "SGD_SAME_LR"):
        actor, optimizer = prepare_branch(
            reference_actor,
            ready["actor_optimizer"],
            kind,
        )
        branches[kind] = {
            "actor": actor,
            "optimizer": optimizer,
            "previous_gradient": None,
            "milestones": {
                "0": milestone_metrics(
                    actor,
                    optimizer,
                    reference_outputs,
                    diagnostic_observations,
                    reference_parameters,
                    counts,
                    device,
                    kind,
                )
            },
        }

    ready_step = int(ready["env_steps"])
    selected_milestones = sorted(
        set(
            milestone
            for milestone in MILESTONES
            if milestone <= int(args.updates)
        )
        | {int(args.updates)}
    )

    traces = []
    for update_index in range(1, int(args.updates) + 1):
        lr, env_step, schedule = lr_at_actor_update(
            config,
            ready_step,
            update_index,
        )
        batch = batch_bank[(update_index - 1) % len(batch_bank)]

        for kind, branch in branches.items():
            record, gradient, delta = one_update(
                branch["actor"],
                branch["optimizer"],
                critic,
                batch,
                scale,
                offset,
                config,
                lr,
                reference_parameters,
                counts,
                branch["previous_gradient"],
            )
            branch["previous_gradient"] = gradient

            should_trace = (
                update_index <= 20
                or update_index % int(args.trace_every) == 0
                or update_index in selected_milestones
            )
            if should_trace:
                row = host_record(record)
                row.update(
                    {
                        "testing_only": True,
                        "branch": kind,
                        "actor_update": int(update_index),
                        "implied_env_step": int(env_step),
                        "actor_lr": float(lr),
                        "batch_bank_index": int((update_index - 1) % len(batch_bank)),
                        "schedule": schedule,
                    }
                )
                traces.append(row)

            if update_index in selected_milestones:
                milestone = milestone_metrics(
                    branch["actor"],
                    branch["optimizer"],
                    reference_outputs,
                    diagnostic_observations,
                    reference_parameters,
                    counts,
                    device,
                    kind,
                )
                milestone["current_gradient"] = group_stats(gradient, counts)
                milestone["current_parameter_step"] = group_stats(delta, counts)
                milestone["current_update_scalars"] = host_record(record)
                branch["milestones"][str(update_index)] = milestone

        if update_index in selected_milestones:
            console = {"actor_update": update_index, "env_step": env_step, "actor_lr": lr}
            for kind, branch in branches.items():
                metric = branch["milestones"][str(update_index)]
                console[kind] = {
                    "parameter_drift_l2": metric[
                        "cumulative_parameter_drift"
                    ]["total"]["l2"],
                    "sampled_action_drift": metric[
                        "policy_drift"
                    ]["sampled_action_l2_mean"],
                    "weighted_action_drift": metric[
                        "policy_drift"
                    ]["weighted_action_l2_mean"],
                }
            print(json.dumps(console, sort_keys=True), flush=True)

    # Convert only compact milestone content to the main summary.
    summary_branches = {
        kind: {"milestones": branch["milestones"]}
        for kind, branch in branches.items()
    }

    # Aggregate the per-update dynamics numerically.
    dynamics = {}
    for kind in branches:
        rows = [row for row in traces if row["branch"] == kind]
        def finite_values(key):
            return np.asarray(
                [
                    row[key]
                    for row in rows
                    if row.get(key) is not None and math.isfinite(float(row[key]))
                ],
                dtype=np.float64,
            )

        amp = finite_values("effective_amplification")
        grad_cos = finite_values("gradient_cosine_previous")
        delta_cos = finite_values("delta_cosine_negative_gradient")
        dynamics[kind] = {
            "actor_updates_total": int(args.updates),
            "trace_rows": len(rows),
            "effective_amplification": {
                "median": float(np.median(amp)),
                "p10": float(np.percentile(amp, 10)),
                "p90": float(np.percentile(amp, 90)),
                "first": float(amp[0]),
                "last": float(amp[-1]),
            },
            "gradient_cosine_previous": {
                "median": float(np.median(grad_cos)) if len(grad_cos) else None,
                "p10": float(np.percentile(grad_cos, 10)) if len(grad_cos) else None,
                "p90": float(np.percentile(grad_cos, 90)) if len(grad_cos) else None,
                "positive_fraction": float(np.mean(grad_cos > 0)) if len(grad_cos) else None,
            },
            "delta_cosine_negative_gradient": {
                "median": float(np.median(delta_cos)),
                "p10": float(np.percentile(delta_cos, 10)),
                "p90": float(np.percentile(delta_cos, 90)),
            },
            "final_cumulative_parameter_drift_l2": float(
                rows[-1]["cumulative_parameter_drift_l2"]
            ),
        }

    final_adam = summary_branches["PRODUCTION_ADAM"]["milestones"][str(args.updates)]
    final_sgd = summary_branches["SGD_SAME_LR"]["milestones"][str(args.updates)]
    adam_action = final_adam["policy_drift"]["sampled_action_l2_mean"]
    sgd_action = final_sgd["policy_drift"]["sampled_action_l2_mean"]
    adam_param = final_adam["cumulative_parameter_drift"]["total"]["l2"]
    sgd_param = final_sgd["cumulative_parameter_drift"]["total"]["l2"]

    result = {
        "testing_only": True,
        "formal_training_modified": False,
        "formal_checkpoint_modified": False,
        "target_mode": "random2q",
        "group": "multi_q",
        "run": str(run),
        "checkpoint": str(ready_path),
        "online_replay_sidecar": str(replay_path),
        "device": str(device),
        "seed": SEED,
        "updates": int(args.updates),
        "batch_bank_size": int(args.batch_bank_size),
        "trace_every": int(args.trace_every),
        "trace_sampling": (
            "updates 1..20, every trace_every updates, and all milestones; "
            "gradient_cosine_previous always compares truly consecutive updates"
        ),
        "mechanism_scope": (
            "frozen critic_ready Critic and replay; production Actor objective, "
            "production LR schedule, same pre-sampled batch order"
        ),
        "limitation": (
            "This is not exact online-training replay: Critic and replay do not "
            "evolve after critic_ready and no environment is executed."
        ),
        "checkpoint_contract": {
            "env_steps": ready_step,
            "critic_updates": int(ready["updates"]),
            "actor_updates": int(ready["actor_updates"]),
            "actor_gate_open": bool(ready["actor_gate_open"]),
            "actor_optimizer": optimizer_audit,
        },
        "schedule_contract": {
            "utd": float(config["utd"]),
            "policy_delay": int(config["policy_delay"]),
            "env_steps_per_actor_update": int(
                round(int(config["policy_delay"]) / float(config["utd"]))
            ),
            "target_actor_lr": float(config["actor_warmup"]["target_actor_lr"]),
            "milestones": selected_milestones,
        },
        "branches": summary_branches,
        "dynamics": dynamics,
        "dynamics_sampling_note": (
            "Distribution summaries use the compact trace sample, not every update. "
            "Each sampled gradient_cosine_previous still compares update t with t-1."
        ),
        "paired_final_comparison": {
            "adam_to_sgd_cumulative_parameter_drift_ratio": (
                float(adam_param / sgd_param) if sgd_param > 0 else None
            ),
            "adam_to_sgd_sampled_action_drift_ratio": (
                float(adam_action / sgd_action) if sgd_action > 0 else None
            ),
            "production_adam_final_sampled_action_drift": float(adam_action),
            "sgd_same_lr_final_sampled_action_drift": float(sgd_action),
        },
        "source_integrity": {
            "reference_actor_hash": reference_hash,
            "critic_hash": critic_hash,
            "reference_actor_hash_unchanged": module_hash(reference_actor) == reference_hash,
            "critic_hash_unchanged": module_hash(critic) == critic_hash,
        },
    }

    if not result["source_integrity"]["reference_actor_hash_unchanged"]:
        raise RuntimeError("Reference Actor changed")
    if not result["source_integrity"]["critic_hash_unchanged"]:
        raise RuntimeError("Frozen Critic changed")

    write_json(args.output.resolve(), result)
    write_jsonl(args.trace.resolve(), traces)

    print(f"WROTE {args.output.resolve()}", flush=True)
    print(f"WROTE {args.trace.resolve()}", flush=True)
    print("FORMAL TRAINING REMAINS STOPPED", flush=True)


if __name__ == "__main__":
    main()
