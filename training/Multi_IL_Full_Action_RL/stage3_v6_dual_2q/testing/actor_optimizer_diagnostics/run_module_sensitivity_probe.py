#!/usr/bin/env python3
"""Testing-only Stage3-v6 Actor module-sensitivity causal probe.

Question
--------
Which Actor parameter subspaces convert a production RL update into the largest
policy-output drift?

This experiment removes Adam preconditioning geometry and equalizes the *actual
global parameter-step L2* across all branches. Every branch:
  * starts from the exact random2q/multi_q critic_ready Actor,
  * sees the same frozen critic_ready Critic,
  * sees the same pre-sampled production-shaped Actor batches in the same order,
  * uses the unchanged production Actor objective and grad clipping,
  * moves along the negative masked RL gradient,
  * receives exactly the same scalar global step budget on each update.

The common step budget is produced by a shadow production Adam driven by the
FULL_GRADIENT branch's clipped gradients. Shadow Adam supplies magnitude only;
no branch uses Adam's parameter-space direction.

Branches
--------
FULL_GRADIENT : all nonzero Actor gradients
RNN_ONLY      : recurrent core only
GMM_MEAN_ONLY : component-mean head only
GMM_LOGITS_ONLY : categorical-logit head only
MEAN_LOGITS   : component means + categorical logits
RNN_MEAN      : recurrent core + component means

No simulator is started. Critic and replay are frozen at critic_ready. This is
a local mechanism test, not an exact online-training replay.
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
from run_adam_dynamics_probe import (  # noqa: E402
    build_batch_bank,
    lr_at_actor_update,
)
from stage3_v5_actor import load_exact_actor, module_hash  # noqa: E402
from stage3_v6_agent import strict_stage2_load  # noqa: E402


SEED = 20261003
DEFAULT_UPDATES = 500
DEFAULT_BATCH_BANK = 64
MILESTONES = (0, 1, 10, 50, 100, 250, 500)

BRANCH_GROUPS = {
    "FULL_GRADIENT": None,
    "RNN_ONLY": frozenset({"rnn"}),
    "GMM_MEAN_ONLY": frozenset({"gmm_mean"}),
    "GMM_LOGITS_ONLY": frozenset({"gmm_logits"}),
    "MEAN_LOGITS": frozenset({"gmm_mean", "gmm_logits"}),
    "RNN_MEAN": frozenset({"rnn", "gmm_mean"}),
}


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def named_parameters_snapshot(actor):
    return {
        name: parameter.detach().clone()
        for name, parameter in actor.named_parameters()
    }


def named_gradients(actor):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in actor.named_parameters()
    }


def sum_squares(values):
    tensors = [
        value.detach().float().square().sum()
        for value in values
        if value is not None
    ]
    if not tensors:
        return torch.tensor(0.0)
    total = tensors[0]
    for tensor in tensors[1:]:
        total = total + tensor
    return total


def vector_l2(values):
    return sum_squares(values.values()).sqrt()


def masked_gradients(gradients, allowed_groups):
    if allowed_groups is None:
        return gradients
    return {
        name: (
            grad
            if grad is not None and actor_group(name) in allowed_groups
            else None
        )
        for name, grad in gradients.items()
    }


def group_stats(named_tensors):
    accum = {}
    counts = {}
    for name, tensor in named_tensors.items():
        if tensor is None:
            continue
        group = actor_group(name)
        value = tensor.detach().float()
        accum[group] = accum.get(group, 0.0) + value.square().sum()
        counts[group] = counts.get(group, 0) + value.numel()
    if accum:
        values = list(accum.values())
        total = values[0]
        for value in values[1:]:
            total = total + value
        accum["total"] = total
        counts["total"] = sum(counts.values())
    result = {}
    for group, value in accum.items():
        l2 = scalar(value.sqrt())
        result[group] = {
            "l2": l2,
            "rms": l2 / math.sqrt(counts[group]),
            "numel": int(counts[group]),
        }
    return result


def parameter_drift(actor, reference):
    values = {
        name: parameter.detach() - reference[name]
        for name, parameter in actor.named_parameters()
    }
    return group_stats(values)


def actual_delta(actor, before):
    return {
        name: parameter.detach() - before[name]
        for name, parameter in actor.named_parameters()
    }


def cosine(first, second):
    dot = None
    first_sq = None
    second_sq = None
    for name in first:
        a = first[name]
        b = second.get(name)
        if a is None or b is None:
            continue
        a = a.detach().float()
        b = b.detach().float()
        current_dot = (a * b).sum()
        current_a = a.square().sum()
        current_b = b.square().sum()
        dot = current_dot if dot is None else dot + current_dot
        first_sq = current_a if first_sq is None else first_sq + current_a
        second_sq = current_b if second_sq is None else second_sq + current_b
    if dot is None:
        return None
    denominator = (first_sq * second_sq).sqrt().clamp_min(1e-30)
    return scalar(dot / denominator)


class ShadowAdamMagnitude:
    """Production-Adam moments used only to generate a common scalar step budget."""

    def __init__(self, actor_optimizer_state):
        groups = actor_optimizer_state["param_groups"]
        if len(groups) != 1:
            raise RuntimeError("Expected one production Actor optimizer param group")
        group = groups[0]
        if actor_optimizer_state.get("state"):
            raise RuntimeError("Expected empty Actor optimizer state at critic_ready")
        if float(group.get("weight_decay", 0.0)) != 0.0:
            raise RuntimeError("Expected production Actor Adam weight_decay=0")
        if bool(group.get("amsgrad", False)):
            raise RuntimeError("Expected production Actor Adam amsgrad=False")
        self.beta1, self.beta2 = [float(x) for x in group["betas"]]
        self.eps = float(group["eps"])
        self.state = {}

    @torch.no_grad()
    def step_l2(self, gradients, lr):
        update_units = {}
        for name, grad in gradients.items():
            if grad is None:
                continue
            state = self.state.get(name)
            if state is None:
                state = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(grad),
                    "exp_avg_sq": torch.zeros_like(grad),
                }
                self.state[name] = state
            state["step"] += 1
            state["exp_avg"].mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            state["exp_avg_sq"].mul_(self.beta2).addcmul_(
                grad, grad, value=1.0 - self.beta2
            )
            step = int(state["step"])
            bias1 = 1.0 - self.beta1 ** step
            bias2 = 1.0 - self.beta2 ** step
            denom = (
                state["exp_avg_sq"].sqrt()
                .div_(math.sqrt(bias2))
                .add_(self.eps)
            )
            update_units[name] = state["exp_avg"].div(bias1).div(denom)
        unit_l2 = vector_l2(update_units)
        return unit_l2 * float(lr), update_units

    def summary(self):
        steps = [int(state["step"]) for state in self.state.values()]
        return {
            "active_parameter_tensors": len(steps),
            "step_min": min(steps) if steps else 0,
            "step_max": max(steps) if steps else 0,
        }


def compute_gradient(actor, critic, batch, scale, offset, config, device):
    actor.train()
    actor.zero_grad(set_to_none=True)
    loss, details = production_actor_loss(
        actor,
        critic,
        batch,
        scale,
        offset,
        config,
        device,
    )
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("Non-finite Actor RL loss")
    loss.backward()
    raw_norm = torch.nn.utils.clip_grad_norm_(
        actor.parameters(),
        float(config["actor_max_grad_norm"]),
    )
    gradients = named_gradients(actor)
    if any(
        grad is not None and not bool(torch.isfinite(grad).all())
        for grad in gradients.values()
    ):
        raise FloatingPointError("Non-finite Actor gradient")
    return gradients, scalar(raw_norm), scalar(details["expected_q1"].mean())


@torch.no_grad()
def apply_normalized_step(actor, gradients, target_l2):
    grad_l2_t = vector_l2(gradients)
    grad_l2 = scalar(grad_l2_t)
    target = scalar(target_l2)
    if target < 0 or not math.isfinite(target):
        raise FloatingPointError("Invalid target parameter-step L2")
    if target > 0 and grad_l2 <= 0:
        raise RuntimeError("Masked gradient is zero but target step is nonzero")
    scale = (
        target_l2 / grad_l2_t.clamp_min(1e-30)
        if target > 0
        else torch.zeros_like(grad_l2_t)
    )
    scale_value = scalar(scale)
    before = named_parameters_snapshot(actor)
    for name, parameter in actor.named_parameters():
        grad = gradients.get(name)
        if grad is not None:
            parameter.add_(grad, alpha=-scale_value)
    delta = actual_delta(actor, before)
    actual_l2 = scalar(vector_l2(delta))
    ratio = actual_l2 / target if target > 0 else 1.0
    return {
        "masked_gradient_l2": grad_l2,
        "matched_scale": scale_value,
        "target_parameter_step_l2": target,
        "actual_parameter_step_l2": actual_l2,
        "step_match_ratio": ratio,
        "delta_cosine_negative_masked_gradient": cosine(
            delta,
            {
                name: None if grad is None else -grad
                for name, grad in gradients.items()
            },
        ),
        "parameter_step_groups": group_stats(delta),
    }


@torch.no_grad()
def fixed_q1(actor, critic, batch, scale, offset, config, device):
    old_mode = actor.training
    try:
        actor.train()
        loss, details = production_actor_loss(
            actor,
            critic,
            batch,
            scale,
            offset,
            config,
            device,
        )
        return scalar(details["expected_q1"].mean()), scalar(loss)
    finally:
        actor.train(old_mode)


def milestone_metrics(
    actor,
    reference_outputs,
    diagnostic_observations,
    reference_parameters,
    critic,
    diagnostic_batch,
    scale,
    offset,
    config,
    device,
    baseline_q1,
):
    outputs = actor_outputs(actor, diagnostic_observations, device)
    drift = policy_drift(outputs, reference_outputs)
    q1, rl_loss = fixed_q1(
        actor, critic, diagnostic_batch, scale, offset, config, device
    )
    return {
        "policy_drift": drift,
        "parameter_drift": parameter_drift(actor, reference_parameters),
        "fixed_q1": q1,
        "fixed_q1_gain": q1 - baseline_q1,
        "fixed_rl_loss": rl_loss,
        "actor_hash": module_hash(actor),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--batch-bank-size", type=int, default=DEFAULT_BATCH_BANK)
    parser.add_argument("--trace-every", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results_module_sensitivity_probe.json",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=HERE / "trace_module_sensitivity.jsonl",
    )
    args = parser.parse_args()

    if not 1 <= int(args.updates) <= 2000:
        raise ValueError("--updates must be in [1,2000]")
    if not 4 <= int(args.batch_bank_size) <= 256:
        raise ValueError("--batch-bank-size must be in [4,256]")
    if not 1 <= int(args.trace_every) <= 500:
        raise ValueError("--trace-every must be in [1,500]")

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
    if int(ready["env_steps"]) != 130000:
        raise RuntimeError(f"Expected random2q critic_ready=130000, got {ready['env_steps']}")
    if int(ready.get("actor_updates", -1)) != 0:
        raise RuntimeError("critic_ready Actor must have zero Actor updates")

    optimizer_audit = checkpoint_optimizer_audit(ready)
    if not optimizer_audit["state_is_empty"]:
        raise RuntimeError("Expected empty Actor Adam state at critic_ready")

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
        config, ready, int(args.batch_bank_size)
    )
    diagnostic_observations = torch.as_tensor(
        diagnostic_batch["observations"],
        dtype=torch.float32,
        device=device,
    )
    reference_outputs = actor_outputs(
        reference_actor, diagnostic_observations, device
    )
    reference_parameters = named_parameters_snapshot(reference_actor)
    baseline_q1, baseline_rl_loss = fixed_q1(
        reference_actor,
        critic,
        diagnostic_batch,
        scale,
        offset,
        config,
        device,
    )

    branches = {}
    for branch_name in BRANCH_GROUPS:
        actor = copy.deepcopy(reference_actor)
        actor.requires_grad_(True)
        actor.train()
        if module_hash(actor) != reference_hash:
            raise RuntimeError(f"Initial branch Actor mismatch: {branch_name}")
        branches[branch_name] = {
            "actor": actor,
            "milestones": {},
        }

    for branch_name, branch in branches.items():
        branch["milestones"]["0"] = milestone_metrics(
            branch["actor"],
            reference_outputs,
            diagnostic_observations,
            reference_parameters,
            critic,
            diagnostic_batch,
            scale,
            offset,
            config,
            device,
            baseline_q1,
        )

    shadow_adam = ShadowAdamMagnitude(ready["actor_optimizer"])
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
    target_step_schedule = []

    for update_index in range(1, int(args.updates) + 1):
        lr, env_step, schedule = lr_at_actor_update(
            config, ready_step, update_index
        )
        batch_index = (update_index - 1) % len(batch_bank)
        batch = batch_bank[batch_index]

        # FULL_GRADIENT is evaluated first. Its clipped gradient drives shadow
        # Adam to generate one scalar step budget shared by every branch.
        full_actor = branches["FULL_GRADIENT"]["actor"]
        full_grad, full_raw_norm, full_q1 = compute_gradient(
            full_actor, critic, batch, scale, offset, config, device
        )
        target_l2, shadow_units = shadow_adam.step_l2(full_grad, lr)
        target_value = scalar(target_l2)
        target_step_schedule.append(
            {
                "actor_update": int(update_index),
                "implied_env_step": int(env_step),
                "actor_lr": float(lr),
                "target_parameter_step_l2": target_value,
            }
        )

        for branch_name, allowed_groups in BRANCH_GROUPS.items():
            branch = branches[branch_name]
            actor = branch["actor"]

            if branch_name == "FULL_GRADIENT":
                gradients = full_grad
                raw_norm = full_raw_norm
                q1_before = full_q1
            else:
                gradients, raw_norm, q1_before = compute_gradient(
                    actor, critic, batch, scale, offset, config, device
                )

            masked = masked_gradients(gradients, allowed_groups)
            full_clipped_l2 = scalar(vector_l2(gradients))
            masked_l2 = scalar(vector_l2(masked))
            if target_value > 0 and masked_l2 <= 0:
                raise RuntimeError(
                    f"{branch_name} has zero masked gradient at update {update_index}"
                )

            step = apply_normalized_step(actor, masked, target_l2)
            should_trace = (
                update_index <= 10
                or update_index % int(args.trace_every) == 0
                or update_index in selected_milestones
            )
            if should_trace:
                row = {
                    "testing_only": True,
                    "branch": branch_name,
                    "actor_update": int(update_index),
                    "implied_env_step": int(env_step),
                    "actor_lr": float(lr),
                    "batch_bank_index": int(batch_index),
                    "raw_grad_norm_before_clip": float(raw_norm),
                    "full_clipped_gradient_l2": full_clipped_l2,
                    "masked_gradient_l2": masked_l2,
                    "masked_to_full_gradient_l2_ratio": (
                        masked_l2 / full_clipped_l2
                        if full_clipped_l2 > 0 else None
                    ),
                    "q1_before_update": float(q1_before),
                    "allowed_groups": (
                        "ALL" if allowed_groups is None
                        else sorted(allowed_groups)
                    ),
                    **step,
                }
                traces.append(row)

            if update_index in selected_milestones:
                metrics = milestone_metrics(
                    actor,
                    reference_outputs,
                    diagnostic_observations,
                    reference_parameters,
                    critic,
                    diagnostic_batch,
                    scale,
                    offset,
                    config,
                    device,
                    baseline_q1,
                )
                metrics["current_step"] = {
                    "target_parameter_step_l2": target_value,
                    "actual_parameter_step_l2": step["actual_parameter_step_l2"],
                    "step_match_ratio": step["step_match_ratio"],
                    "masked_gradient_l2": masked_l2,
                    "full_clipped_gradient_l2": full_clipped_l2,
                }
                branch["milestones"][str(update_index)] = metrics

        if update_index in selected_milestones:
            console = {
                "actor_update": int(update_index),
                "implied_env_step": int(env_step),
                "actor_lr": float(lr),
                "target_parameter_step_l2": target_value,
            }
            for branch_name, branch in branches.items():
                metrics = branch["milestones"][str(update_index)]
                console[branch_name] = {
                    "parameter_drift_l2": metrics[
                        "parameter_drift"
                    ]["total"]["l2"],
                    "sampled_action_drift": metrics[
                        "policy_drift"
                    ]["sampled_action_l2_mean"],
                    "weighted_action_drift": metrics[
                        "policy_drift"
                    ]["weighted_action_l2_mean"],
                    "hidden_l2": metrics[
                        "policy_drift"
                    ]["hidden_l2_mean"],
                    "fixed_q1_gain": metrics["fixed_q1_gain"],
                }
            print(json.dumps(console, sort_keys=True), flush=True)

    # Validate actual global step matching on all sampled rows.
    by_branch = {}
    for branch_name in BRANCH_GROUPS:
        rows = [row for row in traces if row["branch"] == branch_name]
        ratios = np.asarray(
            [float(row["step_match_ratio"]) for row in rows],
            dtype=np.float64,
        )
        by_branch[branch_name] = {
            "trace_rows": len(rows),
            "step_match_ratio": {
                "median": float(np.median(ratios)),
                "max_abs_error_from_1": float(np.max(np.abs(ratios - 1.0))),
            },
            "masked_to_full_gradient_l2_ratio": {
                "median": float(
                    np.median(
                        [
                            row["masked_to_full_gradient_l2_ratio"]
                            for row in rows
                            if row["masked_to_full_gradient_l2_ratio"] is not None
                        ]
                    )
                ),
            },
        }

    final_rows = {}
    for branch_name, branch in branches.items():
        final = branch["milestones"][str(args.updates)]
        final_rows[branch_name] = {
            "parameter_drift_l2": final["parameter_drift"]["total"]["l2"],
            "sampled_action_drift": final["policy_drift"][
                "sampled_action_l2_mean"
            ],
            "weighted_action_drift": final["policy_drift"][
                "weighted_action_l2_mean"
            ],
            "hidden_l2": final["policy_drift"]["hidden_l2_mean"],
            "component_mean_rms": final["policy_drift"][
                "component_mean_rms"
            ],
            "logits_rms": final["policy_drift"]["logits_rms"],
            "categorical_kl_mean": final["policy_drift"][
                "categorical_kl_mean"
            ],
            "top1_mode_change_fraction": final["policy_drift"][
                "top1_mode_change_fraction"
            ],
            "fixed_q1_gain": final["fixed_q1_gain"],
        }

    # Directional sensitivity is output drift per cumulative parameter L2.
    full_final = final_rows["FULL_GRADIENT"]
    for branch_name, row in final_rows.items():
        parameter_l2 = row["parameter_drift_l2"]
        row["weighted_action_drift_per_parameter_l2"] = (
            row["weighted_action_drift"] / parameter_l2
            if parameter_l2 > 0 else None
        )
        row["sampled_action_drift_per_parameter_l2"] = (
            row["sampled_action_drift"] / parameter_l2
            if parameter_l2 > 0 else None
        )
        row["weighted_action_drift_vs_full"] = (
            row["weighted_action_drift"] / full_final["weighted_action_drift"]
            if full_final["weighted_action_drift"] > 0 else None
        )
        row["sampled_action_drift_vs_full"] = (
            row["sampled_action_drift"] / full_final["sampled_action_drift"]
            if full_final["sampled_action_drift"] > 0 else None
        )

    sensitivity_rankings = {
        "weighted_action_drift_descending": sorted(
            BRANCH_GROUPS,
            key=lambda name: final_rows[name]["weighted_action_drift"],
            reverse=True,
        ),
        "sampled_action_drift_descending": sorted(
            BRANCH_GROUPS,
            key=lambda name: final_rows[name]["sampled_action_drift"],
            reverse=True,
        ),
        "hidden_l2_descending": sorted(
            BRANCH_GROUPS,
            key=lambda name: final_rows[name]["hidden_l2"],
            reverse=True,
        ),
        "categorical_kl_descending": sorted(
            BRANCH_GROUPS,
            key=lambda name: final_rows[name]["categorical_kl_mean"],
            reverse=True,
        ),
    }

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
        "branches": {
            branch_name: {
                "allowed_groups": (
                    "ALL" if groups is None else sorted(groups)
                ),
                "milestones": branches[branch_name]["milestones"],
            }
            for branch_name, groups in BRANCH_GROUPS.items()
        },
        "common_step_budget": {
            "source": (
                "shadow production Adam driven only by FULL_GRADIENT clipped gradients; "
                "magnitude only, no Adam parameter direction applied"
            ),
            "schedule": target_step_schedule,
            "shadow_state": shadow_adam.summary(),
        },
        "step_matching": by_branch,
        "baseline_fixed_q1": baseline_q1,
        "baseline_fixed_rl_loss": baseline_rl_loss,
        "final_comparison": final_rows,
        "sensitivity_rankings": sensitivity_rankings,
        "interpretation": {
            "goal": (
                "localize which Actor parameter subspaces convert equal global "
                "parameter-step budgets into the largest policy-output drift"
            ),
            "not_claimed": (
                "This frozen-Critic/replay probe does not by itself prove which "
                "module causes online closed-loop collapse."
            ),
        },
        "source_integrity": {
            "reference_actor_hash": reference_hash,
            "critic_hash": critic_hash,
            "reference_actor_hash_unchanged": module_hash(reference_actor)
            == reference_hash,
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
