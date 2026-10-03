#!/usr/bin/env python3
"""Testing-only Stage3-v6 Actor optimizer-step diagnostic.

This probe is intentionally narrow:
  * random2q / multi_q only
  * starts from the random2q branch's own critic_ready checkpoint
  * freezes the Critic
  * uses one fixed production-shaped Actor batch
  * computes the production Actor gradient once
  * compares the exact same clipped gradient under:
      - fresh production Adam
      - SGD with the same learning rate
      - SGD with a learning rate chosen to match Adam's global parameter-step L2
  * measures immediate parameter and policy-output drift.

No production source or formal checkpoint is modified.
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
for folder in (V5, V6):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from stage3_v5_actor import flat_to_obs, load_exact_actor, module_hash
from stage3_v5_history_critic import component_mean_q, encode_replay_contexts
from stage3_v5_replay import (
    BalancedOfflineDemonstrations,
    OnlineSequenceReplay,
    aligned_sequence_batch,
)
from stage3_v6_agent import strict_stage2_load


SEED = 20261003
DEFAULT_PROBE_OFFSETS = (16, 10000, 20000, 60000)


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


def resolve_device(name):
    if str(name).startswith("npu"):
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but torch.npu is unavailable")
        torch.npu.set_device(str(name))
    return torch.device(name)


def actor_group(name):
    if name.startswith("nets.rnn.nets."):
        return "rnn"
    if name.startswith("nets.decoder.nets.mean."):
        return "gmm_mean"
    if name.startswith("nets.decoder.nets.logits."):
        return "gmm_logits"
    if name.startswith("nets.decoder.nets.scale."):
        return "gmm_std"
    if "encoder" in name:
        return "encoder"
    return "other"


def final_distribution(sequence_distribution):
    base = sequence_distribution.component_distribution.base_dist
    return torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(
            logits=sequence_distribution.mixture_distribution.logits[:, -1]
        ),
        torch.distributions.Independent(
            torch.distributions.Normal(
                base.loc[:, -1],
                base.scale[:, -1],
            ),
            1,
        ),
    )


def production_actor_loss(actor, critic, batch, scale, offset, config, device):
    observations = torch.as_tensor(
        batch["observations"], dtype=torch.float32, device=device
    )
    actions = torch.as_tensor(
        batch["actions"], dtype=torch.float32, device=device
    )
    episode_steps = torch.as_tensor(
        batch["episode_steps"], dtype=torch.long, device=device
    )
    horizon = int(config["actor_source_contract"]["rnn_horizon"])
    if observations.shape[1] != horizon:
        raise RuntimeError(
            f"Actor batch must be exactly one horizon ({horizon}), got {observations.shape}"
        )
    if not bool(torch.all(episode_steps[:, 0].remainder(horizon).eq(0))):
        raise RuntimeError("Actor batch is not reset-boundary aligned")

    sequence_distribution = actor.forward_train(
        flat_to_obs(observations),
        rnn_init_state=None,
        return_state=False,
    )
    distribution = final_distribution(sequence_distribution)
    contexts = encode_replay_contexts(
        critic,
        observations,
        actions,
        episode_steps,
        int(config["horizon"]),
    )
    final_context = (contexts[0][:, -1], contexts[1][:, -1])
    expected, q1, q2, tensors, _ = component_mean_q(
        critic,
        final_context,
        distribution,
        scale,
        offset,
        twin_min=False,
    )
    loss = -expected.mean()
    return loss, {
        "expected_q1": expected,
        "q1_modes": q1,
        "q2_modes": q2,
        "probs": tensors["probs"],
        "means_normalized": tensors["means_normalized"],
    }


def save_rng_state(device):
    state = {
        "cpu": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if device.type == "npu":
        state["npu"] = torch.npu.get_rng_state()
    elif device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state, device):
    torch.set_rng_state(state["cpu"])
    np.random.set_state(state["numpy"])
    if device.type == "npu":
        torch.npu.set_rng_state(state["npu"])
    elif device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def seed_execution(device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if device.type == "npu":
        torch.npu.manual_seed(SEED)
    elif device.type == "cuda":
        torch.cuda.manual_seed(SEED)


def sampled_execution_stream(actor, observations, device):
    """Production-like categorical component sampling with common random numbers."""
    old_mode = actor.training
    state = save_rng_state(device)
    try:
        seed_execution(device)
        actor.eval()
        rnn_state = None
        samples = []
        for timestep in range(observations.shape[1]):
            dist, rnn_state = actor.forward_train_step(
                flat_to_obs(observations[:, timestep]),
                rnn_state=rnn_state,
            )
            samples.append(dist.sample())
        return torch.stack(samples, dim=1)
    finally:
        actor.train(old_mode)
        restore_rng_state(state, device)


@torch.no_grad()
def actor_outputs(actor, observations, device):
    old_mode = actor.training
    hidden = []
    hook = actor.nets["rnn"].nets.register_forward_hook(
        lambda _module, _inputs, output: hidden.append(output[0].detach())
    )
    try:
        actor.eval()
        dist = actor.forward_train(
            flat_to_obs(observations),
            rnn_init_state=None,
            return_state=False,
        )
        means = dist.component_distribution.base_dist.loc
        probs = dist.mixture_distribution.probs
        logits = dist.mixture_distribution.logits
        weighted = (probs.unsqueeze(-1) * means).sum(-2)
        sampled = sampled_execution_stream(actor, observations, device)
        return {
            "means": means.detach().clone(),
            "probs": probs.detach().clone(),
            "logits": logits.detach().clone(),
            "weighted": weighted.detach().clone(),
            "sampled": sampled.detach().clone(),
            "hidden": hidden[0].detach().clone(),
        }
    finally:
        hook.remove()
        actor.train(old_mode)


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def finite_tensor(tensor):
    return bool(torch.isfinite(tensor).all().item())


def named_parameter_snapshot(actor):
    return {
        name: parameter.detach().clone()
        for name, parameter in actor.named_parameters()
    }


def named_grad_snapshot(actor):
    return {
        name: (
            None
            if parameter.grad is None
            else parameter.grad.detach().clone()
        )
        for name, parameter in actor.named_parameters()
    }


def assign_grads(actor, gradients):
    for name, parameter in actor.named_parameters():
        grad = gradients[name]
        parameter.grad = None if grad is None else grad.detach().clone().to(parameter.device)


def aggregate_tensor_map(values, reference_parameters=None):
    by_group = {}
    for name, value in values.items():
        if value is None:
            continue
        group = actor_group(name)
        entry = by_group.setdefault(
            group,
            {"sq": 0.0, "abs_sum": 0.0, "max_abs": 0.0, "count": 0, "nonzero": 0},
        )
        x = value.detach().double().cpu()
        entry["sq"] += float(x.square().sum())
        entry["abs_sum"] += float(x.abs().sum())
        entry["max_abs"] = max(entry["max_abs"], float(x.abs().max()))
        entry["count"] += int(x.numel())
        entry["nonzero"] += int(torch.count_nonzero(x))
    if values:
        total = {"sq": 0.0, "abs_sum": 0.0, "max_abs": 0.0, "count": 0, "nonzero": 0}
        for entry in by_group.values():
            for key in ("sq", "abs_sum", "count", "nonzero"):
                total[key] += entry[key]
            total["max_abs"] = max(total["max_abs"], entry["max_abs"])
        by_group["total"] = total

    result = {}
    for group, entry in by_group.items():
        count = max(1, int(entry["count"]))
        row = {
            "l2": math.sqrt(entry["sq"]),
            "rms": math.sqrt(entry["sq"] / count),
            "mean_abs": entry["abs_sum"] / count,
            "max_abs": entry["max_abs"],
            "numel": int(entry["count"]),
            "nonzero_fraction": entry["nonzero"] / count,
        }
        if reference_parameters is not None and group != "total":
            param_sq = 0.0
            for name, parameter in reference_parameters.items():
                if actor_group(name) == group:
                    param_sq += float(parameter.detach().double().cpu().square().sum())
            row["relative_l2"] = (
                math.sqrt(entry["sq"] / param_sq)
                if param_sq > 0
                else None
            )
        result[group] = row

    if reference_parameters is not None and "total" in result:
        param_sq = sum(
            float(parameter.detach().double().cpu().square().sum())
            for parameter in reference_parameters.values()
        )
        result["total"]["relative_l2"] = (
            math.sqrt(by_group["total"]["sq"] / param_sq)
            if param_sq > 0
            else None
        )
    return result


def delta_map(actor, before):
    return {
        name: parameter.detach() - before[name]
        for name, parameter in actor.named_parameters()
    }


def vector_dot(a, b):
    value = 0.0
    for name in a:
        if a[name] is None or b[name] is None:
            continue
        value += float(
            (a[name].detach().double().cpu() * b[name].detach().double().cpu()).sum()
        )
    return value


def vector_norm(values):
    return math.sqrt(
        sum(
            float(value.detach().double().cpu().square().sum())
            for value in values.values()
            if value is not None
        )
    )


def cosine(a, b):
    na = vector_norm(a)
    nb = vector_norm(b)
    if na == 0 or nb == 0:
        return None
    return vector_dot(a, b) / (na * nb)


def negate(values):
    return {
        name: None if value is None else -value
        for name, value in values.items()
    }


def sign_descent(values):
    return {
        name: None if value is None else -torch.sign(value)
        for name, value in values.items()
    }


def policy_drift(current, reference):
    weighted_l2 = torch.linalg.vector_norm(
        current["weighted"] - reference["weighted"], dim=-1
    )
    sampled_l2 = torch.linalg.vector_norm(
        current["sampled"] - reference["sampled"], dim=-1
    )
    hidden_l2 = torch.linalg.vector_norm(
        current["hidden"] - reference["hidden"], dim=-1
    )
    probs = current["probs"].clamp_min(1e-12)
    ref_probs = reference["probs"].clamp_min(1e-12)
    kl = (ref_probs * (ref_probs.log() - probs.log())).sum(-1)
    top1_change = (
        current["probs"].argmax(-1) != reference["probs"].argmax(-1)
    ).float()
    return {
        "weighted_action_l2_mean": scalar(weighted_l2.mean()),
        "weighted_action_l2_by_timestep": [
            scalar(x) for x in weighted_l2.mean(0)
        ],
        "sampled_action_l2_mean": scalar(sampled_l2.mean()),
        "sampled_action_l2_by_timestep": [
            scalar(x) for x in sampled_l2.mean(0)
        ],
        "component_mean_rms": scalar(
            (current["means"] - reference["means"]).square().mean().sqrt()
        ),
        "logits_rms": scalar(
            (current["logits"] - reference["logits"]).square().mean().sqrt()
        ),
        "categorical_kl_mean": scalar(kl.mean()),
        "top1_mode_change_fraction": scalar(top1_change.mean()),
        "hidden_l2_mean": scalar(hidden_l2.mean()),
    }


def adam_state_stats(optimizer, actor):
    result = {}
    by_name = dict(actor.named_parameters())
    for group_name in sorted(set(actor_group(name) for name in by_name)):
        exp_avg_sq_sum = 0.0
        exp_avg2_sum = 0.0
        count = 0
        steps = set()
        for name, parameter in by_name.items():
            if actor_group(name) != group_name:
                continue
            state = optimizer.state.get(parameter, {})
            if not state:
                continue
            exp_avg = state["exp_avg"].detach().double().cpu()
            exp_avg_sq = state["exp_avg_sq"].detach().double().cpu()
            exp_avg2_sum += float(exp_avg.square().sum())
            exp_avg_sq_sum += float(exp_avg_sq.sum())
            count += int(exp_avg.numel())
            step = state.get("step", 0)
            steps.add(int(step.detach().cpu().item()) if torch.is_tensor(step) else int(step))
        if count:
            result[group_name] = {
                "step_values": sorted(steps),
                "exp_avg_rms": math.sqrt(exp_avg2_sum / count),
                "sqrt_exp_avg_sq_rms": math.sqrt(exp_avg_sq_sum / count),
                "numel": count,
            }
    return result


def clone_with_grads(reference_actor, gradients):
    clone = copy.deepcopy(reference_actor)
    clone.train()
    assign_grads(clone, gradients)
    return clone


def apply_fresh_adam(reference_actor, gradients, lr):
    actor = clone_with_grads(reference_actor, gradients)
    before = named_parameter_snapshot(actor)
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(lr), weight_decay=0.0)
    optimizer.step()
    return actor, delta_map(actor, before), optimizer


def apply_sgd(reference_actor, gradients, lr):
    actor = clone_with_grads(reference_actor, gradients)
    before = named_parameter_snapshot(actor)
    optimizer = torch.optim.SGD(actor.parameters(), lr=float(lr), weight_decay=0.0)
    optimizer.step()
    return actor, delta_map(actor, before), optimizer


def first_step_adam_formula(gradients, lr, eps=1e-8):
    return {
        name: (
            None
            if grad is None
            else -float(lr) * grad / (grad.abs() + float(eps))
        )
        for name, grad in gradients.items()
    }


def map_difference(a, b):
    return {
        name: (
            None
            if a[name] is None or b[name] is None
            else a[name] - b[name]
        )
        for name in a
    }


def lr_for_env_step(config, ready_payload, env_step):
    state = ready_payload.get("training_state") or {}
    ready_step = int(state.get("critic_ready_step", ready_payload["env_steps"]))
    warmup_steps = int(
        state.get(
            "actor_warmup_steps",
            min(300000, max(100000, ready_step)),
        )
    )
    elapsed = max(0, int(env_step) - ready_step)
    progress = min(1.0, elapsed / warmup_steps)
    target = float(config["actor_warmup"]["target_actor_lr"])
    return target * progress, {
        "critic_ready_step": ready_step,
        "actor_warmup_steps": warmup_steps,
        "elapsed": elapsed,
        "warmup_progress": progress,
        "target_actor_lr": target,
    }


def build_fixed_batches(config, ready_payload):
    offline_paths = [
        config["offline_sources"][key]
        for key in ("bc_rnn", "bc_transformer", "bc_gmm")
    ]
    offline = BalancedOfflineDemonstrations(
        offline_paths,
        seed=int(config["training_seed"]),
    )
    replay_path = Path(ready_payload["online_sequence_replay"])
    if not replay_path.is_file():
        raise FileNotFoundError(
            f"critic_ready checkpoint online replay sidecar is missing: {replay_path}"
        )
    online = OnlineSequenceReplay.load(replay_path)
    train_batch = aligned_sequence_batch(
        offline,
        online,
        count=int(config["recurrent_replay"]["actor_sequence_batch_size"]),
        length=int(config["actor_source_contract"]["rnn_horizon"]),
        horizon=int(config["actor_source_contract"]["rnn_horizon"]),
    )
    diagnostic_batch = aligned_sequence_batch(
        offline,
        online,
        count=int(config["recurrent_replay"]["actor_sequence_batch_size"]),
        length=int(config["actor_source_contract"]["rnn_horizon"]),
        horizon=int(config["actor_source_contract"]["rnn_horizon"]),
    )
    return train_batch, diagnostic_batch, replay_path


def checkpoint_optimizer_audit(payload):
    optimizer = payload["actor_optimizer"]
    state = optimizer.get("state", {})
    groups = optimizer.get("param_groups", [])
    return {
        "state_entry_count": len(state),
        "state_is_empty": len(state) == 0,
        "param_group_count": len(groups),
        "group_lrs": [float(group["lr"]) for group in groups],
        "group_weight_decay": [
            float(group.get("weight_decay", 0.0)) for group in groups
        ],
        "betas": [
            list(map(float, group.get("betas", (0.9, 0.999)))) for group in groups
        ],
        "eps": [float(group.get("eps", 1e-8)) for group in groups],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results_optimizer_step_probe.json",
    )
    parser.add_argument(
        "--probe-env-steps",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Absolute env-step probes. If omitted, use offsets "
            "+16/+10K/+20K/+60K from this random2q checkpoint's own critic_ready step."
        ),
    )
    args = parser.parse_args()

    run = args.run.resolve()
    output = args.output.resolve()
    device = resolve_device(args.device)

    config = read_json(run / "shared" / "config_resolved.json")
    manifest = read_json(run / "shared" / "stage2_source_manifest.json")
    ready_path = run / "random2q" / "multi_q" / "checkpoints" / "critic_ready.pth"
    if not ready_path.is_file():
        raise FileNotFoundError(ready_path)
    ready = torch.load(ready_path, map_location="cpu", weights_only=False)

    if ready.get("critic_target_mode") != "random2q":
        raise RuntimeError(
            f"This probe is random2q-only, checkpoint mode={ready.get('critic_target_mode')!r}"
        )
    if ready.get("group") != "multi_q":
        raise RuntimeError(f"This probe requires multi_q, got {ready.get('group')!r}")
    ready_env_steps = int(ready["env_steps"])
    if ready_env_steps <= 0:
        raise RuntimeError(f"Invalid critic_ready env_steps: {ready_env_steps}")
    if int(ready["actor_updates"]) != 0:
        raise RuntimeError(
            f"Expected zero Actor updates at critic_ready, got {ready['actor_updates']}"
        )

    optimizer_audit = checkpoint_optimizer_audit(ready)

    actor, rollout, _ = load_exact_actor(
        run / "shared" / "bc_rnn_gmm_source.pth",
        device,
    )
    actor.load_state_dict(ready["actor"], strict=True)
    actor.train()
    actor_hash_before = module_hash(actor)

    critic, _ = strict_stage2_load(manifest["multi_q"]["checkpoint"], device)
    critic.load_state_dict(ready["q1_q2"], strict=True)
    critic.eval().requires_grad_(False)
    critic_hash_before = module_hash(critic)

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

    train_batch, diagnostic_batch, replay_path = build_fixed_batches(config, ready)
    diagnostic_observations = torch.as_tensor(
        diagnostic_batch["observations"],
        dtype=torch.float32,
        device=device,
    )

    reference_parameters = named_parameter_snapshot(actor)
    reference_outputs = actor_outputs(actor, diagnostic_observations, device)

    # Compute exactly one production Actor gradient at the untouched critic_ready policy.
    actor.zero_grad(set_to_none=True)
    loss, loss_details = production_actor_loss(
        actor,
        critic,
        train_batch,
        scale,
        offset,
        config,
        device,
    )
    if not finite_tensor(loss):
        raise FloatingPointError("Actor RL loss is non-finite")
    loss.backward()

    raw_gradients = named_grad_snapshot(actor)
    for name, grad in raw_gradients.items():
        if grad is not None and not finite_tensor(grad):
            raise FloatingPointError(f"Non-finite raw Actor gradient: {name}")

    raw_stats = aggregate_tensor_map(raw_gradients, reference_parameters)
    raw_global_norm = vector_norm(raw_gradients)
    max_grad_norm = float(config["actor_max_grad_norm"])
    returned_norm = torch.nn.utils.clip_grad_norm_(
        actor.parameters(),
        max_grad_norm,
    )
    clipped_gradients = named_grad_snapshot(actor)
    clipped_stats = aggregate_tensor_map(clipped_gradients, reference_parameters)
    clipped_global_norm = vector_norm(clipped_gradients)
    clip_scale_observed = (
        clipped_global_norm / raw_global_norm if raw_global_norm > 0 else 1.0
    )
    clip_scale_expected = min(1.0, max_grad_norm / (raw_global_norm + 1e-6))

    probe_env_steps = (
        [ready_env_steps + int(offset) for offset in DEFAULT_PROBE_OFFSETS]
        if args.probe_env_steps is None
        else [int(step) for step in args.probe_env_steps]
    )
    if any(step <= ready_env_steps for step in probe_env_steps):
        raise RuntimeError(
            f"All probes must be after critic_ready={ready_env_steps}, got {probe_env_steps}"
        )

    results = {
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
        "checkpoint_contract": {
            "env_steps": int(ready["env_steps"]),
            "actor_updates": int(ready["actor_updates"]),
            "critic_updates": int(ready["updates"]),
            "actor_gate_open": bool(ready["actor_gate_open"]),
            "gate_open_step": ready.get("gate_open_step"),
        },
        "checkpoint_actor_optimizer": optimizer_audit,
        "production_actor_objective": {
            "type": "final_token_Q1_component_mean",
            "actor_rl_loss": scalar(loss),
            "q1_expected_mean": scalar(loss_details["expected_q1"].mean()),
            "q1_mode_mean": scalar(loss_details["q1_modes"].mean()),
            "q2_mode_mean_diagnostic_only": scalar(loss_details["q2_modes"].mean()),
        },
        "gradient": {
            "raw": raw_stats,
            "clipped": clipped_stats,
            "raw_global_l2": raw_global_norm,
            "clip_grad_norm_returned": scalar(returned_norm),
            "max_grad_norm": max_grad_norm,
            "clipped_global_l2": clipped_global_norm,
            "clip_scale_observed": clip_scale_observed,
            "clip_scale_expected": clip_scale_expected,
            "clip_triggered": raw_global_norm > max_grad_norm,
        },
        "probe_schedule_contract": {
            "critic_ready_env_steps": ready_env_steps,
            "probe_env_steps": probe_env_steps,
            "default_offsets_from_ready": list(DEFAULT_PROBE_OFFSETS),
        },
        "probe_env_steps": {},
    }

    # Negative clipped gradient is the plain SGD descent direction.
    descent_gradient = negate(clipped_gradients)
    sign_direction = sign_descent(clipped_gradients)

    for env_step in probe_env_steps:
        lr, schedule = lr_for_env_step(config, ready, int(env_step))

        adam_actor, adam_delta, adam_optimizer = apply_fresh_adam(
            actor,
            clipped_gradients,
            lr,
        )
        adam_delta_stats = aggregate_tensor_map(adam_delta, reference_parameters)
        adam_delta_l2 = vector_norm(adam_delta)

        sgd_actor, sgd_delta, _ = apply_sgd(
            actor,
            clipped_gradients,
            lr,
        )
        sgd_delta_stats = aggregate_tensor_map(sgd_delta, reference_parameters)

        if clipped_global_norm > 0:
            matched_lr = adam_delta_l2 / clipped_global_norm
        else:
            matched_lr = 0.0
        sgd_matched_actor, sgd_matched_delta, _ = apply_sgd(
            actor,
            clipped_gradients,
            matched_lr,
        )
        sgd_matched_stats = aggregate_tensor_map(
            sgd_matched_delta,
            reference_parameters,
        )

        adam_formula = first_step_adam_formula(clipped_gradients, lr)
        adam_formula_error = map_difference(adam_delta, adam_formula)

        adam_outputs = actor_outputs(
            adam_actor,
            diagnostic_observations,
            device,
        )
        sgd_outputs = actor_outputs(
            sgd_actor,
            diagnostic_observations,
            device,
        )
        sgd_matched_outputs = actor_outputs(
            sgd_matched_actor,
            diagnostic_observations,
            device,
        )

        grad_rms = clipped_stats["total"]["rms"]
        adam_rms = adam_delta_stats["total"]["rms"]
        sgd_rms = sgd_delta_stats["total"]["rms"]

        results["probe_env_steps"][str(int(env_step))] = {
            "schedule": dict(schedule, actor_lr=lr),
            "fresh_adam": {
                "parameter_delta": adam_delta_stats,
                "optimizer_state_after_step": adam_state_stats(
                    adam_optimizer,
                    adam_actor,
                ),
                "policy_drift": policy_drift(adam_outputs, reference_outputs),
                "delta_cosine_with_negative_gradient": cosine(
                    adam_delta,
                    descent_gradient,
                ),
                "delta_cosine_with_negative_sign_gradient": cosine(
                    adam_delta,
                    sign_direction,
                ),
                "normalized_update_amplification_vs_sgd_gradient": (
                    adam_rms / (lr * grad_rms)
                    if lr > 0 and grad_rms > 0
                    else None
                ),
                "first_step_formula_error": aggregate_tensor_map(
                    adam_formula_error,
                    reference_parameters,
                ),
            },
            "sgd_same_lr": {
                "parameter_delta": sgd_delta_stats,
                "policy_drift": policy_drift(sgd_outputs, reference_outputs),
                "delta_cosine_with_negative_gradient": cosine(
                    sgd_delta,
                    descent_gradient,
                ),
                "normalized_update_amplification_vs_sgd_gradient": (
                    sgd_rms / (lr * grad_rms)
                    if lr > 0 and grad_rms > 0
                    else None
                ),
            },
            "sgd_global_step_matched": {
                "matched_lr": matched_lr,
                "parameter_delta": sgd_matched_stats,
                "policy_drift": policy_drift(
                    sgd_matched_outputs,
                    reference_outputs,
                ),
                "delta_cosine_with_adam_delta": cosine(
                    sgd_matched_delta,
                    adam_delta,
                ),
            },
        }

        print(
            json.dumps(
                {
                    "env_step": int(env_step),
                    "actor_lr": lr,
                    "raw_grad_l2": raw_global_norm,
                    "clip_triggered": raw_global_norm > max_grad_norm,
                    "adam_delta_l2": adam_delta_l2,
                    "adam_sampled_action_drift": results["probe_env_steps"][
                        str(int(env_step))
                    ]["fresh_adam"]["policy_drift"]["sampled_action_l2_mean"],
                    "sgd_same_lr_sampled_action_drift": results[
                        "probe_env_steps"
                    ][str(int(env_step))]["sgd_same_lr"]["policy_drift"][
                        "sampled_action_l2_mean"
                    ],
                    "sgd_matched_sampled_action_drift": results[
                        "probe_env_steps"
                    ][str(int(env_step))]["sgd_global_step_matched"][
                        "policy_drift"
                    ]["sampled_action_l2_mean"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    # Reconfirm the source objects were untouched.
    if module_hash(actor) != actor_hash_before:
        raise RuntimeError("Reference Actor changed during diagnostic")
    if module_hash(critic) != critic_hash_before:
        raise RuntimeError("Frozen Critic changed during diagnostic")
    results["source_integrity"] = {
        "actor_hash_unchanged": True,
        "critic_hash_unchanged": True,
    }

    write_json(output, results)
    print(f"WROTE {output}", flush=True)
    print("FORMAL TRAINING REMAINS STOPPED", flush=True)


if __name__ == "__main__":
    main()
