#!/usr/bin/env python3
"""Evaluate one Stage3 recurrent GMM Actor checkpoint against the fixed Stage3 protocol.

This is a read-only diagnostic. It does not modify training checkpoints, replay,
optimizer state, or the running experiment. The intent is to compare actor_init,
critic_ready/gate-open, and later Actor checkpoints with the exact same evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V3 = HERE.parent / "stage3_v3_rgmm_td3"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))

from stage3_v5_actor import (  # noqa: E402
    distribution_tensors,
    environment_means,
    flat_to_obs,
    load_exact_actor,
    module_hash,
    obs_to_flat,
)
from stage3_v3_evaluation import evaluate_actor  # noqa: E402
from stage3_new_evaluation import (  # noqa: E402
    build_env,
    close_env,
    reset_seed,
)


DEFAULT_SEEDS = tuple(range(20000, 20010))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-checkpoint", required=True,
                        help="Authoritative BC-RNN-GMM checkpoint used to instantiate the Actor architecture.")
    parser.add_argument("--target-checkpoint", required=True,
                        help="Actor checkpoint to evaluate: actor_init.pth, critic_ready.pth, step_*.pth, etc.")
    parser.add_argument("--reference-actor", default="",
                        help="Optional reference Actor checkpoint for parameter drift. Defaults to the BC Actor.")
    parser.add_argument("--expert-dataset", required=True,
                        help="Dataset used by Stage3 to build the TwoArmTransport evaluation environment.")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--label", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--probe-first-step", action="store_true", default=True)
    parser.add_argument("--no-probe-first-step", dest="probe_first_step", action="store_false")
    return parser.parse_args()


def resolve_device(name: str):
    if name.startswith("npu"):
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but torch.npu is unavailable")
        torch.npu.set_device(name)
    return torch.device(name)


def load_payload(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_actor_state(payload):
    if isinstance(payload, dict):
        for key in ("actor", "actor_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict) and value:
                return value, key
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return payload, "raw_state_dict"
    raise RuntimeError(
        "Could not find Actor weights. Expected checkpoint['actor'], "
        "checkpoint['actor_state_dict'], or a raw tensor state_dict."
    )


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_on_cpu(state):
    return {
        str(key): value.detach().cpu().clone()
        for key, value in state.items()
        if torch.is_tensor(value)
    }


def parameter_drift(reference, target):
    reference = tensor_state_on_cpu(reference)
    target = tensor_state_on_cpu(target)
    if set(reference) != set(target):
        missing = sorted(set(reference) - set(target))
        unexpected = sorted(set(target) - set(reference))
        raise RuntimeError(
            f"Actor state_dict mismatch: missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    sum_sq = 0.0
    ref_sq = 0.0
    max_abs = 0.0
    changed = 0
    total = 0
    groups = {
        "gmm_mean": "nets.decoder.nets.mean",
        "gmm_std": "nets.decoder.nets.scale",
        "gmm_logits": "nets.decoder.nets.logits",
        "rnn": "nets.rnn.nets",
        "encoder": "nets.encoder",
    }
    grouped = {name: {"sum_sq": 0.0, "ref_sq": 0.0, "max_abs": 0.0, "numel": 0}
               for name in groups}
    grouped["other"] = {"sum_sq": 0.0, "ref_sq": 0.0, "max_abs": 0.0, "numel": 0}

    for key in sorted(reference):
        ref = reference[key]
        cur = target[key]
        if ref.shape != cur.shape:
            raise RuntimeError(f"Shape mismatch for {key}: {tuple(ref.shape)} != {tuple(cur.shape)}")
        if not (ref.is_floating_point() or ref.is_complex()):
            if not torch.equal(ref, cur):
                changed += int(ref.numel())
            total += int(ref.numel())
            continue
        ref64 = ref.to(torch.float64)
        delta = cur.to(torch.float64) - ref64
        local_sq = float(delta.square().sum().item())
        local_ref_sq = float(ref64.square().sum().item())
        local_max = float(delta.abs().max().item()) if delta.numel() else 0.0
        local_changed = int(torch.count_nonzero(delta).item())
        local_numel = int(delta.numel())
        sum_sq += local_sq
        ref_sq += local_ref_sq
        max_abs = max(max_abs, local_max)
        changed += local_changed
        total += local_numel

        group_name = "other"
        for candidate, token in groups.items():
            if token in key:
                group_name = candidate
                break
        item = grouped[group_name]
        item["sum_sq"] += local_sq
        item["ref_sq"] += local_ref_sq
        item["max_abs"] = max(item["max_abs"], local_max)
        item["numel"] += local_numel

    result_groups = {}
    for name, item in grouped.items():
        if item["numel"] <= 0:
            continue
        l2 = math.sqrt(item["sum_sq"])
        ref_l2 = math.sqrt(item["ref_sq"])
        result_groups[name] = {
            "l2": l2,
            "relative_l2": l2 / max(ref_l2, 1e-30),
            "max_abs": item["max_abs"],
            "numel": item["numel"],
        }

    l2 = math.sqrt(sum_sq)
    reference_l2 = math.sqrt(ref_sq)
    return {
        "l2": l2,
        "relative_l2": l2 / max(reference_l2, 1e-30),
        "max_abs": max_abs,
        "changed_numel": changed,
        "total_numel": total,
        "changed_fraction": changed / max(total, 1),
        "groups": result_groups,
    }


@torch.no_grad()
def first_step_probe(actor, action_scale, action_offset, env, seeds, device):
    rows = []
    was_training = bool(actor.training)
    actor.eval()
    try:
        for seed in seeds:
            observation = reset_seed(env, int(seed))
            flat = torch.as_tensor(
                obs_to_flat(observation)[None],
                dtype=torch.float32,
                device=device,
            )
            distribution, _ = actor.forward_train_step(flat_to_obs(flat), rnn_state=None)
            tensors = distribution_tensors(distribution)
            probs = tensors["probs"][0]
            means_env = environment_means(distribution, action_scale, action_offset)[0]
            entropy = float((-(probs * probs.clamp_min(1e-12).log()).sum()).item())
            rows.append({
                "seed": int(seed),
                "mixture_probs": probs.detach().cpu().tolist(),
                "mixture_entropy": entropy,
                "argmax_mode": int(torch.argmax(probs).item()),
                "component_means_env": means_env.detach().cpu().tolist(),
                "component_mean_pairwise_distance": float(
                    torch.pdist(means_env.reshape(means_env.shape[0], -1), p=2).mean().item()
                    if means_env.shape[0] > 1 else 0.0
                ),
                "learned_std_mean": float(tensors["scales"][0].mean().item()),
                "learned_std_min": float(tensors["scales"][0].min().item()),
                "learned_std_max": float(tensors["scales"][0].max().item()),
            })
    finally:
        actor.train(was_training)

    return {
        "episodes": rows,
        "mean_entropy": float(np.mean([row["mixture_entropy"] for row in rows])),
        "mean_pairwise_component_distance": float(np.mean(
            [row["component_mean_pairwise_distance"] for row in rows]
        )),
        "mean_learned_std": float(np.mean([row["learned_std_mean"] for row in rows])),
        "argmax_mode_histogram": {
            str(mode): sum(int(row["argmax_mode"] == mode) for row in rows)
            for mode in range(5)
        },
    }


def main():
    args = parse_args()
    seeds = [int(seed) for seed in args.seeds]
    if not seeds:
        raise ValueError("--seeds must not be empty")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds contains duplicates")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")

    bc_checkpoint = Path(args.bc_checkpoint).expanduser().resolve()
    target_checkpoint = Path(args.target_checkpoint).expanduser().resolve()
    expert_dataset = Path(args.expert_dataset).expanduser().resolve()
    reference_path = Path(args.reference_actor).expanduser().resolve() if args.reference_actor else None
    output = Path(args.output).expanduser().resolve()
    for path in (bc_checkpoint, target_checkpoint, expert_dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    if reference_path is not None and not reference_path.is_file():
        raise FileNotFoundError(reference_path)

    device = resolve_device(args.device)
    started = time.monotonic()

    actor, rollout, actor_metadata = load_exact_actor(bc_checkpoint, device)
    action_scale = torch.as_tensor(
        rollout.action_normalization_stats["actions"]["scale"],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 1, -1)
    action_offset = torch.as_tensor(
        rollout.action_normalization_stats["actions"]["offset"],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 1, -1)

    initial_state = tensor_state_on_cpu(actor.state_dict())
    if reference_path is None:
        reference_state = initial_state
        reference_source = "bc_checkpoint"
        reference_sha = sha256_file(bc_checkpoint)
    else:
        reference_payload = load_payload(reference_path)
        reference_state, reference_key = extract_actor_state(reference_payload)
        reference_state = tensor_state_on_cpu(reference_state)
        reference_source = f"{reference_path}:{reference_key}"
        reference_sha = sha256_file(reference_path)

    target_payload = load_payload(target_checkpoint)
    target_state, target_key = extract_actor_state(target_payload)
    incompatible = actor.load_state_dict(target_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict target Actor load failed: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )

    drift = parameter_drift(reference_state, actor.state_dict())
    env = None
    try:
        env = build_env(expert_dataset)
        probe = (
            first_step_probe(actor, action_scale, action_offset, env, seeds, device)
            if args.probe_first_step else None
        )
        evaluation = evaluate_actor(
            actor,
            action_scale,
            action_offset,
            env,
            seeds,
            args.horizon,
            args.retries,
            None,
            None,
        )
    finally:
        if env is not None:
            close_env(env)

    label = args.label or target_checkpoint.stem
    result = {
        "diagnostic": "stage3_actor_checkpoint_same_evaluator_ab",
        "label": label,
        "device": str(device),
        "bc_checkpoint": str(bc_checkpoint),
        "bc_checkpoint_sha256": sha256_file(bc_checkpoint),
        "target_checkpoint": str(target_checkpoint),
        "target_checkpoint_sha256": sha256_file(target_checkpoint),
        "target_actor_state_key": target_key,
        "target_actor_hash": module_hash(actor),
        "reference_actor": reference_source,
        "reference_actor_sha256": reference_sha,
        "actor_metadata": actor_metadata,
        "parameter_drift_from_reference": drift,
        "evaluation_seeds": seeds,
        "horizon": int(args.horizon),
        "first_step_gmm_probe": probe,
        "evaluation": evaluation,
        "elapsed_sec": time.monotonic() - started,
    }
    for key in ("env_steps", "updates", "actor_updates", "actor_gate_open", "gate_open_step"):
        if isinstance(target_payload, dict) and key in target_payload:
            result[key] = target_payload[key]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)

    summary = {
        "label": label,
        "target_checkpoint": str(target_checkpoint),
        "env_steps": result.get("env_steps"),
        "actor_updates": result.get("actor_updates"),
        "success_count": evaluation["success_count"],
        "success_rate": evaluation["success_rate"],
        "mean_length": evaluation["mean_length"],
        "mean_return": evaluation["mean_return"],
        "parameter_drift_l2": drift["l2"],
        "parameter_drift_relative_l2": drift["relative_l2"],
        "parameter_drift_max_abs": drift["max_abs"],
        "mean_gmm_entropy": None if probe is None else probe["mean_entropy"],
        "mean_component_distance": None if probe is None else probe["mean_pairwise_component_distance"],
        "output": str(output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
