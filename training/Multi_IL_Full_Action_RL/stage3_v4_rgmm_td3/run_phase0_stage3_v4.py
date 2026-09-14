#!/usr/bin/env python3
"""Mandatory no-update transfer equivalence and 20-episode competence gate."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

V3 = Path(__file__).resolve().parents[1] / "stage3_v3_rgmm_td3"
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))
from stage3_v3_actor import (distribution_tensors, environment_means,
                             flat_to_obs, load_exact_actor, module_hash,
                             recurrent_distributions)
from stage3_v3_evaluation import build_env, close_env, evaluate_actor
from stage3_v4_phase0_reuse import PHASE0_FILES, validate_phase0_reuse
from stage3_v3_replay import OfflineDemonstrations


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def resolve_device(name):
    if name.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(name)
    return torch.device(name)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def differences(left, right):
    delta = (left - right).detach().abs()
    return {"max_abs_diff": float(delta.max().cpu()),
            "mean_abs_diff": float(delta.mean().cpu())}


def compare_hidden(left, right):
    if isinstance(left, tuple):
        rows = [differences(a, b) for a, b in zip(left, right)]
        return {"max_abs_diff": max(row["max_abs_diff"] for row in rows),
                "mean_abs_diff": float(np.mean([row["mean_abs_diff"] for row in rows])),
                "per_state": rows}
    return differences(left, right)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    pair = Path(args.pair_run_dir).resolve()
    config = read_json(pair / "shared" / "config_resolved.json")
    if all((pair / "shared" / name).is_file() for name in PHASE0_FILES):
        fairness = read_json(pair / "shared" / "pair_fairness.json")
        reuse = validate_phase0_reuse(pair, config, fairness["actor_hash"])
        print(json.dumps({"status": "PASS", "phase0_already_complete": True,
                          "success_count": reuse["success_count"],
                          "success_rate": reuse["success_rate"]}, indent=2))
        return
    device = resolve_device(args.device)
    seed_all(config["training_seed"])
    source, rollout, metadata = load_exact_actor(config["bc_rnn_checkpoint"], device)
    actor, _, _ = load_exact_actor(config["bc_rnn_checkpoint"], device)
    payload = torch.load(pair / "shared" / "actor_init.pth", map_location=device)
    incompatible = actor.load_state_dict(payload["actor_state_dict"], strict=True)
    source.eval(); actor.eval()
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict Actor initialization returned incompatible keys")

    offline = OfflineDemonstrations(config["expert_dataset"], config["training_seed"] + 17)
    batch = offline.sample_sequences(8, 10)
    observations = torch.as_tensor(batch["observations"], dtype=torch.float32, device=device)
    episode_steps = torch.as_tensor(batch["episode_steps"], dtype=torch.long, device=device)
    left_dist, left_hidden = recurrent_distributions(source, observations, episode_steps, 10)
    right_dist, right_hidden = recurrent_distributions(actor, observations, episode_steps, 10)
    per_tensor = {"means": [], "scales": [], "logits": [], "probabilities": []}
    for left, right in zip(left_dist, right_dist):
        lt, rt = distribution_tensors(left), distribution_tensors(right)
        per_tensor["means"].append(differences(lt["means_normalized"], rt["means_normalized"]))
        per_tensor["scales"].append(differences(lt["scales"], rt["scales"]))
        per_tensor["logits"].append(differences(lt["logits"], rt["logits"]))
        per_tensor["probabilities"].append(differences(lt["probs"], rt["probs"]))
    state = torch.get_rng_state()
    npu_state = torch.npu.get_rng_state() if hasattr(torch, "npu") and torch.npu.is_available() else None
    left_action = left_dist[-1].sample()
    torch.set_rng_state(state)
    if npu_state is not None:
        torch.npu.set_rng_state(npu_state)
    right_action = right_dist[-1].sample()
    scale = torch.as_tensor(rollout.action_normalization_stats["actions"]["scale"],
                            dtype=torch.float32, device=device).reshape(1, 1, 1, 14)
    offset = torch.as_tensor(rollout.action_normalization_stats["actions"]["offset"],
                             dtype=torch.float32, device=device).reshape(1, 1, 1, 14)
    action_diff = differences(left_action * scale.reshape(1, 14) + offset.reshape(1, 14),
                              right_action * scale.reshape(1, 14) + offset.reshape(1, 14))
    hidden_diff = compare_hidden(left_hidden, right_hidden)
    maxima = [action_diff["max_abs_diff"], hidden_diff["max_abs_diff"]]
    for rows in per_tensor.values():
        maxima.extend(row["max_abs_diff"] for row in rows)
    maximum = max(maxima)
    tensor_summary = {
        key: {"max_abs_diff": max(row["max_abs_diff"] for row in rows),
              "mean_abs_diff": float(np.mean([row["mean_abs_diff"] for row in rows]))}
        for key, rows in per_tensor.items()
    }
    tolerance = float(config["actor_gate"]["equivalence_tolerance"])
    equivalence_pass = bool(maximum <= tolerance and module_hash(source) == module_hash(actor))
    transfer = {
        "checkpoint_source": config["bc_rnn_checkpoint_source"],
        "checkpoint_sha256": config["bc_rnn_checkpoint_sha256"],
        "network_structure": metadata, "loaded_parameter_count": metadata["parameter_count"],
        "missing_keys": [], "unexpected_keys": [], "source_actor_hash": module_hash(source),
        "new_actor_hash": module_hash(actor), "per_tensor_diff": per_tensor,
        "means_max_abs_diff": tensor_summary["means"]["max_abs_diff"],
        "scales_max_abs_diff": tensor_summary["scales"]["max_abs_diff"],
        "logits_max_abs_diff": tensor_summary["logits"]["max_abs_diff"],
        "probabilities_max_abs_diff": tensor_summary["probabilities"]["max_abs_diff"],
        "hidden_max_abs_diff": hidden_diff["max_abs_diff"],
        "action_max_abs_diff": action_diff["max_abs_diff"],
        "hidden": hidden_diff, "action": action_diff, "max_abs_diff": maximum,
        "tolerance": tolerance, "equivalence_pass": equivalence_pass,
    }
    write_json(pair / "shared" / "transfer_validation.json", transfer)
    if not equivalence_pass:
        raise RuntimeError("TRANSFER_EQUIVALENCE_FAIL")

    env = build_env(config["expert_dataset"])
    try:
        competence = evaluate_actor(actor, scale, offset, env,
                                    config["evaluation_seeds"], config["horizon"],
                                    config["sim_error_handling"]["evaluation_retry_count"])
    finally:
        close_env(env)
    competence_pass = bool(
        competence["valid_episodes"] == config["actor_gate"]["competence_episodes"]
        and competence["success_count"] >= config["actor_gate"]["competence_min_successes"])
    competence.update({"stage": "stage3-v4", "env_steps": 0,
                       "competence_pass": competence_pass,
                       "minimum_successes": config["actor_gate"]["competence_min_successes"]})
    write_json(pair / "shared" / "step0_competence.json", competence)
    gate = {"env_steps": 0, "eval_success_count": competence["success_count"],
            "eval_success_rate": competence["success_rate"],
            "equivalence_pass": True, "competence_pass": competence_pass,
            "warmup_pass": False, "gate_open": False, "latched": True}
    write_json(pair / "shared" / "phase0_gate.json", gate)
    if not competence_pass:
        raise RuntimeError("TRANSFER_COMPETENCE_FAIL")
    print(json.dumps({"status": "PASS", "equivalence_max_abs_diff": maximum,
                      "success_count": competence["success_count"],
                      "success_rate": competence["success_rate"],
                      "actor_rl_gate_open": False}, indent=2))


if __name__ == "__main__":
    main()
