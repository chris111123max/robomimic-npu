#!/usr/bin/env python3
"""Prepare immutable shared artifacts for a paired Stage3-v2 run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch

from stage3_v2_agent import build_actor, state_hash, strict_stage2_load
from stage3_v2_behavior import validate_behavior_schedule


HERE = Path(__file__).resolve().parent


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "stage3_v2_config.json"))
    parser.add_argument("--source-pair-run-dir", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root")
    parser.add_argument("--total-env-steps", type=int)
    return parser.parse_args()


def validate_contract(config):
    fixed = {
        "obs_dim": 59, "action_dim": 14, "hidden_dims": [256, 256],
        "activation": "relu", "critic_layer_norm": True,
        "actor_lr": 3e-4, "critic_lr": 3e-4, "alpha_lr": 3e-4,
        "critic_weight_decay": 1e-4, "gamma": 0.99, "tau": 0.005,
        "actor_weight_decay": 0.0, "alpha_weight_decay": 0.0,
        "target_update_interval": 1,
        "target_entropy": -14.0, "alpha_init": 0.01,
        "automatic_entropy_tuning": True, "batch_size": 256, "utd": 1,
        "offline_fraction": 0.5, "online_fraction": 0.5,
    }
    for key, value in fixed.items():
        if config.get(key) != value:
            raise RuntimeError(f"Fixed Stage3-v2 contract changed: {key}")
    validate_behavior_schedule(config["episode_behavior_schedule"])
    cql = config["cql"]
    if cql != {
        "enabled": True, "lambda": 0.1, "num_random_actions": 10,
        "num_policy_actions": 1, "apply_to_expert": True,
        "apply_to_online": True, "detach_policy_actions": True,
        "source_diagnostic_interval_env_steps": 5000,
        "source_diagnostic_sample_size": 256,
    }:
        raise RuntimeError("Stage3-v2 CQL-lite contract changed")
    if config.get("anchor", {}).get("enabled", False):
        raise RuntimeError("Stage3-v2 forbids value anchors")


def main():
    args = arguments()
    config = read_json(args.config)
    if args.total_env_steps is not None:
        config["total_env_steps"] = int(args.total_env_steps)
    if args.output_root:
        config["output_root"] = str(Path(args.output_root).resolve())
    validate_contract(config)

    source_pair = Path(args.source_pair_run_dir).resolve()
    source_shared = source_pair / "shared"
    source_config = read_json(source_shared / "config_resolved.json")
    source_manifest = read_json(source_shared / "stage2_source_manifest.json")
    seed_manifest = read_json(source_shared / "seed_manifest.json")
    actor_source = source_shared / "actor_init.pth"
    cache_source = source_shared / "expert_rnn_proposal_cache.npz"
    cache_manifest_source = cache_source.with_suffix(".manifest.json")
    required = [actor_source, cache_source, cache_manifest_source]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage3-v2 source artifact: {path}")
    if seed_manifest.get("evaluation_seeds") != list(range(20000, 20010)):
        raise RuntimeError("Source pair does not use evaluation seeds 20000..20009")

    config["expert_dataset"] = str(Path(source_config["expert_dataset"]).resolve())
    config["bc_rnn_checkpoint"] = str(Path(source_config["bc_rnn_checkpoint"]).resolve())
    config["expert_rnn_proposal_cache"] = None
    config["stage2_run_dir"] = str(Path(source_config["stage2_run_dir"]).resolve())
    config["training_seed"] = int(seed_manifest["training_seed"])
    config["train_seed_base"] = int(seed_manifest["train_seed_base"])

    run_id = args.run_id or datetime.now().strftime("stage3v2_%Y%m%d_%H%M%S")
    run = Path(config["output_root"]) / run_id
    if run.exists():
        raise FileExistsError(run)
    shared = run / "shared"
    shared.mkdir(parents=True)
    (run / "rnn_q").mkdir()
    (run / "multi_q").mkdir()
    shutil.copyfile(actor_source, shared / "actor_init.pth")
    shutil.copyfile(source_shared / "seed_manifest.json", shared / "seed_manifest.json")
    shutil.copyfile(cache_source, shared / "expert_rnn_proposal_cache.npz")
    shutil.copyfile(cache_manifest_source, shared / "expert_rnn_proposal_cache.manifest.json")
    config["expert_rnn_proposal_cache"] = str(
        (shared / "expert_rnn_proposal_cache.npz").resolve()
    )

    actor_payload = torch.load(shared / "actor_init.pth", map_location="cpu")
    actor = build_actor(config, "cpu")
    actor.load_state_dict(actor_payload["actor_state_dict"], strict=True)
    actor_hash = state_hash(actor)
    if actor_hash != actor_payload.get("actor_hash"):
        raise RuntimeError("Shared Actor hash mismatch")

    stage2_sources = {}
    for branch in ("rnn_q", "multi_q"):
        checkpoint = Path(source_manifest[branch]["checkpoint"]).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing Stage2 {branch} checkpoint: {checkpoint}")
        _, payload = strict_stage2_load(checkpoint, "cpu", config)
        stage2_sources[branch] = {
            "checkpoint": str(checkpoint),
            "sha256": sha256(checkpoint),
            "model_config": payload["model_config"],
            "gamma": payload["gamma"],
        }

    actor_sha = sha256(shared / "actor_init.pth")
    seed_sha = sha256(shared / "seed_manifest.json")
    pair_contract = {
        "stage": "stage3-v2",
        "only_primary_variable": "stage2_critic_initialization",
        "branches": ["rnn_q", "multi_q"],
        "actor_init": {
            "rnn_q_hash": actor_hash, "multi_q_hash": actor_hash,
            "artifact_sha256": actor_sha,
        },
        "actor_hashes_identical": True,
        "bc_rnn_checkpoint": config["bc_rnn_checkpoint"],
        "expert_dataset": config["expert_dataset"],
        "seed_manifest_sha256": seed_sha,
        "evaluation_seeds": seed_manifest["evaluation_seeds"],
        "episode_behavior_schedule": config["episode_behavior_schedule"],
        "bc_regularization_schedule": config["bc_regularization_schedule"],
        "shared_hyperparameters": {
            key: config[key] for key in (
                "hidden_dims", "actor_lr", "critic_lr", "alpha_lr", "gamma",
                "tau", "batch_size", "utd", "target_entropy", "alpha_init",
                "critic_weight_decay", "offline_fraction", "online_fraction",
                "parallel_env", "cql",
            )
        },
        "critic_initialization": stage2_sources,
        "forbidden_mechanisms": [
            "timestep_q_selector", "hybrid_bootstrap", "q_filtered_bc",
            "awac", "anchor",
        ],
    }
    write_json(shared / "config_resolved.json", config)
    write_json(shared / "stage2_source_manifest.json", stage2_sources)
    write_json(shared / "pair_contract.json", pair_contract)
    write_json(shared / "source_artifacts.json", {
        "source_pair_run_dir": str(source_pair),
        "actor_init_source": str(actor_source),
        "actor_init_sha256": actor_sha,
        "expert_rnn_cache_source": str(cache_source),
        "expert_rnn_cache_sha256": sha256(cache_source),
    })
    print(json.dumps({
        "status": "PREPARED",
        "pair_run_dir": str(run.resolve()),
        "actor_init_checkpoint": str((shared / "actor_init.pth").resolve()),
        "actor_init_hash": actor_hash,
        "rnn_q_critic_checkpoint": stage2_sources["rnn_q"]["checkpoint"],
        "multi_q_critic_checkpoint": stage2_sources["multi_q"]["checkpoint"],
        "bc_rnn_checkpoint": config["bc_rnn_checkpoint"],
    }, indent=2))


if __name__ == "__main__":
    main()
