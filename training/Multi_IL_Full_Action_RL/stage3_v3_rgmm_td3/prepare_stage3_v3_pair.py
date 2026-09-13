#!/usr/bin/env python3
"""Prepare immutable shared inputs for the paired Stage3-v3 experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch

from stage3_v3_actor import load_exact_actor, module_hash
from stage3_v3_agent import strict_stage2_load
from stage3_v3_phase0_reuse import PHASE0_FILES, validate_phase0_reuse

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
    parser.add_argument("--config", default=str(HERE / "stage3_v3_config.json"))
    parser.add_argument("--run-id")
    parser.add_argument("--output-root")
    parser.add_argument("--bc-rnn-checkpoint", required=True)
    parser.add_argument("--expert-dataset")
    parser.add_argument("--rnn-q-checkpoint", required=True)
    parser.add_argument("--multi-q-checkpoint", required=True)
    parser.add_argument("--total-env-steps", type=int)
    parser.add_argument("--reuse-phase0-from", help="completed compatible Stage3-v3 pair")
    return parser.parse_args()


def validate_config(config):
    fixed = {
        "obs_dim": 59, "action_dim": 14, "hidden_dims": [256, 256],
        "activation": "relu", "critic_layer_norm": True, "gamma": 0.99,
        "tau": 0.005, "critic_lr": 3e-4, "critic_weight_decay": 1e-4,
        "batch_size": 256, "offline_fraction": 0.5, "online_fraction": 0.5,
        "utd": 1, "policy_delay": 2,
    }
    for key, value in fixed.items():
        if config.get(key) != value:
            raise RuntimeError(f"Stage3-v3 fixed contract changed: {key}")
    if config["online_cql"]["enabled"]:
        raise RuntimeError("Stage3-v3 v1 forbids online CQL")
    adaptive = config["adaptive_bc"]
    if not (0 <= adaptive["min_weight"] <= adaptive["initial_weight"] <= adaptive["max_weight"]
            and 0 < adaptive["ema_rate"] <= 1 and 0 <= adaptive["target_success_rate"] <= 1
            and adaptive["kp"] >= 0 and adaptive["kd"] >= 0
            and adaptive["feedback_source"] == "fixed_seed_evaluation_success_rate"):
        raise RuntimeError("Invalid adaptive BC configuration")
    normalization = config["q_scale_normalization"]
    if not (normalization["enabled"] is True and normalization["alpha"] > 0
            and normalization["epsilon"] > 0):
        raise RuntimeError("Invalid Q scale normalization configuration")
    if config["actor_gate"] != {
        "equivalence_tolerance": 1e-5, "competence_episodes": 20,
        "competence_min_successes": 10, "warmup_env_steps": 10000,
        "latched": True,
    }:
        raise RuntimeError("Stage3-v3 competence gate contract changed")


def main():
    args = arguments()
    config = read_json(args.config)
    validate_config(config)
    if args.output_root:
        config["output_root"] = str(Path(args.output_root).resolve())
    if args.total_env_steps is not None:
        config["total_env_steps"] = int(args.total_env_steps)

    bc_checkpoint = Path(args.bc_rnn_checkpoint).resolve()
    if not bc_checkpoint.is_file():
        raise FileNotFoundError(bc_checkpoint)
    checkpoint_sha = sha256(bc_checkpoint)
    expected_sha = config["actor_source_contract"]["checkpoint_sha256"]
    if checkpoint_sha != expected_sha:
        raise RuntimeError(
            f"BC-RNN-GMM checkpoint SHA256 mismatch: {checkpoint_sha} != {expected_sha}")
    actor, rollout, actor_metadata = load_exact_actor(bc_checkpoint, torch.device("cpu"))
    if actor_metadata["parameter_count"] != int(config["actor_source_contract"]["parameter_count"]):
        raise RuntimeError("Checkpoint Actor parameter count differs from audited contract")
    checkpoint_config = json.loads(rollout.policy.global_config.dump())
    configured_data = checkpoint_config["train"]["data"]
    embedded_dataset = configured_data[0]["path"] if isinstance(configured_data, list) else configured_data
    expert_dataset = Path(args.expert_dataset or embedded_dataset).resolve()
    if not expert_dataset.is_file():
        raise FileNotFoundError(f"Expert demonstrations not found: {expert_dataset}")

    critic_paths = {"rnn_q": Path(args.rnn_q_checkpoint).resolve(),
                    "multi_q": Path(args.multi_q_checkpoint).resolve()}
    critic_sources = {}
    for branch, path in critic_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        critic, payload = strict_stage2_load(path, torch.device("cpu"))
        critic_sources[branch] = {
            "checkpoint": str(path), "sha256": sha256(path),
            "model_config": payload["model_config"], "gamma": payload["gamma"],
            "initial_hash": module_hash(critic),
        }

    config["bc_rnn_checkpoint_source"] = str(bc_checkpoint)
    config["bc_rnn_checkpoint_sha256"] = checkpoint_sha
    config["expert_dataset"] = str(expert_dataset)
    config["expert_dataset_sha256"] = sha256(expert_dataset)
    actor_hash = module_hash(actor)
    phase0_reuse = (validate_phase0_reuse(args.reuse_phase0_from, config, actor_hash)
                    if args.reuse_phase0_from else None)
    run_id = args.run_id or datetime.now().strftime("stage3v3_%Y%m%d_%H%M%S")
    run = Path(config["output_root"]) / run_id
    if run.exists():
        raise FileExistsError(run)
    shared = run / "shared"
    shared.mkdir(parents=True)
    for branch in critic_paths:
        (run / branch).mkdir()

    immutable_bc = shared / "bc_rnn_gmm_source.pth"
    shutil.copyfile(bc_checkpoint, immutable_bc)
    actor_payload = {
        "stage": "stage3-v3", "source_checkpoint": str(bc_checkpoint),
        "source_checkpoint_sha256": config["bc_rnn_checkpoint_sha256"],
        "actor_state_dict": {key: value.detach().cpu() for key, value in actor.state_dict().items()},
        "actor_hash": actor_hash, "metadata": actor_metadata,
        "action_normalization_stats": rollout.action_normalization_stats,
    }
    torch.save(actor_payload, shared / "actor_init.pth")
    config["bc_rnn_checkpoint"] = str(immutable_bc.resolve())

    seed_manifest = {
        "training_seed": int(config["training_seed"]),
        "train_seed_base": int(config["train_seed_base"]),
        "train_seed_rule": "train_seed_base + generation * num_envs + env_id",
        "evaluation_seeds": config["evaluation_seeds"],
    }
    fairness = {
        "stage": "stage3-v3", "status": "PREPARED",
        "only_primary_variable": "Stage2 Critic initialization checkpoint",
        "branches": ["rnn_q", "multi_q"], "actor_hashes_identical": True,
        "actor_hash": actor_hash, "actor_checkpoint_sha256": sha256(immutable_bc),
        "actor_optimizer": {"type": "Adam", "lr": config["actor_lr"], "weight_decay": 0.0},
        "critic_optimizer": {"type": "AdamW", "lr": config["critic_lr"],
                             "weight_decay": config["critic_weight_decay"]},
        "environment_seeds_identical": True, "evaluation_seeds_identical": True,
        "offline_dataset_identical": True, "replay_settings_identical": True,
        "training_schedules_identical": True, "critic_initialization": critic_sources,
    }
    write_json(shared / "config_resolved.json", config)
    write_json(shared / "bc_checkpoint_config.json", checkpoint_config)
    write_json(shared / "actor_source_metadata.json", actor_metadata)
    write_json(shared / "stage2_source_manifest.json", critic_sources)
    write_json(shared / "seed_manifest.json", seed_manifest)
    write_json(shared / "pair_fairness.json", fairness)
    if phase0_reuse is not None:
        source_shared = Path(phase0_reuse["source_pair_run_dir"]) / "shared"
        for name in PHASE0_FILES:
            destination = shared / name
            shutil.copyfile(source_shared / name, destination)
            if sha256(destination) != phase0_reuse["artifact_sha256"][name]:
                destination.unlink()
                raise RuntimeError(f"Phase-0 artifact changed while copying: {name}")
        write_json(shared / "phase0_reuse_manifest.json", phase0_reuse)
    print(json.dumps({"status": "PREPARED", "pair_run_dir": str(run.resolve()),
                      "actor_hash": actor_hash, "phase0_reused": phase0_reuse is not None,
                      "next": ("train_stage3_v3_vector.py" if phase0_reuse
                               else "run_phase0_stage3_v3.py")}, indent=2))


if __name__ == "__main__":
    main()
