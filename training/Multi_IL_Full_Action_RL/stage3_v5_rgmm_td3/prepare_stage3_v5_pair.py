#!/usr/bin/env python3
"""Prepare immutable shared inputs for the paired Stage3-v5 experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch

V3 = Path(__file__).resolve().parents[1] / "stage3_v3_rgmm_td3"
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))
from stage3_v5_actor import load_exact_actor, module_hash
from stage3_v5_agent import strict_stage2_load

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
    parser.add_argument("--config", default=str(HERE / "stage3_v5_config.json"))
    parser.add_argument("--run-id")
    parser.add_argument("--output-root")
    parser.add_argument("--bc-rnn-checkpoint", required=True)
    parser.add_argument("--expert-dataset")
    parser.add_argument("--rnn-q-checkpoint", required=True)
    parser.add_argument("--multi-q-checkpoint", required=True)
    parser.add_argument("--total-env-steps", type=int)
    parser.add_argument("--reuse-phase0-from", help="deprecated; V5 rejects Phase-0 reuse")
    return parser.parse_args()


def validate_config(config):
    """V5 locks the TD3 math but delegates timing to the explicit handoff FSM."""
    if config.get("stage") != "stage3-v5-rgmm-td3":
        raise RuntimeError("Expected Stage3-v5 resolved configuration")
    fixed = {"gamma": 0.99, "tau": 0.005, "batch_size": 256,
             "offline_fraction": 0.5, "online_fraction": 0.5,
             "utd": 0.25, "policy_delay": 4}
    for key, value in fixed.items():
        if config.get(key) != value:
            raise RuntimeError(f"Stage3-v5 fixed contract changed: {key}")
    if config.get("adaptive_bc_enabled") or config.get("bc_weight") != 0:
        raise RuntimeError("Stage3-v5 forbids BC regularization")
    if config["parallel_env"].get("num_envs") != 16 or config["parallel_env"].get("startup_parallelism") != 4:
        raise RuntimeError("Formal V5 requires 16 environments started in 4-wide batches")
    if int(config["parallel_env"].get("max_collector_lag_transitions", 0)) < 16:
        raise RuntimeError("V5 collector lag bound must cover one 16-env dispatch")
    if set(config.get("offline_sources", {})) != {"bc_rnn", "bc_transformer", "bc_gmm"}:
        raise RuntimeError("V5 requires the three audited Stage1 offline sources")
    if config["evaluation"]["seeds"] != list(range(20000, 20010)):
        raise RuntimeError("V5 requires the fixed ten evaluation seeds")


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
    expected_sha = config["actor_source_contract"].get("checkpoint_sha256")
    if expected_sha and checkpoint_sha != expected_sha:
        raise RuntimeError(f"BC-RNN-GMM checkpoint SHA256 mismatch: {checkpoint_sha} != {expected_sha}")
    actor, rollout, actor_metadata = load_exact_actor(bc_checkpoint, torch.device("cpu"))
    if (config["actor_source_contract"].get("parameter_count") is not None
            and actor_metadata["parameter_count"] != int(config["actor_source_contract"]["parameter_count"])):
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
    offline_source_metadata = {}
    for source_name, source_path in config["offline_sources"].items():
        source_file = Path(source_path).resolve()
        if not source_file.is_file():
            raise FileNotFoundError(f"Offline Stage1 source not found: {source_file}")
        offline_source_metadata[source_name] = {
            "path": str(source_file), "sha256": sha256(source_file),
        }
    config["offline_source_metadata"] = offline_source_metadata
    actor_hash = module_hash(actor)
    if args.reuse_phase0_from:
        raise RuntimeError("V5 does not reuse Phase-0 policy evaluations; Critic readiness is replay-only")
    run_id = args.run_id or datetime.now().strftime("stage3v5_%Y%m%d_%H%M%S")
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
        "stage": "stage3-v5", "source_checkpoint": str(bc_checkpoint),
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
        "evaluation_seeds": config["evaluation"]["seeds"],
    }
    fairness = {
        "stage": "stage3-v5", "status": "PREPARED",
        "primary_branch_variables": ["Stage2 Critic initialization checkpoint",
                                      "offline replay composition"],
        "branches": ["rnn_q", "multi_q"], "actor_hashes_identical": True,
        "actor_hash": actor_hash, "actor_checkpoint_sha256": sha256(immutable_bc),
        "actor_optimizer": {"type": "Adam", "lr": config["actor_lr"], "weight_decay": 0.0},
        "critic_optimizer": {"type": "AdamW", "lr": config["critic_lr"],
                             "weight_decay": config["critic_weight_decay"]},
        "environment_seeds_identical": True, "evaluation_seeds_identical": True,
        "offline_dataset_identical": False,
        "offline_sources": offline_source_metadata,
        "offline_replay_composition": {
            "rnn_q": "RNN-only Stage1 rollout source",
            "multi_q": "balanced rotation of RNN, Transformer, and GMM Stage1 sources",
        },
        "replay_settings_identical": True,
        "training_schedules_identical": True, "critic_initialization": critic_sources,
    }
    write_json(shared / "config_resolved.json", config)
    write_json(shared / "bc_checkpoint_config.json", checkpoint_config)
    write_json(shared / "actor_source_metadata.json", actor_metadata)
    write_json(shared / "stage2_source_manifest.json", critic_sources)
    write_json(shared / "seed_manifest.json", seed_manifest)
    write_json(shared / "pair_fairness.json", fairness)
    print(json.dumps({"status": "PREPARED", "pair_run_dir": str(run.resolve()),
                      "actor_hash": actor_hash, "next": "train_stage3_v5_vector.py"}, indent=2))


if __name__ == "__main__":
    main()
