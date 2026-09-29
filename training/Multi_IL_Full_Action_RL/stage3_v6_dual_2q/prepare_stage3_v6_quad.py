#!/usr/bin/env python3
"""Prepare immutable shared inputs for the four-run Stage3-v6 experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
V5 = HERE.parent / "stage3_v5_rgmm_td3"
V3 = HERE.parent / "stage3_v3_rgmm_td3"
for directory in (HERE, V5, V3):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from stage3_v5_actor import load_exact_actor, module_hash  # noqa: E402
from stage3_v6_agent import strict_stage2_load  # noqa: E402
from stage3_v6_target import TARGET_MODES  # noqa: E402


NPU_MAPPING = {
    "mean2q/multi_q": "npu:0",
    "mean2q/rnn_q": "npu:1",
    "random2q/multi_q": "npu:2",
    "random2q/rnn_q": "npu:3",
}


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
    parser.add_argument("--config", default=str(HERE / "stage3_v6_config.json"))
    parser.add_argument("--run-id")
    parser.add_argument("--output-root")
    parser.add_argument("--bc-rnn-checkpoint", required=True)
    parser.add_argument("--expert-dataset")
    parser.add_argument("--rnn-q-checkpoint", required=True)
    parser.add_argument("--multi-q-checkpoint", required=True)
    parser.add_argument("--total-env-steps", type=int)
    return parser.parse_args()


def validate_config(config):
    if config.get("stage") != "stage3-v6-rgmm-td3":
        raise RuntimeError("Expected Stage3-v6 resolved configuration")
    fixed = {
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 256,
        "offline_fraction": 0.5,
        "online_fraction": 0.5,
        "utd": 0.25,
        "policy_delay": 4,
    }
    for key, value in fixed.items():
        if config.get(key) != value:
            raise RuntimeError(f"Stage3-v6 fixed contract changed: {key}")
    if tuple(config.get("critic_target_modes", ())) != TARGET_MODES:
        raise RuntimeError("Stage3-v6 target modes must be mean2q/random2q")
    selector = config.get("random_one_selector", {})
    if selector.get("granularity") != "one_target_q_for_entire_critic_update_minibatch":
        raise RuntimeError("Stage3-v6 random-one selector granularity changed")
    if selector.get("distribution") != "uniform_q1_q2":
        raise RuntimeError("Stage3-v6 random-one selector distribution changed")
    if selector.get("rng_stream") != "dedicated_numpy_generator_not_global_training_rng":
        raise RuntimeError("Stage3-v6 selector must use an isolated RNG stream")
    if config.get("adaptive_bc_enabled") or config.get("bc_weight") != 0:
        raise RuntimeError("Stage3-v6 forbids BC regularization")
    parallel = config["parallel_env"]
    if parallel.get("num_envs") != 16 or parallel.get("startup_parallelism") != 4:
        raise RuntimeError("Formal Stage3-v6 requires 16 envs, startup 4-wide")
    if int(parallel.get("max_collector_lag_transitions", 0)) < 16:
        raise RuntimeError("Stage3-v6 collector lag bound is too small")
    if set(config.get("offline_sources", {})) != {
        "bc_rnn", "bc_transformer", "bc_gmm"
    }:
        raise RuntimeError("Stage3-v6 requires all three audited Stage1 sources")
    if config["evaluation"]["seeds"] != list(range(20000, 20010)):
        raise RuntimeError("Stage3-v6 requires fixed evaluation seeds 20000..20009")
    if "hard_clipped_min_critic_target" not in config.get("forbidden_mechanisms", []):
        raise RuntimeError("Stage3-v6 must explicitly forbid hard clipped-min target")
    v2 = config.get("critic_readiness_v2", {})
    if (v2.get("version") != 2 or v2.get("min_online_steps") != 100000
            or v2.get("check_interval_steps") != 10000
            or v2.get("data") != {"min_completed_episodes": 150,
                                   "min_success_episodes": 30,
                                   "min_failure_episodes": 30}
            or v2.get("rank") != {"metric": "qmean_mc_spearman", "min_spearman": 0.7}
            or v2.get("td_health") != {
                "metric": "member_max_mae", "sustained_worsening_fraction": 0.35,
                "required_consecutive_worsening_intervals": 2,
                "required_history_points": 3}
            or v2.get("numeric") != {
                "qmean_mean_shift_z_max": 0.5, "qmean_std_ratio_max": 1.5,
                "finite_required": True, "epsilon": 1e-6}
            or v2.get("consecutive_passes") != 2
            or v2.get("critic_only_warning_steps") != 300000
            or v2.get("hard_stop_on_readiness_timeout") is not False):
        raise RuntimeError("Stage3-v6 Critic Readiness V2 contract changed")
    if config.get("critic_readiness", {}).get("diagnostic_only") is not True:
        raise RuntimeError("Legacy V5 readiness fields must be diagnostic-only")


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
        raise RuntimeError(
            f"BC-RNN-GMM checkpoint SHA256 mismatch: {checkpoint_sha} != {expected_sha}"
        )

    actor, rollout, actor_metadata = load_exact_actor(
        bc_checkpoint, torch.device("cpu")
    )
    if (
        config["actor_source_contract"].get("parameter_count") is not None
        and actor_metadata["parameter_count"]
        != int(config["actor_source_contract"]["parameter_count"])
    ):
        raise RuntimeError("Checkpoint Actor parameter count differs")

    checkpoint_config = json.loads(rollout.policy.global_config.dump())
    configured_data = checkpoint_config["train"]["data"]
    embedded_dataset = (
        configured_data[0]["path"]
        if isinstance(configured_data, list)
        else configured_data
    )
    expert_dataset = Path(args.expert_dataset or embedded_dataset).resolve()
    if not expert_dataset.is_file():
        raise FileNotFoundError(f"Expert demonstrations not found: {expert_dataset}")

    critic_paths = {
        "rnn_q": Path(args.rnn_q_checkpoint).resolve(),
        "multi_q": Path(args.multi_q_checkpoint).resolve(),
    }
    critic_sources = {}
    for group, critic_path in critic_paths.items():
        if not critic_path.is_file():
            raise FileNotFoundError(critic_path)
        critic, payload = strict_stage2_load(critic_path, torch.device("cpu"))
        expected_step = {"rnn_q": 46000, "multi_q": 5000}[group]
        if int(payload.get("checkpoint_step", -1)) != expected_step:
            raise RuntimeError(
                f"{group} must use selected h10 Stage2.2 step {expected_step}; "
                f"got {payload.get('checkpoint_step')}"
            )
        critic_sources[group] = {
            "checkpoint": str(critic_path),
            "sha256": sha256(critic_path),
            "architecture": payload["architecture"],
            "gamma": payload["gamma"],
            "checkpoint_step": int(payload["checkpoint_step"]),
            "critic_type": payload["critic_type"],
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
            raise FileNotFoundError(f"Offline source not found: {source_file}")
        offline_source_metadata[source_name] = {
            "path": str(source_file),
            "sha256": sha256(source_file),
        }
    config["offline_source_metadata"] = offline_source_metadata

    actor_hash = module_hash(actor)
    run_id = args.run_id or datetime.now().strftime("stage3v6_%Y%m%d_%H%M%S")
    run = Path(config["output_root"]) / run_id
    if run.exists():
        raise FileExistsError(run)

    shared = run / "shared"
    shared.mkdir(parents=True)
    for target_mode in TARGET_MODES:
        for group in ("multi_q", "rnn_q"):
            (run / target_mode / group).mkdir(parents=True)

    immutable_bc = shared / "bc_rnn_gmm_source.pth"
    shutil.copyfile(bc_checkpoint, immutable_bc)
    actor_payload = {
        "stage": "stage3-v6",
        "source_checkpoint": str(bc_checkpoint),
        "source_checkpoint_sha256": checkpoint_sha,
        "actor_state_dict": {
            key: value.detach().cpu()
            for key, value in actor.state_dict().items()
        },
        "actor_hash": actor_hash,
        "metadata": actor_metadata,
        "action_normalization_stats": rollout.action_normalization_stats,
    }
    torch.save(actor_payload, shared / "actor_init.pth")
    config["bc_rnn_checkpoint"] = str(immutable_bc.resolve())

    seed_manifest = {
        "training_seed": int(config["training_seed"]),
        "train_seed_base": int(config["train_seed_base"]),
        "train_seed_rule": "train_seed_base + generation * num_envs + env_id",
        "evaluation_seeds": config["evaluation"]["seeds"],
        "random_one_selector_seed": (
            int(config["training_seed"])
            + int(config["random_one_selector"]["seed_offset"])
        ),
        "selector_rng_isolated_from_global_training_rng": True,
    }
    fairness = {
        "stage": "stage3-v6",
        "status": "PREPARED",
        "runs": [
            "mean2q/multi_q",
            "mean2q/rnn_q",
            "random2q/multi_q",
            "random2q/rnn_q",
        ],
        "npu_mapping": NPU_MAPPING,
        "target_modes": list(TARGET_MODES),
        "actor_hashes_identical": True,
        "actor_hash": actor_hash,
        "actor_checkpoint_sha256": sha256(immutable_bc),
        "environment_seeds_identical_across_target_modes": True,
        "evaluation_seeds_identical_across_target_modes": True,
        "replay_settings_identical_across_target_modes": True,
        "training_schedules_identical_across_target_modes": True,
        "selector_rng_isolated": True,
        "actor_optimizer": {
            "type": "Adam",
            "lr": config["actor_lr"],
            "weight_decay": 0.0,
        },
        "critic_optimizer": {
            "type": "AdamW",
            "lr": config["critic_lr"],
            "weight_decay": config["critic_weight_decay"],
        },
        "offline_replay_composition": {
            "rnn_q": "RNN-only Stage1 rollout source",
            "multi_q": "balanced RNN/Transformer/GMM Stage1 sources",
        },
        "critic_initialization": critic_sources,
        "only_intended_mean_vs_random_difference": (
            "Bellman target estimator: twin mean versus one uniformly sampled "
            "target Q per Critic update"
        ),
        "actor_objective_unchanged_from_v5": (
            "Q1 component-mean objective; intentionally unchanged so target "
            "mechanism is the only mean-vs-random variable"
        ),
    }

    write_json(shared / "config_resolved.json", config)
    write_json(shared / "bc_checkpoint_config.json", checkpoint_config)
    write_json(shared / "actor_source_metadata.json", actor_metadata)
    write_json(shared / "stage2_source_manifest.json", critic_sources)
    write_json(shared / "seed_manifest.json", seed_manifest)
    write_json(shared / "quad_fairness.json", fairness)

    print(json.dumps({
        "status": "PREPARED",
        "quad_run_dir": str(run.resolve()),
        "actor_hash": actor_hash,
        "npu_mapping": NPU_MAPPING,
        "next": "launch_stage3_v6_4npu.py",
    }, indent=2))


if __name__ == "__main__":
    main()
