#!/usr/bin/env python3
"""Train and compare random, RNN-only, and Multi-IL twin SAC critics."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_ROOT = Path(
    "/data/home/3220251075/lerobot_workspace/training_runs/"
    "Multi_IL_Full_Action_RL/stage2_critic_pretraining"
)
POLICIES = ("bc_rnn", "bc_transformer", "bc_gmm")

from critic_network import make_critic_pair, soft_update  # noqa: E402
from stage2_dataset import Stage2PolicyDataset, cache_rnn_targets  # noqa: E402
from stage2_evaluation import evaluate_critic  # noqa: E402
from stage2_sampler import EpisodeBalancedSampler, MultiPolicyBalancedSampler  # noqa: E402
from validate_frozen_rnn_target import (  # noqa: E402
    checkpoint_from_rnn_dataset,
    load_frozen_policy,
    select_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "analysis/stage1_5_selected_datasets.json"),
    )
    parser.add_argument(
        "--sanity-report",
        default=str(PROJECT_ROOT / "analysis/stage2/frozen_rnn_target_validation.json"),
    )
    parser.add_argument("--config", default=str(SCRIPT_DIR / "stage2_config.json"))
    parser.add_argument("--seed-split", default=str(SCRIPT_DIR / "stage2_seed_split.json"))
    parser.add_argument("--output-root")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def now():
    return datetime.now(timezone.utc).astimezone().isoformat()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    npu = getattr(torch, "npu", None)
    if npu is not None and hasattr(npu, "manual_seed_all"):
        npu.manual_seed_all(seed)


def dataset_seed_set(path):
    seeds = []
    with h5py.File(path, "r") as handle:
        if "episodes" not in handle:
            raise RuntimeError(f"missing /episodes: {path}")
        for group in handle["episodes"].values():
            seed = group.attrs.get("initial_seed")
            if seed is None:
                seed = group["initial_seed"][0]
            seeds.append(int(seed))
    if len(seeds) != len(set(seeds)):
        raise RuntimeError(f"duplicate seeds: {path}")
    return set(seeds)


def load_inputs(args):
    config = read_json(args.config)
    split = read_json(args.seed_split)
    manifest = read_json(args.manifest)
    sanity = read_json(args.sanity_report)
    if sanity.get("final_status") != "PASS":
        raise RuntimeError(
            f"Stage 2.0 frozen target sanity status must be PASS, got {sanity.get('final_status')}"
        )
    if manifest.get("status") != "selected":
        raise RuntimeError(f"Stage 1.5 manifest is not selected: {manifest.get('status')}")
    training_runs_root = Path(manifest["training_runs_root"]).resolve()
    datasets = {policy: str(Path(manifest["datasets"][policy]).resolve()) for policy in POLICIES}
    for path in datasets.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        if os.path.commonpath((str(training_runs_root), path)) != str(training_runs_root):
            raise RuntimeError(f"Selected dataset is outside training_runs: {path}")
    train_seeds = [int(value) for value in split["train_seeds"]]
    validation_seeds = [int(value) for value in split["validation_seeds"]]
    if train_seeds != list(range(10000, 10080)):
        raise RuntimeError("Formal train split must be seeds 10000..10079")
    if validation_seeds != list(range(10080, 10100)):
        raise RuntimeError("Formal validation split must be seeds 10080..10099")
    if set(train_seeds) & set(validation_seeds):
        raise RuntimeError("Train and validation seeds overlap")
    seed_sets = {policy: dataset_seed_set(path) for policy, path in datasets.items()}
    if any(seed_sets[policy] != seed_sets[POLICIES[0]] for policy in POLICIES[1:]):
        raise RuntimeError("Three policy seed sets are not identical")
    if seed_sets[POLICIES[0]] != set(range(10000, 10100)):
        raise RuntimeError("Formal Stage 1 datasets must contain exactly seeds 10000..10099")
    required = set(train_seeds) | set(validation_seeds)
    if any(required - values for values in seed_sets.values()):
        raise RuntimeError("Stage 2 split contains seeds absent from Stage 1 datasets")
    if args.max_updates is not None:
        config["max_updates"] = int(args.max_updates)
    if args.batch_size is not None:
        config["batch_size"] = int(args.batch_size)
    if args.smoke_test:
        config.update({
            "max_updates": 20,
            "batch_size": 32,
            "validation_every": 10,
            "validation_batch_size": 1024,
        })
        train_seeds = train_seeds[:5]
        validation_seeds = validation_seeds[:2]
    for key in ("max_updates", "batch_size", "validation_every", "validation_batch_size"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0.0 <= float(config["gamma"]) <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 < float(config["tau"]) <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    return config, split, datasets, train_seeds, validation_seeds, training_runs_root


def to_tensor_batch(batch, device):
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in batch.items()
    }


class TrainingAccumulator:
    KEYS = ("critic_loss", "q1_loss", "q2_loss", "target_q_mean", "target_q_std", "q1_mean", "q2_mean")

    def __init__(self):
        self.rows = []
        self.q_min = np.inf
        self.q_max = -np.inf
        self.nonfinite = 0

    def add(self, row):
        self.rows.append(row)
        self.q_min = min(self.q_min, row["q_min"])
        self.q_max = max(self.q_max, row["q_max"])
        self.nonfinite += row["nan_inf_count"]

    def flush(self, update):
        result = {"update": update}
        for key in self.KEYS:
            result[key] = float(np.mean([row[key] for row in self.rows]))
        result.update({
            "q_min": self.q_min,
            "q_max": self.q_max,
            "nan_inf_count": self.nonfinite,
        })
        self.__init__()
        return result


def train_step(critic, target, optimizer, batch, gamma, tau):
    state, action = batch["state"], batch["action"]
    with torch.no_grad():
        target_q1, target_q2 = target(batch["next_state"], batch["next_target_action"])
        target_q = torch.minimum(target_q1, target_q2)
        bellman = batch["reward"] + gamma * batch["bootstrap_mask"] * target_q
    q1, q2 = critic(state, action)
    q1_loss = torch.mean(torch.square(q1 - bellman))
    q2_loss = torch.mean(torch.square(q2 - bellman))
    loss = q1_loss + q2_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    soft_update(critic, target, tau)
    tensors = (loss, q1_loss, q2_loss, target_q, q1, q2)
    nonfinite = sum(int((~torch.isfinite(value)).sum().item()) for value in tensors)
    if nonfinite:
        raise RuntimeError(f"NaN/Inf detected during critic update: count={nonfinite}")
    q_values = torch.cat((q1, q2), dim=0)
    return {
        "critic_loss": float(loss.item()),
        "q1_loss": float(q1_loss.item()),
        "q2_loss": float(q2_loss.item()),
        "target_q_mean": float(target_q.mean().item()),
        "target_q_std": float(target_q.std(unbiased=False).item()),
        "q1_mean": float(q1.mean().item()),
        "q2_mean": float(q2.mean().item()),
        "q_min": float(q_values.min().item()),
        "q_max": float(q_values.max().item()),
        "nan_inf_count": nonfinite,
    }


def checkpoint_payload(critic, target, optimizer, update, config, validation):
    return {
        "critic_state_dict": critic.state_dict(),
        "target_state_dict": target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "update": update,
        "config": config,
        "validation": validation,
        "architecture": {
            "state_dim": config["state_dim"],
            "action_dim": config["action_dim"],
            "hidden_dims": config["hidden_dims"],
            "twin_q": True,
        },
    }


def train_variant(name, sampler, validation_data, base_state, config, device, output_dir):
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir()
    atomic_json(output_dir / "config.json", {**config, "variant": name})
    critic, target = make_critic_pair(
        config["state_dim"], config["action_dim"], config["hidden_dims"], device
    )
    critic.load_state_dict(base_state)
    target.load_state_dict(base_state)
    target.requires_grad_(False)
    optimizer = torch.optim.Adam(critic.parameters(), lr=float(config["critic_lr"]))
    train_rows, validation_rows = [], []
    best_mse = np.inf
    best_update = None
    accumulator = TrainingAccumulator()
    critic.train()
    for update in range(1, int(config["max_updates"]) + 1):
        batch = to_tensor_batch(sampler.sample(config["batch_size"]), device)
        metrics = train_step(
            critic, target, optimizer, batch, float(config["gamma"]), float(config["tau"])
        )
        accumulator.add(metrics)
        if update % int(config["validation_every"]) == 0 or update == int(config["max_updates"]):
            training_metrics = accumulator.flush(update)
            validation = evaluate_critic(
                critic, target, validation_data, config["gamma"], device,
                config["validation_batch_size"],
            )
            validation_row = {
                key: value for key, value in validation.items()
                if key not in {"pairwise_details", "trajectory_scores", "q_by_trajectory_quartile"}
            }
            validation_row.update({
                "q_quartile_0_25": validation["q_by_trajectory_quartile"]["0-25%"],
                "q_quartile_25_50": validation["q_by_trajectory_quartile"]["25-50%"],
                "q_quartile_50_75": validation["q_by_trajectory_quartile"]["50-75%"],
                "q_quartile_75_100": validation["q_by_trajectory_quartile"]["75-100%"],
            })
            validation_row["update"] = update
            train_rows.append(training_metrics)
            validation_rows.append(validation_row)
            write_csv(output_dir / "train_metrics.csv", train_rows)
            write_csv(output_dir / "validation_metrics.csv", validation_rows)
            if validation["td_mse"] < best_mse:
                best_mse, best_update = validation["td_mse"], update
                torch.save(
                    checkpoint_payload(critic, target, optimizer, update, config, validation),
                    checkpoints / "best.pth",
                )
            print(
                f"update {update}/{config['max_updates']} | "
                f"loss {training_metrics['critic_loss']:.6f} | "
                f"val_mse {validation['td_mse']:.6f} | "
                f"q_range [{validation['predicted_q_min']:.6f}, "
                f"{validation['predicted_q_max']:.6f}]",
                flush=True,
            )
            critic.train()
    final_validation = validation
    torch.save(
        checkpoint_payload(
            critic, target, optimizer, config["max_updates"], config, final_validation
        ),
        checkpoints / "last.pth",
    )
    best = torch.load(checkpoints / "best.pth", map_location=device)
    critic.load_state_dict(best["critic_state_dict"])
    target.load_state_dict(best["target_state_dict"])
    evaluation = evaluate_critic(
        critic, target, validation_data, config["gamma"], device,
        config["validation_batch_size"],
    )
    evaluation.update({"best_update": best_update, "best_val_td_mse": best_mse})
    atomic_json(output_dir / "evaluation.json", evaluation)
    return evaluation


def dataset_statistics(train_data, validation_data):
    result = {}
    for policy in POLICIES:
        datasets = (train_data[policy], validation_data[policy])
        result[policy] = {
            "train_episodes": len(datasets[0].episodes),
            "validation_episodes": len(datasets[1].episodes),
            "train_transitions": datasets[0].transition_count,
            "validation_transitions": datasets[1].transition_count,
            "reward_unique_values": sorted(set().union(*(item.reward_values for item in datasets))),
            "terminated_count": sum(item.terminated_count for item in datasets),
            "truncated_count": sum(item.truncated_count for item in datasets),
            "truncated_only_count": sum(item.truncated_only_count for item in datasets),
            "terminated_or_truncated_count": sum(
                item.terminated_or_truncated_count for item in datasets
            ),
            "bootstrap_count": sum(item.bootstrap_count for item in datasets),
            "non_bootstrap_count": sum(item.non_bootstrap_count for item in datasets),
            "truncated_bootstrap_violation_count": sum(
                item.truncated_bootstrap_violation_count for item in datasets
            ),
            "bootstrapped_truncated_count": sum(
                item.truncated_bootstrap_violation_count for item in datasets
            ),
        }
    return result


def terminal_mask_statistics(train_data, validation_data, terminal_mask_mode):
    datasets = [*train_data.values(), *validation_data.values()]
    result = {
        "terminal_mask_mode": terminal_mask_mode,
        "total_transitions": sum(item.transition_count for item in datasets),
        "terminated_count": sum(item.terminated_count for item in datasets),
        "truncated_count": sum(item.truncated_count for item in datasets),
        "terminated_or_truncated_count": sum(
            item.terminated_or_truncated_count for item in datasets
        ),
        "bootstrap_count": sum(item.bootstrap_count for item in datasets),
        "non_bootstrap_count": sum(item.non_bootstrap_count for item in datasets),
        "truncated_bootstrap_violations": sum(
            item.truncated_bootstrap_violation_count for item in datasets
        ),
    }
    if terminal_mask_mode == "terminated_or_truncated":
        result["status"] = (
            "PASS" if result["truncated_bootstrap_violations"] == 0 else "FAIL"
        )
    else:
        result["status"] = "NOT_APPLICABLE_STAGE2_V1_RULE"
    return result


def compact_summary(evaluation):
    return {
        key: evaluation.get(key) for key in (
            "best_update", "best_val_td_mse", "td_mse", "td_mae",
            "pairwise_ranking_accuracy", "success_trajectory_q_mean",
            "failure_trajectory_q_mean", "q_by_trajectory_quartile",
            "predicted_q_min", "predicted_q_max",
        )
    }


def stage2_v1_comparison(v1_run_dir, rnn_evaluation, multi_evaluation):
    v1_run_dir = Path(v1_run_dir).resolve()
    summary_path = v1_run_dir / "stage2_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    v1_summary = read_json(summary_path)
    result = {"stage2_v1_run_dir": str(v1_run_dir), "stage2_v1_summary": str(summary_path)}
    for label, directory, current in (
        ("rnn_only", "rnn_only_critic", rnn_evaluation),
        ("multi_il", "multi_il_critic", multi_evaluation),
    ):
        v1_evaluation_path = v1_run_dir / directory / "evaluation.json"
        v1_evaluation = read_json(v1_evaluation_path)
        result[label] = {
            "v1_final_td_mse": v1_evaluation.get("td_mse"),
            "v2_final_td_mse": current.get("td_mse"),
            "v1_final_q_min": v1_evaluation.get("predicted_q_min"),
            "v2_final_q_min": current.get("predicted_q_min"),
            "v1_final_q_max": v1_evaluation.get("predicted_q_max"),
            "v2_final_q_max": current.get("predicted_q_max"),
            "v1_pairwise_ranking_accuracy": v1_evaluation.get("pairwise_ranking_accuracy"),
            "v2_pairwise_ranking_accuracy": current.get("pairwise_ranking_accuracy"),
        }
    result["stage2_v1_summary_loaded"] = bool(v1_summary)
    return result


def main():
    args = parse_args()
    config, _, dataset_paths, train_seeds, validation_seeds, training_runs_root = load_inputs(args)
    experiment_name = config.get("experiment_name", "stage2")
    terminal_mask_mode = config.get("terminal_mask_mode", "terminated_only")
    terminal_rule = config.get("terminal_rule", terminal_mask_mode)
    configured_output_root = Path(
        args.output_root or config.get("output_root", str(DEFAULT_OUTPUT_ROOT))
    )
    if args.smoke_test and not config.get("smoke_output_in_output_root", False):
        output_root = Path("/tmp/multi_il_full_action_rl_stage2_smoke")
    else:
        output_root = configured_output_root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_run_id = f"smoke_test_{timestamp}" if args.smoke_test else timestamp
    run_id = args.run_id or default_run_id
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    latest_filename = config.get("latest_run_filename")
    if latest_filename:
        latest_path = PROJECT_ROOT / "analysis/stage2" / latest_filename
    else:
        latest_path = None if args.smoke_test else PROJECT_ROOT / "analysis/stage2/latest_stage2_run.json"
    if latest_path is not None:
        atomic_json(latest_path, {
            "run_dir": str(run_dir),
            "created_at": now(),
            "terminal_rule": terminal_rule,
            "status": "smoke_running" if args.smoke_test else "running",
        })
    try:
        effective_split = {
            "train_seeds": train_seeds,
            "validation_seeds": validation_seeds,
            "split_unit": "initial_seed / complete trajectory",
            "transition_level_random_split": False,
            "formal_split_source": str(Path(args.seed_split).resolve()),
            "smoke_subset": bool(args.smoke_test),
        }
        run_config = {
            **config,
            "experiment_name": experiment_name,
            "terminal_mask_mode": terminal_mask_mode,
            "manifest": str(Path(args.manifest).resolve()),
            "datasets": dataset_paths,
            "smoke_test": bool(args.smoke_test),
        }
        atomic_json(run_dir / "config.json", run_config)
        atomic_json(run_dir / "seed_split.json", effective_split)
        checkpoint = checkpoint_from_rnn_dataset(
            dataset_paths["bc_rnn"], training_runs_root
        )
        run_config["rnn_checkpoint"] = str(checkpoint)
        all_seeds = train_seeds + validation_seeds
        if config.get("reuse_cached_rnn_targets", False):
            cache_root = Path(config["rnn_target_cache_source"]).resolve()
            if not cache_root.is_dir():
                raise NotADirectoryError(cache_root)
            cache_manifest_path = cache_root / "manifest.json"
            cache_manifest = read_json(cache_manifest_path)
            if Path(cache_manifest.get("checkpoint", "")).resolve() != checkpoint:
                raise RuntimeError("Reused RNN target cache checkpoint does not match Stage 1 metadata")
            cached_seeds = {int(seed) for seed in cache_manifest.get("seeds", [])}
            if not set(all_seeds).issubset(cached_seeds):
                raise RuntimeError(
                    f"Reused RNN target cache is missing seeds: "
                    f"{sorted(set(all_seeds) - cached_seeds)}"
                )
            device = select_device(args.device)
            frozen_parameters = None
            frame_stack = int(cache_manifest.get("frame_stack", 1))
            cache_summary = {
                policy_id: {
                    "episodes": len(all_seeds),
                    "reused": len(all_seeds),
                    "created": 0,
                } for policy_id in POLICIES
            }
            run_config.update({
                "reused_rnn_target_cache": True,
                "rnn_target_cache_source": str(cache_root),
                "rnn_target_cache_source_manifest": str(cache_manifest_path),
            })
        else:
            cache_root = run_dir / "cached_rnn_targets"
            policy, _, _, device, frozen_parameters, frame_stack, _ = load_frozen_policy(
                checkpoint, args.device
            )
            cache_summary = cache_rnn_targets(
                policy, dataset_paths, all_seeds, cache_root, frame_stack
            )
            atomic_json(cache_root / "manifest.json", {
                "target_policy": "frozen_bc_rnn",
                "checkpoint": str(checkpoint),
                "frame_stack": frame_stack,
                "seeds": all_seeds,
                "sources": cache_summary,
                "stage1_data_modified": False,
            })
            del policy
            if device.type == "npu" and hasattr(torch.npu, "empty_cache"):
                torch.npu.empty_cache()
            run_config.update({
                "reused_rnn_target_cache": False,
                "rnn_target_cache_source": str(cache_root),
            })
        atomic_json(run_dir / "config.json", run_config)

        train_data = {
            policy_id: Stage2PolicyDataset(
                policy_id, path, train_seeds, cache_root, terminal_mask_mode
            ) for policy_id, path in dataset_paths.items()
        }
        validation_data = {
            policy_id: Stage2PolicyDataset(
                policy_id, path, validation_seeds, cache_root, terminal_mask_mode
            ) for policy_id, path in dataset_paths.items()
        }
        dimensions = {
            (dataset.state_dim, dataset.action_dim)
            for dataset in (*train_data.values(), *validation_data.values())
        }
        if dimensions != {(59, 14)}:
            raise RuntimeError(f"Expected state/action dimensions (59, 14), got {dimensions}")
        if config["state_dim"] != 59 or config["action_dim"] != 14:
            raise RuntimeError("Formal config must use state_dim=59 and action_dim=14")
        stats = dataset_statistics(train_data, validation_data)
        atomic_json(run_dir / "dataset_statistics.json", stats)
        terminal_stats = terminal_mask_statistics(
            train_data, validation_data, terminal_mask_mode
        )
        atomic_json(run_dir / "terminal_mask_statistics.json", terminal_stats)
        if (
            terminal_mask_mode == "terminated_or_truncated"
            and terminal_stats["truncated_bootstrap_violations"] != 0
        ):
            raise RuntimeError(
                "Stage 2.1 terminal mask sanity failed: "
                f"truncated_bootstrap_violations="
                f"{terminal_stats['truncated_bootstrap_violations']}"
            )

        print("=" * 66)
        print("Stage 2.1 Critic Pretraining" if experiment_name == "stage2_1" else "Stage 2 SAC Critic Pretraining")
        print("=" * 66)
        if experiment_name == "stage2_1":
            print("Change from Stage 2: truncated transitions no longer bootstrap")
        print("Device:", device)
        print("Selected datasets:")
        for policy_id in POLICIES:
            print(f"  {policy_id}: {dataset_paths[policy_id]}")
        print("RNN checkpoint:", checkpoint)
        print("state_dim = 59")
        print("action_dim = 14")
        print("Train seeds:", len(train_seeds))
        print("Val seeds:", len(validation_seeds))
        print("RNN-only train episodes:", len(train_data["bc_rnn"].episodes))
        print("Multi train episodes per policy:", {
            policy_id: len(train_data[policy_id].episodes) for policy_id in POLICIES
        })
        print("Reward unique values:", {
            policy_id: stats[policy_id]["reward_unique_values"] for policy_id in POLICIES
        })
        print("Terminated count:", sum(row["terminated_count"] for row in stats.values()))
        print("Truncated count:", sum(row["truncated_count"] for row in stats.values()))
        print("Frozen RNN cache:", "REUSED" if run_config["reused_rnn_target_cache"] else "CREATED")
        print("Cache source:", cache_root)
        if frozen_parameters is not None:
            print("All RNN parameters frozen:", frozen_parameters > 0)
        print("RNN target cache:", cache_summary)
        print("Terminal mask:")
        print("  terminated -> 0")
        print("  truncated ->", 0 if terminal_mask_mode == "terminated_or_truncated" else 1)
        print("  other -> 1")
        print("Terminal mask sanity:")
        print("  terminated transitions:", terminal_stats["terminated_count"])
        print("  truncated transitions:", terminal_stats["truncated_count"])
        print("  truncated bootstrap violations:", terminal_stats["truncated_bootstrap_violations"])

        seed_all(int(config["random_seed"]))
        base_critic, base_target = make_critic_pair(59, 14, config["hidden_dims"], device)
        base_state = copy.deepcopy(base_critic.state_dict())
        random_dir = run_dir / "random_critic"
        random_dir.mkdir()
        torch.save({
            "critic_state_dict": base_state,
            "target_state_dict": base_target.state_dict(),
            "config": config,
            "update": 0,
        }, random_dir / "model_init.pth")
        random_evaluation = evaluate_critic(
            base_critic, base_target, validation_data, config["gamma"], device,
            config["validation_batch_size"],
        )
        random_evaluation.update({"best_update": 0, "best_val_td_mse": random_evaluation["td_mse"]})
        atomic_json(random_dir / "evaluation.json", random_evaluation)
        atomic_json(random_dir / "metrics.json", compact_summary(random_evaluation))
        del base_critic, base_target

        print("\n" + "-" * 66)
        print("Training RNN-only Critic")
        print("-" * 66)
        rnn_sampler = EpisodeBalancedSampler(train_data["bc_rnn"], config["random_seed"])
        rnn_evaluation = train_variant(
            "rnn_only_critic", rnn_sampler, validation_data, base_state,
            config, device, run_dir / "rnn_only_critic",
        )

        print("\n" + "-" * 66)
        print("Training Multi-IL Critic")
        print("sampling = RNN:T:GMM = 1:1:1")
        print("-" * 66)
        multi_sampler = MultiPolicyBalancedSampler(train_data, config["random_seed"])
        multi_evaluation = train_variant(
            "multi_il_critic", multi_sampler, validation_data, base_state,
            config, device, run_dir / "multi_il_critic",
        )

        stage_results = {
            "random_critic": compact_summary(random_evaluation),
            "rnn_only_critic": compact_summary(rnn_evaluation),
            "multi_il_critic": compact_summary(multi_evaluation),
            "comparison": {
                "rnn_only_vs_multi": {
                    "val_td_mse_delta_multi_minus_rnn": (
                        multi_evaluation["td_mse"] - rnn_evaluation["td_mse"]
                    ),
                    "pairwise_accuracy_delta_multi_minus_rnn": (
                        None if multi_evaluation["pairwise_ranking_accuracy"] is None
                        or rnn_evaluation["pairwise_ranking_accuracy"] is None
                        else multi_evaluation["pairwise_ranking_accuracy"]
                        - rnn_evaluation["pairwise_ranking_accuracy"]
                    ),
                }
            },
        }
        if experiment_name == "stage2_1":
            comparison_to_v1 = stage2_v1_comparison(
                config["stage2_v1_run_dir"], rnn_evaluation, multi_evaluation
            )
            summary = {
                "stage": "stage2_1_critic_pretraining",
                "terminal_mask_mode": terminal_mask_mode,
                "reused_rnn_target_cache": run_config["reused_rnn_target_cache"],
                "rnn_target_cache_source": str(cache_root),
                "terminal_mask_statistics": terminal_stats,
                "stage2_1": stage_results,
                "comparison_to_stage2_v1": comparison_to_v1,
                "run_dir": str(run_dir),
                "smoke_test": bool(args.smoke_test),
                "completed_at": now(),
                "stage3_started": False,
                "stage4_started": False,
            }
        else:
            summary = {
                **stage_results,
                "run_dir": str(run_dir),
                "completed_at": now(),
                "stage3_started": False,
                "stage4_started": False,
            }
            comparison_to_v1 = None
        summary_filename = config.get("summary_filename", "stage2_summary.json")
        atomic_json(run_dir / summary_filename, summary)
        if latest_path is not None:
            atomic_json(latest_path, {
                "run_dir": str(run_dir),
                "created_at": now(),
                "terminal_rule": terminal_rule,
                "status": "smoke_complete" if args.smoke_test else "complete",
            })
        print("\n" + "-" * 66)
        print("Stage 2.1 Summary" if experiment_name == "stage2_1" else "Stage 2 Summary")
        print("-" * 66)
        for label, evaluation in (
            ("Random", random_evaluation), ("RNN-only", rnn_evaluation), ("Multi", multi_evaluation)
        ):
            print(
                f"{label}: val_mse={evaluation['td_mse']:.6f} "
                f"pair_ranking={evaluation['pairwise_ranking_accuracy']} "
                f"success_q={evaluation['success_trajectory_q_mean']} "
                f"failure_q={evaluation['failure_trajectory_q_mean']}"
            )
        if comparison_to_v1 is not None:
            print("Comparison to Stage 2 v1:")
            for label in ("rnn_only", "multi_il"):
                row = comparison_to_v1[label]
                print(
                    f"  {label}: td_mse {row['v1_final_td_mse']} -> "
                    f"{row['v2_final_td_mse']}, q_range "
                    f"[{row['v1_final_q_min']}, {row['v1_final_q_max']}] -> "
                    f"[{row['v2_final_q_min']}, {row['v2_final_q_max']}], "
                    f"ranking {row['v1_pairwise_ranking_accuracy']} -> "
                    f"{row['v2_pairwise_ranking_accuracy']}"
                )
        print("Run directory:", run_dir)
        print("=" * 66)
    except BaseException as exception:
        if latest_path is not None:
            atomic_json(latest_path, {
                "run_dir": str(run_dir),
                "created_at": now(),
                "terminal_rule": terminal_rule,
                "status": "smoke_failed" if args.smoke_test else "failed",
                "error": f"{type(exception).__name__}: {exception}",
            })
        raise


if __name__ == "__main__":
    main()
