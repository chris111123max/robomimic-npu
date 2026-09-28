#!/usr/bin/env python3
"""Read-only TD-target bias decomposition for the historical 300K Stage3-v5 run.

Uses the run's Stage2.2 architecture and 11-token Stage3 successor context.
No environment, optimizer, or checkpoint write is performed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STAGE3 = HERE.parent
for directory in (HERE, STAGE3):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from stage3_v5_actor import load_exact_actor  # noqa: E402
from stage3_v5_agent import (  # noqa: E402
    _last_reset_starts_from_numpy,
    target_final_distribution_vectorized,
)
from stage3_v5_history_critic import component_mean_q  # noqa: E402
from stage3_v5_readiness import discounted_returns  # noqa: E402
from test_stage2_stage3_readiness_compare import resolve_device, sync  # noqa: E402
from test_stage3_300k_critic_stage_adaptation_matrix import (  # noqa: E402
    CHECKPOINTS,
    load_stage2_source_critic,
    old_dataset,
    stage_dataset,
)

DATASETS = ("old", "late")
EXPECTED_SEMANTICS = "full_episode_prefix_unroll_learning_mask"
ROW_KEYS = (
    "behavior_q1", "behavior_q2", "behavior_qmean", "behavior_qmin",
    "policy_q1", "policy_q2", "policy_qmean", "policy_qmin",
    "zero_first_qmin", "mc_identity_target", "behavior_mean_target",
    "behavior_min_target", "policy_mean_target", "production_target",
    "zero_first_behavior_target",
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-run-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", default=str(HERE))
    return parser.parse_args()


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def terminal_mask(episode):
    if "terminals" in episode:
        mask = np.asarray(episode["terminals"], dtype=np.float64).reshape(-1)
    elif "terminated" in episode and "truncated" in episode:
        mask = (
            np.asarray(episode["terminated"], dtype=bool).reshape(-1)
            | np.asarray(episode["truncated"], dtype=bool).reshape(-1)
        ).astype(np.float64)
    elif "dones" in episode:
        mask = np.asarray(episode["dones"], dtype=np.float64).reshape(-1)
    else:
        raise RuntimeError("Episode lacks Stage3 terminal mask")
    if "terminated" in episode and "truncated" in episode:
        expected = (
            np.asarray(episode["terminated"], dtype=bool).reshape(-1)
            | np.asarray(episode["truncated"], dtype=bool).reshape(-1)
        )
        if not np.array_equal(mask.astype(bool), expected):
            raise RuntimeError("Stage3 terminal mask differs from terminated|truncated")
    return mask


def prepare_dataset(episodes, manifest, context_length, gamma):
    references = {length: [] for length in range(1, context_length + 1)}
    returns = []
    max_identity_error = 0.0
    terminal_count = 0
    terminated_count = 0
    truncated_count = 0
    transitions = 0
    for ep_index, episode in enumerate(episodes):
        length = len(episode["actions"])
        rewards = np.asarray(episode["rewards"], dtype=np.float64).reshape(-1)
        mask = terminal_mask(episode)
        if len(mask) != length or len(rewards) != length:
            raise RuntimeError(f"Episode {ep_index}: reward/terminal length mismatch")
        if not np.all((mask == 0) | (mask == 1)):
            raise RuntimeError(f"Episode {ep_index}: non-binary terminal mask")
        if np.any(mask[:-1]) or mask[-1] != 1:
            raise RuntimeError(f"Episode {ep_index}: not a complete finite episode")
        mc = discounted_returns(episode, gamma)
        next_mc = np.zeros_like(mc)
        next_mc[:-1] = mc[1:]
        identity_error = rewards + gamma * (1.0 - mask) * next_mc - mc
        max_identity_error = max(max_identity_error, float(np.max(np.abs(identity_error))))
        returns.append(mc)
        terminal_count += int(mask.sum())
        terminated_count += int(np.asarray(episode.get("terminated", []), bool).sum())
        truncated_count += int(np.asarray(episode.get("truncated", []), bool).sum())
        transitions += length
        for step in range(length):
            references[min(context_length, step + 1)].append((ep_index, step))
    if max_identity_error > 1e-9:
        raise RuntimeError(f"Finite MC identity failed: {max_identity_error}")
    return {
        "episodes": episodes,
        "manifest": manifest,
        "references": references,
        "returns": returns,
        "contract": {
            "episodes": len(episodes),
            "transitions": transitions,
            "terminal_transitions": terminal_count,
            "terminated_transitions": terminated_count,
            "truncated_transitions": truncated_count,
            "finite_mc_identity_max_abs": max_identity_error,
            "last_transition_masked": True,
        },
    }


def make_batch(dataset, references, length):
    episodes, returns = dataset["episodes"], dataset["returns"]
    count = len(references)
    observations = np.empty((count, length, 59), np.float32)
    next_observations = np.empty_like(observations)
    actions = np.empty((count, length, 14), np.float32)
    episode_steps = np.empty((count, length), np.int64)
    next_actions = np.zeros((count, 14), np.float32)
    rewards = np.empty(count, np.float32)
    terminals = np.empty(count, np.float32)
    mc = np.empty(count, np.float64)
    mc_next = np.zeros(count, np.float64)
    for row, (ep_index, target) in enumerate(references):
        episode = episodes[ep_index]
        start = target - length + 1
        ep_actions = np.asarray(episode["actions"], np.float32)
        observations[row] = np.asarray(episode["observations"][start:target + 1], np.float32)
        next_observations[row] = np.asarray(
            episode["next_observations"][start:target + 1], np.float32
        )
        actions[row] = ep_actions[start:target + 1]
        episode_steps[row] = np.asarray(
            episode.get("episode_steps", np.arange(len(ep_actions))), np.int64
        )[start:target + 1]
        rewards[row] = np.asarray(episode["rewards"], np.float32).reshape(-1)[target]
        terminals[row] = terminal_mask(episode)[target]
        mc[row] = returns[ep_index][target]
        if terminals[row] < 0.5:
            if target + 1 >= len(ep_actions):
                raise RuntimeError("Nonterminal transition has no next replay action")
            next_actions[row] = ep_actions[target + 1]
            mc_next[row] = returns[ep_index][target + 1]
    return {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "episode_steps": episode_steps,
        "next_actions": next_actions,
        "rewards": rewards,
        "terminals": terminals,
        "mc": mc,
        "mc_next": mc_next,
    }


def error_stats(values):
    x = np.asarray(values, np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all():
        raise RuntimeError("Empty or nonfinite error array")
    return {
        "count": len(x),
        "mean_bias": float(x.mean()),
        "median_bias": float(np.median(x)),
        "mae": float(np.abs(x).mean()),
        "rmse": float(np.sqrt(np.square(x).mean())),
        "std": float(x.std()),
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "negative_fraction": float(np.mean(x < -1e-10)),
        "positive_fraction": float(np.mean(x > 1e-10)),
        "finite": True,
    }


def value_stats(values):
    x = np.asarray(values, np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all():
        raise RuntimeError("Empty or nonfinite value array")
    return {
        "count": len(x),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "finite": True,
    }


def scope_stats(a, mask):
    g = a["mc"][mask]
    identity = a["mc_identity_target"][mask] - g
    learned_mean = a["behavior_mean_target"][mask] - a["mc_identity_target"][mask]
    min_behavior = a["behavior_min_target"][mask] - a["behavior_mean_target"][mask]
    policy_mean = a["policy_mean_target"][mask] - a["behavior_mean_target"][mask]
    min_policy = a["production_target"][mask] - a["policy_mean_target"][mask]
    total = a["production_target"][mask] - g
    components = {
        "mc_identity_minus_mc": identity,
        "learned_future_mean_td_minus_mc_identity": learned_mean,
        "behavior_min_minus_behavior_mean": min_behavior,
        "policy_mean_minus_behavior_mean": policy_mean,
        "policy_min_minus_policy_mean": min_policy,
        "behavior_min_td_minus_mc": a["behavior_min_target"][mask] - g,
        "policy_min_minus_behavior_min": (
            a["production_target"][mask] - a["behavior_min_target"][mask]
        ),
        "production_td_minus_mc": total,
        "first_prior_action_effect_on_behavior_td": (
            a["behavior_min_target"][mask] - a["zero_first_behavior_target"][mask]
        ),
        "zero_first_behavior_td_minus_mc": a["zero_first_behavior_target"][mask] - g,
    }
    future_mask = mask & (a["terminals"] < 0.5)
    future = {
        "q1_target_behavior": value_stats(a["behavior_q1"][future_mask]),
        "q2_target_behavior": value_stats(a["behavior_q2"][future_mask]),
        "qmean_target_behavior": value_stats(a["behavior_qmean"][future_mask]),
        "qmin_target_behavior": value_stats(a["behavior_qmin"][future_mask]),
        "policy_q1_expectation": value_stats(a["policy_q1"][future_mask]),
        "policy_q2_expectation": value_stats(a["policy_q2"][future_mask]),
        "policy_qmean_expectation": value_stats(a["policy_qmean"][future_mask]),
        "policy_qmin_expectation": value_stats(a["policy_qmin"][future_mask]),
        "mc_next": value_stats(a["mc_next"][future_mask]),
        "qmin_behavior_minus_mc_next": error_stats(
            a["behavior_qmin"][future_mask] - a["mc_next"][future_mask]
        ),
        "qmean_behavior_minus_mc_next": error_stats(
            a["behavior_qmean"][future_mask] - a["mc_next"][future_mask]
        ),
    }
    # Exact five-term path: the policy-side min term must be incremental
    # relative to the behavior-side min term, avoiding double counting.
    incremental_min = min_policy - min_behavior
    components["incremental_min_effect_policy_vs_behavior"] = incremental_min
    residual = total - (
        identity + learned_mean + min_behavior + policy_mean + incremental_min
    )
    direct_residual = total - (
        components["behavior_min_td_minus_mc"] + components["policy_min_minus_behavior_min"]
    )
    return {
        "count": int(mask.sum()),
        "nonterminal_count": int(future_mask.sum()),
        "components": {key: error_stats(value) for key, value in components.items()},
        "future_q": future,
        "algebra_max_abs_residual": float(np.max(np.abs(residual))),
        "direct_min_path_max_abs_residual": float(np.max(np.abs(direct_residual))),
    }


def bucket_stats(a, eligible):
    positive = a["mc"][eligible & (a["mc"] > 1e-12)]
    if not len(positive):
        raise RuntimeError("No positive finite MC returns for bucket analysis")
    p50, p90 = [float(v) for v in np.percentile(positive, (50, 90))]
    g = a["mc"]
    buckets = {
        "zero": eligible & (np.abs(g) <= 1e-12),
        "positive_low_to_p50": eligible & (g > 1e-12) & (g <= p50),
        "positive_p50_to_p90": eligible & (g > p50) & (g <= p90),
        "positive_top10": eligible & (g > p90),
    }
    if np.any(eligible & (g < -1e-12)):
        buckets["negative"] = eligible & (g < -1e-12)
    result = {}
    for name, mask in buckets.items():
        if not mask.any():
            result[name] = {"count": 0}
            continue
        result[name] = {
            "count": int(mask.sum()),
            "mc": value_stats(g[mask]),
            "behavior_min_td_minus_mc": error_stats(a["behavior_min_target"][mask] - g[mask]),
            "production_td_minus_mc": error_stats(a["production_target"][mask] - g[mask]),
            "policy_min_minus_behavior_min": error_stats(
                a["production_target"][mask] - a["behavior_min_target"][mask]
            ),
        }
    return {"positive_p50": p50, "positive_p90": p90, "buckets": result}


@torch.no_grad()
def evaluate_dataset(critic, actor, payload, dataset, device, batch_size):
    config = payload["config"]
    gamma = float(config["gamma"])
    context_length = int(config["recurrent_replay"]["critic_context_length"])
    horizon = int(config["horizon"])
    actor_horizon = int(config["actor_source_contract"]["rnn_horizon"])
    norm = payload["action_normalization_stats"]
    scale = torch.as_tensor(norm["scale"], dtype=torch.float32, device=device)
    offset = torch.as_tensor(norm["offset"], dtype=torch.float32, device=device)
    arrays = {key: [] for key in ROW_KEYS}
    arrays.update({key: [] for key in ("mc", "mc_next", "rewards", "terminals", "context_length")})
    max_component_contract_error = 0.0
    for length, references in dataset["references"].items():
        for first in range(0, len(references), batch_size):
            batch = make_batch(dataset, references[first:first + batch_size], length)
            nxt = torch.as_tensor(batch["next_observations"], device=device)
            actions = torch.as_tensor(batch["actions"], device=device)
            steps = torch.as_tensor(batch["episode_steps"], device=device)
            next_actions = torch.as_tensor(batch["next_actions"], device=device)
            reward = torch.as_tensor(batch["rewards"], device=device)
            terminals = torch.as_tensor(batch["terminals"], device=device)
            mc_next = torch.as_tensor(batch["mc_next"], dtype=torch.float32, device=device)
            progress = (steps + 1).to(dtype=nxt.dtype).unsqueeze(-1) / float(horizon)

            # Exact historical production successor context: next_obs and
            # the executed action at each predecessor state, including token 0.
            contexts, _ = critic.encode_history(nxt, actions, progress)
            final = (contexts[0][:, -1], contexts[1][:, -1])
            q1, q2 = critic.q_from_context(final, next_actions)
            bq1, bq2 = q1.reshape(-1), q2.reshape(-1)
            bmean = 0.5 * (bq1 + bq2)
            bmin = torch.minimum(bq1, bq2)

            starts = _last_reset_starts_from_numpy(batch["episode_steps"], actor_horizon)
            distribution, _ = target_final_distribution_vectorized(
                actor, nxt, horizon=actor_horizon, starts=starts
            )
            pmin, q1_modes, q2_modes, params, _ = component_mean_q(
                critic, final, distribution, scale, offset
            )
            probs = params["probs"]
            p1 = (probs * q1_modes).sum(-1)
            p2 = (probs * q2_modes).sum(-1)
            pmean = 0.5 * (p1 + p2)
            component_contract_error = torch.max(
                torch.abs(pmin - (probs * torch.minimum(q1_modes, q2_modes)).sum(-1))
            ).item()
            max_component_contract_error = max(max_component_contract_error, component_contract_error)

            # Control: same target successor state and zero-state LSTM window,
            # but set its first unavailable previous action to zero.
            zero_first_actions = actions.clone()
            zero_first_actions[:, 0] = 0.0
            zero_contexts, _ = critic.encode_history(nxt, zero_first_actions, progress)
            zero_final = (zero_contexts[0][:, -1], zero_contexts[1][:, -1])
            z1, z2 = critic.q_from_context(zero_final, next_actions)
            zero_min = torch.minimum(z1.reshape(-1), z2.reshape(-1))

            mask = 1.0 - terminals
            y_id = reward + gamma * mask * mc_next
            y_bmean = reward + gamma * mask * bmean
            y_bmin = reward + gamma * mask * bmin
            y_pmean = reward + gamma * mask * pmean
            y_pmin = reward + gamma * mask * pmin
            y_zero = reward + gamma * mask * zero_min
            values = (
                bq1, bq2, bmean, bmin, p1, p2, pmean, pmin,
                zero_min, y_id, y_bmean, y_bmin, y_pmean, y_pmin, y_zero,
            )
            packed = torch.stack(values, dim=1).cpu().numpy()
            for column, key in enumerate(ROW_KEYS):
                arrays[key].append(packed[:, column])
            for key in ("mc", "mc_next", "rewards", "terminals"):
                arrays[key].append(batch[key])
            arrays["context_length"].append(
                np.full(len(batch["rewards"]), length, dtype=np.int64)
            )
    a = {key: np.concatenate(parts).reshape(-1) for key, parts in arrays.items()}
    if any(not np.isfinite(value).all() for value in a.values()):
        raise FloatingPointError("Nonfinite decomposition tensor")
    all_mask = np.ones(len(a["mc"]), dtype=bool)
    eligible = (a["context_length"] == context_length) & (a["terminals"] < 0.5)
    if not eligible.any():
        raise RuntimeError("No full-context nonterminal transitions")
    all_stats = scope_stats(a, all_mask)
    eligible_stats = scope_stats(a, eligible)
    if all_stats["components"]["mc_identity_minus_mc"]["mae"] > 5e-6:
        raise RuntimeError("MC identity target differs from finite MC return")
    if max(all_stats["algebra_max_abs_residual"],
           eligible_stats["algebra_max_abs_residual"],
           all_stats["direct_min_path_max_abs_residual"]) > 5e-6:
        raise RuntimeError("TD decomposition does not add back to production target")
    if max_component_contract_error > 5e-6:
        raise RuntimeError("Component-mean production min contract mismatch")
    return {
        "transition_counts": {
            "all": len(a["mc"]),
            "nonterminal": int((a["terminals"] < 0.5).sum()),
            "terminal": int((a["terminals"] >= 0.5).sum()),
            "train_eligible_full_context_nonterminal": int(eligible.sum()),
        },
        "contracts": {
            "mc_identity_max_abs": float(np.max(np.abs(a["mc_identity_target"] - a["mc"]))),
            "algebra_max_abs_residual": all_stats["algebra_max_abs_residual"],
            "direct_min_path_max_abs_residual": all_stats["direct_min_path_max_abs_residual"],
            "component_mean_min_max_abs_residual": max_component_contract_error,
            "all_arrays_finite": True,
        },
        "all_transitions": all_stats,
        "train_eligible_full_context_nonterminal": eligible_stats,
        "return_buckets": bucket_stats(a, eligible),
    }


def write_outputs(path, report):
    path.mkdir(parents=True, exist_ok=True)
    json_path = path / "stage3_300k_td_bias_decomposition.json"
    csv_path = path / "stage3_300k_td_bias_decomposition.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    fields = (
        "group", "checkpoint", "dataset", "scope", "component", "count",
        "mean_bias", "median_bias", "mae", "rmse", "std", "p05", "p25",
        "p50", "p75", "p95", "negative_fraction", "positive_fraction", "finite",
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for group, group_result in report["groups"].items():
            for checkpoint, checkpoint_result in group_result["checkpoints"].items():
                for dataset, dataset_result in checkpoint_result["datasets"].items():
                    for scope in ("all_transitions", "train_eligible_full_context_nonterminal"):
                        for component, metrics in dataset_result[scope]["components"].items():
                            writer.writerow({
                                "group": group,
                                "checkpoint": checkpoint,
                                "dataset": dataset,
                                "scope": scope,
                                "component": component,
                                **metrics,
                            })
    return json_path, csv_path


def main():
    args = arguments()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    run_dir = Path(args.stage3_run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    config = json.loads((run_dir / "shared" / "config_resolved.json").read_text())
    sources = json.loads((run_dir / "shared" / "stage2_source_manifest.json").read_text())
    device = resolve_device(args.device)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report = {
        "status": "RUNNING",
        "head_commit": head,
        "stage3_run_dir": str(run_dir),
        "device": str(device),
        "datasets_evaluated": list(DATASETS),
        "historical_contract": {
            "stage2_history_semantics": EXPECTED_SEMANTICS,
            "stage3_critic_context_length": 11,
            "successor_tokens": "next_observations[s:t+1]",
            "successor_previous_actions": "actions[s:t+1], including first token",
            "actor_horizon": 10,
            "terminal_mask": "terminated OR truncated",
            "gamma": 0.99,
            "decomposition": (
                "production-MC = (MC-id-MC) + "
                "(behavior-mean-MC-id) + (behavior-min-behavior-mean) + "
                "(policy-mean-behavior-mean) + "
                "[(policy-min-policy-mean)-(behavior-min-behavior-mean)]"
            ),
        },
        "safety": {
            "environment_steps": 0,
            "optimizer_steps": 0,
            "rollout_started": False,
            "training_checkpoint_writes": 0,
            "production_source_modified": False,
            "fusion_result_touched": False,
        },
        "groups": {},
    }
    for group in ("multi_q", "rnn_q"):
        stage2_path = Path(sources[group]["checkpoint"]).resolve()
        if not stage2_path.is_file():
            raise FileNotFoundError(stage2_path)
        group_dir = run_dir / group
        old_eps, old_manifest = old_dataset(config, group)
        late_eps, late_manifest = stage_dataset(
            group_dir, "late", 200_000, 300_000, "step_0300000.sequences.npy"
        )
        datasets = {
            "old": (old_eps, old_manifest),
            "late": (late_eps, late_manifest),
        }
        actor, _, actor_metadata = load_exact_actor(
            config["bc_rnn_checkpoint"], device
        )
        actor.eval()
        actor.requires_grad_(False)
        prepared = None
        group_result = {
            "stage2_source_checkpoint": str(stage2_path),
            "stage2_source_manifest_architecture": sources[group]["architecture"],
            "dataset_manifest": {name: data[1] for name, data in datasets.items()},
            "actor_source_metadata": actor_metadata,
            "checkpoints": {},
        }
        reference_config = None
        reference_norm = None
        for checkpoint, filename in CHECKPOINTS.items():
            checkpoint_path = group_dir / "checkpoints" / filename
            payload = torch.load(checkpoint_path, map_location="cpu")
            if payload.get("stage") != "stage3-v5" or payload.get("group") != group:
                raise RuntimeError(f"{checkpoint_path}: Stage3 group contract changed")
            cp_config = payload["config"]
            contract_keys = (
                "gamma", "horizon", "recurrent_replay",
                "actor_source_contract", "bc_rnn_checkpoint",
            )
            selected_config = {key: cp_config[key] for key in contract_keys}
            if reference_config is None:
                reference_config = selected_config
                reference_norm = payload["action_normalization_stats"]
            elif selected_config != reference_config or payload["action_normalization_stats"] != reference_norm:
                raise RuntimeError(f"{checkpoint_path}: checkpoint config/normalization drift")
            context_length = int(cp_config["recurrent_replay"]["critic_context_length"])
            gamma = float(cp_config["gamma"])
            if context_length != 11 or not math.isclose(gamma, 0.99, abs_tol=1e-12):
                raise RuntimeError(f"{checkpoint_path}: historical context/gamma mismatch")
            if int(cp_config["actor_source_contract"]["rnn_horizon"]) != 10:
                raise RuntimeError(f"{checkpoint_path}: historical Actor horizon mismatch")
            if prepared is None:
                prepared = {
                    name: prepare_dataset(episodes, manifest, context_length, gamma)
                    for name, (episodes, manifest) in datasets.items()
                }
            critic, stage2_payload = load_stage2_source_critic(stage2_path, device)
            semantics = stage2_payload.get(
                "history_semantics",
                stage2_payload.get("architecture", {}).get("history_semantics"),
            )
            if semantics != EXPECTED_SEMANTICS:
                raise RuntimeError(f"{stage2_path}: historical Stage2 semantics mismatch")
            critic.load_state_dict(payload["target_q1_q2"], strict=True)
            critic.eval()
            critic.requires_grad_(False)
            actor.load_state_dict(payload["target_actor"], strict=True)
            actor.eval()
            actor.low_noise_eval = True
            actor_hash = state_hash(payload["actor"])
            target_actor_hash = state_hash(payload["target_actor"])
            checkpoint_result = {
                "path": str(checkpoint_path.resolve()),
                "env_steps": int(payload["env_steps"]),
                "critic_updates": int(payload["updates"]),
                "actor_updates": int(payload["actor_updates"]),
                "critic_context_length": context_length,
                "stage2_history_semantics": semantics,
                "actor_hash": actor_hash,
                "target_actor_hash": target_actor_hash,
                "actor_equals_target_actor": actor_hash == target_actor_hash,
                "target_critic_hash": state_hash(payload["target_q1_q2"]),
                "datasets": {},
            }
            for name, dataset in prepared.items():
                print(
                    f"[EVAL] {group} {checkpoint} {name} "
                    f"episodes={len(dataset['episodes'])} transitions={dataset['contract']['transitions']}",
                    flush=True,
                )
                checkpoint_result["datasets"][name] = evaluate_dataset(
                    critic, actor, payload, dataset, device, args.batch_size
                )
            group_result["checkpoints"][checkpoint] = checkpoint_result
            del critic, stage2_payload, payload
            gc.collect()
            if device.type == "npu":
                torch.npu.empty_cache()
            sync(device)
        hashes_actor = {x["actor_hash"] for x in group_result["checkpoints"].values()}
        hashes_target = {x["target_actor_hash"] for x in group_result["checkpoints"].values()}
        group_result["actor_verification"] = {
            "actor_hashes_unchanged": len(hashes_actor) == 1,
            "target_actor_hashes_unchanged": len(hashes_target) == 1,
            "actor_equals_target_at_all_checkpoints": all(
                x["actor_equals_target_actor"] for x in group_result["checkpoints"].values()
            ),
            "all_actor_updates_zero": all(
                x["actor_updates"] == 0 for x in group_result["checkpoints"].values()
            ),
        }
        report["groups"][group] = group_result
        del actor, prepared, datasets, old_eps, late_eps
        gc.collect()
        if device.type == "npu":
            torch.npu.empty_cache()
    report["validity"] = {
        "groups_complete": set(report["groups"]) == {"multi_q", "rnn_q"},
        "all_actor_updates_zero": all(
            g["actor_verification"]["all_actor_updates_zero"]
            for g in report["groups"].values()
        ),
        "all_mc_identity_valid": all(
            result["contracts"]["mc_identity_max_abs"] <= 5e-6
            for g in report["groups"].values()
            for cp in g["checkpoints"].values()
            for result in cp["datasets"].values()
        ),
        "all_algebra_valid": all(
            result["contracts"]["algebra_max_abs_residual"] <= 5e-6
            for g in report["groups"].values()
            for cp in g["checkpoints"].values()
            for result in cp["datasets"].values()
        ),
        "actor_and_target_actor_unchanged": all(
            g["actor_verification"]["actor_hashes_unchanged"]
            and g["actor_verification"]["target_actor_hashes_unchanged"]
            for g in report["groups"].values()
        ),
    }
    report["status"] = "PASS" if all(report["validity"].values()) else "INCONCLUSIVE"
    json_path, csv_path = write_outputs(Path(args.output_dir).resolve(), report)
    print(f"[STATUS] {report['status']}")
    print(f"[JSON] {json_path}")
    print(f"[CSV] {csv_path}")


if __name__ == "__main__":
    main()
