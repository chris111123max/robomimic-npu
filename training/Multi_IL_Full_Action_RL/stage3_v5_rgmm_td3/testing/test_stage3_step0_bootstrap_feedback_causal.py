#!/usr/bin/env python3
"""Causal short-run test: moving bootstrap feedback vs clipped-min seed.

Historical target:
  stage3v5_stage22_rnn4k_multi6k_20260922

Four branches start from the exact same Stage3 step0 Critic / optimizer state
and consume the exact same frozen 50/50 old+late minibatch schedule:

  A production_moving_min : production component-mean target, twin min, Polyak.
  B moving_mean           : same moving target Critic, twin mean instead of min.
  C frozen_step0_min      : production twin-min target, target Critic frozen.
  D oracle_mc             : exact finite-MC target; target Critic still Polyak-updated
                            only for diagnostics (it is never used for supervision).

No environment or rollout is created. No production checkpoint is written.
Only diagnostic optimizer steps on frozen replay data are performed.
"""
from __future__ import annotations

import argparse
import copy
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
from stage3_v5_readiness import correlation, discounted_returns  # noqa: E402
from test_stage2_stage3_readiness_compare import resolve_device, sync  # noqa: E402
from test_stage3_300k_critic_stage_adaptation_matrix import (  # noqa: E402
    load_stage2_source_critic,
    old_dataset,
    stage_dataset,
)
from test_stage3_300k_td_bias_decomposition import terminal_mask  # noqa: E402


BRANCHES = (
    "production_moving_min",
    "moving_mean",
    "frozen_step0_min",
    "oracle_mc",
)
EXPECTED_SEMANTICS = "full_episode_prefix_unroll_learning_mask"
DEFAULT_MILESTONES = (0, 1, 10, 50, 100, 250, 500, 1000, 2000, 3000, 5000, 10000)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-run-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--groups", nargs="+", choices=("multi_q", "rnn_q"),
                        default=("multi_q", "rnn_q"))
    parser.add_argument("--branches", nargs="+", choices=BRANCHES,
                        default=BRANCHES)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probe-size", type=int, default=4096)
    parser.add_argument("--probe-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument(
        "--output-dir",
        help=("Default: testing/_runtime/"
              "stage3_step0_bootstrap_feedback_causal"),
    )
    return parser.parse_args()


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def module_hash(module):
    return state_hash(module.state_dict())


def cleanup(device):
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def load_complete_dataset(run_config, run_dir, group):
    old_eps, old_manifest = old_dataset(run_config, group)
    late_eps, late_manifest = stage_dataset(
        run_dir / group,
        "late",
        200_000,
        300_000,
        "step_0300000.sequences.npy",
    )
    return {
        "old": {"episodes": old_eps, "manifest": old_manifest},
        "late": {"episodes": late_eps, "manifest": late_manifest},
    }


def returns_cache(episodes, gamma):
    result = []
    for index, episode in enumerate(episodes):
        mask = terminal_mask(episode)
        if np.any(mask[:-1]) or mask[-1] != 1:
            raise RuntimeError(f"episode {index}: expected complete finite episode")
        mc = discounted_returns(episode, gamma)
        rewards = np.asarray(episode["rewards"], np.float64).reshape(-1)
        next_mc = np.zeros_like(mc)
        next_mc[:-1] = mc[1:]
        identity = rewards + gamma * (1.0 - mask) * next_mc - mc
        if float(np.max(np.abs(identity))) > 5e-6:
            raise RuntimeError(
                f"episode {index}: finite-MC identity mismatch "
                f"{float(np.max(np.abs(identity)))}"
            )
        result.append(np.asarray(mc, np.float64))
    return result


def eligible_episode_indices(episodes, length):
    ids = [
        index for index, episode in enumerate(episodes)
        if len(np.asarray(episode["actions"])) >= int(length)
    ]
    if not ids:
        raise RuntimeError("no episode is long enough for recurrent sampling")
    return np.asarray(ids, dtype=np.int32)


def make_ref_schedule(old_eps, late_eps, updates, batch_size, length, seed):
    if int(batch_size) != 256:
        raise ValueError("historical Stage3-v5 Critic batch must remain 256")
    half = int(batch_size) // 2
    old_ids = eligible_episode_indices(old_eps, length)
    late_ids = eligible_episode_indices(late_eps, length)
    rng = np.random.default_rng(int(seed))
    schedule = []
    for _ in range(int(updates)):
        sides = []
        for episodes, ids in ((old_eps, old_ids), (late_eps, late_ids)):
            chosen = ids[rng.integers(len(ids), size=half)]
            refs = np.empty((half, 2), np.int32)
            for row, episode_id in enumerate(chosen):
                n_starts = len(episodes[int(episode_id)]["actions"]) - int(length) + 1
                refs[row] = (int(episode_id), int(rng.integers(n_starts)))
            sides.append(refs)
        schedule.append((sides[0], sides[1]))
    return schedule


def make_probe_refs(episodes, count, length, seed):
    all_refs = []
    for episode_id, episode in enumerate(episodes):
        n_starts = len(episode["actions"]) - int(length) + 1
        for start in range(max(0, n_starts)):
            all_refs.append((episode_id, start))
    if not all_refs:
        raise RuntimeError("no full-context probe transition")
    rng = np.random.default_rng(int(seed))
    count = min(int(count), len(all_refs))
    choice = rng.choice(len(all_refs), size=count, replace=False)
    return np.asarray([all_refs[int(i)] for i in choice], dtype=np.int32)


def stack_refs(episodes, mc_returns, refs, length):
    count = len(refs)
    observations = np.empty((count, length, 59), np.float32)
    next_observations = np.empty_like(observations)
    actions = np.empty((count, length, 14), np.float32)
    episode_steps = np.empty((count, length), np.int64)
    rewards = np.empty((count, 1), np.float32)
    terminals = np.empty((count, 1), np.float32)
    mc = np.empty((count, 1), np.float32)
    mc_next = np.zeros((count, 1), np.float32)
    next_actions = np.zeros((count, 14), np.float32)
    success = np.empty(count, np.int64)
    episode_ids = np.empty(count, np.int64)

    for row, (episode_id, start) in enumerate(np.asarray(refs, np.int64)):
        episode_id, start = int(episode_id), int(start)
        episode = episodes[episode_id]
        target = start + int(length) - 1
        stop = target + 1
        ep_actions = np.asarray(episode["actions"], np.float32)
        steps = np.asarray(
            episode.get("episode_steps", np.arange(len(ep_actions))), np.int64
        )
        observations[row] = np.asarray(
            episode["observations"][start:stop], np.float32
        )
        next_observations[row] = np.asarray(
            episode["next_observations"][start:stop], np.float32
        )
        actions[row] = ep_actions[start:stop]
        episode_steps[row] = steps[start:stop]
        rewards[row, 0] = np.asarray(
            episode["rewards"], np.float32
        ).reshape(-1)[target]
        terminals[row, 0] = terminal_mask(episode)[target]
        mc[row, 0] = mc_returns[episode_id][target]
        if terminals[row, 0] < 0.5:
            if target + 1 >= len(ep_actions):
                raise RuntimeError("nonterminal target lacks replay next action")
            mc_next[row, 0] = mc_returns[episode_id][target + 1]
            next_actions[row] = ep_actions[target + 1]
        success[row] = int(bool(episode.get("success", False)))
        episode_ids[row] = episode_id

    return {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "episode_steps": episode_steps,
        "rewards": rewards,
        "terminals": terminals,
        "mc": mc,
        "mc_next": mc_next,
        "next_actions": next_actions,
        "success": success,
        "episode_ids": episode_ids,
    }


def merge_batches(left, right):
    keys = (
        "observations", "next_observations", "actions", "episode_steps",
        "rewards", "terminals", "mc", "mc_next", "next_actions",
        "success", "episode_ids",
    )
    return {key: np.concatenate((left[key], right[key]), axis=0) for key in keys}


def tensor_batch(batch, device):
    result = {}
    for key in (
        "observations", "next_observations", "actions",
        "rewards", "terminals", "mc", "mc_next", "next_actions",
    ):
        result[key] = torch.as_tensor(
            batch[key], dtype=torch.float32, device=device
        )
    result["episode_steps"] = torch.as_tensor(
        batch["episode_steps"], dtype=torch.long, device=device
    )
    return result


def action_norm_tensors(payload, device):
    norm = payload["action_normalization_stats"]
    scale = torch.as_tensor(
        norm["scale"], dtype=torch.float32, device=device
    )
    offset = torch.as_tensor(
        norm["offset"], dtype=torch.float32, device=device
    )
    return scale, offset


def encode_diagnostic_contexts(
    critic,
    observations,
    actions,
    episode_steps,
    horizon,
    next_observations=None,
):
    """Encode the historical 11-token Stage3 contract for this test.

    The shared Stage3 adapter only accepts at most 10 tokens and its successor
    path zeroes / shifts the first predecessor-action token. The historical
    Stage3 11-token successor contract instead feeds actions[s:t+1] directly.
    This test-local adapter preserves the trained recurrent encoder and progress
    features while honoring that recorded 11-token contract.
    """
    if observations.ndim != 3 or actions.ndim != 3:
        raise ValueError("diagnostic history inputs must be rank-three")
    if observations.shape[1] != 11 or actions.shape[:2] != observations.shape[:2]:
        raise ValueError("historical Stage3 diagnostic requires aligned 11-token windows")
    if episode_steps.shape != observations.shape[:2]:
        raise ValueError("episode steps must align with the 11-token window")

    if next_observations is None:
        tokens = observations
        predecessor_actions = torch.zeros_like(actions)
        predecessor_actions[:, 1:] = actions[:, :-1]
        predecessor_actions = predecessor_actions.masked_fill(
            episode_steps.eq(0).unsqueeze(-1), 0.0
        )
        steps = episode_steps
    else:
        if next_observations.shape != observations.shape:
            raise ValueError("successor observations must match current windows")
        tokens = next_observations
        predecessor_actions = actions
        steps = episode_steps + 1

    progress = steps.to(dtype=observations.dtype).unsqueeze(-1) / float(horizon)
    contexts, _ = critic.encode_history(tokens, predecessor_actions, progress)
    return contexts


@torch.no_grad()
def policy_expected_values(
    target_critic,
    target_actor,
    batch_t,
    config,
    scale,
    offset,
):
    contexts = encode_diagnostic_contexts(
        target_critic,
        batch_t["observations"],
        batch_t["actions"],
        batch_t["episode_steps"],
        int(config["horizon"]),
        next_observations=batch_t["next_observations"],
    )
    final = (contexts[0][:, -1], contexts[1][:, -1])
    actor_horizon = int(config["actor_source_contract"]["rnn_horizon"])
    starts = _last_reset_starts_from_numpy(
        batch_t["episode_steps"].detach().cpu().numpy(), actor_horizon
    )
    distribution, _ = target_final_distribution_vectorized(
        target_actor,
        batch_t["next_observations"],
        horizon=actor_horizon,
        starts=starts,
    )
    qmin, q1_modes, q2_modes, params, _ = component_mean_q(
        target_critic, final, distribution, scale, offset
    )
    probs = params["probs"]
    qmean = (
        probs * (0.5 * (q1_modes + q2_modes))
    ).sum(-1)
    return qmin.reshape(-1, 1), qmean.reshape(-1, 1)


@torch.no_grad()
def behavior_future_values(target_critic, batch_t, config):
    contexts = encode_diagnostic_contexts(
        target_critic,
        batch_t["observations"],
        batch_t["actions"],
        batch_t["episode_steps"],
        int(config["horizon"]),
        next_observations=batch_t["next_observations"],
    )
    final = (contexts[0][:, -1], contexts[1][:, -1])
    q1, q2 = target_critic.q_from_context(final, batch_t["next_actions"])
    q1, q2 = q1.reshape(-1, 1), q2.reshape(-1, 1)
    return torch.minimum(q1, q2), 0.5 * (q1 + q2)


def online_values(critic, batch_t, config):
    contexts = encode_diagnostic_contexts(
        critic,
        batch_t["observations"],
        batch_t["actions"],
        batch_t["episode_steps"],
        int(config["horizon"]),
    )
    final = (contexts[0][:, -1], contexts[1][:, -1])
    current_actions = batch_t["actions"][:, -1]
    return critic.q_from_context(final, current_actions)


def make_agent_parts(stage2_path, step0_payload, device):
    critic, stage2_payload = load_stage2_source_critic(stage2_path, device)
    semantics = stage2_payload.get(
        "history_semantics",
        stage2_payload.get("architecture", {}).get("history_semantics"),
    )
    if semantics != EXPECTED_SEMANTICS:
        raise RuntimeError(f"historical Stage2 semantics changed: {semantics!r}")
    critic.load_state_dict(step0_payload["q1_q2"], strict=True)
    critic.train()

    target_critic = copy.deepcopy(critic).to(device)
    target_critic.load_state_dict(step0_payload["target_q1_q2"], strict=True)
    target_critic.eval()
    target_critic.requires_grad_(False)

    config = step0_payload["config"]
    actor, rollout, _ = load_exact_actor(config["bc_rnn_checkpoint"], device)
    del rollout
    actor.load_state_dict(step0_payload["target_actor"], strict=True)
    actor.eval()
    actor.low_noise_eval = True
    actor.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=float(config["critic_lr"]),
        weight_decay=float(config["critic_weight_decay"]),
    )
    optimizer.load_state_dict(step0_payload["critic_optimizer"])
    return critic, target_critic, actor, optimizer, stage2_payload


def error_stats(values):
    x = np.asarray(values, np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all():
        raise RuntimeError("empty or nonfinite metric array")
    return {
        "count": int(len(x)),
        "mean": float(x.mean()),
        "mae": float(np.abs(x).mean()),
        "rmse": float(np.sqrt(np.square(x).mean())),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
    }


@torch.no_grad()
def evaluate_probe(
    critic,
    target_critic,
    target_actor,
    episodes,
    mc_returns,
    refs,
    length,
    payload,
    device,
    batch_size,
    branch,
):
    config = payload["config"]
    gamma = float(config["gamma"])
    scale, offset = action_norm_tensors(payload, device)

    qmin_parts = []
    mc_parts = []
    behavior_future_error = []
    production_error = []
    mean_target_error = []
    min_minus_mean = []
    training_target_error = []

    critic.eval()
    target_critic.eval()
    for first in range(0, len(refs), int(batch_size)):
        batch = stack_refs(
            episodes, mc_returns, refs[first:first + int(batch_size)], length
        )
        b = tensor_batch(batch, device)
        q1, q2 = online_values(critic, b, config)
        qmin = torch.minimum(q1, q2)
        policy_min, policy_mean = policy_expected_values(
            target_critic, target_actor, b, config, scale, offset
        )
        behavior_min, _ = behavior_future_values(target_critic, b, config)

        mask = 1.0 - b["terminals"]
        y_prod = b["rewards"] + gamma * mask * policy_min
        y_mean = b["rewards"] + gamma * mask * policy_mean
        if branch == "production_moving_min" or branch == "frozen_step0_min":
            y_train = y_prod
        elif branch == "moving_mean":
            y_train = y_mean
        elif branch == "oracle_mc":
            y_train = b["mc"]
        else:
            raise ValueError(branch)

        nonterminal = b["terminals"].reshape(-1) < 0.5
        if nonterminal.any():
            behavior_future_error.append(
                (
                    behavior_min.reshape(-1)[nonterminal]
                    - b["mc_next"].reshape(-1)[nonterminal]
                ).cpu().numpy()
            )
        qmin_parts.append(qmin.reshape(-1).cpu().numpy())
        mc_parts.append(b["mc"].reshape(-1).cpu().numpy())
        production_error.append((y_prod - b["mc"]).reshape(-1).cpu().numpy())
        mean_target_error.append((y_mean - b["mc"]).reshape(-1).cpu().numpy())
        min_minus_mean.append((y_prod - y_mean).reshape(-1).cpu().numpy())
        training_target_error.append((y_train - b["mc"]).reshape(-1).cpu().numpy())

    critic.train()
    qmin = np.concatenate(qmin_parts).astype(np.float64)
    mc = np.concatenate(mc_parts).astype(np.float64)
    spearman, pearson = correlation(qmin, mc)
    return {
        "count": int(len(qmin)),
        "spearman_qmin_mc": float(spearman),
        "pearson_qmin_mc": float(pearson),
        "qmin_minus_mc": error_stats(qmin - mc),
        "behavior_future_qmin_minus_mc_next": error_stats(
            np.concatenate(behavior_future_error)
        ),
        "production_target_minus_mc": error_stats(np.concatenate(production_error)),
        "mean_target_minus_mc": error_stats(np.concatenate(mean_target_error)),
        "production_min_minus_mean": error_stats(np.concatenate(min_minus_mean)),
        "training_target_minus_mc": error_stats(np.concatenate(training_target_error)),
        "qmin_mean": float(qmin.mean()),
        "qmin_std": float(qmin.std()),
        "mc_mean": float(mc.mean()),
        "mc_std": float(mc.std()),
    }


def train_step(
    branch,
    critic,
    target_critic,
    target_actor,
    optimizer,
    batch,
    payload,
    device,
):
    config = payload["config"]
    gamma = float(config["gamma"])
    tau = float(config["tau"])
    scale, offset = action_norm_tensors(payload, device)
    b = tensor_batch(batch, device)

    with torch.no_grad():
        if branch == "oracle_mc":
            td_target = b["mc"]
        else:
            policy_min, policy_mean = policy_expected_values(
                target_critic, target_actor, b, config, scale, offset
            )
            expected = policy_mean if branch == "moving_mean" else policy_min
            td_target = (
                b["rewards"]
                + gamma * (1.0 - b["terminals"]) * expected
            )

    q1, q2 = online_values(critic, b, config)
    loss_q1 = torch.nn.functional.mse_loss(q1, td_target)
    loss_q2 = torch.nn.functional.mse_loss(q2, td_target)
    loss = loss_q1 + loss_q2
    if not torch.isfinite(loss):
        raise FloatingPointError(f"{branch}: nonfinite Critic loss")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        critic.parameters(), float(config["critic_max_grad_norm"])
    )
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        raise FloatingPointError(f"{branch}: nonfinite Critic grad")
    optimizer.step()

    if branch != "frozen_step0_min":
        with torch.no_grad():
            for source, target in zip(
                critic.parameters(), target_critic.parameters()
            ):
                target.mul_(1.0 - tau).add_(source, alpha=tau)

    return {
        "loss_q1": float(loss_q1.detach().cpu()),
        "loss_q2": float(loss_q2.detach().cpu()),
        "td_target_mean": float(td_target.mean().detach().cpu()),
        "qmin_mean": float(torch.minimum(q1, q2).mean().detach().cpu()),
        "grad_norm_preclip": float(torch.as_tensor(grad_norm).detach().cpu()),
    }


def milestone_values(updates):
    values = [value for value in DEFAULT_MILESTONES if value <= int(updates)]
    if int(updates) not in values:
        values.append(int(updates))
    return tuple(sorted(set(values)))


def run_branch(
    branch,
    stage2_path,
    step0_payload,
    datasets,
    caches,
    schedule,
    probes,
    length,
    device,
    milestones,
    probe_batch_size,
):
    critic, target_critic, target_actor, optimizer, stage2_payload = make_agent_parts(
        stage2_path, step0_payload, device
    )
    initial = {
        "online_hash": module_hash(critic),
        "target_hash": module_hash(target_critic),
        "actor_hash": module_hash(target_actor),
    }
    trajectory = []
    last_train = None

    def record(update):
        sync(device)
        current_target_hash = module_hash(target_critic)
        row = {
            "update": int(update),
            "online_hash": module_hash(critic),
            "target_hash": current_target_hash,
            "target_changed_from_initial": (
                current_target_hash != initial["target_hash"]
            ),
            "last_train_metrics": last_train,
            "datasets": {},
        }
        for name in ("old", "late"):
            row["datasets"][name] = evaluate_probe(
                critic,
                target_critic,
                target_actor,
                datasets[name]["episodes"],
                caches[name],
                probes[name],
                length,
                step0_payload,
                device,
                probe_batch_size,
                branch,
            )
        trajectory.append(row)
        late = row["datasets"]["late"]
        print(
            f"[{branch}] update={update} "
            f"late_spear={late['spearman_qmin_mc']:.6f} "
            f"late_qbias={late['qmin_minus_mc']['mean']:.6f} "
            f"future_bias={late['behavior_future_qmin_minus_mc_next']['mean']:.6f} "
            f"train_target_bias={late['training_target_minus_mc']['mean']:.6f}",
            flush=True,
        )

    record(0)
    milestone_set = set(int(x) for x in milestones)
    failure = None
    completed = 0

    for update_index, (old_refs, late_refs) in enumerate(schedule, 1):
        left = stack_refs(
            datasets["old"]["episodes"], caches["old"], old_refs, length
        )
        right = stack_refs(
            datasets["late"]["episodes"], caches["late"], late_refs, length
        )
        batch = merge_batches(left, right)
        try:
            last_train = train_step(
                branch,
                critic,
                target_critic,
                target_actor,
                optimizer,
                batch,
                step0_payload,
                device,
            )
        except Exception as exc:
            failure = {
                "update": int(update_index),
                "type": type(exc).__name__,
                "message": str(exc),
            }
            break
        completed = update_index
        if completed in milestone_set:
            record(completed)

    result = {
        "branch": branch,
        "completed": bool(failure is None and completed == len(schedule)),
        "updates_requested": int(len(schedule)),
        "updates_completed": int(completed),
        "failure": failure,
        "initial_hashes": initial,
        "final_online_hash": module_hash(critic),
        "final_target_hash": module_hash(target_critic),
        "target_changed": module_hash(target_critic) != initial["target_hash"],
        "stage2_history_semantics": stage2_payload.get(
            "history_semantics",
            stage2_payload.get("architecture", {}).get("history_semantics"),
        ),
        "trajectory": trajectory,
    }

    del critic, target_critic, target_actor, optimizer, stage2_payload
    cleanup(device)
    return result


def compare_branches(branch_results):
    lookup = {}
    for branch, result in branch_results.items():
        lookup[branch] = {
            int(row["update"]): row for row in result["trajectory"]
        }
    common = set.intersection(
        *(set(rows) for rows in lookup.values())
    )
    rows = []
    for update in sorted(common):
        item = {"update": int(update)}
        for branch in branch_results:
            late = lookup[branch][update]["datasets"]["late"]
            item[f"{branch}_late_spearman"] = float(
                late["spearman_qmin_mc"]
            )
            item[f"{branch}_late_q_bias"] = float(
                late["qmin_minus_mc"]["mean"]
            )
            item[f"{branch}_late_future_bias"] = float(
                late["behavior_future_qmin_minus_mc_next"]["mean"]
            )
            item[f"{branch}_late_training_target_bias"] = float(
                late["training_target_minus_mc"]["mean"]
            )
        rows.append(item)
    return rows


def write_outputs(output_dir, report):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "stage3_step0_bootstrap_feedback_causal.json"
    csv_path = output_dir / "stage3_step0_bootstrap_feedback_causal.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    rows = []
    for group, group_result in report["groups"].items():
        for branch, branch_result in group_result["branches"].items():
            for milestone in branch_result["trajectory"]:
                for dataset, metrics in milestone["datasets"].items():
                    rows.append({
                        "group": group,
                        "branch": branch,
                        "update": milestone["update"],
                        "dataset": dataset,
                        "spearman_qmin_mc": metrics["spearman_qmin_mc"],
                        "qmin_bias": metrics["qmin_minus_mc"]["mean"],
                        "qmin_mae": metrics["qmin_minus_mc"]["mae"],
                        "future_qmin_bias": metrics[
                            "behavior_future_qmin_minus_mc_next"
                        ]["mean"],
                        "production_target_bias": metrics[
                            "production_target_minus_mc"
                        ]["mean"],
                        "mean_target_bias": metrics[
                            "mean_target_minus_mc"
                        ]["mean"],
                        "min_minus_mean": metrics[
                            "production_min_minus_mean"
                        ]["mean"],
                        "training_target_bias": metrics[
                            "training_target_minus_mc"
                        ]["mean"],
                    })
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


def main():
    args = arguments()
    if args.updates < 1:
        raise ValueError("--updates must be >= 1")
    if args.batch_size != 256:
        raise ValueError("--batch-size must remain the historical 256")
    if args.probe_size < 256:
        raise ValueError("--probe-size must be >= 256")
    if args.probe_batch_size < 1:
        raise ValueError("--probe-batch-size must be positive")

    run_dir = Path(args.stage3_run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    config = json.loads(
        (run_dir / "shared" / "config_resolved.json").read_text()
    )
    sources = json.loads(
        (run_dir / "shared" / "stage2_source_manifest.json").read_text()
    )
    device = resolve_device(args.device)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else HERE / "_runtime" / "stage3_step0_bootstrap_feedback_causal"
    )
    report = {
        "status": "RUNNING",
        "head_commit": head,
        "stage3_run_dir": str(run_dir),
        "device": str(device),
        "design": {
            "branches": list(args.branches),
            "frozen_dataset_mix": "128 old + 128 late per update",
            "same_minibatch_schedule_across_branches": True,
            "same_step0_initialization_across_branches": True,
            "actor_learning": False,
            "environment_or_rollout": False,
            "purpose": (
                "separate recursive moving-target bootstrap feedback "
                "from clipped twin-min bias"
            ),
        },
        "safety": {
            "environment_steps": 0,
            "rollout_started": False,
            "production_checkpoint_writes": 0,
            "production_source_modified": False,
            "fusion_result_touched": False,
            "diagnostic_optimizer_steps_per_completed_branch": int(args.updates),
        },
        "groups": {},
    }

    milestones = milestone_values(args.updates)
    for group_index, group in enumerate(args.groups):
        if group not in sources:
            raise RuntimeError(f"stage2 source manifest misses {group}")
        step0_path = (
            run_dir / group / "checkpoints" / "step0_transfer.pth"
        )
        stage2_path = Path(sources[group]["checkpoint"]).resolve()
        if not step0_path.is_file() or not stage2_path.is_file():
            raise FileNotFoundError(
                step0_path if not step0_path.is_file() else stage2_path
            )
        step0 = torch.load(step0_path, map_location="cpu")
        if step0.get("stage") != "stage3-v5" or step0.get("group") != group:
            raise RuntimeError(f"{step0_path}: invalid Stage3 group")
        if int(step0.get("env_steps", -1)) != 0:
            raise RuntimeError(f"{step0_path}: not env-step zero")
        if int(step0.get("updates", -1)) != 0:
            raise RuntimeError(f"{step0_path}: Critic already updated")
        if int(step0.get("actor_updates", -1)) != 0:
            raise RuntimeError(f"{step0_path}: Actor already updated")
        if state_hash(step0["actor"]) != state_hash(step0["target_actor"]):
            raise RuntimeError(
                f"{step0_path}: step0 Actor and target Actor are not identical"
            )

        cp_config = step0["config"]
        length = int(cp_config["recurrent_replay"]["critic_context_length"])
        if length != 11:
            raise RuntimeError(
                f"{group}: expected historical context length 11, got {length}"
            )
        if not math.isclose(float(cp_config["gamma"]), 0.99, abs_tol=1e-12):
            raise RuntimeError(f"{group}: historical gamma changed")
        if int(cp_config["actor_source_contract"]["rnn_horizon"]) != 10:
            raise RuntimeError(f"{group}: Actor horizon changed")
        if not math.isclose(float(cp_config["tau"]), 0.005, abs_tol=1e-12):
            raise RuntimeError(f"{group}: historical tau changed")

        datasets = load_complete_dataset(config, run_dir, group)
        caches = {
            name: returns_cache(value["episodes"], float(cp_config["gamma"]))
            for name, value in datasets.items()
        }
        group_seed = int(args.seed) + group_index * 100_003
        schedule = make_ref_schedule(
            datasets["old"]["episodes"],
            datasets["late"]["episodes"],
            args.updates,
            args.batch_size,
            length,
            group_seed,
        )
        probes = {
            "old": make_probe_refs(
                datasets["old"]["episodes"],
                args.probe_size,
                length,
                group_seed + 1_000_003,
            ),
            "late": make_probe_refs(
                datasets["late"]["episodes"],
                args.probe_size,
                length,
                group_seed + 2_000_003,
            ),
        }

        branches = {}
        initial_hash_sets = {"online": set(), "target": set(), "actor": set()}
        for branch in args.branches:
            print(
                f"[GROUP={group}] branch={branch} updates={args.updates} "
                f"milestones={milestones}",
                flush=True,
            )
            result = run_branch(
                branch,
                stage2_path,
                step0,
                datasets,
                caches,
                schedule,
                probes,
                length,
                device,
                milestones,
                args.probe_batch_size,
            )
            branches[branch] = result
            initial_hash_sets["online"].add(
                result["initial_hashes"]["online_hash"]
            )
            initial_hash_sets["target"].add(
                result["initial_hashes"]["target_hash"]
            )
            initial_hash_sets["actor"].add(
                result["initial_hashes"]["actor_hash"]
            )

        group_result = {
            "step0_checkpoint": str(step0_path.resolve()),
            "stage2_source_checkpoint": str(stage2_path),
            "stage2_history_semantics": EXPECTED_SEMANTICS,
            "critic_context_length": length,
            "gamma": float(cp_config["gamma"]),
            "tau": float(cp_config["tau"]),
            "critic_lr": float(cp_config["critic_lr"]),
            "critic_weight_decay": float(cp_config["critic_weight_decay"]),
            "critic_max_grad_norm": float(cp_config["critic_max_grad_norm"]),
            "actor_updates": int(step0["actor_updates"]),
            "dataset_manifest": {
                name: value["manifest"] for name, value in datasets.items()
            },
            "probe_counts": {
                name: int(len(refs)) for name, refs in probes.items()
            },
            "branches": branches,
            "comparison": compare_branches(branches),
            "validity": {
                "same_initial_online_hash": len(initial_hash_sets["online"]) == 1,
                "same_initial_target_hash": len(initial_hash_sets["target"]) == 1,
                "same_initial_target_actor_hash": len(initial_hash_sets["actor"]) == 1,
                "all_requested_branches_completed": (
                    set(branches) == set(args.branches)
                    and all(value["completed"] for value in branches.values())
                ),
                "frozen_target_really_frozen": (
                    not branches["frozen_step0_min"]["target_changed"]
                    if "frozen_step0_min" in branches else True
                ),
                "moving_min_target_changed": (
                    branches["production_moving_min"]["target_changed"]
                    if "production_moving_min" in branches else True
                ),
                "moving_mean_target_changed": (
                    branches["moving_mean"]["target_changed"]
                    if "moving_mean" in branches else True
                ),
                "oracle_diagnostic_target_changed": (
                    branches["oracle_mc"]["target_changed"]
                    if "oracle_mc" in branches else True
                ),
            },
        }
        report["groups"][group] = group_result

        report["status"] = "RUNNING"
        write_outputs(output_dir, report)
        del step0, datasets, caches, schedule, probes, branches
        cleanup(device)

    report["validity"] = {
        "all_requested_groups_complete": set(report["groups"]) == set(args.groups),
        "all_group_contracts_pass": all(
            all(value["validity"].values())
            for value in report["groups"].values()
        ),
        "no_environment_or_rollout": True,
        "production_untouched": True,
    }
    report["status"] = (
        "PASS" if all(report["validity"].values()) else "INCONCLUSIVE"
    )
    json_path, csv_path = write_outputs(output_dir, report)
    print(f"[STATUS] {report['status']}", flush=True)
    print(f"[JSON] {json_path}", flush=True)
    print(f"[CSV] {csv_path}", flush=True)


if __name__ == "__main__":
    main()