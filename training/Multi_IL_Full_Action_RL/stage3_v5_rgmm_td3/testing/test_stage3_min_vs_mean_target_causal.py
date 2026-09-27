#!/usr/bin/env python3
"""Paired causal test: production min target vs twin-mean target.

Both branches start from the exact same Stage3-v5 Multi step0 checkpoint,
restore the exact same Critic optimizer state, consume the exact same
precomputed canonical replay sequence batches in the exact same order, keep the
Actor and target Actor frozen, and use the same production Polyak target-Critic
update after every Critic step.

The ONLY intended intervention is the twin aggregation inside the Bellman
bootstrap target:

  min branch (production):
      E_pi[min(Q1_target, Q2_target)]

  mean branch (counterfactual):
      E_pi[(Q1_target + Q2_target) / 2]

This determines whether clipped-double-Q pessimism merely appears as a small
static penalty, or causally seeds the later common bootstrap drift.

Diagnostic only: no environment, rollout, Actor update, or training checkpoint
write occurs.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STAGE3 = HERE.parent
for directory in (HERE, STAGE3):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from stage3_v5_agent import (
    _last_reset_starts_from_numpy,
    target_final_distribution_vectorized,
)
from stage3_v5_history_critic import encode_replay_contexts
from stage3_v5_replay import final_transition
from test_stage2_stage3_readiness_compare import resolve_device, sync
from test_stage3_frozen_target_causal import (
    DEFAULT_STAGE2,
    build_agent,
    load_canonical,
    make_probe_refs,
    make_reference_schedule,
    module_digest,
    probe_reference_returns,
    stack_sequence_batch,
)
from stage3_v5_readiness import auc, correlation


DEFAULT_MILESTONES = (0, 100, 250, 500, 1000, 2000)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-run-dir", required=True)
    parser.add_argument("--stage2-checkpoint", default=DEFAULT_STAGE2)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probe-size", type=int, default=8192)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument(
        "--diagnostic-replay",
        help="Canonical replay; defaults to Multi Stage3 200K sidecar.",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def cleanup_device(device):
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def nested_state_equal(left, right):
    if torch.is_tensor(left) and torch.is_tensor(right):
        return bool(
            left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(
            nested_state_equal(left[key], right[key])
            for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(
            right, (list, tuple)):
        return (
            len(left) == len(right)
            and all(
                nested_state_equal(a, b)
                for a, b in zip(left, right)
            )
        )
    return left == right


@torch.no_grad()
def mean_bellman_target(self, b, target_sequence):
    """Stage3 production target path with ONLY twin min replaced by twin mean."""
    horizon = int(self.config["actor_source_contract"]["rnn_horizon"])
    target_starts = _last_reset_starts_from_numpy(
        target_sequence["episode_steps"], horizon)
    target_keys = ["next_observations"]
    if hasattr(self.target_critic, "encode_history"):
        target_keys += ["observations", "actions", "episode_steps"]
    target = self._tensor_batch({
        key: target_sequence[key] for key in target_keys
    })

    distribution, _ = target_final_distribution_vectorized(
        self.target_actor,
        target["next_observations"],
        horizon=horizon,
        starts=target_starts,
    )

    if not hasattr(self.target_critic, "encode_history"):
        raise RuntimeError(
            "Mean-target causal test requires the audited history-aware Critic")

    target_contexts = encode_replay_contexts(
        self.target_critic,
        target["observations"],
        target["actions"],
        target["episode_steps"],
        self.config["horizon"],
        next_observations=target["next_observations"],
    )
    final_contexts = (
        target_contexts[0][:, -1],
        target_contexts[1][:, -1],
    )

    base = distribution.component_distribution.base_dist
    means = base.loc
    probabilities = distribution.mixture_distribution.probs
    action_dim = int(means.shape[-1])
    broadcast = (1,) * (means.ndim - 1) + (action_dim,)
    component_actions = (
        means * self.action_scale.reshape(broadcast)
        + self.action_offset.reshape(broadcast)
    )

    q1, q2 = self.target_critic.q_from_context(
        final_contexts, component_actions)
    q1 = q1.squeeze(-1)
    q2 = q2.squeeze(-1)
    twin_mean = 0.5 * (q1 + q2)
    expected_next = (probabilities * twin_mean).sum(-1)

    td_target = (
        b["rewards"]
        + float(self.config["gamma"])
        * (1.0 - b["terminals"])
        * expected_next.reshape(-1, 1)
    )
    return (
        td_target,
        distribution,
        expected_next,
        target,
        target_starts,
        target_contexts,
    )


@torch.no_grad()
def evaluate_probe_twin(
    critic,
    episodes,
    refs,
    returns,
    labels,
    episode_ids,
    device,
    horizon,
    length,
    batch_size,
):
    critic.eval()
    q1_parts, q2_parts = [], []
    for first in range(0, len(refs), int(batch_size)):
        selected = refs[first:first + int(batch_size)]
        sequence = stack_sequence_batch(
            episodes, selected, length)
        obs = torch.as_tensor(
            sequence["observations"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(
            sequence["actions"], dtype=torch.float32, device=device)
        steps = torch.as_tensor(
            sequence["episode_steps"], dtype=torch.long, device=device)
        contexts = encode_replay_contexts(
            critic, obs, actions, steps, int(horizon))
        executed = actions[:, -1]
        q1, q2 = critic.q_from_context(
            (contexts[0][:, -1], contexts[1][:, -1]), executed)
        q1_parts.append(q1.reshape(-1).detach().cpu().numpy())
        q2_parts.append(q2.reshape(-1).detach().cpu().numpy())

    q1 = np.concatenate(q1_parts).astype(np.float64)
    q2 = np.concatenate(q2_parts).astype(np.float64)
    qmean = 0.5 * (q1 + q2)
    qmin = np.minimum(q1, q2)
    reference = np.asarray(returns, dtype=np.float64)

    def metrics(values):
        spearman, pearson = correlation(values, reference)
        return {
            "spearman_q_return": float(spearman),
            "pearson_q_return": float(pearson),
            "mae_q_return": float(np.mean(np.abs(values - reference))),
            "signed_bias_q_return": float(np.mean(values - reference)),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    unique = np.unique(episode_ids)
    episode_q = []
    episode_labels = []
    for episode_index in unique:
        mask = episode_ids == episode_index
        episode_q.append(float(qmin[mask].mean()))
        episode_labels.append(
            int(labels[np.flatnonzero(mask)[0]]))
    episode_q = np.asarray(episode_q, dtype=np.float64)
    episode_labels = np.asarray(episode_labels, dtype=np.int64)
    episode_auc = (
        float(auc(episode_labels, episode_q))
        if len(np.unique(episode_labels)) == 2 else None
    )

    return {
        "count": int(len(reference)),
        "q1": metrics(q1),
        "q2": metrics(q2),
        "qmean": metrics(qmean),
        "qmin": metrics(qmin),
        "probe_episode_auc_qmin": episode_auc,
        "twin_abs_gap_mean": float(np.mean(np.abs(q1 - q2))),
        "twin_abs_gap_p95": float(np.percentile(np.abs(q1 - q2), 95)),
        "mean_minus_min_mean": float(np.mean(qmean - qmin)),
    }


def target_contract_check(
    min_agent,
    mean_agent,
    episodes,
    schedule,
    length,
):
    """Verify same state/batch and isolate the intended target aggregation."""
    sequence = stack_sequence_batch(
        episodes, schedule[0], length)
    final = final_transition(sequence)

    b_min = min_agent._tensor_batch(final)
    b_mean = mean_agent._tensor_batch(final)
    min_out = min_agent.bellman_target(b_min, sequence)
    mean_out = mean_agent.bellman_target(b_mean, sequence)

    min_td, min_dist, min_next = min_out[:3]
    mean_td, mean_dist, mean_next = mean_out[:3]

    base_min = min_dist.component_distribution.base_dist
    base_mean = mean_dist.component_distribution.base_dist
    actor_distribution_max_abs = max(
        float((base_min.loc - base_mean.loc).abs().max().item()),
        float((base_min.scale - base_mean.scale).abs().max().item()),
        float((
            min_dist.mixture_distribution.logits
            - mean_dist.mixture_distribution.logits
        ).abs().max().item()),
    )

    reward = torch.as_tensor(
        final["rewards"], dtype=torch.float32, device=min_td.device)
    terminal = torch.as_tensor(
        final["terminals"], dtype=torch.float32, device=min_td.device)
    nonterminal = (terminal.reshape(-1) < 0.5)

    mean_minus_min = mean_next - min_next
    return {
        "same_actor_distribution_max_abs": actor_distribution_max_abs,
        "min_td_shape": list(min_td.shape),
        "mean_td_shape": list(mean_td.shape),
        "min_expected_next_mean": float(min_next.mean().item()),
        "mean_expected_next_mean": float(mean_next.mean().item()),
        "mean_minus_min_expected_next_mean": float(
            mean_minus_min.mean().item()),
        "mean_minus_min_expected_next_abs_mean": float(
            mean_minus_min.abs().mean().item()),
        "mean_minus_min_expected_next_min": float(
            mean_minus_min.min().item()),
        "mean_expected_ge_min_fraction": float(
            (mean_minus_min >= -1e-7).float().mean().item()),
        "terminal_td_equals_reward_min_max_abs": float(
            (min_td.reshape(-1)[~nonterminal]
             - reward.reshape(-1)[~nonterminal]).abs().max().item()
            if bool((~nonterminal).any()) else 0.0),
        "terminal_td_equals_reward_mean_max_abs": float(
            (mean_td.reshape(-1)[~nonterminal]
             - reward.reshape(-1)[~nonterminal]).abs().max().item()
            if bool((~nonterminal).any()) else 0.0),
    }


def branch_run(
    mode,
    stage2_path,
    step0_payload,
    episodes,
    schedule,
    probe_refs,
    probe_returns,
    probe_labels,
    probe_episode_ids,
    device,
    milestones,
    probe_batch_size,
):
    if mode not in ("min", "mean"):
        raise ValueError(mode)

    config = step0_payload["config"]
    length = int(config["recurrent_replay"]["critic_context_length"])
    horizon = int(config["horizon"])
    agent = build_agent(stage2_path, step0_payload, device)

    if mode == "mean":
        agent.bellman_target = types.MethodType(
            mean_bellman_target, agent)

    initial_online_hash = module_digest(agent.critic)
    initial_target_hash = module_digest(agent.target_critic)
    initial_target_actor_hash = module_digest(agent.target_actor)

    trajectory = []

    def record(update):
        sync(device)
        online = evaluate_probe_twin(
            agent.critic,
            episodes,
            probe_refs,
            probe_returns,
            probe_labels,
            probe_episode_ids,
            device,
            horizon,
            length,
            probe_batch_size,
        )
        target = evaluate_probe_twin(
            agent.target_critic,
            episodes,
            probe_refs,
            probe_returns,
            probe_labels,
            probe_episode_ids,
            device,
            horizon,
            length,
            probe_batch_size,
        )
        sync(device)
        trajectory.append({
            "update": int(update),
            "online": online,
            "target": target,
        })
        print(
            f"[{mode}] update={update} "
            f"online_qmin_spear="
            f"{online['qmin']['spearman_q_return']:.6f} "
            f"online_qmean_spear="
            f"{online['qmean']['spearman_q_return']:.6f} "
            f"qmin_bias="
            f"{online['qmin']['signed_bias_q_return']:.6f}",
            flush=True,
        )

    record(0)
    failure = None
    completed_updates = 0
    milestone_set = set(map(int, milestones))

    for index in range(len(schedule)):
        sequence = stack_sequence_batch(
            episodes, schedule[index], length)
        final = final_transition(sequence)
        try:
            agent.critic_update(
                final, sequence, collect_metrics=False)
        except FloatingPointError as exc:
            failure = {
                "failed_update": int(index + 1),
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            print(
                f"[{mode}] NONFINITE at update={index + 1}: {exc}",
                flush=True,
            )
            break

        # Same production target-Critic feedback in both branches.
        agent.polyak_update()
        completed_updates = int(index + 1)
        if completed_updates in milestone_set:
            record(completed_updates)

    result = {
        "mode": mode,
        "updates_requested": int(len(schedule)),
        "updates_completed": int(completed_updates),
        "completed": bool(
            failure is None
            and completed_updates == len(schedule)),
        "failure": failure,
        "initial_online_hash": initial_online_hash,
        "final_online_hash": module_digest(agent.critic),
        "initial_target_critic_hash": initial_target_hash,
        "final_target_critic_hash": module_digest(
            agent.target_critic),
        "initial_target_actor_hash": initial_target_actor_hash,
        "final_target_actor_hash": module_digest(
            agent.target_actor),
        "target_critic_changed": bool(
            module_digest(agent.target_critic)
            != initial_target_hash),
        "target_actor_changed": bool(
            module_digest(agent.target_actor)
            != initial_target_actor_hash),
        "trajectory": trajectory,
    }

    del agent
    cleanup_device(device)
    return result


def main():
    args = arguments()
    if args.updates != 2000:
        raise ValueError(
            "Keep --updates 2000 for this paired causal test")
    if args.batch_size != 256:
        raise ValueError(
            "Keep --batch-size 256 to match production Critic updates")
    if args.probe_size < 100:
        raise ValueError("--probe-size must be >= 100")
    if args.probe_batch_size < 1:
        raise ValueError("--probe-batch-size must be >= 1")

    run_dir = Path(args.stage3_run_dir).resolve()
    stage2_path = Path(args.stage2_checkpoint).resolve()
    step0_path = (
        run_dir / "multi_q" / "checkpoints"
        / "step0_transfer.pth"
    )
    diagnostic_replay = (
        Path(args.diagnostic_replay).resolve()
        if args.diagnostic_replay
        else run_dir / "multi_q" / "checkpoints"
        / "step_0200000.sequences.npy"
    )

    for path in (stage2_path, step0_path, diagnostic_replay):
        if not path.exists():
            raise FileNotFoundError(path)

    step0_payload = torch.load(
        step0_path, map_location="cpu")
    if step0_payload.get("stage") != "stage3-v5":
        raise RuntimeError("step0 checkpoint is not Stage3-v5")
    if step0_payload.get("group") != "multi_q":
        raise RuntimeError("step0 checkpoint is not multi_q")
    if int(step0_payload.get("env_steps", -1)) != 0:
        raise RuntimeError("Reference checkpoint is not env-step zero")
    if int(step0_payload.get("updates", -1)) != 0:
        raise RuntimeError(
            "Reference checkpoint already contains Critic updates")
    if int(step0_payload.get("actor_updates", -1)) != 0:
        raise RuntimeError(
            "Reference checkpoint already contains Actor updates")

    config = step0_payload["config"]
    length = int(
        config["recurrent_replay"]["critic_context_length"])
    if length != 10:
        raise RuntimeError("Expected audited context length 10")
    if float(config["tau"]) != 0.005:
        raise RuntimeError(
            "Expected production tau=0.005")

    fixed, episodes = load_canonical(diagnostic_replay)
    schedule = make_reference_schedule(
        episodes,
        args.updates,
        args.batch_size,
        length,
        args.seed,
    )
    probe_refs = make_probe_refs(
        episodes,
        args.probe_size,
        length,
        args.seed + 1000003,
    )
    probe_returns, probe_labels, probe_episode_ids = (
        probe_reference_returns(
            episodes,
            probe_refs,
            length,
            float(config["gamma"]),
        )
    )

    device = resolve_device(args.device)
    milestones = DEFAULT_MILESTONES

    # Contract check on two freshly reconstructed agents before either branch
    # is allowed to train.
    min_contract_agent = build_agent(
        stage2_path, step0_payload, device)
    mean_contract_agent = build_agent(
        stage2_path, step0_payload, device)
    mean_contract_agent.bellman_target = types.MethodType(
        mean_bellman_target, mean_contract_agent)

    initial_contract = {
        "same_initial_online_hash": bool(
            module_digest(min_contract_agent.critic)
            == module_digest(mean_contract_agent.critic)),
        "same_initial_target_critic_hash": bool(
            module_digest(min_contract_agent.target_critic)
            == module_digest(mean_contract_agent.target_critic)),
        "same_initial_actor_hash": bool(
            module_digest(min_contract_agent.actor)
            == module_digest(mean_contract_agent.actor)),
        "same_initial_target_actor_hash": bool(
            module_digest(min_contract_agent.target_actor)
            == module_digest(mean_contract_agent.target_actor)),
        "same_initial_optimizer_state": nested_state_equal(
            min_contract_agent.critic_optimizer.state_dict(),
            mean_contract_agent.critic_optimizer.state_dict(),
        ),
        "target_semantics": target_contract_check(
            min_contract_agent,
            mean_contract_agent,
            episodes,
            schedule,
            length,
        ),
    }
    del min_contract_agent, mean_contract_agent
    cleanup_device(device)

    print(
        f"[SETUP] episodes={len(episodes)} "
        f"updates={args.updates} batch={args.batch_size} "
        f"probe={len(probe_refs)} device={device}",
        flush=True,
    )

    min_branch = branch_run(
        "min",
        stage2_path,
        step0_payload,
        episodes,
        schedule,
        probe_refs,
        probe_returns,
        probe_labels,
        probe_episode_ids,
        device,
        milestones,
        args.probe_batch_size,
    )
    mean_branch = branch_run(
        "mean",
        stage2_path,
        step0_payload,
        episodes,
        schedule,
        probe_refs,
        probe_returns,
        probe_labels,
        probe_episode_ids,
        device,
        milestones,
        args.probe_batch_size,
    )

    min_by_update = {
        int(row["update"]): row
        for row in min_branch["trajectory"]
    }
    mean_by_update = {
        int(row["update"]): row
        for row in mean_branch["trajectory"]
    }
    common_updates = sorted(
        set(min_by_update) & set(mean_by_update))
    if not common_updates:
        raise RuntimeError(
            "Branches share no recorded milestone")

    comparison = []
    for update in common_updates:
        a = min_by_update[update]
        b = mean_by_update[update]
        comparison.append({
            "update": int(update),
            "min_online_qmin_spearman": float(
                a["online"]["qmin"]["spearman_q_return"]),
            "mean_online_qmin_spearman": float(
                b["online"]["qmin"]["spearman_q_return"]),
            "mean_minus_min_qmin_spearman": float(
                b["online"]["qmin"]["spearman_q_return"]
                - a["online"]["qmin"]["spearman_q_return"]),
            "min_online_qmean_spearman": float(
                a["online"]["qmean"]["spearman_q_return"]),
            "mean_online_qmean_spearman": float(
                b["online"]["qmean"]["spearman_q_return"]),
            "min_online_qmin_mae": float(
                a["online"]["qmin"]["mae_q_return"]),
            "mean_online_qmin_mae": float(
                b["online"]["qmin"]["mae_q_return"]),
            "min_online_qmin_bias": float(
                a["online"]["qmin"]["signed_bias_q_return"]),
            "mean_online_qmin_bias": float(
                b["online"]["qmin"]["signed_bias_q_return"]),
            "min_target_qmin_spearman": float(
                a["target"]["qmin"]["spearman_q_return"]),
            "mean_target_qmin_spearman": float(
                b["target"]["qmin"]["spearman_q_return"]),
        })

    validity = {
        "same_initial_online_hash": bool(
            min_branch["initial_online_hash"]
            == mean_branch["initial_online_hash"]),
        "same_initial_target_critic_hash": bool(
            min_branch["initial_target_critic_hash"]
            == mean_branch["initial_target_critic_hash"]),
        "same_initial_target_actor_hash": bool(
            min_branch["initial_target_actor_hash"]
            == mean_branch["initial_target_actor_hash"]),
        "same_initial_probe_qmin_spearman": bool(
            abs(
                min_branch["trajectory"][0]["online"]["qmin"][
                    "spearman_q_return"]
                - mean_branch["trajectory"][0]["online"]["qmin"][
                    "spearman_q_return"]
            ) <= 1e-12),
        "target_critic_changed_both": bool(
            min_branch["target_critic_changed"]
            and mean_branch["target_critic_changed"]),
        "target_actor_unchanged_both": bool(
            not min_branch["target_actor_changed"]
            and not mean_branch["target_actor_changed"]),
        "contract_same_actor_distribution": bool(
            initial_contract["target_semantics"][
                "same_actor_distribution_max_abs"] <= 1e-7),
        "contract_terminal_semantics_match": bool(
            initial_contract["target_semantics"][
                "terminal_td_equals_reward_min_max_abs"] <= 1e-7
            and initial_contract["target_semantics"][
                "terminal_td_equals_reward_mean_max_abs"] <= 1e-7),
        "contract_mean_expected_not_below_min": bool(
            initial_contract["target_semantics"][
                "mean_minus_min_expected_next_min"] >= -1e-7
            and initial_contract["target_semantics"][
                "mean_expected_ge_min_fraction"] >= 1.0),
        "same_initial_optimizer_state": bool(
            initial_contract["same_initial_optimizer_state"]),
        "initial_contract_hashes_match": bool(
            initial_contract["same_initial_online_hash"]
            and initial_contract["same_initial_target_critic_hash"]
            and initial_contract["same_initial_actor_hash"]
            and initial_contract["same_initial_target_actor_hash"]),
    }
    valid = bool(all(validity.values()))

    last_common = int(common_updates[-1])
    first = comparison[0]
    last = comparison[-1]
    initial_qmin_spearman = float(
        first["min_online_qmin_spearman"])

    summary = {
        "last_common_update": last_common,
        "initial_online_qmin_spearman": initial_qmin_spearman,
        "final_min_online_qmin_spearman": float(
            last["min_online_qmin_spearman"]),
        "final_mean_online_qmin_spearman": float(
            last["mean_online_qmin_spearman"]),
        "final_mean_minus_min_qmin_spearman": float(
            last["mean_minus_min_qmin_spearman"]),
        "min_change_from_initial": float(
            last["min_online_qmin_spearman"]
            - initial_qmin_spearman),
        "mean_change_from_initial": float(
            last["mean_online_qmin_spearman"]
            - initial_qmin_spearman),
        "final_min_qmin_mae": float(
            last["min_online_qmin_mae"]),
        "final_mean_qmin_mae": float(
            last["mean_online_qmin_mae"]),
        "final_min_qmin_bias": float(
            last["min_online_qmin_bias"]),
        "final_mean_qmin_bias": float(
            last["mean_online_qmin_bias"]),
    }

    output = {
        "status": (
            "PASS"
            if valid
            and min_branch["completed"]
            and mean_branch["completed"]
            else "PARTIAL"
            if valid
            else "INVALID"
        ),
        "experiment": "stage3_v5_min_vs_mean_target_causal",
        "read_only_environment": True,
        "environment_steps_performed": 0,
        "actor_updates_performed": 0,
        "training_checkpoints_written": 0,
        "optimizer_steps_requested_per_branch": int(
            args.updates),
        "min_optimizer_steps_completed": int(
            min_branch["updates_completed"]),
        "mean_optimizer_steps_completed": int(
            mean_branch["updates_completed"]),
        "device": str(device),
        "stage2_architecture_checkpoint": str(stage2_path),
        "stage3_step0_checkpoint": str(step0_path),
        "canonical_diagnostic_replay": str(
            diagnostic_replay),
        "canonical_seed": fixed.get("seed"),
        "canonical_episode_count": int(len(episodes)),
        "sequence_length": int(length),
        "batch_size": int(args.batch_size),
        "updates": int(args.updates),
        "schedule_seed": int(args.seed),
        "probe_size": int(len(probe_refs)),
        "probe_seed": int(args.seed + 1000003),
        "milestones": list(map(int, milestones)),
        "gamma": float(config["gamma"]),
        "tau": float(config["tau"]),
        "critic_lr": float(config["critic_lr"]),
        "critic_weight_decay": float(
            config["critic_weight_decay"]),
        "same_precomputed_batch_schedule": True,
        "isolation_scope": (
            "fixed canonical online replay only; paired causal "
            "test of twin aggregation in the Bellman target"
        ),
        "intervention": {
            "min": (
                "exact production critic_update target using "
                "E[min(Q1_target,Q2_target)], followed by "
                "production polyak_update"
            ),
            "mean": (
                "same critic_update and production polyak_update, "
                "with only target twin aggregation changed to "
                "E[(Q1_target+Q2_target)/2]"
            ),
            "actor": "frozen and identical in both branches",
            "target_actor": (
                "frozen and identical in both branches"),
            "only_intended_difference": (
                "min versus arithmetic mean across target twin "
                "Critics inside Bellman expected-next-Q"
            ),
        },
        "initial_contract": initial_contract,
        "validity": validity,
        "min_branch": min_branch,
        "mean_branch": mean_branch,
        "comparison": comparison,
        "comparison_scope": {
            "common_milestones": list(
                map(int, common_updates)),
            "last_common_update": last_common,
            "min_failure": min_branch["failure"],
            "mean_failure": mean_branch["failure"],
        },
        "summary": summary,
    }

    out_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "testing"
        / "stage2_vs_stage3_readiness"
        / "multi_min_vs_mean_target_causal.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n")

    print(
        "\nupdate\tmin_qmin_spear\tmean_qmin_spear\t"
        "mean-min\tmin_qmean_spear\tmean_qmean_spear\t"
        "min_bias\tmean_bias"
    )
    for row in comparison:
        print(
            f"{row['update']}\t"
            f"{row['min_online_qmin_spearman']:.6f}\t"
            f"{row['mean_online_qmin_spearman']:.6f}\t"
            f"{row['mean_minus_min_qmin_spearman']:.6f}\t"
            f"{row['min_online_qmean_spearman']:.6f}\t"
            f"{row['mean_online_qmean_spearman']:.6f}\t"
            f"{row['min_online_qmin_bias']:.6f}\t"
            f"{row['mean_online_qmin_bias']:.6f}"
        )

    print("\n[VALIDITY]")
    print(json.dumps(validity, indent=2, sort_keys=True))
    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[SAVED] {out_path}", flush=True)

    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
