#!/usr/bin/env python3
"""Paired causal test: moving learned-Q bootstrap vs exact MC target.

Both branches start from the exact same Stage3-v5 Multi step0 checkpoint,
restore the exact same Critic optimizer state, consume the exact same
precomputed canonical replay sequence batches in the exact same order, keep
Actor learning disabled, and perform the same production Polyak target-Critic
update after every Critic optimizer step.

The ONLY intended difference is the Critic regression target.

Branch A -- moving_qmean_replay_bootstrap
    y_t = r_t + gamma * (1-terminal_t)
          * mean(Q1_target,Q2_target)(h_(t+1), a_(t+1)^replay)

This deliberately removes target-Actor continuation and clipped-min
pessimism, both already diagnosed separately, while preserving the central
moving learned-Q bootstrap feedback loop.

Branch B -- oracle_mc
    y_t = G_t
        = r_t + gamma * (1-terminal_t) * G_(t+1)

Thus the oracle branch keeps the same optimizer, model, replay schedule,
number of gradient steps, and Polyak dynamics, but replaces learned-Q
bootstrap supervision with the exact finite-episode MC target used by
Stage2.2.

Interpretation:
  * If bootstrap drifts while oracle MC stays near the initial MC geometry,
    learned-Q bootstrap feedback is the causal source.
  * If oracle MC also drifts strongly, repeated optimization / finite replay
    projection / function approximation remain important independent causes.

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

from stage3_v5_history_critic import encode_replay_contexts  # noqa: E402
from stage3_v5_readiness import discounted_returns  # noqa: E402
from stage3_v5_replay import final_transition  # noqa: E402
from test_stage2_stage3_readiness_compare import resolve_device, sync  # noqa: E402
from test_stage3_frozen_target_causal import (  # noqa: E402
    DEFAULT_STAGE2,
    build_agent,
    load_canonical,
    make_probe_refs,
    make_reference_schedule,
    module_digest,
    probe_reference_returns,
    stack_sequence_batch,
)
from test_stage3_min_vs_mean_target_causal import (  # noqa: E402
    evaluate_probe_twin,
    nested_state_equal,
)
from test_stage3_td_target_decomposition import episode_terminals  # noqa: E402


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


def mc_return_cache(episodes, gamma):
    return {
        int(index): discounted_returns(episode, float(gamma))
        for index, episode in enumerate(episodes)
    }


def augment_training_sequence(
    episodes,
    refs,
    length,
    returns_by_episode,
):
    """Attach exact G_t, G_(t+1), and replay a_(t+1) to one fixed batch."""
    sequence = stack_sequence_batch(episodes, refs, length)
    refs = np.asarray(refs, dtype=np.int64)

    batch = len(refs)
    next_actions = np.zeros((batch, 14), dtype=np.float32)
    mc_current = np.empty(batch, dtype=np.float32)
    mc_next = np.zeros(batch, dtype=np.float32)

    for row, (episode_index, start) in enumerate(refs):
        episode_index = int(episode_index)
        start = int(start)
        episode = episodes[episode_index]
        target = start + int(length) - 1

        actions = np.asarray(episode["actions"], dtype=np.float32)
        terminals = episode_terminals(episode)
        returns = returns_by_episode[episode_index]

        if target >= len(actions):
            raise RuntimeError("Scheduled target exceeds episode length")

        mc_current[row] = returns[target]

        if terminals[target] < 0.5:
            if target + 1 >= len(actions):
                raise RuntimeError(
                    "Non-terminal scheduled transition has no replay successor")
            next_actions[row] = actions[target + 1]
            mc_next[row] = returns[target + 1]

    sequence["diagnostic_replay_next_actions"] = next_actions
    sequence["diagnostic_mc_current"] = mc_current
    sequence["diagnostic_mc_next"] = mc_next
    return sequence


@torch.no_grad()
def replay_qmean_bellman_target(self, b, target_sequence):
    """Moving target-Critic bootstrap using replay next action and twin mean."""
    target = self._tensor_batch({
        key: target_sequence[key]
        for key in (
            "observations",
            "actions",
            "episode_steps",
            "next_observations",
            "diagnostic_replay_next_actions",
        )
    })

    successor_context = encode_replay_contexts(
        self.target_critic,
        target["observations"],
        target["actions"],
        target["episode_steps"],
        self.config["horizon"],
        next_observations=target["next_observations"],
    )
    final_context = (
        successor_context[0][:, -1],
        successor_context[1][:, -1],
    )
    q1, q2 = self.target_critic.q_from_context(
        final_context,
        target["diagnostic_replay_next_actions"],
    )
    q1 = q1.reshape(-1)
    q2 = q2.reshape(-1)
    expected_next = 0.5 * (q1 + q2)

    td_target = (
        b["rewards"]
        + float(self.config["gamma"])
        * (1.0 - b["terminals"])
        * expected_next.reshape(-1, 1)
    )

    # critic_update(..., collect_metrics=False) returns immediately after the
    # optimizer step, so only td_target is consumed downstream. Preserve the
    # production return arity to keep the update path identical.
    return (
        td_target,
        None,
        expected_next,
        target,
        None,
        successor_context,
    )


@torch.no_grad()
def oracle_mc_bellman_target(self, b, target_sequence):
    """Exact finite-episode MC target used by Stage2.2: y_t = G_t."""
    oracle = torch.as_tensor(
        target_sequence["diagnostic_mc_current"],
        dtype=torch.float32,
        device=self.device,
    ).reshape(-1, 1)

    return (
        oracle,
        None,
        oracle.reshape(-1),
        None,
        None,
        None,
    )


def canonical_mc_identity(episodes, gamma):
    residuals = []
    count = 0
    for episode in episodes:
        rewards = np.asarray(
            episode["rewards"], dtype=np.float64).reshape(-1)
        terminals = episode_terminals(episode).astype(np.float64)
        returns = np.asarray(
            discounted_returns(episode, float(gamma)),
            dtype=np.float64,
        ).reshape(-1)

        if not (
            len(rewards) == len(terminals) == len(returns)
        ):
            raise RuntimeError("Canonical episode length mismatch")

        for index in range(len(rewards)):
            if terminals[index] >= 0.5:
                identity = rewards[index]
            else:
                if index + 1 >= len(returns):
                    raise RuntimeError(
                        "Non-terminal transition lacks MC successor")
                identity = (
                    rewards[index]
                    + float(gamma) * returns[index + 1]
                )
            residuals.append(returns[index] - identity)
            count += 1

    residuals = np.asarray(residuals, dtype=np.float64)
    return {
        "count": int(count),
        "signed_mean": float(residuals.mean()),
        "abs_mean": float(np.abs(residuals).mean()),
        "abs_max": float(np.abs(residuals).max()),
        "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
    }


@torch.no_grad()
def target_contract_check(
    bootstrap_agent,
    oracle_agent,
    episodes,
    refs,
    length,
    returns_by_episode,
):
    """Check the first paired batch before either branch performs an update."""
    sequence = augment_training_sequence(
        episodes, refs, length, returns_by_episode)
    final = final_transition(sequence)

    b_bootstrap = bootstrap_agent._tensor_batch(final)
    b_oracle = oracle_agent._tensor_batch(final)

    bootstrap_out = bootstrap_agent.bellman_target(
        b_bootstrap, sequence)
    oracle_out = oracle_agent.bellman_target(
        b_oracle, sequence)

    bootstrap_td = bootstrap_out[0].reshape(-1)
    oracle_td = oracle_out[0].reshape(-1)

    mc_current = torch.as_tensor(
        sequence["diagnostic_mc_current"],
        dtype=torch.float32,
        device=oracle_td.device,
    ).reshape(-1)
    mc_next = torch.as_tensor(
        sequence["diagnostic_mc_next"],
        dtype=torch.float32,
        device=oracle_td.device,
    ).reshape(-1)
    reward = torch.as_tensor(
        final["rewards"],
        dtype=torch.float32,
        device=oracle_td.device,
    ).reshape(-1)
    terminal = torch.as_tensor(
        final["terminals"],
        dtype=torch.float32,
        device=oracle_td.device,
    ).reshape(-1)

    mc_identity = (
        reward
        + float(bootstrap_agent.config["gamma"])
        * (1.0 - terminal)
        * mc_next
    )

    gap = bootstrap_td - oracle_td
    terminal_mask = terminal >= 0.5

    return {
        "oracle_equals_mc_current_max_abs": float(
            (oracle_td - mc_current).abs().max().item()),
        "mc_identity_max_abs": float(
            (mc_current - mc_identity).abs().max().item()),
        "bootstrap_minus_oracle_signed_mean": float(
            gap.mean().item()),
        "bootstrap_minus_oracle_abs_mean": float(
            gap.abs().mean().item()),
        "bootstrap_minus_oracle_abs_p95": float(
            torch.quantile(gap.abs(), 0.95).item()),
        "terminal_bootstrap_equals_reward_max_abs": float(
            (bootstrap_td[terminal_mask] - reward[terminal_mask])
            .abs().max().item()
            if bool(terminal_mask.any()) else 0.0),
        "terminal_oracle_equals_reward_max_abs": float(
            (oracle_td[terminal_mask] - reward[terminal_mask])
            .abs().max().item()
            if bool(terminal_mask.any()) else 0.0),
    }


def branch_run(
    mode,
    stage2_path,
    step0_payload,
    episodes,
    schedule,
    returns_by_episode,
    probe_refs,
    probe_returns,
    probe_labels,
    probe_episode_ids,
    device,
    milestones,
    probe_batch_size,
):
    if mode not in ("moving_qmean_replay_bootstrap", "oracle_mc"):
        raise ValueError(mode)

    config = step0_payload["config"]
    length = int(
        config["recurrent_replay"]["critic_context_length"])
    horizon = int(config["horizon"])
    agent = build_agent(stage2_path, step0_payload, device)

    if mode == "moving_qmean_replay_bootstrap":
        agent.bellman_target = types.MethodType(
            replay_qmean_bellman_target, agent)
    else:
        agent.bellman_target = types.MethodType(
            oracle_mc_bellman_target, agent)

    initial_online_hash = module_digest(agent.critic)
    initial_target_hash = module_digest(agent.target_critic)
    initial_actor_hash = module_digest(agent.actor)
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
            f"qmin_spear="
            f"{online['qmin']['spearman_q_return']:.6f} "
            f"qmean_spear="
            f"{online['qmean']['spearman_q_return']:.6f} "
            f"qmin_mae={online['qmin']['mae_q_return']:.6f} "
            f"qmin_bias="
            f"{online['qmin']['signed_bias_q_return']:.6f}",
            flush=True,
        )

    record(0)

    milestone_set = set(map(int, milestones))
    failure = None
    completed_updates = 0

    for index in range(len(schedule)):
        sequence = augment_training_sequence(
            episodes,
            schedule[index],
            length,
            returns_by_episode,
        )
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

        # Deliberately identical in both branches. The oracle branch does not
        # use target_critic to form its target, but still updates it so the
        # optimization/update schedule differs only in supervision source.
        agent.polyak_update()

        completed_updates = int(index + 1)
        if completed_updates in milestone_set:
            record(completed_updates)

    final_online_hash = module_digest(agent.critic)
    final_target_hash = module_digest(agent.target_critic)
    final_actor_hash = module_digest(agent.actor)
    final_target_actor_hash = module_digest(agent.target_actor)

    result = {
        "mode": mode,
        "updates_requested": int(len(schedule)),
        "updates_completed": int(completed_updates),
        "completed": bool(
            failure is None
            and completed_updates == len(schedule)),
        "failure": failure,
        "initial_online_hash": initial_online_hash,
        "final_online_hash": final_online_hash,
        "initial_target_critic_hash": initial_target_hash,
        "final_target_critic_hash": final_target_hash,
        "target_critic_changed": bool(
            final_target_hash != initial_target_hash),
        "initial_actor_hash": initial_actor_hash,
        "final_actor_hash": final_actor_hash,
        "actor_changed": bool(
            final_actor_hash != initial_actor_hash),
        "initial_target_actor_hash": initial_target_actor_hash,
        "final_target_actor_hash": final_target_actor_hash,
        "target_actor_changed": bool(
            final_target_actor_hash != initial_target_actor_hash),
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

    stage2_payload = torch.load(
        stage2_path, map_location="cpu")
    if stage2_payload.get("stage_version") != "2.2":
        raise RuntimeError("Expected Stage2.2 checkpoint")
    if stage2_payload.get("training_target") != (
        "finite_episode_monte_carlo_return_no_bootstrap"
    ):
        raise RuntimeError(
            "Stage2.2 training target contract changed unexpectedly")

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
    gamma = float(config["gamma"])
    length = int(
        config["recurrent_replay"]["critic_context_length"])
    if length != 10:
        raise RuntimeError("Expected audited context length 10")
    if abs(float(stage2_payload.get("gamma", -1.0)) - gamma) > 1e-12:
        raise RuntimeError("Stage2.2 / Stage3 gamma mismatch")
    if float(config["tau"]) != 0.005:
        raise RuntimeError("Expected production tau=0.005")

    fixed, episodes = load_canonical(diagnostic_replay)
    schedule = make_reference_schedule(
        episodes,
        args.updates,
        args.batch_size,
        length,
        args.seed,
    )
    returns_by_episode = mc_return_cache(
        episodes, gamma)

    probe_refs = make_probe_refs(
        episodes,
        args.probe_size,
        length,
        args.seed + 1000003,
    )
    (
        probe_returns,
        probe_labels,
        probe_episode_ids,
    ) = probe_reference_returns(
        episodes,
        probe_refs,
        length,
        gamma,
    )

    mc_identity = canonical_mc_identity(
        episodes, gamma)
    device = resolve_device(args.device)
    milestones = DEFAULT_MILESTONES

    # Rebuild two pristine agents solely to prove paired initial conditions and
    # target semantics before either branch is allowed to optimize.
    bootstrap_contract_agent = build_agent(
        stage2_path, step0_payload, device)
    oracle_contract_agent = build_agent(
        stage2_path, step0_payload, device)

    bootstrap_contract_agent.bellman_target = types.MethodType(
        replay_qmean_bellman_target,
        bootstrap_contract_agent,
    )
    oracle_contract_agent.bellman_target = types.MethodType(
        oracle_mc_bellman_target,
        oracle_contract_agent,
    )

    initial_contract = {
        "same_initial_online_hash": bool(
            module_digest(bootstrap_contract_agent.critic)
            == module_digest(oracle_contract_agent.critic)),
        "same_initial_target_critic_hash": bool(
            module_digest(bootstrap_contract_agent.target_critic)
            == module_digest(oracle_contract_agent.target_critic)),
        "same_initial_actor_hash": bool(
            module_digest(bootstrap_contract_agent.actor)
            == module_digest(oracle_contract_agent.actor)),
        "same_initial_target_actor_hash": bool(
            module_digest(bootstrap_contract_agent.target_actor)
            == module_digest(oracle_contract_agent.target_actor)),
        "same_initial_optimizer_state": nested_state_equal(
            bootstrap_contract_agent.critic_optimizer.state_dict(),
            oracle_contract_agent.critic_optimizer.state_dict(),
        ),
        "target_semantics": target_contract_check(
            bootstrap_contract_agent,
            oracle_contract_agent,
            episodes,
            schedule[0],
            length,
            returns_by_episode,
        ),
    }

    del bootstrap_contract_agent, oracle_contract_agent
    cleanup_device(device)

    print(
        f"[SETUP] episodes={len(episodes)} "
        f"updates={args.updates} batch={args.batch_size} "
        f"probe={len(probe_refs)} device={device}",
        flush=True,
    )

    bootstrap = branch_run(
        "moving_qmean_replay_bootstrap",
        stage2_path,
        step0_payload,
        episodes,
        schedule,
        returns_by_episode,
        probe_refs,
        probe_returns,
        probe_labels,
        probe_episode_ids,
        device,
        milestones,
        args.probe_batch_size,
    )

    oracle = branch_run(
        "oracle_mc",
        stage2_path,
        step0_payload,
        episodes,
        schedule,
        returns_by_episode,
        probe_refs,
        probe_returns,
        probe_labels,
        probe_episode_ids,
        device,
        milestones,
        args.probe_batch_size,
    )

    bootstrap_by_update = {
        int(row["update"]): row
        for row in bootstrap["trajectory"]
    }
    oracle_by_update = {
        int(row["update"]): row
        for row in oracle["trajectory"]
    }
    common_updates = sorted(
        set(bootstrap_by_update) & set(oracle_by_update))
    if not common_updates:
        raise RuntimeError(
            "Branches share no recorded milestone")

    comparison = []
    for update in common_updates:
        left = bootstrap_by_update[update]
        right = oracle_by_update[update]
        comparison.append({
            "update": int(update),
            "bootstrap_online_qmin_spearman": float(
                left["online"]["qmin"]["spearman_q_return"]),
            "oracle_online_qmin_spearman": float(
                right["online"]["qmin"]["spearman_q_return"]),
            "oracle_minus_bootstrap_qmin_spearman": float(
                right["online"]["qmin"]["spearman_q_return"]
                - left["online"]["qmin"]["spearman_q_return"]),
            "bootstrap_online_qmean_spearman": float(
                left["online"]["qmean"]["spearman_q_return"]),
            "oracle_online_qmean_spearman": float(
                right["online"]["qmean"]["spearman_q_return"]),
            "bootstrap_online_qmin_mae": float(
                left["online"]["qmin"]["mae_q_return"]),
            "oracle_online_qmin_mae": float(
                right["online"]["qmin"]["mae_q_return"]),
            "bootstrap_online_qmin_bias": float(
                left["online"]["qmin"]["signed_bias_q_return"]),
            "oracle_online_qmin_bias": float(
                right["online"]["qmin"]["signed_bias_q_return"]),
            "bootstrap_target_qmin_spearman": float(
                left["target"]["qmin"]["spearman_q_return"]),
            "oracle_target_qmin_spearman": float(
                right["target"]["qmin"]["spearman_q_return"]),
        })

    validity = {
        "same_initial_online_hash": bool(
            bootstrap["initial_online_hash"]
            == oracle["initial_online_hash"]),
        "same_initial_target_critic_hash": bool(
            bootstrap["initial_target_critic_hash"]
            == oracle["initial_target_critic_hash"]),
        "same_initial_actor_hash": bool(
            bootstrap["initial_actor_hash"]
            == oracle["initial_actor_hash"]),
        "same_initial_target_actor_hash": bool(
            bootstrap["initial_target_actor_hash"]
            == oracle["initial_target_actor_hash"]),
        "same_initial_probe_qmin_spearman": bool(
            abs(
                bootstrap["trajectory"][0]["online"]["qmin"][
                    "spearman_q_return"]
                - oracle["trajectory"][0]["online"]["qmin"][
                    "spearman_q_return"]
            ) <= 1e-12),
        "target_critic_changed_both": bool(
            bootstrap["target_critic_changed"]
            and oracle["target_critic_changed"]),
        "actor_unchanged_both": bool(
            not bootstrap["actor_changed"]
            and not oracle["actor_changed"]),
        "target_actor_unchanged_both": bool(
            not bootstrap["target_actor_changed"]
            and not oracle["target_actor_changed"]),
        "same_initial_optimizer_state": bool(
            initial_contract["same_initial_optimizer_state"]),
        "initial_contract_hashes_match": bool(
            initial_contract["same_initial_online_hash"]
            and initial_contract["same_initial_target_critic_hash"]
            and initial_contract["same_initial_actor_hash"]
            and initial_contract["same_initial_target_actor_hash"]),
        "canonical_mc_identity_max_abs_le_2e_6": bool(
            mc_identity["abs_max"] <= 2e-6),
        "oracle_target_equals_mc_max_abs_le_2e_6": bool(
            initial_contract["target_semantics"][
                "oracle_equals_mc_current_max_abs"] <= 2e-6),
        "first_batch_mc_identity_max_abs_le_2e_6": bool(
            initial_contract["target_semantics"][
                "mc_identity_max_abs"] <= 2e-6),
        "terminal_semantics_match": bool(
            initial_contract["target_semantics"][
                "terminal_bootstrap_equals_reward_max_abs"] <= 2e-6
            and initial_contract["target_semantics"][
                "terminal_oracle_equals_reward_max_abs"] <= 2e-6),
    }
    valid = bool(all(validity.values()))

    last_common = int(common_updates[-1])
    initial_spearman = float(
        comparison[0]["bootstrap_online_qmin_spearman"])
    last = comparison[-1]

    summary = {
        "last_common_update": last_common,
        "initial_online_qmin_spearman": initial_spearman,
        "final_bootstrap_online_qmin_spearman": float(
            last["bootstrap_online_qmin_spearman"]),
        "final_oracle_online_qmin_spearman": float(
            last["oracle_online_qmin_spearman"]),
        "final_oracle_minus_bootstrap_qmin_spearman": float(
            last["oracle_minus_bootstrap_qmin_spearman"]),
        "bootstrap_change_from_initial": float(
            last["bootstrap_online_qmin_spearman"]
            - initial_spearman),
        "oracle_change_from_initial": float(
            last["oracle_online_qmin_spearman"]
            - initial_spearman),
        "final_bootstrap_qmin_mae": float(
            last["bootstrap_online_qmin_mae"]),
        "final_oracle_qmin_mae": float(
            last["oracle_online_qmin_mae"]),
        "final_bootstrap_qmin_bias": float(
            last["bootstrap_online_qmin_bias"]),
        "final_oracle_qmin_bias": float(
            last["oracle_online_qmin_bias"]),
    }

    output = {
        "status": (
            "PASS"
            if valid
            and bootstrap["completed"]
            and oracle["completed"]
            else "PARTIAL"
            if valid
            else "INVALID"
        ),
        "experiment": (
            "stage3_v5_moving_qmean_replay_bootstrap_vs_oracle_mc"
        ),
        "read_only_environment": True,
        "environment_steps_performed": 0,
        "actor_updates_performed": 0,
        "training_checkpoints_written": 0,
        "optimizer_steps_requested_per_branch": int(
            args.updates),
        "bootstrap_optimizer_steps_completed": int(
            bootstrap["updates_completed"]),
        "oracle_optimizer_steps_completed": int(
            oracle["updates_completed"]),
        "device": str(device),
        "stage2_checkpoint": str(stage2_path),
        "stage2_training_target": stage2_payload.get(
            "training_target"),
        "stage3_step0_checkpoint": str(step0_path),
        "canonical_diagnostic_replay": str(
            diagnostic_replay),
        "canonical_seed": fixed.get("seed"),
        "canonical_episode_count": int(len(episodes)),
        "sequence_length": int(length),
        "batch_size": int(args.batch_size),
        "updates": int(args.updates),
        "schedule_seed": int(args.seed),
        "same_precomputed_batch_schedule": True,
        "probe_size": int(len(probe_refs)),
        "probe_seed": int(args.seed + 1000003),
        "milestones": list(map(int, milestones)),
        "gamma": gamma,
        "tau": float(config["tau"]),
        "critic_lr": float(config["critic_lr"]),
        "critic_weight_decay": float(
            config["critic_weight_decay"]),
        "canonical_mc_identity": mc_identity,
        "isolation_scope": (
            "fixed canonical online replay only; target Actor continuation "
            "and clipped twin-min are deliberately removed from the bootstrap "
            "branch so the intervention isolates moving learned-Q feedback "
            "versus exact finite-episode MC supervision"
        ),
        "intervention": {
            "moving_qmean_replay_bootstrap": (
                "y=r+gamma*(1-terminal)*mean(Q1_target,Q2_target)"
                "(successor_history,replay_next_action), followed by "
                "production polyak_update"
            ),
            "oracle_mc": (
                "y=G_t exact finite-episode Monte-Carlo return, with the "
                "same critic optimizer step and the same production "
                "polyak_update"
            ),
            "actor": "frozen and unused in target construction",
            "target_actor": (
                "frozen and unused in target construction"),
            "only_intended_difference": (
                "moving learned-Q replay-continuation target versus exact "
                "finite-episode MC target"
            ),
        },
        "initial_contract": initial_contract,
        "validity": validity,
        "bootstrap_branch": bootstrap,
        "oracle_branch": oracle,
        "comparison": comparison,
        "comparison_scope": {
            "common_milestones": list(
                map(int, common_updates)),
            "last_common_update": last_common,
            "bootstrap_failure": bootstrap["failure"],
            "oracle_failure": oracle["failure"],
        },
        "summary": summary,
    }

    out_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "testing"
        / "stage2_vs_stage3_readiness"
        / "multi_bootstrap_vs_oracle_mc_causal.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n")

    print(
        "\nupdate\tbootstrap_qmin_spear\toracle_qmin_spear\t"
        "oracle-bootstrap\tbootstrap_qmean_spear\t"
        "oracle_qmean_spear\tbootstrap_mae\toracle_mae\t"
        "bootstrap_bias\toracle_bias"
    )
    for row in comparison:
        print(
            f"{row['update']}\t"
            f"{row['bootstrap_online_qmin_spearman']:.6f}\t"
            f"{row['oracle_online_qmin_spearman']:.6f}\t"
            f"{row['oracle_minus_bootstrap_qmin_spearman']:.6f}\t"
            f"{row['bootstrap_online_qmean_spearman']:.6f}\t"
            f"{row['oracle_online_qmean_spearman']:.6f}\t"
            f"{row['bootstrap_online_qmin_mae']:.6f}\t"
            f"{row['oracle_online_qmin_mae']:.6f}\t"
            f"{row['bootstrap_online_qmin_bias']:.6f}\t"
            f"{row['oracle_online_qmin_bias']:.6f}"
        )

    print("\n[VALIDITY]")
    print(json.dumps(validity, indent=2, sort_keys=True))
    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[STATUS] {output['status']}")
    print(f"[SAVED] {out_path}", flush=True)

    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
