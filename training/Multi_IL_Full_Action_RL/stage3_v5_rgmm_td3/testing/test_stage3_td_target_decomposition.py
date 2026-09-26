#!/usr/bin/env python3
"""Read-only Stage3-v5 TD-target decomposition on one canonical replay set.

Purpose
-------
Determine whether the Stage3 component-mean bootstrap target loses alignment
with the Monte-Carlo return, and compare it with a counterfactual control that
uses the *executed next replay action* under the exact same target Critic.

For each Stage3 checkpoint (step0, 100K, 200K), this script computes on the
same canonical frozen diagnostic episodes:

  current_q:
      min(Q1,Q2)(s_t, a_t)

  component_mean_td:
      r_t + gamma * (1-terminal_t)
            * sum_k p_k(s_{t+1}) *
              min(Q1_target,Q2_target)(s_{t+1}, mu_k)

  executed_next_action_td:
      r_t + gamma * (1-terminal_t)
            * min(Q1_target,Q2_target)(s_{t+1}, a_{t+1}^{replay})

The first target is exactly the Stage3-v5 Case-A Bellman target. The second is
a diagnostic control only; it is never used for training.

No environment, rollout, optimizer step, Actor update, or Critic update occurs.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STAGE3 = HERE.parent
for directory in (HERE, STAGE3):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from stage3_v5_actor import load_exact_actor, module_hash  # noqa: E402
from stage3_v5_agent import (  # noqa: E402
    _last_reset_starts_from_numpy,
    strict_stage2_load,
    target_final_distribution_vectorized,
)
from stage3_v5_history_critic import (  # noqa: E402
    component_mean_q,
    encode_replay_contexts,
)
from stage3_v5_readiness import correlation, discounted_returns  # noqa: E402
from stage3_v5_replay import OnlineSequenceReplay  # noqa: E402
from test_stage2_stage3_readiness_compare import resolve_device, sync  # noqa: E402


DEFAULT_STAGE2 = (
    "/data/home/3220251075/lerobot_workspace/training_runs/"
    "Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/"
    "stage2_2_h10_multi_20260923_150749/multi_q/checkpoints/step_00005000.pth"
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-run-dir", required=True)
    parser.add_argument("--stage2-checkpoint", default=DEFAULT_STAGE2)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--diagnostic-replay",
        help="Canonical .sequences.npy; defaults to Multi Stage3 200K sidecar.",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def cleanup_device(device):
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def load_canonical_episodes(path):
    replay = OnlineSequenceReplay.load(path)
    fixed = replay.fixed_critic_diagnostic_set
    if fixed is None:
        raise RuntimeError(f"No fixed_critic_diagnostic_set in {path}")
    episodes = fixed.get("episodes", [])
    if not episodes:
        raise RuntimeError("Canonical diagnostic set contains no episodes")
    return fixed, episodes


def episode_terminals(episode):
    if "terminals" in episode:
        result = np.asarray(episode["terminals"], dtype=np.float32).reshape(-1)
    elif "dones" in episode:
        result = np.asarray(episode["dones"], dtype=np.float32).reshape(-1)
    elif "terminated" in episode and "truncated" in episode:
        result = (
            np.asarray(episode["terminated"], bool).reshape(-1)
            | np.asarray(episode["truncated"], bool).reshape(-1)
        ).astype(np.float32)
    else:
        raise RuntimeError("Episode has no Stage3 terminal mask")
    return result


def scalar_metrics(values, reference):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if values.shape != reference.shape or not len(values):
        raise ValueError("Metric arrays must be non-empty and shape matched")
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise FloatingPointError("Non-finite diagnostic values")
    spearman, pearson = correlation(values, reference)
    error = values - reference
    return {
        "count": int(len(values)),
        "spearman_vs_mc": float(spearman),
        "pearson_vs_mc": float(pearson),
        "mae_vs_mc": float(np.mean(np.abs(error))),
        "rmse_vs_mc": float(np.sqrt(np.mean(np.square(error)))),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def distribution_gap_metrics(component_td, executed_td):
    component_td = np.asarray(component_td, np.float64).reshape(-1)
    executed_td = np.asarray(executed_td, np.float64).reshape(-1)
    spearman, pearson = correlation(component_td, executed_td)
    gap = component_td - executed_td
    return {
        "spearman_component_vs_executed": float(spearman),
        "pearson_component_vs_executed": float(pearson),
        "component_minus_executed_mean": float(gap.mean()),
        "component_minus_executed_abs_mean": float(np.abs(gap).mean()),
        "component_minus_executed_p95_abs": float(np.percentile(np.abs(gap), 95)),
    }


def build_index(episodes, context_length, gamma):
    items_by_length = {length: [] for length in range(1, context_length + 1)}
    returns = []
    transition_count = 0
    nonterminal_count = 0
    for episode_index, episode in enumerate(episodes):
        actions = np.asarray(episode["actions"], dtype=np.float32)
        observations = np.asarray(episode["observations"], dtype=np.float32)
        next_observations = np.asarray(episode["next_observations"], dtype=np.float32)
        if observations.shape != (len(actions), 59):
            raise RuntimeError(f"Episode {episode_index}: observation shape mismatch")
        if next_observations.shape != observations.shape or actions.shape[1:] != (14,):
            raise RuntimeError(f"Episode {episode_index}: transition shape mismatch")
        terminal = episode_terminals(episode)
        if len(terminal) != len(actions):
            raise RuntimeError(f"Episode {episode_index}: terminal length mismatch")
        mc = discounted_returns(episode, gamma)
        returns.append(mc)
        for target in range(len(actions)):
            length = min(context_length, target + 1)
            items_by_length[length].append((episode_index, target))
        transition_count += len(actions)
        nonterminal_count += int(np.sum(terminal < 0.5))
    return items_by_length, returns, transition_count, nonterminal_count


def make_batch(episodes, references, length, mc_returns):
    batch = len(references)
    observations = np.empty((batch, length, 59), np.float32)
    next_observations = np.empty((batch, length, 59), np.float32)
    actions = np.empty((batch, length, 14), np.float32)
    episode_steps = np.empty((batch, length), np.int64)
    current_actions = np.empty((batch, 14), np.float32)
    next_actions = np.zeros((batch, 14), np.float32)
    rewards = np.empty(batch, np.float32)
    terminals = np.empty(batch, np.float32)
    mc = np.empty(batch, np.float32)

    for row, (episode_index, target) in enumerate(references):
        episode = episodes[episode_index]
        ep_actions = np.asarray(episode["actions"], np.float32)
        start = target - length + 1
        observations[row] = np.asarray(
            episode["observations"][start:target + 1], np.float32)
        next_observations[row] = np.asarray(
            episode["next_observations"][start:target + 1], np.float32)
        actions[row] = ep_actions[start:target + 1]
        steps = np.asarray(
            episode.get("episode_steps", np.arange(len(ep_actions))),
            dtype=np.int64,
        )
        episode_steps[row] = steps[start:target + 1]
        current_actions[row] = ep_actions[target]

        terminal = episode_terminals(episode)
        terminals[row] = terminal[target]
        rewards[row] = np.asarray(episode["rewards"], np.float32).reshape(-1)[target]
        mc[row] = mc_returns[episode_index][target]

        if terminal[target] < 0.5:
            if target + 1 >= len(ep_actions):
                raise RuntimeError(
                    "Non-terminal transition has no replay next action; "
                    "cannot form executed-action control target"
                )
            next_actions[row] = ep_actions[target + 1]

    return {
        "observations": observations,
        "next_observations": next_observations,
        "actions": actions,
        "episode_steps": episode_steps,
        "current_actions": current_actions,
        "next_actions": next_actions,
        "rewards": rewards,
        "terminals": terminals,
        "mc_returns": mc,
    }


@torch.no_grad()
def evaluate_checkpoint(
    payload,
    online_critic,
    target_critic,
    target_actor,
    episodes,
    items_by_length,
    mc_returns,
    device,
    batch_size,
):
    config = payload["config"]
    gamma = float(config["gamma"])
    horizon = int(config["horizon"])
    actor_horizon = int(config["actor_source_contract"]["rnn_horizon"])
    context_length = int(config["recurrent_replay"]["critic_context_length"])
    if context_length != 10 or actor_horizon != 10:
        raise RuntimeError("Diagnostic requires the audited horizon-10 Stage3 contract")

    normalization = payload["action_normalization_stats"]
    scale = torch.as_tensor(
        normalization["scale"], dtype=torch.float32, device=device
    ).reshape(1, 1, 1, 14)
    offset = torch.as_tensor(
        normalization["offset"], dtype=torch.float32, device=device
    ).reshape(1, 1, 1, 14)

    target_actor.eval()
    target_actor.requires_grad_(False)
    target_actor.low_noise_eval = True
    online_critic.eval()
    target_critic.eval()
    target_critic.requires_grad_(False)

    outputs = {
        "current_q": [],
        "component_td": [],
        "executed_td": [],
        "component_next_q": [],
        "executed_next_q": [],
        "mc": [],
        "terminal": [],
        "closest_component_action_l2": [],
        "weighted_component_action_l2": [],
        "context_length": [],
    }

    for length in range(1, context_length + 1):
        references = items_by_length[length]
        for first in range(0, len(references), batch_size):
            selected = references[first:first + batch_size]
            batch = make_batch(episodes, selected, length, mc_returns)

            obs = torch.as_tensor(batch["observations"], device=device)
            nxt = torch.as_tensor(batch["next_observations"], device=device)
            actions = torch.as_tensor(batch["actions"], device=device)
            steps = torch.as_tensor(batch["episode_steps"], device=device)
            current_action = torch.as_tensor(batch["current_actions"], device=device)
            next_action = torch.as_tensor(batch["next_actions"], device=device)
            reward = torch.as_tensor(batch["rewards"], device=device)
            terminal = torch.as_tensor(batch["terminals"], device=device)

            # Current online Critic Q(s_t, a_t), under exactly the same sliding
            # zero-state history semantics used by Stage3 readiness.
            current_contexts = encode_replay_contexts(
                online_critic, obs, actions, steps, horizon)
            q1, q2 = online_critic.q_from_context(
                (current_contexts[0][:, -1], current_contexts[1][:, -1]),
                current_action,
            )
            current_q = torch.minimum(q1, q2).reshape(-1)

            # Exact Stage3 successor target-Critic context.
            successor_contexts = encode_replay_contexts(
                target_critic,
                obs,
                actions,
                steps,
                horizon,
                next_observations=nxt,
            )
            final_successor = (
                successor_contexts[0][:, -1],
                successor_contexts[1][:, -1],
            )

            # Exact production component-mean target actor semantics.
            starts = _last_reset_starts_from_numpy(
                batch["episode_steps"], actor_horizon)
            distribution, _ = target_final_distribution_vectorized(
                target_actor, nxt, horizon=actor_horizon, starts=starts)
            component_next_q, _, _, params, component_actions = component_mean_q(
                target_critic,
                final_successor,
                distribution,
                scale,
                offset,
                twin_min=True,
            )
            component_next_q = component_next_q.reshape(-1)

            # Diagnostic control: same target Critic / successor context, but
            # evaluate the action actually executed at the replay next state.
            eq1, eq2 = target_critic.q_from_context(final_successor, next_action)
            executed_next_q = torch.minimum(eq1, eq2).reshape(-1)

            bootstrap_mask = 1.0 - terminal
            component_td = reward + gamma * bootstrap_mask * component_next_q
            executed_td = reward + gamma * bootstrap_mask * executed_next_q

            # Action-space distance is diagnostic only. Terminal rows are
            # masked out later because their next action is intentionally zero.
            probabilities = params["probs"]
            weighted_action = (
                probabilities.unsqueeze(-1) * component_actions
            ).sum(dim=-2)
            closest_l2 = torch.linalg.vector_norm(
                component_actions - next_action.unsqueeze(-2), dim=-1
            ).min(dim=-1).values
            weighted_l2 = torch.linalg.vector_norm(
                weighted_action - next_action, dim=-1)

            outputs["current_q"].append(current_q.cpu().numpy())
            outputs["component_td"].append(component_td.cpu().numpy())
            outputs["executed_td"].append(executed_td.cpu().numpy())
            outputs["component_next_q"].append(component_next_q.cpu().numpy())
            outputs["executed_next_q"].append(executed_next_q.cpu().numpy())
            outputs["mc"].append(batch["mc_returns"])
            outputs["terminal"].append(batch["terminals"])
            outputs["closest_component_action_l2"].append(closest_l2.cpu().numpy())
            outputs["weighted_component_action_l2"].append(weighted_l2.cpu().numpy())
            outputs["context_length"].append(
                np.full(len(selected), length, dtype=np.int64))

    outputs = {
        key: np.concatenate(value).reshape(-1)
        for key, value in outputs.items()
    }
    nonterminal = outputs["terminal"] < 0.5
    full_horizon_nonterminal = (
        nonterminal & (outputs["context_length"] == context_length))
    if not np.any(nonterminal):
        raise RuntimeError("Canonical set contains no non-terminal transitions")
    if not np.any(full_horizon_nonterminal):
        raise RuntimeError("Canonical set contains no train-eligible horizon-10 transitions")

    all_metrics = {
        "current_q": scalar_metrics(outputs["current_q"], outputs["mc"]),
        "component_mean_td": scalar_metrics(outputs["component_td"], outputs["mc"]),
        "executed_next_action_td": scalar_metrics(outputs["executed_td"], outputs["mc"]),
        "target_gap": distribution_gap_metrics(
            outputs["component_td"], outputs["executed_td"]),
    }
    def masked_metrics(mask):
        return {
            "current_q": scalar_metrics(
                outputs["current_q"][mask], outputs["mc"][mask]),
            "component_mean_td": scalar_metrics(
                outputs["component_td"][mask], outputs["mc"][mask]),
            "executed_next_action_td": scalar_metrics(
                outputs["executed_td"][mask], outputs["mc"][mask]),
            "target_gap": distribution_gap_metrics(
                outputs["component_td"][mask],
                outputs["executed_td"][mask],
            ),
            "component_next_q": {
                "mean": float(outputs["component_next_q"][mask].mean()),
                "std": float(outputs["component_next_q"][mask].std()),
            },
            "executed_next_q": {
                "mean": float(outputs["executed_next_q"][mask].mean()),
                "std": float(outputs["executed_next_q"][mask].std()),
            },
            "action_distance": {
                "closest_component_mean_l2_mean": float(
                    outputs["closest_component_action_l2"][mask].mean()),
                "closest_component_mean_l2_p95": float(np.percentile(
                    outputs["closest_component_action_l2"][mask], 95)),
                "weighted_component_mean_l2_mean": float(
                    outputs["weighted_component_action_l2"][mask].mean()),
                "weighted_component_mean_l2_p95": float(np.percentile(
                    outputs["weighted_component_action_l2"][mask], 95)),
            },
        }

    nonterminal_metrics = masked_metrics(nonterminal)
    train_eligible_metrics = masked_metrics(full_horizon_nonterminal)

    return {
        "transition_count": int(len(outputs["mc"])),
        "nonterminal_transition_count": int(nonterminal.sum()),
        "full_horizon_nonterminal_transition_count": int(
            full_horizon_nonterminal.sum()),
        "terminal_transition_count": int((~nonterminal).sum()),
        "all_transitions": all_metrics,
        "nonterminal_transitions": nonterminal_metrics,
        "full_horizon_nonterminal_transitions": train_eligible_metrics,
    }


def validate_payload(reference, candidate, name):
    if candidate.get("stage") != "stage3-v5" or candidate.get("group") != "multi_q":
        raise RuntimeError(f"{name}: expected Stage3-v5 multi_q checkpoint")
    ref_config = reference["config"]
    config = candidate["config"]
    for key in ("gamma", "horizon", "recurrent_replay", "actor_source_contract",
                "bc_rnn_checkpoint"):
        if config.get(key) != ref_config.get(key):
            raise RuntimeError(f"{name}: config mismatch for {key}")
    if candidate.get("action_normalization_stats") != reference.get(
            "action_normalization_stats"):
        raise RuntimeError(f"{name}: action normalization changed")


def main():
    args = arguments()
    run_dir = Path(args.stage3_run_dir).resolve()
    stage2_path = Path(args.stage2_checkpoint).resolve()
    checkpoint_paths = {
        "stage3_step0": run_dir / "multi_q" / "checkpoints" / "step0_transfer.pth",
        "stage3_100k": run_dir / "multi_q" / "checkpoints" / "step_0100000.pth",
        "stage3_200k": run_dir / "multi_q" / "checkpoints" / "step_0200000.pth",
    }
    diagnostic_replay = (
        Path(args.diagnostic_replay).resolve()
        if args.diagnostic_replay
        else run_dir / "multi_q" / "checkpoints" / "step_0200000.sequences.npy"
    )

    if not stage2_path.exists():
        raise FileNotFoundError(stage2_path)
    for name, path in checkpoint_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    if not diagnostic_replay.exists():
        raise FileNotFoundError(diagnostic_replay)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    # Load payloads on CPU first and validate immutable contracts before
    # allocating any NPU models.
    payloads = {
        name: torch.load(path, map_location="cpu")
        for name, path in checkpoint_paths.items()
    }
    reference_payload = payloads["stage3_step0"]
    for name, payload in payloads.items():
        validate_payload(reference_payload, payload, name)

    config = reference_payload["config"]
    gamma = float(config["gamma"])
    context_length = int(config["recurrent_replay"]["critic_context_length"])
    if context_length != 10:
        raise RuntimeError("Expected Stage3-v5 critic_context_length=10")

    fixed, episodes = load_canonical_episodes(diagnostic_replay)
    items_by_length, mc_returns, transition_count, nonterminal_count = build_index(
        episodes, context_length, gamma)

    device = resolve_device(args.device)

    # Build Actor architecture once. Each checkpoint's exact target_actor
    # state_dict is loaded before that checkpoint is evaluated.
    target_actor, _, actor_metadata = load_exact_actor(
        config["bc_rnn_checkpoint"], device)

    results = {}
    actor_hashes = {}
    for name in ("stage3_step0", "stage3_100k", "stage3_200k"):
        payload = payloads[name]

        incompatible = target_actor.load_state_dict(
            payload["target_actor"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"{name}: strict target Actor load failed")
        target_actor.eval()
        target_actor.low_noise_eval = True
        actor_hashes[name] = module_hash(target_actor)

        online_critic, _ = strict_stage2_load(stage2_path, device)
        target_critic, _ = strict_stage2_load(stage2_path, device)
        online_critic.load_state_dict(payload["q1_q2"], strict=True)
        target_critic.load_state_dict(payload["target_q1_q2"], strict=True)

        sync(device)
        metrics = evaluate_checkpoint(
            payload,
            online_critic,
            target_critic,
            target_actor,
            episodes,
            items_by_length,
            mc_returns,
            device,
            int(args.batch_size),
        )
        sync(device)

        results[name] = {
            "checkpoint": str(checkpoint_paths[name]),
            "env_steps": int(payload.get("env_steps", -1)),
            "critic_updates": int(payload.get("updates", -1)),
            "actor_updates": int(payload.get("actor_updates", -1)),
            "target_actor_hash": actor_hashes[name],
            "metrics": metrics,
        }
        del online_critic, target_critic
        cleanup_device(device)

    # Compact deltas that directly answer the hypothesis test.
    deltas = {}
    order = ("stage3_step0", "stage3_100k", "stage3_200k")
    for left, right in zip(order, order[1:]):
        left_m = results[left]["metrics"]["full_horizon_nonterminal_transitions"]
        right_m = results[right]["metrics"]["full_horizon_nonterminal_transitions"]
        deltas[f"{left}_to_{right}"] = {
            "current_q_spearman_change": float(
                right_m["current_q"]["spearman_vs_mc"]
                - left_m["current_q"]["spearman_vs_mc"]),
            "component_td_spearman_change": float(
                right_m["component_mean_td"]["spearman_vs_mc"]
                - left_m["component_mean_td"]["spearman_vs_mc"]),
            "executed_td_spearman_change": float(
                right_m["executed_next_action_td"]["spearman_vs_mc"]
                - left_m["executed_next_action_td"]["spearman_vs_mc"]),
            "component_minus_executed_spearman_margin_change": float(
                (
                    right_m["component_mean_td"]["spearman_vs_mc"]
                    - right_m["executed_next_action_td"]["spearman_vs_mc"]
                ) - (
                    left_m["component_mean_td"]["spearman_vs_mc"]
                    - left_m["executed_next_action_td"]["spearman_vs_mc"]
                )
            ),
        }

    output = {
        "status": "PASS",
        "read_only": True,
        "environment_steps_performed": 0,
        "optimizer_steps_performed": 0,
        "actor_updates_performed": 0,
        "critic_updates_performed": 0,
        "device": str(device),
        "stage2_architecture_checkpoint": str(stage2_path),
        "canonical_diagnostic_replay": str(diagnostic_replay),
        "canonical_seed": fixed.get("seed"),
        "canonical_episode_count": int(len(episodes)),
        "canonical_success_episodes": int(fixed.get("success_episode_count", 0)),
        "canonical_failure_episodes": int(fixed.get("failure_episode_count", 0)),
        "canonical_transition_count": int(transition_count),
        "canonical_nonterminal_transition_count": int(nonterminal_count),
        "history_contract": "sliding_horizon_10_zero_state",
        "primary_analysis_subset": (
            "full_horizon_nonterminal_transitions: exactly the final transition "
            "of train-eligible length-10 windows"
        ),
        "production_component_target": (
            "r + gamma*(1-terminal)*sum_k p_k*"
            "min(Q1_target,Q2_target)(s_next,component_mean_k)"
        ),
        "diagnostic_control_target": (
            "r + gamma*(1-terminal)*"
            "min(Q1_target,Q2_target)(s_next,replay_executed_next_action)"
        ),
        "actor_metadata": actor_metadata,
        "target_actor_hashes": actor_hashes,
        "results": results,
        "deltas": deltas,
    }

    out_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "testing" / "stage2_vs_stage3_readiness"
        / "multi_td_target_decomposition.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    print(
        f"[CANONICAL] episodes={len(episodes)} transitions={transition_count} "
        f"nonterminal={nonterminal_count} seed={fixed.get('seed')} device={device}"
    )
    print(
        "\ncheckpoint\tupdates\tQ_spear\tcomponentTD_spear\t"
        "executedTD_spear\tcomponent_MAE\texecuted_MAE\ttarget_gap_abs"
    )
    for name in order:
        row = results[name]
        m = row["metrics"]["full_horizon_nonterminal_transitions"]
        print(
            f"{name}\t{row['critic_updates']}\t"
            f"{m['current_q']['spearman_vs_mc']:.6f}\t"
            f"{m['component_mean_td']['spearman_vs_mc']:.6f}\t"
            f"{m['executed_next_action_td']['spearman_vs_mc']:.6f}\t"
            f"{m['component_mean_td']['mae_vs_mc']:.6f}\t"
            f"{m['executed_next_action_td']['mae_vs_mc']:.6f}\t"
            f"{m['target_gap']['component_minus_executed_abs_mean']:.6f}"
        )
    print("\n[TARGET ACTOR HASHES]")
    for name in order:
        print(f"{name}: {actor_hashes[name]}")
    print(f"\n[SAVED] {out_path}", flush=True)


if __name__ == "__main__":
    main()
