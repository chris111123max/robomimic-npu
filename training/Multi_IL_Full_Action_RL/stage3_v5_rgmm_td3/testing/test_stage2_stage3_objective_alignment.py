#!/usr/bin/env python3
"""Read-only Stage2.2-vs-Stage3 objective-alignment diagnostic.

The purpose is to distinguish two possibilities:

1. the Stage3 Critic is simply failing to optimize its own Bellman objective;
2. the Stage3 Critic is becoming more Bellman-consistent while moving away
   from the finite-episode Monte-Carlo-return geometry learned in Stage2.2.

For each Stage3 checkpoint (step0, 100K, 200K), evaluate the same canonical
replay transitions and construct four targets:

  mc_return:
      G_t

  mc_identity_td:
      r_t + gamma * (1-terminal_t) * G_{t+1}
      This must equal G_t up to floating-point error.

  behavior_q_td:
      r_t + gamma * (1-terminal_t)
            * min(Q1_target,Q2_target)(h_{t+1}, a_{t+1}^{replay})
      Same target Critic and successor history as Stage3, but continue with the
      action actually executed by the replay trajectory.

  production_td:
      r_t + gamma * (1-terminal_t)
            * E_{a~pi_target}[min(Q1_target,Q2_target)(h_{t+1}, a)]
      Exact Stage3-v5 production component-mean Bellman target.

The decomposition

  production_td - mc_return
    = (mc_identity_td - mc_return)
    + (behavior_q_td - mc_identity_td)
    + (production_td - behavior_q_td)

separates:
  * MC identity / terminal semantics,
  * bootstrap value approximation error under the replay continuation action,
  * target-Actor continuation substitution under the same target Critic.

The script also measures whether the Stage3 TD gradient direction conflicts
with the direction that would move Q toward the Stage2.2 MC target.

No environment, rollout, optimizer step, Actor update, Critic update, or
checkpoint write occurs.
"""
from __future__ import annotations

import argparse
import gc
import json
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
from stage3_v5_history_critic import component_mean_q, encode_replay_contexts  # noqa: E402
from stage3_v5_readiness import correlation  # noqa: E402
from test_stage2_stage3_readiness_compare import resolve_device, sync  # noqa: E402
from test_stage3_td_target_decomposition import (  # noqa: E402
    DEFAULT_STAGE2,
    build_index,
    episode_terminals,
    load_canonical_episodes,
    make_batch,
    validate_payload,
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


def alignment_metrics(values, reference):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if values.shape != reference.shape or not len(values):
        raise ValueError("Metric arrays must be non-empty and shape matched")
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise FloatingPointError("Non-finite alignment values")
    spearman, pearson = correlation(values, reference)
    error = values - reference
    return {
        "count": int(len(values)),
        "spearman": float(spearman),
        "pearson": float(pearson),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "signed_mean_error": float(np.mean(error)),
        "value_mean": float(np.mean(values)),
        "value_std": float(np.std(values)),
        "reference_mean": float(np.mean(reference)),
        "reference_std": float(np.std(reference)),
    }


def gap_metrics(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not len(left):
        raise ValueError("Gap arrays must be non-empty and shape matched")
    gap = left - right
    return {
        "count": int(len(gap)),
        "signed_mean": float(np.mean(gap)),
        "abs_mean": float(np.mean(np.abs(gap))),
        "abs_p95": float(np.percentile(np.abs(gap), 95)),
        "abs_max": float(np.max(np.abs(gap))),
    }


def gradient_direction_metrics(q, td_target, mc_target, eps=1e-10):
    """Compare the TD regression direction with the MC regression direction."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    td = np.asarray(td_target, dtype=np.float64).reshape(-1)
    mc = np.asarray(mc_target, dtype=np.float64).reshape(-1)
    if q.shape != td.shape or q.shape != mc.shape or not len(q):
        raise ValueError("Gradient-direction arrays must be shape matched")

    td_direction = td - q
    mc_direction = mc - q
    product = td_direction * mc_direction
    informative = (np.abs(td_direction) > eps) & (np.abs(mc_direction) > eps)
    if not np.any(informative):
        return {
            "count": int(len(q)),
            "informative_count": 0,
            "conflict_fraction": 0.0,
            "aligned_fraction": 0.0,
            "zero_or_neutral_fraction": 1.0,
        }

    conflict = informative & (product < 0.0)
    aligned = informative & (product > 0.0)
    neutral = ~(conflict | aligned)
    return {
        "count": int(len(q)),
        "informative_count": int(np.sum(informative)),
        "conflict_fraction": float(np.mean(conflict[informative])),
        "aligned_fraction": float(np.mean(aligned[informative])),
        "zero_or_neutral_fraction": float(np.mean(neutral)),
        "mean_td_direction": float(np.mean(td_direction)),
        "mean_mc_direction": float(np.mean(mc_direction)),
    }


def next_mc_values(episodes, references, mc_returns):
    result = np.zeros(len(references), dtype=np.float32)
    for row, (episode_index, target) in enumerate(references):
        terminal = episode_terminals(episodes[episode_index])
        if terminal[target] < 0.5:
            if target + 1 >= len(mc_returns[episode_index]):
                raise RuntimeError(
                    "Non-terminal transition has no successor MC return")
            result[row] = mc_returns[episode_index][target + 1]
    return result


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
        raise RuntimeError("Expected audited horizon-10 Stage3 contract")

    normalization = payload["action_normalization_stats"]
    scale = torch.as_tensor(
        normalization["scale"], dtype=torch.float32, device=device
    ).reshape(1, 1, 1, 14)
    offset = torch.as_tensor(
        normalization["offset"], dtype=torch.float32, device=device
    ).reshape(1, 1, 1, 14)

    online_critic.eval()
    target_critic.eval()
    target_critic.requires_grad_(False)
    target_actor.eval()
    target_actor.requires_grad_(False)
    target_actor.low_noise_eval = True

    outputs = {
        "q1": [],
        "q2": [],
        "qmin": [],
        "mc": [],
        "mc_next": [],
        "mc_identity_td": [],
        "behavior_next_q": [],
        "policy_next_q": [],
        "behavior_q_td": [],
        "production_td": [],
        "terminal": [],
        "context_length": [],
    }

    for length in range(1, context_length + 1):
        references = items_by_length[length]
        for first in range(0, len(references), int(batch_size)):
            selected = references[first:first + int(batch_size)]
            batch = make_batch(
                episodes, selected, length, mc_returns)
            mc_next = next_mc_values(episodes, selected, mc_returns)

            obs = torch.as_tensor(
                batch["observations"], dtype=torch.float32, device=device)
            nxt = torch.as_tensor(
                batch["next_observations"], dtype=torch.float32, device=device)
            actions = torch.as_tensor(
                batch["actions"], dtype=torch.float32, device=device)
            steps = torch.as_tensor(
                batch["episode_steps"], dtype=torch.long, device=device)
            current_action = torch.as_tensor(
                batch["current_actions"], dtype=torch.float32, device=device)
            next_action = torch.as_tensor(
                batch["next_actions"], dtype=torch.float32, device=device)
            reward = torch.as_tensor(
                batch["rewards"], dtype=torch.float32, device=device)
            terminal = torch.as_tensor(
                batch["terminals"], dtype=torch.float32, device=device)
            mc_next_tensor = torch.as_tensor(
                mc_next, dtype=torch.float32, device=device)

            current_context = encode_replay_contexts(
                online_critic, obs, actions, steps, horizon)
            q1, q2 = online_critic.q_from_context(
                (current_context[0][:, -1], current_context[1][:, -1]),
                current_action,
            )
            q1 = q1.reshape(-1)
            q2 = q2.reshape(-1)
            qmin = torch.minimum(q1, q2)

            successor_context = encode_replay_contexts(
                target_critic,
                obs,
                actions,
                steps,
                horizon,
                next_observations=nxt,
            )
            successor_final = (
                successor_context[0][:, -1],
                successor_context[1][:, -1],
            )

            bq1, bq2 = target_critic.q_from_context(
                successor_final, next_action)
            behavior_next_q = torch.minimum(
                bq1, bq2).reshape(-1)

            starts = _last_reset_starts_from_numpy(
                batch["episode_steps"], actor_horizon)
            distribution, _ = target_final_distribution_vectorized(
                target_actor,
                nxt,
                horizon=actor_horizon,
                starts=starts,
            )
            policy_next_q, _, _, _, _ = component_mean_q(
                target_critic,
                successor_final,
                distribution,
                scale,
                offset,
                twin_min=True,
            )
            policy_next_q = policy_next_q.reshape(-1)

            bootstrap_mask = 1.0 - terminal
            mc_identity_td = (
                reward + gamma * bootstrap_mask * mc_next_tensor)
            behavior_q_td = (
                reward + gamma * bootstrap_mask * behavior_next_q)
            production_td = (
                reward + gamma * bootstrap_mask * policy_next_q)

            outputs["q1"].append(q1.cpu().numpy())
            outputs["q2"].append(q2.cpu().numpy())
            outputs["qmin"].append(qmin.cpu().numpy())
            outputs["mc"].append(batch["mc_returns"])
            outputs["mc_next"].append(mc_next)
            outputs["mc_identity_td"].append(
                mc_identity_td.cpu().numpy())
            outputs["behavior_next_q"].append(
                behavior_next_q.cpu().numpy())
            outputs["policy_next_q"].append(
                policy_next_q.cpu().numpy())
            outputs["behavior_q_td"].append(
                behavior_q_td.cpu().numpy())
            outputs["production_td"].append(
                production_td.cpu().numpy())
            outputs["terminal"].append(batch["terminals"])
            outputs["context_length"].append(
                np.full(len(selected), length, dtype=np.int64))

    outputs = {
        key: np.concatenate(value).reshape(-1)
        for key, value in outputs.items()
    }

    nonterminal = outputs["terminal"] < 0.5
    full_horizon_nonterminal = (
        nonterminal & (outputs["context_length"] == context_length))
    if not np.any(full_horizon_nonterminal):
        raise RuntimeError(
            "Canonical replay has no full-horizon non-terminal transitions")

    # MC Bellman identity is checked across all transitions, including terminal
    # and truncated rows. If this fails, the objective comparison is invalid.
    identity_gap_all = gap_metrics(
        outputs["mc_identity_td"], outputs["mc"])

    def subset_metrics(mask):
        mc = outputs["mc"][mask]
        mc_identity = outputs["mc_identity_td"][mask]
        behavior_td = outputs["behavior_q_td"][mask]
        production_td = outputs["production_td"][mask]
        q1 = outputs["q1"][mask]
        q2 = outputs["q2"][mask]
        qmin = outputs["qmin"][mask]
        mc_next = outputs["mc_next"][mask]
        behavior_next = outputs["behavior_next_q"][mask]
        policy_next = outputs["policy_next_q"][mask]

        bootstrap_gap = behavior_td - mc_identity
        policy_gap = production_td - behavior_td
        identity_gap = mc_identity - mc
        reconstructed = identity_gap + bootstrap_gap + policy_gap
        total_gap = production_td - mc

        return {
            "count": int(np.sum(mask)),
            "targets_vs_mc": {
                "mc_identity_td": alignment_metrics(
                    mc_identity, mc),
                "behavior_q_td": alignment_metrics(
                    behavior_td, mc),
                "production_td": alignment_metrics(
                    production_td, mc),
            },
            "successor_value_vs_behavior_mc_next": {
                "target_q_replay_next_action": alignment_metrics(
                    behavior_next, mc_next),
                "target_q_policy_component_mean": alignment_metrics(
                    policy_next, mc_next),
            },
            "target_gap_decomposition": {
                "mc_identity_gap": gap_metrics(
                    mc_identity, mc),
                "bootstrap_value_gap": gap_metrics(
                    behavior_td, mc_identity),
                "policy_continuation_gap": gap_metrics(
                    production_td, behavior_td),
                "total_stage3_vs_mc_gap": gap_metrics(
                    production_td, mc),
                "decomposition_reconstruction_max_abs": float(
                    np.max(np.abs(reconstructed - total_gap))),
            },
            "online_q_vs_mc": {
                "q1": alignment_metrics(q1, mc),
                "q2": alignment_metrics(q2, mc),
                "qmin": alignment_metrics(qmin, mc),
            },
            "online_q_vs_production_td": {
                "q1": alignment_metrics(q1, production_td),
                "q2": alignment_metrics(q2, production_td),
                "qmin": alignment_metrics(qmin, production_td),
            },
            "training_loss_like_residual": {
                "q1_td_mse": float(
                    np.mean(np.square(q1 - production_td))),
                "q2_td_mse": float(
                    np.mean(np.square(q2 - production_td))),
                "q1_td_mae": float(
                    np.mean(np.abs(q1 - production_td))),
                "q2_td_mae": float(
                    np.mean(np.abs(q2 - production_td))),
            },
            "mc_residual": {
                "q1_mc_mse": float(np.mean(np.square(q1 - mc))),
                "q2_mc_mse": float(np.mean(np.square(q2 - mc))),
                "qmin_mc_mae": float(np.mean(np.abs(qmin - mc))),
            },
            "td_vs_mc_gradient_direction": {
                "q1": gradient_direction_metrics(
                    q1, production_td, mc),
                "q2": gradient_direction_metrics(
                    q2, production_td, mc),
                "qmin_diagnostic": gradient_direction_metrics(
                    qmin, production_td, mc),
            },
        }

    primary = subset_metrics(full_horizon_nonterminal)
    nonterminal_metrics = subset_metrics(nonterminal)

    return {
        "transition_count": int(len(outputs["mc"])),
        "nonterminal_transition_count": int(np.sum(nonterminal)),
        "full_horizon_nonterminal_transition_count": int(
            np.sum(full_horizon_nonterminal)),
        "mc_identity_all_transitions": identity_gap_all,
        "nonterminal_transitions": nonterminal_metrics,
        "full_horizon_nonterminal_transitions": primary,
    }


def main():
    args = arguments()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    run_dir = Path(args.stage3_run_dir).resolve()
    stage2_path = Path(args.stage2_checkpoint).resolve()
    checkpoint_paths = {
        "stage3_step0": (
            run_dir / "multi_q" / "checkpoints" / "step0_transfer.pth"),
        "stage3_100k": (
            run_dir / "multi_q" / "checkpoints" / "step_0100000.pth"),
        "stage3_200k": (
            run_dir / "multi_q" / "checkpoints" / "step_0200000.pth"),
    }
    diagnostic_replay = (
        Path(args.diagnostic_replay).resolve()
        if args.diagnostic_replay
        else run_dir / "multi_q" / "checkpoints"
        / "step_0200000.sequences.npy"
    )

    for name, path in {
        "stage2": stage2_path,
        **checkpoint_paths,
        "diagnostic_replay": diagnostic_replay,
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    payloads = {
        name: torch.load(path, map_location="cpu")
        for name, path in checkpoint_paths.items()
    }
    reference_payload = payloads["stage3_step0"]
    for name, payload in payloads.items():
        validate_payload(reference_payload, payload, name)

    stage2_payload = torch.load(stage2_path, map_location="cpu")
    if stage2_payload.get("stage_version") != "2.2":
        raise RuntimeError("Expected Stage2.2 checkpoint")
    if stage2_payload.get("training_target") != (
        "finite_episode_monte_carlo_return_no_bootstrap"
    ):
        raise RuntimeError(
            "Stage2.2 training target contract changed unexpectedly")
    if float(stage2_payload.get("gamma", -1.0)) != float(
        reference_payload["config"]["gamma"]
    ):
        raise RuntimeError("Stage2.2 / Stage3 gamma mismatch")

    config = reference_payload["config"]
    gamma = float(config["gamma"])
    context_length = int(
        config["recurrent_replay"]["critic_context_length"])
    if context_length != 10:
        raise RuntimeError("Expected Stage3 critic_context_length=10")

    fixed, episodes = load_canonical_episodes(diagnostic_replay)
    (
        items_by_length,
        mc_returns,
        transition_count,
        nonterminal_count,
    ) = build_index(episodes, context_length, gamma)

    device = resolve_device(args.device)
    target_actor, _, actor_metadata = load_exact_actor(
        config["bc_rnn_checkpoint"], device)

    results = {}
    actor_hashes = {}
    order = ("stage3_step0", "stage3_100k", "stage3_200k")

    for name in order:
        payload = payloads[name]
        incompatible = target_actor.load_state_dict(
            payload["target_actor"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"{name}: strict target Actor load failed")
        target_actor.eval()
        target_actor.low_noise_eval = True
        actor_hashes[name] = module_hash(target_actor)

        online_critic, _ = strict_stage2_load(
            stage2_path, device)
        target_critic, _ = strict_stage2_load(
            stage2_path, device)
        online_critic.load_state_dict(
            payload["q1_q2"], strict=True)
        target_critic.load_state_dict(
            payload["target_q1_q2"], strict=True)

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

        identity_max = metrics[
            "mc_identity_all_transitions"]["abs_max"]
        if identity_max > 2e-6:
            raise RuntimeError(
                f"{name}: MC Bellman identity failed: {identity_max}")

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

    deltas = {}
    for left, right in zip(order, order[1:]):
        lm = results[left]["metrics"][
            "full_horizon_nonterminal_transitions"]
        rm = results[right]["metrics"][
            "full_horizon_nonterminal_transitions"]
        deltas[f"{left}_to_{right}"] = {
            "qmin_mc_spearman_change": float(
                rm["online_q_vs_mc"]["qmin"]["spearman"]
                - lm["online_q_vs_mc"]["qmin"]["spearman"]),
            "qmin_mc_mae_change": float(
                rm["online_q_vs_mc"]["qmin"]["mae"]
                - lm["online_q_vs_mc"]["qmin"]["mae"]),
            "production_td_mc_spearman_change": float(
                rm["targets_vs_mc"]["production_td"]["spearman"]
                - lm["targets_vs_mc"]["production_td"]["spearman"]),
            "production_td_mc_mae_change": float(
                rm["targets_vs_mc"]["production_td"]["mae"]
                - lm["targets_vs_mc"]["production_td"]["mae"]),
            "q1_td_mae_change": float(
                rm["training_loss_like_residual"]["q1_td_mae"]
                - lm["training_loss_like_residual"]["q1_td_mae"]),
            "q2_td_mae_change": float(
                rm["training_loss_like_residual"]["q2_td_mae"]
                - lm["training_loss_like_residual"]["q2_td_mae"]),
            "bootstrap_gap_abs_mean_change": float(
                rm["target_gap_decomposition"][
                    "bootstrap_value_gap"]["abs_mean"]
                - lm["target_gap_decomposition"][
                    "bootstrap_value_gap"]["abs_mean"]),
            "policy_gap_abs_mean_change": float(
                rm["target_gap_decomposition"][
                    "policy_continuation_gap"]["abs_mean"]
                - lm["target_gap_decomposition"][
                    "policy_continuation_gap"]["abs_mean"]),
            "q1_gradient_conflict_fraction_change": float(
                rm["td_vs_mc_gradient_direction"]["q1"][
                    "conflict_fraction"]
                - lm["td_vs_mc_gradient_direction"]["q1"][
                    "conflict_fraction"]),
            "q2_gradient_conflict_fraction_change": float(
                rm["td_vs_mc_gradient_direction"]["q2"][
                    "conflict_fraction"]
                - lm["td_vs_mc_gradient_direction"]["q2"][
                    "conflict_fraction"]),
        }

    validity = {
        "target_actor_hash_constant": (
            len(set(actor_hashes.values())) == 1
        ),
        "actor_updates_zero_all": all(
            int(results[name]["actor_updates"]) == 0
            for name in order
        ),
        "mc_identity_max_abs_le_2e_6": all(
            results[name]["metrics"][
                "mc_identity_all_transitions"]["abs_max"] <= 2e-6
            for name in order
        ),
        "decomposition_reconstruction_max_abs_le_2e_6": all(
            results[name]["metrics"][
                "full_horizon_nonterminal_transitions"
            ]["target_gap_decomposition"][
                "decomposition_reconstruction_max_abs"
            ] <= 2e-6
            for name in order
        ),
        "stage3_step0_is_zero_update": bool(
            results["stage3_step0"]["env_steps"] == 0
            and results["stage3_step0"]["critic_updates"] == 0
        ),
    }
    valid = bool(all(validity.values()))

    output = {
        "status": "PASS" if valid else "INVALID",
        "read_only": True,
        "environment_steps_performed": 0,
        "optimizer_steps_performed": 0,
        "actor_updates_performed": 0,
        "critic_updates_performed": 0,
        "training_checkpoints_written": 0,
        "device": str(device),
        "stage2_checkpoint": str(stage2_path),
        "stage2_objective_contract": {
            "training_target": stage2_payload.get(
                "training_target"),
            "loss": stage2_payload.get("loss"),
            "gamma": float(stage2_payload.get("gamma")),
            "history_semantics": stage2_payload.get(
                "history_semantics"),
            "terminated_truncated_semantics": stage2_payload.get(
                "terminated_truncated_semantics"),
        },
        "stage3_objective_contract": {
            "training_target": (
                "r + gamma*(1-terminal)*"
                "E_target_actor[min(target_Q1,target_Q2)]"
            ),
            "loss": "mse_q1_plus_mse_q2",
            "gamma": gamma,
            "history_semantics": config[
                "recurrent_replay"]["critic_history_semantics"],
            "objective_revision": config.get("objective_revision"),
        },
        "canonical_diagnostic_replay": str(diagnostic_replay),
        "canonical_seed": fixed.get("seed"),
        "canonical_episode_count": int(len(episodes)),
        "canonical_success_episodes": int(
            fixed.get("success_episode_count", 0)),
        "canonical_failure_episodes": int(
            fixed.get("failure_episode_count", 0)),
        "canonical_transition_count": int(transition_count),
        "canonical_nonterminal_transition_count": int(
            nonterminal_count),
        "primary_analysis_subset": (
            "full_horizon_nonterminal_transitions"
        ),
        "decomposition": {
            "mc_return": "G_t",
            "mc_identity_td": (
                "r_t + gamma*(1-terminal_t)*G_(t+1)"
            ),
            "behavior_q_td": (
                "r_t + gamma*(1-terminal_t)*"
                "min(target_Q1,target_Q2)"
                "(h_(t+1), replay_action_(t+1))"
            ),
            "production_td": (
                "r_t + gamma*(1-terminal_t)*"
                "E_target_actor[min(target_Q1,target_Q2)]"
            ),
            "bootstrap_value_gap": (
                "behavior_q_td - mc_identity_td"
            ),
            "policy_continuation_gap": (
                "production_td - behavior_q_td"
            ),
        },
        "actor_metadata": actor_metadata,
        "target_actor_hashes": actor_hashes,
        "validity": validity,
        "results": results,
        "deltas": deltas,
    }

    out_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "testing"
        / "stage2_vs_stage3_readiness"
        / "multi_stage2_stage3_objective_alignment.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n")

    print(
        f"[CANONICAL] episodes={len(episodes)} "
        f"transitions={transition_count} "
        f"nonterminal={nonterminal_count} "
        f"seed={fixed.get('seed')} device={device}"
    )
    print(
        "\ncheckpoint\tupdates\tQmc_spear\tQmc_mae\t"
        "TDmc_spear\tTDmc_mae\tbehaviorTD_mae\t"
        "bootstrap_gap\tpolicy_gap\tq1_td_mae\t"
        "q1_conflict"
    )
    for name in order:
        row = results[name]
        m = row["metrics"][
            "full_horizon_nonterminal_transitions"]
        print(
            f"{name}\t{row['critic_updates']}\t"
            f"{m['online_q_vs_mc']['qmin']['spearman']:.6f}\t"
            f"{m['online_q_vs_mc']['qmin']['mae']:.6f}\t"
            f"{m['targets_vs_mc']['production_td']['spearman']:.6f}\t"
            f"{m['targets_vs_mc']['production_td']['mae']:.6f}\t"
            f"{m['targets_vs_mc']['behavior_q_td']['mae']:.6f}\t"
            f"{m['target_gap_decomposition']['bootstrap_value_gap']['abs_mean']:.6f}\t"
            f"{m['target_gap_decomposition']['policy_continuation_gap']['abs_mean']:.6f}\t"
            f"{m['training_loss_like_residual']['q1_td_mae']:.6f}\t"
            f"{m['td_vs_mc_gradient_direction']['q1']['conflict_fraction']:.6f}"
        )

    print("\n[MC IDENTITY MAX ABS]")
    for name in order:
        value = results[name]["metrics"][
            "mc_identity_all_transitions"]["abs_max"]
        print(f"{name}: {value:.9g}")

    print("\n[TARGET ACTOR HASHES]")
    for name in order:
        print(f"{name}: {actor_hashes[name]}")

    print("\n[VALIDITY]")
    print(json.dumps(validity, indent=2, sort_keys=True))
    print(f"\n[SAVED] {out_path}", flush=True)
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
