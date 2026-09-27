#!/usr/bin/env python3
"""Read-only Bellman-closure diagnostic for the original Stage2.2 MC Critic.

Purpose
-------
Before any Stage3 update, test whether the Stage2.2 horizon-10 Critic that was
trained directly on finite-episode Monte-Carlo returns is also approximately
closed under the one-step replay Bellman recursion used later by Stage3.

For each non-terminal canonical replay transition, evaluate the SAME Stage2.2
Critic on:

  current value:
      Q(h_t, a_t)

  successor replay value:
      Q(h_(t+1), a_(t+1)^replay)

and compare:

  direct MC target:
      G_t

  exact MC identity:
      r_t + gamma * G_(t+1)

  Stage2.2 self-bootstrap target:
      r_t + gamma * Q(h_(t+1), a_(t+1)^replay)

  Bellman closure residual:
      Q(h_t,a_t) - [r_t + gamma*Q(h_(t+1),a_(t+1)^replay)]

The diagnostic is computed separately for Q1, Q2, their arithmetic mean, and
their minimum. The primary analysis subset is full horizon-10 non-terminal
transitions. Results are also bucketed by episode step.

Important interpretation boundary
---------------------------------
A non-zero self-bootstrap residual does NOT by itself prove that the horizon-10
representation is non-Markov. With exact MC identity,

  Q_t - (r + gamma Q_next)
    = (Q_t - G_t) - gamma (Q_next - G_next)

so finite function-approximation / regression errors can also produce a closure
residual. This script therefore records both current and successor MC errors,
their correlation, and verifies the algebraic decomposition exactly. It is a
diagnostic for Bellman closure, not a standalone proof of state aliasing.

No environment, rollout, optimizer step, Actor update, Critic update, or
training checkpoint write occurs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
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

from stage3_v5_agent import strict_stage2_load  # noqa: E402
from stage3_v5_history_critic import encode_replay_contexts  # noqa: E402
from stage3_v5_readiness import correlation  # noqa: E402
from test_stage2_stage3_readiness_compare import resolve_device, sync  # noqa: E402
from test_stage3_td_target_decomposition import (  # noqa: E402
    DEFAULT_STAGE2,
    build_index,
    episode_terminals,
    load_canonical_episodes,
    make_batch,
)


PROGRESS_BINS = (
    ("step_9_49", 9, 50),
    ("step_50_99", 50, 100),
    ("step_100_199", 100, 200),
    ("step_200_399", 200, 400),
    ("step_400_plus", 400, None),
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


def state_dict_digest(state_dict):
    digest = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        digest.update(
            value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def next_mc_values(episodes, references, mc_returns):
    result = np.zeros(len(references), dtype=np.float32)
    for row, (episode_index, target) in enumerate(references):
        terminal = episode_terminals(episodes[episode_index])
        if terminal[target] >= 0.5:
            continue
        if target + 1 >= len(mc_returns[episode_index]):
            raise RuntimeError(
                "Non-terminal transition has no successor MC return")
        result[row] = mc_returns[episode_index][target + 1]
    return result


def metric_against_reference(values, reference):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if values.shape != reference.shape or not len(values):
        raise ValueError("Metric arrays must be non-empty and shape matched")
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise FloatingPointError("Non-finite Bellman-closure metric")
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


def residual_metrics(residual):
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    if not len(residual) or not np.isfinite(residual).all():
        raise FloatingPointError("Invalid Bellman-closure residual")
    return {
        "count": int(len(residual)),
        "signed_mean": float(np.mean(residual)),
        "abs_mean": float(np.mean(np.abs(residual))),
        "abs_p50": float(np.percentile(np.abs(residual), 50)),
        "abs_p95": float(np.percentile(np.abs(residual), 95)),
        "abs_max": float(np.max(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "std": float(np.std(residual)),
    }


def error_pair_metrics(current_error, successor_error, gamma):
    current_error = np.asarray(
        current_error, dtype=np.float64).reshape(-1)
    successor_error = np.asarray(
        successor_error, dtype=np.float64).reshape(-1)
    if current_error.shape != successor_error.shape or not len(current_error):
        raise ValueError("Error-pair arrays must be shape matched")
    spearman, pearson = correlation(current_error, successor_error)
    predicted_closure = current_error - float(gamma) * successor_error
    return {
        "current_error_mean": float(np.mean(current_error)),
        "current_error_mae": float(np.mean(np.abs(current_error))),
        "successor_error_mean": float(np.mean(successor_error)),
        "successor_error_mae": float(np.mean(np.abs(successor_error))),
        "current_successor_error_spearman": float(spearman),
        "current_successor_error_pearson": float(pearson),
        "predicted_closure_abs_mean": float(
            np.mean(np.abs(predicted_closure))),
        "predicted_closure_signed_mean": float(
            np.mean(predicted_closure)),
    }


def bin_mask(steps, lower, upper):
    mask = steps >= int(lower)
    if upper is not None:
        mask &= steps < int(upper)
    return mask


@torch.no_grad()
def evaluate_stage2_closure(
    critic,
    episodes,
    items_by_length,
    mc_returns,
    device,
    batch_size,
    gamma,
    horizon,
    context_length,
):
    critic.eval()
    critic.requires_grad_(False)

    outputs = {
        "q1_current": [],
        "q2_current": [],
        "qmean_current": [],
        "qmin_current": [],
        "q1_next": [],
        "q2_next": [],
        "qmean_next": [],
        "qmin_next": [],
        "mc_current": [],
        "mc_next": [],
        "reward": [],
        "terminal": [],
        "episode_step": [],
        "context_length": [],
    }

    for length in range(1, context_length + 1):
        references = items_by_length[length]
        for first in range(0, len(references), int(batch_size)):
            selected = references[first:first + int(batch_size)]
            batch = make_batch(
                episodes, selected, length, mc_returns)
            mc_next = next_mc_values(
                episodes, selected, mc_returns)

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

            current_context = encode_replay_contexts(
                critic, obs, actions, steps, horizon)
            current_final = (
                current_context[0][:, -1],
                current_context[1][:, -1],
            )
            q1_current, q2_current = critic.q_from_context(
                current_final, current_action)
            q1_current = q1_current.reshape(-1)
            q2_current = q2_current.reshape(-1)

            successor_context = encode_replay_contexts(
                critic,
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
            q1_next, q2_next = critic.q_from_context(
                successor_final, next_action)
            q1_next = q1_next.reshape(-1)
            q2_next = q2_next.reshape(-1)

            outputs["q1_current"].append(
                q1_current.cpu().numpy())
            outputs["q2_current"].append(
                q2_current.cpu().numpy())
            outputs["qmean_current"].append(
                (0.5 * (q1_current + q2_current)).cpu().numpy())
            outputs["qmin_current"].append(
                torch.minimum(q1_current, q2_current).cpu().numpy())
            outputs["q1_next"].append(q1_next.cpu().numpy())
            outputs["q2_next"].append(q2_next.cpu().numpy())
            outputs["qmean_next"].append(
                (0.5 * (q1_next + q2_next)).cpu().numpy())
            outputs["qmin_next"].append(
                torch.minimum(q1_next, q2_next).cpu().numpy())
            outputs["mc_current"].append(batch["mc_returns"])
            outputs["mc_next"].append(mc_next)
            outputs["reward"].append(batch["rewards"])
            outputs["terminal"].append(batch["terminals"])
            outputs["episode_step"].append(
                np.asarray(batch["episode_steps"][:, -1], dtype=np.int64))
            outputs["context_length"].append(
                np.full(len(selected), length, dtype=np.int64))

    outputs = {
        key: np.concatenate(value).reshape(-1)
        for key, value in outputs.items()
    }

    bootstrap_mask = 1.0 - outputs["terminal"]
    mc_identity_td = (
        outputs["reward"]
        + float(gamma) * bootstrap_mask * outputs["mc_next"]
    )
    mc_identity_residual = outputs["mc_current"] - mc_identity_td

    nonterminal = outputs["terminal"] < 0.5
    full_horizon_nonterminal = (
        nonterminal
        & (outputs["context_length"] == int(context_length))
    )
    if not np.any(full_horizon_nonterminal):
        raise RuntimeError(
            "Canonical replay has no full-horizon non-terminal transitions")

    estimators = ("q1", "q2", "qmean", "qmin")

    def subset_metrics(mask):
        result = {
            "count": int(np.sum(mask)),
            "episode_step_min": int(outputs["episode_step"][mask].min()),
            "episode_step_max": int(outputs["episode_step"][mask].max()),
            "mc_identity": {
                "target_vs_mc": metric_against_reference(
                    mc_identity_td[mask],
                    outputs["mc_current"][mask],
                ),
                "residual": residual_metrics(
                    mc_identity_residual[mask]),
            },
            "estimators": {},
        }

        for estimator in estimators:
            current = outputs[f"{estimator}_current"][mask]
            successor = outputs[f"{estimator}_next"][mask]
            mc_current = outputs["mc_current"][mask]
            mc_next = outputs["mc_next"][mask]
            reward = outputs["reward"][mask]

            self_bootstrap = (
                reward + float(gamma) * successor)
            closure = current - self_bootstrap
            current_error = current - mc_current
            successor_error = successor - mc_next
            algebraic_prediction = (
                current_error
                - float(gamma) * successor_error
            )

            result["estimators"][estimator] = {
                "current_q_vs_mc": metric_against_reference(
                    current, mc_current),
                "successor_q_vs_mc_next": metric_against_reference(
                    successor, mc_next),
                "self_bootstrap_target_vs_mc": metric_against_reference(
                    self_bootstrap, mc_current),
                "bellman_closure_residual": residual_metrics(
                    closure),
                "mc_error_pair": error_pair_metrics(
                    current_error, successor_error, gamma),
                "closure_reconstruction_max_abs": float(
                    np.max(np.abs(
                        closure - algebraic_prediction))),
            }
        return result

    primary = subset_metrics(full_horizon_nonterminal)

    progress = {}
    full_steps = outputs["episode_step"]
    for label, lower, upper in PROGRESS_BINS:
        mask = (
            full_horizon_nonterminal
            & bin_mask(full_steps, lower, upper)
        )
        if np.any(mask):
            progress[label] = {
                "lower_inclusive": int(lower),
                "upper_exclusive": (
                    None if upper is None else int(upper)),
                **subset_metrics(mask),
            }

    return {
        "transition_count": int(len(outputs["mc_current"])),
        "nonterminal_transition_count": int(np.sum(nonterminal)),
        "full_horizon_nonterminal_transition_count": int(
            np.sum(full_horizon_nonterminal)),
        "mc_identity_all_transitions": {
            "target_vs_mc": metric_against_reference(
                mc_identity_td, outputs["mc_current"]),
            "residual": residual_metrics(
                mc_identity_residual),
        },
        "full_horizon_nonterminal_transitions": primary,
        "progress_bins": progress,
    }


def main():
    args = arguments()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

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

    for name, path in (
        ("stage2", stage2_path),
        ("stage3_step0", step0_path),
        ("diagnostic_replay", diagnostic_replay),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    stage2_payload_cpu = torch.load(
        stage2_path, map_location="cpu")
    if stage2_payload_cpu.get("stage_version") != "2.2":
        raise RuntimeError("Expected Stage2.2 checkpoint")
    if stage2_payload_cpu.get("training_target") != (
        "finite_episode_monte_carlo_return_no_bootstrap"
    ):
        raise RuntimeError(
            "Stage2.2 training target contract changed unexpectedly")

    step0_payload = torch.load(step0_path, map_location="cpu")
    if step0_payload.get("stage") != "stage3-v5":
        raise RuntimeError("step0 checkpoint is not Stage3-v5")
    if step0_payload.get("group") != "multi_q":
        raise RuntimeError("step0 checkpoint is not multi_q")
    if int(step0_payload.get("env_steps", -1)) != 0:
        raise RuntimeError("Stage3 step0 has non-zero env_steps")
    if int(step0_payload.get("updates", -1)) != 0:
        raise RuntimeError("Stage3 step0 has Critic updates")
    if int(step0_payload.get("actor_updates", -1)) != 0:
        raise RuntimeError("Stage3 step0 has Actor updates")

    config = step0_payload["config"]
    gamma = float(config["gamma"])
    horizon = int(config["horizon"])
    context_length = int(
        config["recurrent_replay"]["critic_context_length"])
    if context_length != 10:
        raise RuntimeError(
            "Expected audited Stage3 horizon-10 Critic contract")
    if abs(float(stage2_payload_cpu.get("gamma", -1.0)) - gamma) > 1e-12:
        raise RuntimeError("Stage2.2 / Stage3 gamma mismatch")

    fixed, episodes = load_canonical_episodes(
        diagnostic_replay)
    (
        items_by_length,
        mc_returns,
        transition_count,
        nonterminal_count,
    ) = build_index(
        episodes, context_length, gamma)

    device = resolve_device(args.device)
    critic, loaded_stage2_payload = strict_stage2_load(
        stage2_path, device)

    stage2_hash = state_dict_digest(
        critic.state_dict())
    step0_online_hash = state_dict_digest(
        step0_payload["q1_q2"])
    step0_target_hash = state_dict_digest(
        step0_payload["target_q1_q2"])

    sync(device)
    metrics = evaluate_stage2_closure(
        critic,
        episodes,
        items_by_length,
        mc_returns,
        device,
        int(args.batch_size),
        gamma,
        horizon,
        context_length,
    )
    sync(device)

    primary = metrics[
        "full_horizon_nonterminal_transitions"]
    max_reconstruction = max(
        primary["estimators"][name][
            "closure_reconstruction_max_abs"]
        for name in ("q1", "q2", "qmean", "qmin")
    )

    validity = {
        "stage2_mc_no_bootstrap_contract": bool(
            loaded_stage2_payload.get("training_target")
            == "finite_episode_monte_carlo_return_no_bootstrap"),
        "stage3_step0_is_zero_update": bool(
            int(step0_payload.get("env_steps", -1)) == 0
            and int(step0_payload.get("updates", -1)) == 0
            and int(step0_payload.get("actor_updates", -1)) == 0),
        "stage2_equals_step0_online_critic": bool(
            stage2_hash == step0_online_hash),
        "stage2_equals_step0_target_critic": bool(
            stage2_hash == step0_target_hash),
        "mc_identity_max_abs_le_2e_6": bool(
            metrics["mc_identity_all_transitions"][
                "residual"]["abs_max"] <= 2e-6),
        "closure_algebra_reconstruction_max_abs_le_2e_6": bool(
            max_reconstruction <= 2e-6),
        "finite_primary_metrics": bool(
            all(
                np.isfinite(
                    primary["estimators"][estimator][section][metric]
                )
                for estimator in ("q1", "q2", "qmean", "qmin")
                for section, metric in (
                    ("current_q_vs_mc", "mae"),
                    ("successor_q_vs_mc_next", "mae"),
                    ("self_bootstrap_target_vs_mc", "mae"),
                    ("bellman_closure_residual", "abs_mean"),
                )
            )
        ),
    }
    valid = bool(all(validity.values()))

    output = {
        "status": "PASS" if valid else "INVALID",
        "experiment": "stage2_2_bellman_closure",
        "read_only": True,
        "environment_steps_performed": 0,
        "optimizer_steps_performed": 0,
        "actor_updates_performed": 0,
        "critic_updates_performed": 0,
        "training_checkpoints_written": 0,
        "device": str(device),
        "stage2_checkpoint": str(stage2_path),
        "stage2_checkpoint_step": int(
            loaded_stage2_payload.get(
                "checkpoint_step",
                loaded_stage2_payload.get("step", -1),
            )
        ),
        "stage2_training_target": loaded_stage2_payload.get(
            "training_target"),
        "stage3_step0_checkpoint": str(step0_path),
        "stage2_critic_hash": stage2_hash,
        "stage3_step0_online_critic_hash": step0_online_hash,
        "stage3_step0_target_critic_hash": step0_target_hash,
        "canonical_diagnostic_replay": str(
            diagnostic_replay),
        "canonical_seed": fixed.get("seed"),
        "canonical_episode_count": int(len(episodes)),
        "canonical_success_episodes": int(
            fixed.get("success_episode_count", 0)),
        "canonical_failure_episodes": int(
            fixed.get("failure_episode_count", 0)),
        "canonical_transition_count": int(transition_count),
        "canonical_nonterminal_transition_count": int(
            nonterminal_count),
        "gamma": gamma,
        "horizon": horizon,
        "context_length": context_length,
        "history_contract": "sliding_horizon_10_zero_state",
        "primary_analysis_subset": (
            "full_horizon_nonterminal_transitions"
        ),
        "interpretation_boundary": (
            "A non-zero Stage2.2 self-bootstrap residual shows lack of exact "
            "Bellman closure for the learned finite-history function on these "
            "samples, but does not by itself prove non-Markov state aliasing. "
            "The residual is algebraically current MC regression error minus "
            "gamma times successor MC regression error."
        ),
        "validity": validity,
        "metrics": metrics,
    }

    out_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "testing"
        / "stage2_vs_stage3_readiness"
        / "multi_stage2_bellman_closure.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n")

    print(
        f"[CANONICAL] episodes={len(episodes)} "
        f"transitions={transition_count} "
        f"nonterminal={nonterminal_count} "
        f"device={device}"
    )
    print(
        "\nestimator\tQ_vs_MC_spear\tQ_vs_MC_MAE\t"
        "Qnext_vs_Gnext_spear\tQnext_vs_Gnext_MAE\t"
        "bootstrap_vs_MC_spear\tbootstrap_vs_MC_MAE\t"
        "closure_MAE\tclosure_p95\terr_corr"
    )
    for estimator in ("q1", "q2", "qmean", "qmin"):
        row = primary["estimators"][estimator]
        print(
            f"{estimator}\t"
            f"{row['current_q_vs_mc']['spearman']:.6f}\t"
            f"{row['current_q_vs_mc']['mae']:.6f}\t"
            f"{row['successor_q_vs_mc_next']['spearman']:.6f}\t"
            f"{row['successor_q_vs_mc_next']['mae']:.6f}\t"
            f"{row['self_bootstrap_target_vs_mc']['spearman']:.6f}\t"
            f"{row['self_bootstrap_target_vs_mc']['mae']:.6f}\t"
            f"{row['bellman_closure_residual']['abs_mean']:.6f}\t"
            f"{row['bellman_closure_residual']['abs_p95']:.6f}\t"
            f"{row['mc_error_pair']['current_successor_error_pearson']:.6f}"
        )

    print("\n[PROGRESS BINS: QMIN]")
    for label, row in metrics["progress_bins"].items():
        q = row["estimators"]["qmin"]
        print(
            f"{label}\tcount={row['count']}\t"
            f"Q_MC_MAE={q['current_q_vs_mc']['mae']:.6f}\t"
            f"Qnext_MC_MAE={q['successor_q_vs_mc_next']['mae']:.6f}\t"
            f"closure_MAE="
            f"{q['bellman_closure_residual']['abs_mean']:.6f}\t"
            f"closure_p95="
            f"{q['bellman_closure_residual']['abs_p95']:.6f}"
        )

    print("\n[VALIDITY]")
    print(json.dumps(validity, indent=2, sort_keys=True))
    print(f"[STATUS] {output['status']}")
    print(f"[SAVED] {out_path}", flush=True)

    del critic
    cleanup_device(device)

    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
