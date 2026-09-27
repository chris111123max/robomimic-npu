#!/usr/bin/env python3
"""Read-only decomposition of Stage3 target-Critic bootstrap drift.

For the same canonical replay successors used by the existing Stage3-v5
diagnostics, evaluate the replay next action with the target Critic at:

  * Stage3 step0
  * Stage3 100K
  * Stage3 200K

and compare four successor-value estimators against the empirical continuation
return G_(t+1):

  Q1_target
  Q2_target
  mean(Q1_target, Q2_target)
  min(Q1_target, Q2_target)

The test answers whether the large bootstrap mismatch previously observed is
primarily:

  A) clipped-double-Q pessimism introduced by min(Q1,Q2), or
  B) common drift of the individual target critics themselves.

The primary subset is exactly the full horizon-10, non-terminal transitions.
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


def module_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def next_mc_values(episodes, references, mc_returns):
    result = np.zeros(len(references), dtype=np.float32)
    for row, (episode_index, target) in enumerate(references):
        terminal = episode_terminals(episodes[episode_index])
        if terminal[target] >= 0.5:
            continue
        if target + 1 >= len(mc_returns[episode_index]):
            raise RuntimeError(
                "Non-terminal transition has no successor Monte-Carlo return")
        result[row] = mc_returns[episode_index][target + 1]
    return result


def value_metrics(values, reference):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    if values.shape != reference.shape or not len(values):
        raise ValueError("Metric arrays must be non-empty and shape matched")
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise FloatingPointError("Non-finite successor-value diagnostic")
    spearman, pearson = correlation(values, reference)
    error = values - reference
    return {
        "count": int(len(values)),
        "spearman_vs_mc_next": float(spearman),
        "pearson_vs_mc_next": float(pearson),
        "mae_vs_mc_next": float(np.mean(np.abs(error))),
        "rmse_vs_mc_next": float(np.sqrt(np.mean(np.square(error)))),
        "signed_mean_error_vs_mc_next": float(np.mean(error)),
        "value_mean": float(np.mean(values)),
        "value_std": float(np.std(values)),
        "mc_next_mean": float(np.mean(reference)),
        "mc_next_std": float(np.std(reference)),
    }


def twin_metrics(q1, q2, qmean, qmin):
    q1 = np.asarray(q1, dtype=np.float64).reshape(-1)
    q2 = np.asarray(q2, dtype=np.float64).reshape(-1)
    qmean = np.asarray(qmean, dtype=np.float64).reshape(-1)
    qmin = np.asarray(qmin, dtype=np.float64).reshape(-1)
    if not (q1.shape == q2.shape == qmean.shape == qmin.shape):
        raise ValueError("Twin arrays must be shape matched")

    disagreement = np.abs(q1 - q2)
    min_minus_mean = qmin - qmean
    return {
        "q1_less_than_q2_fraction": float(np.mean(q1 < q2)),
        "q2_less_than_q1_fraction": float(np.mean(q2 < q1)),
        "exact_equal_fraction": float(np.mean(q1 == q2)),
        "absolute_twin_gap_mean": float(np.mean(disagreement)),
        "absolute_twin_gap_p95": float(np.percentile(disagreement, 95)),
        "absolute_twin_gap_max": float(np.max(disagreement)),
        "min_minus_mean_signed_mean": float(np.mean(min_minus_mean)),
        "mean_minus_min_mean": float(np.mean(qmean - qmin)),
        "mean_minus_min_p95": float(np.percentile(qmean - qmin, 95)),
        "mean_minus_min_max": float(np.max(qmean - qmin)),
    }


@torch.no_grad()
def evaluate_checkpoint(
    payload,
    target_critic,
    episodes,
    items_by_length,
    mc_returns,
    device,
    batch_size,
):
    config = payload["config"]
    horizon = int(config["horizon"])
    context_length = int(config["recurrent_replay"]["critic_context_length"])
    if context_length != 10:
        raise RuntimeError("Expected audited horizon-10 Stage3 contract")

    target_critic.eval()
    target_critic.requires_grad_(False)

    outputs = {
        "q1": [],
        "q2": [],
        "qmean": [],
        "qmin": [],
        "mc_next": [],
        "terminal": [],
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
            next_action = torch.as_tensor(
                batch["next_actions"], dtype=torch.float32, device=device)

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
            q1, q2 = target_critic.q_from_context(
                successor_final, next_action)
            q1 = q1.reshape(-1)
            q2 = q2.reshape(-1)
            qmean = 0.5 * (q1 + q2)
            qmin = torch.minimum(q1, q2)

            outputs["q1"].append(q1.cpu().numpy())
            outputs["q2"].append(q2.cpu().numpy())
            outputs["qmean"].append(qmean.cpu().numpy())
            outputs["qmin"].append(qmin.cpu().numpy())
            outputs["mc_next"].append(mc_next)
            outputs["terminal"].append(batch["terminals"])
            outputs["context_length"].append(
                np.full(len(selected), length, dtype=np.int64))

    outputs = {
        key: np.concatenate(value).reshape(-1)
        for key, value in outputs.items()
    }

    nonterminal = outputs["terminal"] < 0.5
    full_horizon_nonterminal = (
        nonterminal
        & (outputs["context_length"] == context_length)
    )
    if not np.any(full_horizon_nonterminal):
        raise RuntimeError(
            "Canonical replay has no full-horizon non-terminal transitions")

    def subset_metrics(mask):
        q1 = outputs["q1"][mask]
        q2 = outputs["q2"][mask]
        qmean = outputs["qmean"][mask]
        qmin = outputs["qmin"][mask]
        mc_next = outputs["mc_next"][mask]

        metrics = {
            "count": int(np.sum(mask)),
            "q1": value_metrics(q1, mc_next),
            "q2": value_metrics(q2, mc_next),
            "qmean": value_metrics(qmean, mc_next),
            "qmin": value_metrics(qmin, mc_next),
            "twin": twin_metrics(q1, q2, qmean, qmin),
        }
        metrics["min_penalty_relative_to_mean"] = {
            "spearman_change_min_minus_mean": float(
                metrics["qmin"]["spearman_vs_mc_next"]
                - metrics["qmean"]["spearman_vs_mc_next"]
            ),
            "pearson_change_min_minus_mean": float(
                metrics["qmin"]["pearson_vs_mc_next"]
                - metrics["qmean"]["pearson_vs_mc_next"]
            ),
            "mae_change_min_minus_mean": float(
                metrics["qmin"]["mae_vs_mc_next"]
                - metrics["qmean"]["mae_vs_mc_next"]
            ),
            "signed_bias_change_min_minus_mean": float(
                metrics["qmin"]["signed_mean_error_vs_mc_next"]
                - metrics["qmean"]["signed_mean_error_vs_mc_next"]
            ),
        }
        return metrics

    return {
        "transition_count": int(len(outputs["mc_next"])),
        "nonterminal_transition_count": int(np.sum(nonterminal)),
        "full_horizon_nonterminal_transition_count": int(
            np.sum(full_horizon_nonterminal)),
        "nonterminal_transitions": subset_metrics(nonterminal),
        "full_horizon_nonterminal_transitions": subset_metrics(
            full_horizon_nonterminal),
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

    config = reference_payload["config"]
    gamma = float(config["gamma"])
    if float(stage2_payload.get("gamma", -1.0)) != gamma:
        raise RuntimeError("Stage2.2 / Stage3 gamma mismatch")
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
    ) = build_index(
        episodes, context_length, gamma)

    device = resolve_device(args.device)
    order = ("stage3_step0", "stage3_100k", "stage3_200k")
    results = {}

    # Reference digest verifies that Stage3 step0 target Critic is exactly the
    # audited Stage2.2 Critic before any Stage3 update.
    stage2_critic, _ = strict_stage2_load(stage2_path, device)
    stage2_digest = module_digest(stage2_critic)
    del stage2_critic
    cleanup_device(device)

    for name in order:
        payload = payloads[name]
        target_critic, _ = strict_stage2_load(
            stage2_path, device)
        target_critic.load_state_dict(
            payload["target_q1_q2"], strict=True)
        target_digest = module_digest(target_critic)

        sync(device)
        metrics = evaluate_checkpoint(
            payload,
            target_critic,
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
            "target_critic_hash": target_digest,
            "metrics": metrics,
        }

        del target_critic
        cleanup_device(device)

    primary = "full_horizon_nonterminal_transitions"

    validity = {
        "actor_updates_zero_all": all(
            results[name]["actor_updates"] == 0 for name in order),
        "stage3_step0_is_zero_update": bool(
            results["stage3_step0"]["env_steps"] == 0
            and results["stage3_step0"]["critic_updates"] == 0),
        "stage3_step0_target_equals_stage2_2": bool(
            results["stage3_step0"]["target_critic_hash"]
            == stage2_digest),
        "same_primary_count_all": (
            len({
                results[name]["metrics"][primary]["count"]
                for name in order
            }) == 1
        ),
        "finite_metrics_all": all(
            np.isfinite(
                results[name]["metrics"][primary][estimator][metric]
            )
            for name in order
            for estimator in ("q1", "q2", "qmean", "qmin")
            for metric in (
                "spearman_vs_mc_next",
                "pearson_vs_mc_next",
                "mae_vs_mc_next",
                "signed_mean_error_vs_mc_next",
            )
        ),
    }
    valid = bool(all(validity.values()))

    deltas = {}
    for left, right in zip(order, order[1:]):
        lm = results[left]["metrics"][primary]
        rm = results[right]["metrics"][primary]
        row = {}
        for estimator in ("q1", "q2", "qmean", "qmin"):
            row[f"{estimator}_spearman_change"] = float(
                rm[estimator]["spearman_vs_mc_next"]
                - lm[estimator]["spearman_vs_mc_next"])
            row[f"{estimator}_mae_change"] = float(
                rm[estimator]["mae_vs_mc_next"]
                - lm[estimator]["mae_vs_mc_next"])
            row[f"{estimator}_signed_bias_change"] = float(
                rm[estimator]["signed_mean_error_vs_mc_next"]
                - lm[estimator]["signed_mean_error_vs_mc_next"])
        row["mean_minus_min_mean_change"] = float(
            rm["twin"]["mean_minus_min_mean"]
            - lm["twin"]["mean_minus_min_mean"])
        row["min_vs_mean_mae_penalty_change"] = float(
            rm["min_penalty_relative_to_mean"][
                "mae_change_min_minus_mean"]
            - lm["min_penalty_relative_to_mean"][
                "mae_change_min_minus_mean"])
        deltas[f"{left}_to_{right}"] = row

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
        "stage2_target_critic_hash": stage2_digest,
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
        "primary_analysis_subset": primary,
        "reference_value": "empirical finite-episode G_(t+1)",
        "evaluated_action": "replay executed action_(t+1)",
        "estimators": [
            "Q1_target",
            "Q2_target",
            "mean(Q1_target,Q2_target)",
            "min(Q1_target,Q2_target)",
        ],
        "validity": validity,
        "results": results,
        "deltas": deltas,
    }

    out_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "testing"
        / "stage2_vs_stage3_readiness"
        / "multi_twin_bootstrap_decomposition.json"
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
        "\ncheckpoint\tupdates\t"
        "q1_spear\tq2_spear\tmean_spear\tmin_spear\t"
        "q1_mae\tq2_mae\tmean_mae\tmin_mae\t"
        "q1_bias\tq2_bias\tmean_bias\tmin_bias\t"
        "mean_minus_min"
    )
    for name in order:
        row = results[name]
        m = row["metrics"][primary]
        print(
            f"{name}\t{row['critic_updates']}\t"
            f"{m['q1']['spearman_vs_mc_next']:.6f}\t"
            f"{m['q2']['spearman_vs_mc_next']:.6f}\t"
            f"{m['qmean']['spearman_vs_mc_next']:.6f}\t"
            f"{m['qmin']['spearman_vs_mc_next']:.6f}\t"
            f"{m['q1']['mae_vs_mc_next']:.6f}\t"
            f"{m['q2']['mae_vs_mc_next']:.6f}\t"
            f"{m['qmean']['mae_vs_mc_next']:.6f}\t"
            f"{m['qmin']['mae_vs_mc_next']:.6f}\t"
            f"{m['q1']['signed_mean_error_vs_mc_next']:.6f}\t"
            f"{m['q2']['signed_mean_error_vs_mc_next']:.6f}\t"
            f"{m['qmean']['signed_mean_error_vs_mc_next']:.6f}\t"
            f"{m['qmin']['signed_mean_error_vs_mc_next']:.6f}\t"
            f"{m['twin']['mean_minus_min_mean']:.6f}"
        )

    print("\n[MIN PENALTY RELATIVE TO MEAN]")
    for name in order:
        p = results[name]["metrics"][primary][
            "min_penalty_relative_to_mean"]
        print(
            f"{name}: "
            f"dSpearman={p['spearman_change_min_minus_mean']:+.6f} "
            f"dMAE={p['mae_change_min_minus_mean']:+.6f} "
            f"dBias={p['signed_bias_change_min_minus_mean']:+.6f}"
        )

    print("\n[VALIDITY]")
    print(json.dumps(validity, indent=2, sort_keys=True))
    print(f"[STATUS] {output['status']}")
    print(f"[SAVED] {out_path}", flush=True)

    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
