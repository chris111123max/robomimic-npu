#!/usr/bin/env python3
"""Read-only Stage2.2 -> Stage3-v5 Critic readiness comparison.

This diagnostic never creates an environment and never performs optimizer
updates. It reuses the frozen Stage3-v5 diagnostic episode set stored in the
Stage3 online replay and evaluates:

  1. the selected Stage2.2 horizon-10 Critic checkpoint before any Stage3 TD;
  2. the Critic weights from a Stage3-v5 checkpoint.

Both models see the exact same episodes and the exact same sliding horizon-10
zero-state history encoding. The script is intentionally single-device and
loads the two models sequentially so it is safe to run on a server with one NPU.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STAGE3 = HERE.parent
ROOT = HERE.parents[4]
STAGE2 = ROOT / "training" / "Multi_IL_Full_Action_RL" / "stage2_2_history_aware_critic"
for directory in (STAGE3, STAGE2):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from stage3_v5_history_critic import strict_stage2_load, encode_replay_contexts  # noqa: E402
from stage3_v5_replay import OnlineSequenceReplay  # noqa: E402
from stage3_v5_readiness import auc, correlation, discounted_returns  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-checkpoint", required=True,
                        help="Selected Stage2.2 horizon-10 Critic checkpoint.")
    parser.add_argument("--stage3-checkpoint", required=True,
                        help="Stage3-v5 checkpoint whose replay and current Critic are compared.")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output",
                        help="Optional JSON output path. Default: beside Stage3 checkpoint.")
    return parser.parse_args()


def resolve_device(name):
    if name.startswith("npu"):
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but unavailable")
        torch.npu.set_device(name)
    return torch.device(name)


def sync(device):
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def release(model, device):
    del model
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def q_values_for_episode(critic, episode, device, horizon=700, context_length=10):
    """Exact Stage3-v5 sliding-horizon Critic evaluation on executed actions."""
    observations = np.asarray(episode["observations"], dtype=np.float32)
    actions = np.asarray(episode["actions"], dtype=np.float32)
    episode_steps = np.asarray(
        episode.get("episode_steps", np.arange(len(actions))), dtype=np.int64)
    length = len(actions)
    if observations.shape != (length, 59) or actions.shape != (length, 14):
        raise RuntimeError("Unexpected episode tensor shape")

    # Chunk transitions to bound accelerator memory. Each transition receives
    # its own <=10-step zero-state window, exactly as Stage3 readiness does.
    q1_all, q2_all = [], []
    chunk = 2048
    for first in range(0, length, chunk):
        last = min(length, first + chunk)
        count = last - first
        obs_windows = np.zeros((count, context_length, 59), np.float32)
        action_windows = np.zeros((count, context_length, 14), np.float32)
        step_windows = np.zeros((count, context_length), np.int64)
        final_indices = np.empty(count, np.int64)

        for row, target in enumerate(range(first, last)):
            start = max(0, target - context_length + 1)
            n = target - start + 1
            obs_windows[row, :n] = observations[start:target + 1]
            action_windows[row, :n] = actions[start:target + 1]
            step_windows[row, :n] = episode_steps[start:target + 1]
            final_indices[row] = n - 1

        obs_t = torch.as_tensor(obs_windows, device=device)
        act_t = torch.as_tensor(action_windows, device=device)
        steps_t = torch.as_tensor(step_windows, device=device)
        contexts = encode_replay_contexts(
            critic, obs_t, act_t, steps_t, horizon)
        rows = torch.arange(count, device=device)
        indices = torch.as_tensor(final_indices, device=device)
        final_contexts = (
            contexts[0][rows, indices],
            contexts[1][rows, indices],
        )
        executed = torch.as_tensor(actions[first:last], device=device)
        q1, q2 = critic.q_from_context(final_contexts, executed)
        q1_all.append(q1.reshape(-1).cpu().numpy())
        q2_all.append(q2.reshape(-1).cpu().numpy())

    return np.concatenate(q1_all), np.concatenate(q2_all)


def evaluate_critic(critic, episodes, device, gamma):
    values = []
    returns = []
    labels = []
    episode_q = []
    q1_values = []
    q2_values = []

    critic.eval()
    for episode in episodes:
        q1, q2 = q_values_for_episode(critic, episode, device)
        qmin = np.minimum(q1, q2)
        ret = discounted_returns(episode, gamma)

        q1_values.append(q1)
        q2_values.append(q2)
        values.append(qmin)
        returns.append(ret)
        episode_q.append(float(np.mean(qmin)))
        labels.append(int(bool(episode.get("success", False))))

    q1 = np.concatenate(q1_values)
    q2 = np.concatenate(q2_values)
    qmin = np.concatenate(values)
    returns = np.concatenate(returns)
    labels = np.asarray(labels, dtype=np.int64)
    episode_q = np.asarray(episode_q, dtype=np.float64)
    spearman, pearson = correlation(qmin, returns)

    success_scores = episode_q[labels == 1]
    failure_scores = episode_q[labels == 0]
    disagreement = np.abs(q1 - q2) / (np.abs(q1) + np.abs(q2) + 1e-8)

    return {
        "episodes": int(len(episodes)),
        "success_episodes": int(labels.sum()),
        "failure_episodes": int(len(labels) - labels.sum()),
        "transitions": int(len(qmin)),
        "spearman_q_return": float(spearman),
        "pearson_q_return": float(pearson),
        "success_failure_auc": float(auc(labels, episode_q)),
        "mean_q_success": float(success_scores.mean()) if len(success_scores) else None,
        "mean_q_failure": float(failure_scores.mean()) if len(failure_scores) else None,
        "delta_q": (
            float(success_scores.mean() - failure_scores.mean())
            if len(success_scores) and len(failure_scores) else None
        ),
        "q1_mean": float(q1.mean()),
        "q1_std": float(q1.std()),
        "q1_min": float(q1.min()),
        "q1_max": float(q1.max()),
        "q2_mean": float(q2.mean()),
        "q2_std": float(q2.std()),
        "q2_min": float(q2.min()),
        "q2_max": float(q2.max()),
        "qmin_mean": float(qmin.mean()),
        "qmin_std": float(qmin.std()),
        "qmin_min": float(qmin.min()),
        "qmin_max": float(qmin.max()),
        "twin_q_disagreement_mean": float(disagreement.mean()),
        "twin_q_disagreement_median": float(np.median(disagreement)),
        "twin_q_disagreement_p95": float(np.percentile(disagreement, 95)),
        "finite": bool(
            np.isfinite(q1).all() and np.isfinite(q2).all()
            and np.isfinite(returns).all()
        ),
    }


def load_frozen_diagnostic(stage3_payload):
    replay_path = stage3_payload.get("online_sequence_replay")
    if not replay_path:
        raise RuntimeError("Stage3 checkpoint has no online_sequence_replay path")
    replay_path = Path(replay_path)
    if not replay_path.exists():
        raise FileNotFoundError(
            f"Stage3 online replay is missing: {replay_path}\n"
            "Use a Stage3 checkpoint from the same run directory."
        )

    replay = OnlineSequenceReplay.load(replay_path)
    fixed = replay.fixed_critic_diagnostic_set
    if fixed is None:
        raise RuntimeError(
            "Stage3 replay has no frozen_critic_diagnostic_set. "
            "Use a checkpoint saved after readiness diagnostics became available."
        )
    episodes = fixed.get("episodes", [])
    if not episodes:
        raise RuntimeError("Frozen diagnostic set contains no episodes")
    return replay_path, fixed, episodes


def main():
    args = arguments()
    stage2_path = Path(args.stage2_checkpoint).resolve()
    stage3_path = Path(args.stage3_checkpoint).resolve()
    if not stage2_path.exists():
        raise FileNotFoundError(stage2_path)
    if not stage3_path.exists():
        raise FileNotFoundError(stage3_path)

    device = resolve_device(args.device)
    stage3_payload = torch.load(stage3_path, map_location="cpu")
    if stage3_payload.get("stage") != "stage3-v5":
        raise RuntimeError("Reference checkpoint is not Stage3-v5")

    replay_path, fixed, episodes = load_frozen_diagnostic(stage3_payload)
    config = stage3_payload["config"]
    if int(config["recurrent_replay"]["critic_context_length"]) != 10:
        raise RuntimeError("Reference Stage3 run is not horizon-10 Critic")
    gamma = float(config["gamma"])

    print(
        f"[DIAGNOSTIC] episodes={len(episodes)} "
        f"success={fixed.get('success_episode_count')} "
        f"failure={fixed.get('failure_episode_count')} "
        f"device={device}",
        flush=True,
    )

    # A. Stage2.2 initial checkpoint: no Stage3 TD updates.
    stage2_critic, stage2_payload = strict_stage2_load(stage2_path, device)
    sync(device)
    stage2_metrics = evaluate_critic(stage2_critic, episodes, device, gamma)
    sync(device)
    stage2_step = int(stage2_payload.get(
        "checkpoint_step", stage2_payload.get("step", -1)))
    release(stage2_critic, device)

    # B. Current Stage3 Critic. Rebuild from the exact same Stage2.2
    # architecture, then load only Stage3 Critic weights.
    stage3_critic, _ = strict_stage2_load(stage2_path, device)
    stage3_critic.load_state_dict(stage3_payload["q1_q2"], strict=True)
    sync(device)
    stage3_metrics = evaluate_critic(stage3_critic, episodes, device, gamma)
    sync(device)
    release(stage3_critic, device)

    result = {
        "status": "PASS",
        "read_only": True,
        "optimizer_steps_performed": 0,
        "environment_steps_performed": 0,
        "history_contract": "sliding_horizon_10_zero_state",
        "device": str(device),
        "stage2_checkpoint": str(stage2_path),
        "stage2_checkpoint_step": stage2_step,
        "stage3_checkpoint": str(stage3_path),
        "stage3_env_steps": int(stage3_payload.get("env_steps", -1)),
        "stage3_critic_updates": int(stage3_payload.get("updates", -1)),
        "diagnostic_replay": str(replay_path),
        "diagnostic_seed": fixed.get("seed"),
        "diagnostic_created_from_episode_count": fixed.get(
            "created_from_episode_count"),
        "stage2_initial": stage2_metrics,
        "stage3_current": stage3_metrics,
        "change_stage3_minus_stage2": {
            key: (
                float(stage3_metrics[key] - stage2_metrics[key])
                if isinstance(stage2_metrics.get(key), (int, float))
                and isinstance(stage3_metrics.get(key), (int, float))
                and not isinstance(stage2_metrics.get(key), bool)
                and not isinstance(stage3_metrics.get(key), bool)
                else None
            )
            for key in (
                "spearman_q_return",
                "pearson_q_return",
                "success_failure_auc",
                "delta_q",
                "qmin_mean",
                "qmin_std",
                "twin_q_disagreement_median",
                "twin_q_disagreement_p95",
            )
        },
        "gate_reference": {
            "min_spearman": float(config["critic_readiness"]["min_spearman"]),
            "min_auc": float(config["critic_readiness"]["min_auc"]),
        },
    }

    output = (
        Path(args.output).resolve()
        if args.output
        else stage3_path.parent.parent / "diagnostics"
             / f"stage2_vs_stage3_readiness_{stage2_path.stem}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[SAVED] {output}", flush=True)


if __name__ == "__main__":
    main()
