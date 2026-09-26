#!/usr/bin/env python3
"""Read-only Multi Critic evolution diagnostic on one canonical fixed episode set.

Compare:
  Stage2.2 selected checkpoint
  Stage3 step0_transfer
  Stage3 100K
  Stage3 200K

All four critics are evaluated on the exact same frozen diagnostic episodes
loaded from a single user-selected Stage3 replay sidecar (default: 200K).
No environment, optimizer, Actor update, or Critic update is performed.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
STAGE3 = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(STAGE3) not in sys.path:
    sys.path.insert(0, str(STAGE3))

from test_stage2_stage3_readiness_compare import evaluate_critic, resolve_device, sync  # noqa: E402
from stage3_v5_agent import strict_stage2_load  # noqa: E402
from stage3_v5_replay import OnlineSequenceReplay  # noqa: E402

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
    parser.add_argument(
        "--diagnostic-replay",
        help="Canonical .sequences.npy. Defaults to Stage3 200K sidecar."
    )
    parser.add_argument("--output")
    return parser.parse_args()


def release(model, device):
    del model
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


def load_stage3_critic(stage2_path, stage3_path, device):
    critic, _ = strict_stage2_load(stage2_path, device)
    payload = torch.load(stage3_path, map_location="cpu")
    if payload.get("stage") != "stage3-v5":
        raise RuntimeError(f"Not a Stage3-v5 checkpoint: {stage3_path}")
    critic.load_state_dict(payload["q1_q2"], strict=True)
    return critic, payload


def main():
    args = arguments()
    run_dir = Path(args.stage3_run_dir).resolve()
    stage2_path = Path(args.stage2_checkpoint).resolve()
    checkpoints = {
        "stage2_2_initial": stage2_path,
        "stage3_step0": run_dir / "multi_q" / "checkpoints" / "step0_transfer.pth",
        "stage3_100k": run_dir / "multi_q" / "checkpoints" / "step_0100000.pth",
        "stage3_200k": run_dir / "multi_q" / "checkpoints" / "step_0200000.pth",
    }
    diagnostic_replay = Path(args.diagnostic_replay).resolve() if args.diagnostic_replay else (
        run_dir / "multi_q" / "checkpoints" / "step_0200000.sequences.npy"
    )

    for name, path in checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    if not diagnostic_replay.exists():
        raise FileNotFoundError(diagnostic_replay)

    device = resolve_device(args.device)
    fixed, episodes = load_canonical_episodes(diagnostic_replay)
    gamma = 0.99

    print(
        f"[CANONICAL SET] replay={diagnostic_replay} episodes={len(episodes)} "
        f"success={fixed.get('success_episode_count')} "
        f"failure={fixed.get('failure_episode_count')} "
        f"seed={fixed.get('seed')}",
        flush=True,
    )

    results = {}

    # Stage2.2 initial.
    critic, payload = strict_stage2_load(stage2_path, device)
    sync(device)
    results["stage2_2_initial"] = {
        "checkpoint": str(stage2_path),
        "checkpoint_step": int(payload.get("checkpoint_step", payload.get("step", -1))),
        "env_steps": 0,
        "critic_updates": 0,
        "metrics": evaluate_critic(critic, episodes, device, gamma),
    }
    sync(device)
    release(critic, device)

    # Stage3 checkpoints, all on the same canonical episodes.
    for name in ("stage3_step0", "stage3_100k", "stage3_200k"):
        path = checkpoints[name]
        critic, payload = load_stage3_critic(stage2_path, path, device)
        sync(device)
        results[name] = {
            "checkpoint": str(path),
            "env_steps": int(payload.get("env_steps", -1)),
            "critic_updates": int(payload.get("updates", -1)),
            "metrics": evaluate_critic(critic, episodes, device, gamma),
        }
        sync(device)
        release(critic, device)

    order = ["stage2_2_initial", "stage3_step0", "stage3_100k", "stage3_200k"]
    keys = (
        "spearman_q_return",
        "pearson_q_return",
        "success_failure_auc",
        "delta_q",
        "qmin_mean",
        "qmin_std",
        "twin_q_disagreement_median",
        "twin_q_disagreement_p95",
    )
    deltas = {}
    for left, right in zip(order, order[1:]):
        deltas[f"{left}_to_{right}"] = {
            key: float(results[right]["metrics"][key] - results[left]["metrics"][key])
            for key in keys
            if results[left]["metrics"].get(key) is not None
            and results[right]["metrics"].get(key) is not None
        }

    output = {
        "status": "PASS",
        "read_only": True,
        "environment_steps_performed": 0,
        "optimizer_steps_performed": 0,
        "history_contract": "sliding_horizon_10_zero_state",
        "canonical_diagnostic_replay": str(diagnostic_replay),
        "canonical_diagnostic_seed": fixed.get("seed"),
        "canonical_episode_count": len(episodes),
        "canonical_success_episodes": fixed.get("success_episode_count"),
        "canonical_failure_episodes": fixed.get("failure_episode_count"),
        "results": results,
        "deltas": deltas,
    }

    out_path = Path(args.output).resolve() if args.output else (
        run_dir / "testing" / "stage2_vs_stage3_readiness" / "multi_critic_evolution.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    # Compact human-readable table.
    print("\nname\tenv_steps\tupdates\tspearman\tpearson\tauc\tqmean\tqstd")
    for name in order:
        row = results[name]
        m = row["metrics"]
        print(
            f"{name}\t{row['env_steps']}\t{row['critic_updates']}\t"
            f"{m['spearman_q_return']:.6f}\t{m['pearson_q_return']:.6f}\t"
            f"{m['success_failure_auc']:.6f}\t{m['qmin_mean']:.6f}\t"
            f"{m['qmin_std']:.6f}"
        )
    print(f"\n[SAVED] {out_path}", flush=True)


if __name__ == "__main__":
    main()
