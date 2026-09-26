#!/usr/bin/env python3
"""Single-NPU launcher for Stage2.2-vs-Stage3 readiness comparisons.

Runs RNN-only first and Multi second on the same accelerator. The selected
Stage2.2 checkpoints are the user-approved defaults. No environment is created
and no training update is performed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPARE = HERE / "test_stage2_stage3_readiness_compare.py"

DEFAULT_RNN_STAGE2 = (
    "/data/home/3220251075/lerobot_workspace/training_runs/"
    "Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/"
    "stage2_2_h10_rnn_20260923_150633/rnn_q/checkpoints/best.pth"
)
DEFAULT_MULTI_STAGE2 = (
    "/data/home/3220251075/lerobot_workspace/training_runs/"
    "Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/"
    "stage2_2_h10_multi_20260923_150749/multi_q/checkpoints/step_00005000.pth"
)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage3-run-dir", required=True,
        help="Stage3-v5 pair run directory containing rnn_q/ and multi_q/."
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--rnn-stage2", default=DEFAULT_RNN_STAGE2)
    parser.add_argument("--multi-stage2", default=DEFAULT_MULTI_STAGE2)
    parser.add_argument(
        "--stage3-checkpoint-name", default="latest.pth",
        help="Checkpoint filename under each group's checkpoints/ directory."
    )
    return parser.parse_args()


def run_one(group, stage2, stage3, device, output):
    command = [
        sys.executable, str(COMPARE),
        "--stage2-checkpoint", str(stage2),
        "--stage3-checkpoint", str(stage3),
        "--device", device,
        "--output", str(output),
    ]
    print(f"[PAIR TEST] START {group}", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)
    print(f"[PAIR TEST] DONE  {group}", flush=True)


def main():
    args = arguments()
    run_dir = Path(args.stage3_run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    selected = {
        "rnn_q": {
            "stage2": Path(args.rnn_stage2).resolve(),
            "stage3": run_dir / "rnn_q" / "checkpoints" / args.stage3_checkpoint_name,
        },
        "multi_q": {
            "stage2": Path(args.multi_stage2).resolve(),
            "stage3": run_dir / "multi_q" / "checkpoints" / args.stage3_checkpoint_name,
        },
    }

    for group, paths in selected.items():
        for kind, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(f"{group} {kind} checkpoint missing: {path}")

    output_dir = run_dir / "testing" / "stage2_vs_stage3_readiness"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Deliberately serial: one NPU only.
    for group in ("rnn_q", "multi_q"):
        paths = selected[group]
        run_one(
            group,
            paths["stage2"],
            paths["stage3"],
            args.device,
            output_dir / f"{group}.json",
        )

    summary = {}
    for group in ("rnn_q", "multi_q"):
        payload = json.loads((output_dir / f"{group}.json").read_text())
        summary[group] = {
            "stage2_checkpoint": payload["stage2_checkpoint"],
            "stage2_checkpoint_step": payload["stage2_checkpoint_step"],
            "stage3_checkpoint": payload["stage3_checkpoint"],
            "stage3_env_steps": payload["stage3_env_steps"],
            "stage3_critic_updates": payload["stage3_critic_updates"],
            "stage2_initial": {
                key: payload["stage2_initial"][key]
                for key in (
                    "spearman_q_return",
                    "pearson_q_return",
                    "success_failure_auc",
                    "qmin_mean",
                    "qmin_std",
                )
            },
            "stage3_current": {
                key: payload["stage3_current"][key]
                for key in (
                    "spearman_q_return",
                    "pearson_q_return",
                    "success_failure_auc",
                    "qmin_mean",
                    "qmin_std",
                )
            },
            "change_stage3_minus_stage2":
                payload["change_stage3_minus_stage2"],
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[PAIR TEST] SUMMARY {summary_path}", flush=True)


if __name__ == "__main__":
    main()
