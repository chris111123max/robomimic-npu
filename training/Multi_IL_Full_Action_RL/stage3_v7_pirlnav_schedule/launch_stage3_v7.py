#!/usr/bin/env python3
"""Single-NPU Stage3-V7 launcher; finds an existing prepared V6 random2q source.

No simulator, optimizer or training is started until source selection succeeds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V6_CONFIG = HERE.parent / "stage3_v6_dual_2q" / "stage3_v6_config.json"


def select_prepared(explicit=None):
    if explicit:
        pair = Path(explicit).expanduser().resolve()
        if not (pair / "shared" / "config_resolved.json").is_file():
            raise FileNotFoundError("V6 prepared run not found: " + str(pair))
        return pair
    config = json.loads(V6_CONFIG.read_text(encoding="utf-8"))
    root = Path(config["output_root"])
    prepared = sorted(
        (p.parent for p in root.glob("*/shared/quad_fairness.json")
         if (p.parent / "config_resolved.json").is_file()),
        key=lambda p: p.parent.stat().st_mtime, reverse=True)
    # p.parent is 'shared', so map to the experiment directory.
    runs = [x.parent for x in prepared]
    if not runs:
        raise FileNotFoundError(
            "No prepared Stage3-V6 experiment found. Prepare original "
            "BC Actor/Stage2.2 source inputs first; cannot invent checkpoint paths.")
    with_100k = [
        p for p in runs
        if (p / "random2q" / "multi_q" / "checkpoints" /
            "step_0100000.pth").is_file()
        and (p / "random2q" / "multi_q" / "checkpoints" /
             "step_0100000.sequences.npy").is_file()
    ]
    if len(with_100k) > 1:
        raise RuntimeError("Multiple 100K source runs found; pass --v6-run-dir explicitly: "
                           + ", ".join(str(p) for p in with_100k))
    if with_100k:
        return with_100k[0]
    if len(runs) > 1:
        raise RuntimeError("Multiple prepared V6 runs found, no 100K source. "
                           "Pass --v6-run-dir to choose scratch initialization: "
                           + ", ".join(str(p) for p in runs))
    return runs[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v6-run-dir", help="Prepared V6 run containing shared/ manifest")
    p.add_argument("--resume", help="Explicit Stage3-V7 checkpoint for continuation")
    p.add_argument("--total-env-steps", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--no-prefetch", action="store_true")
    args = p.parse_args()

    pair = select_prepared(args.v6_run_dir)
    manifest = json.loads((pair / "shared" / "stage2_source_manifest.json").read_text())
    critic = Path(manifest["multi_q"]["checkpoint"]).resolve()
    if not critic.is_file():
        raise FileNotFoundError("Prepared multi_q Stage2.2 Critic missing: " + str(critic))
    if not (pair / "shared" / "bc_rnn_gmm_source.pth").is_file():
        raise FileNotFoundError("Prepared Stage1 BC Actor missing")
    print("[STAGE3-V7] source=", pair, "device=npu:0", flush=True)
    if args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
    else:
        resume_path = None
    original = list(sys.argv)
    sys.argv = [
        str(HERE / "train_stage3_v7_vector.py"), "--group", "multi_q",
        "--target-mode", "random2q", "--device", "npu:0",
        "--quad-run-dir", str(pair), "--critic-init-checkpoint", str(critic)]
    if resume_path:
        sys.argv += ["--resume", str(resume_path)]
    if args.total_env_steps is not None:
        sys.argv += ["--total-env-steps", str(args.total_env_steps)]
    if args.num_envs is not None:
        sys.argv += ["--num-envs", str(args.num_envs)]
    if args.smoke:
        sys.argv.append("--smoke")
    if args.no_prefetch:
        sys.argv.append("--no-prefetch")
    try:
        from train_stage3_v7_vector import main as run_train
        run_train()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
