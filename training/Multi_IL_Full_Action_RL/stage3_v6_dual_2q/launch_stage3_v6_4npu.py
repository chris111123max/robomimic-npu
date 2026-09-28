#!/usr/bin/env python3
"""Launch the four formal Stage3-v6 runs on the fixed NPU0..3 mapping."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNS = (
    ("mean2q", "multi_q", "npu:0"),
    ("mean2q", "rnn_q", "npu:1"),
    ("random2q", "multi_q", "npu:2"),
    ("random2q", "rnn_q", "npu:3"),
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quad-run-dir", required=True)
    parser.add_argument("--total-env-steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--compile-backend", choices=("none", "torchair"))
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main():
    args = arguments()
    quad = Path(args.quad_run_dir).resolve()
    fairness = read_json(quad / "shared" / "quad_fairness.json")
    sources = read_json(quad / "shared" / "stage2_source_manifest.json")

    expected = {
        f"{mode}/{group}": device for mode, group, device in RUNS
    }
    if fairness.get("npu_mapping") != expected:
        raise RuntimeError(
            f"prepared NPU mapping differs: {fairness.get('npu_mapping')} != {expected}"
        )

    if args.smoke:
        if args.num_envs is None:
            args.num_envs = 2
        if args.total_env_steps is None:
            args.total_env_steps = 12000
    else:
        if args.num_envs not in (None, 16):
            raise RuntimeError("Formal Stage3-v6 requires 16 envs per run")
        args.num_envs = 16

    log_dir = quad / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    manifest = {
        "status": "STARTING",
        "quad_run_dir": str(quad),
        "smoke": bool(args.smoke),
        "runs": {},
    }

    for mode, group, device in RUNS:
        key = f"{mode}/{group}"
        group_dir = quad / mode / group
        if not args.smoke and (group_dir / "checkpoints" / "step0_transfer.pth").exists():
            raise RuntimeError(
                f"{key}: formal output already exists; refusing accidental overwrite"
            )

        command = [
            sys.executable,
            str(HERE / "train_stage3_v6_vector.py"),
            "--group", group,
            "--target-mode", mode,
            "--device", device,
            "--quad-run-dir", str(quad),
            "--critic-init-checkpoint", sources[group]["checkpoint"],
            "--num-envs", str(args.num_envs),
        ]
        if args.total_env_steps is not None:
            command += ["--total-env-steps", str(args.total_env_steps)]
        if args.smoke:
            command.append("--smoke")
        if args.compile_backend:
            command += ["--compile-backend", args.compile_backend]
        if args.no_prefetch:
            command.append("--no-prefetch")
        if args.profile:
            command.append("--profile")

        log_path = log_dir / f"{mode}_{group}_{device.replace(':', '')}.log"
        log_handle = open(log_path, "w", encoding="utf-8", buffering=1)
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=str(HERE.parents[3]),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        processes.append((key, device, process, log_handle, log_path, command))
        manifest["runs"][key] = {
            "device": device,
            "pid": int(process.pid),
            "log": str(log_path),
            "command": command,
        }
        print(
            f"[STAGE3-V6] started {key} on {device} pid={process.pid} "
            f"log={log_path}",
            flush=True,
        )

    manifest["status"] = "RUNNING"
    manifest_path = log_dir / "launcher_manifest.json"
    write_json(manifest_path, manifest)

    stopping = False

    def stop(signum, _frame):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print(f"[STAGE3-V6] launcher signal {signum}; terminating children", flush=True)
        for _, _, process, _, _, _ in processes:
            if process.poll() is None:
                process.terminate()

    old_int = signal.signal(signal.SIGINT, stop)
    old_term = signal.signal(signal.SIGTERM, stop)
    try:
        while any(process.poll() is None for _, _, process, _, _, _ in processes):
            time.sleep(5)
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        for _, _, process, handle, _, _ in processes:
            if process.poll() is None:
                process.terminate()
            handle.close()

    return_codes = {
        key: int(process.returncode)
        for key, _, process, _, _, _ in processes
    }
    manifest["return_codes"] = return_codes
    manifest["status"] = (
        "COMPLETE" if all(code == 0 for code in return_codes.values())
        else "PARTIAL_FAILURE"
    )
    write_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "return_codes": return_codes,
        "manifest": str(manifest_path),
    }, indent=2), flush=True)
    if manifest["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
