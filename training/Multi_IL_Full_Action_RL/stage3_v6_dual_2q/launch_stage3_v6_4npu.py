#!/usr/bin/env python3
"""Launch Stage3-v6 in two startup waves on the fixed NPU0..3 mapping.

Wave 1 starts both Multi runs together:
  npu:0 mean2q/multi_q
  npu:2 random2q/multi_q

Only after BOTH Multi trainers write an explicit startup-ready marker proving
that all requested simulator envs and the trainer runtime are initialized does
Wave 2 start both RNN runs together:
  npu:1 mean2q/rnn_q
  npu:3 random2q/rnn_q

This prevents all four trainers from creating simulator environments at once.
"""
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
WAVES = (
    (
        ("mean2q", "multi_q", "npu:0"),
        ("random2q", "multi_q", "npu:2"),
    ),
    (
        ("mean2q", "rnn_q", "npu:1"),
        ("random2q", "rnn_q", "npu:3"),
    ),
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
    parser.add_argument(
        "--startup-wave-timeout-sec",
        type=float,
        default=1200.0,
        help=(
            "Maximum time to wait for both runs in one startup wave to write "
            "READY markers. Wave 2 is never launched before Wave 1 is ready."
        ),
    )
    return parser.parse_args()


def validate_ready_marker(path, mode, group, device, num_envs):
    payload = read_json(path)
    expected = {
        "status": "READY",
        "stage": "stage3-v6",
        "target_mode": mode,
        "group": group,
        "device": device,
        "num_envs": int(num_envs),
        "all_vector_envs_initialized": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"{path}: startup-ready contract mismatch for {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )
    if int(payload.get("vector_env_initial_observation_count", -1)) != int(num_envs):
        raise RuntimeError(f"{path}: not all vector envs initialized")
    if int(payload.get("startup_parallelism", -1)) < 1:
        raise RuntimeError(f"{path}: invalid startup_parallelism")
    return payload


def main():
    args = arguments()
    if args.startup_wave_timeout_sec <= 0:
        raise ValueError("--startup-wave-timeout-sec must be positive")

    quad = Path(args.quad_run_dir).resolve()
    fairness = read_json(quad / "shared" / "quad_fairness.json")
    sources = read_json(quad / "shared" / "stage2_source_manifest.json")

    expected = {
        f"{mode}/{group}": device for mode, group, device in RUNS
    }
    if fairness.get("npu_mapping") != expected:
        raise RuntimeError(
            f"prepared NPU mapping differs: "
            f"{fairness.get('npu_mapping')} != {expected}"
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
    marker_dir = log_dir / "startup_markers"
    log_dir.mkdir(parents=True, exist_ok=True)
    marker_dir.mkdir(parents=True, exist_ok=True)

    launch_nonce = str(time.time_ns())
    processes = []
    manifest = {
        "status": "STARTING",
        "quad_run_dir": str(quad),
        "smoke": bool(args.smoke),
        "startup_policy": (
            "wave1 multi pair -> wait both READY -> wave2 rnn pair"
        ),
        "waves": [
            ["mean2q/multi_q", "random2q/multi_q"],
            ["mean2q/rnn_q", "random2q/rnn_q"],
        ],
        "runs": {},
    }
    manifest_path = log_dir / "launcher_manifest.json"
    stopping = False

    def stop(signum, _frame):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print(
            f"[STAGE3-V6] launcher signal {signum}; terminating children",
            flush=True,
        )
        for item in processes:
            process = item["process"]
            if process.poll() is None:
                process.terminate()

    def start_run(mode, group, device, wave_index):
        key = f"{mode}/{group}"
        group_dir = quad / mode / group
        if (
            not args.smoke
            and (group_dir / "checkpoints" / "step0_transfer.pth").exists()
        ):
            raise RuntimeError(
                f"{key}: formal output already exists; refusing overwrite"
            )

        marker = (
            marker_dir
            / f"{launch_nonce}_wave{wave_index}_{mode}_{group}_{device.replace(':', '')}.json"
        )
        if marker.exists():
            marker.unlink()

        command = [
            sys.executable,
            str(HERE / "train_stage3_v6_vector.py"),
            "--group", group,
            "--target-mode", mode,
            "--device", device,
            "--quad-run-dir", str(quad),
            "--critic-init-checkpoint", sources[group]["checkpoint"],
            "--num-envs", str(args.num_envs),
            "--startup-ready-file", str(marker),
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

        log_path = (
            log_dir / f"{mode}_{group}_{device.replace(':', '')}.log"
        )
        log_handle = open(log_path, "w", encoding="utf-8", buffering=1)
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            command,
            cwd=str(HERE.parents[2]),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
        item = {
            "key": key,
            "mode": mode,
            "group": group,
            "device": device,
            "wave": int(wave_index),
            "process": process,
            "log_handle": log_handle,
            "log_path": log_path,
            "ready_marker": marker,
            "command": command,
        }
        processes.append(item)
        manifest["runs"][key] = {
            "device": device,
            "wave": int(wave_index),
            "pid": int(process.pid),
            "log": str(log_path),
            "startup_ready_marker": str(marker),
            "startup_ready": False,
            "command": command,
        }
        write_json(manifest_path, manifest)
        print(
            f"[STAGE3-V6] wave={wave_index} started {key} on {device} "
            f"pid={process.pid} log={log_path}",
            flush=True,
        )
        return item

    def wait_wave_ready(items, wave_index):
        deadline = time.monotonic() + float(args.startup_wave_timeout_sec)
        pending = {item["key"]: item for item in items}
        while pending:
            if stopping:
                raise RuntimeError("launcher stopping during startup wave")
            for key, item in list(pending.items()):
                process = item["process"]
                if process.poll() is not None:
                    raise RuntimeError(
                        f"wave {wave_index} {key} exited with "
                        f"return code {process.returncode} before READY; "
                        f"see {item['log_path']}"
                    )
                marker = item["ready_marker"]
                if marker.is_file():
                    ready = validate_ready_marker(
                        marker,
                        item["mode"],
                        item["group"],
                        item["device"],
                        args.num_envs,
                    )
                    manifest["runs"][key]["startup_ready"] = True
                    manifest["runs"][key]["startup_ready_payload"] = ready
                    del pending[key]
                    print(
                        f"[STAGE3-V6] wave={wave_index} READY {key}: "
                        f"{ready['vector_env_initial_observation_count']} envs "
                        f"initialized with startup_parallelism="
                        f"{ready['startup_parallelism']}",
                        flush=True,
                    )
            write_json(manifest_path, manifest)
            if pending:
                if time.monotonic() >= deadline:
                    names = ", ".join(sorted(pending))
                    raise TimeoutError(
                        f"wave {wave_index} startup timeout waiting for: {names}"
                    )
                time.sleep(2)

        manifest[f"wave_{wave_index}_ready"] = True
        manifest[f"wave_{wave_index}_ready_time_ns"] = time.time_ns()
        write_json(manifest_path, manifest)
        print(
            f"[STAGE3-V6] wave={wave_index} ALL READY",
            flush=True,
        )

    old_int = signal.signal(signal.SIGINT, stop)
    old_term = signal.signal(signal.SIGTERM, stop)
    try:
        # Wave 1: both Multi runs start together.
        wave1 = [
            start_run(mode, group, device, 1)
            for mode, group, device in WAVES[0]
        ]
        manifest["status"] = "WAITING_MULTI_STARTUP"
        write_json(manifest_path, manifest)
        wait_wave_ready(wave1, 1)

        # Only after BOTH Multi runs have fully initialized their vector envs
        # and trainer runtime may the RNN pair begin creating environments.
        manifest["status"] = "STARTING_RNN_AFTER_MULTI_READY"
        write_json(manifest_path, manifest)
        wave2 = [
            start_run(mode, group, device, 2)
            for mode, group, device in WAVES[1]
        ]
        manifest["status"] = "WAITING_RNN_STARTUP"
        write_json(manifest_path, manifest)
        wait_wave_ready(wave2, 2)

        manifest["status"] = "RUNNING"
        write_json(manifest_path, manifest)
        print(
            "[STAGE3-V6] startup barrier complete: both Multi were READY "
            "before either RNN was launched",
            flush=True,
        )

        while any(
            item["process"].poll() is None for item in processes
        ):
            time.sleep(5)
    except BaseException:
        for item in processes:
            if item["process"].poll() is None:
                item["process"].terminate()
        raise
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
        for item in processes:
            process = item["process"]
            if process.poll() is None:
                process.terminate()
            item["log_handle"].close()

    return_codes = {
        item["key"]: int(item["process"].returncode)
        for item in processes
    }
    manifest["return_codes"] = return_codes
    manifest["status"] = (
        "COMPLETE"
        if len(return_codes) == 4 and all(code == 0 for code in return_codes.values())
        else "PARTIAL_FAILURE"
    )
    write_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "startup_policy": manifest["startup_policy"],
        "wave_1_ready": manifest.get("wave_1_ready", False),
        "wave_2_ready": manifest.get("wave_2_ready", False),
        "return_codes": return_codes,
        "manifest": str(manifest_path),
    }, indent=2), flush=True)
    if manifest["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
