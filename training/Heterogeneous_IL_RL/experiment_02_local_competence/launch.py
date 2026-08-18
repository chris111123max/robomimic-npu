#!/usr/bin/env python3
"""Unified, resumable launcher for Experiment 02."""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

from utils.result_utils import atomic_json, ensure_dirs, read_json

DEFAULT_CONFIG = EXPERIMENT_DIR / "config" / "experiment_config.json"
STAGES = ("validate", "select", "recheck", "build", "reconstruct", "branch", "analyze", "resume", "all")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def effective_config(args):
    config = read_json(Path(args.config).expanduser().resolve())
    config["source_run"] = str(Path(args.source_run or config["source_run"]).expanduser())
    if args.branch_steps:
        config["branch_steps"] = [int(x) for x in args.branch_steps.split(",") if x]
    if args.branch_repeats is not None:
        config["branch_repeats"] = int(args.branch_repeats)
    if args.recheck_repeats is not None:
        config["recheck_repeats"] = int(args.recheck_repeats)
    if args.num_workers is not None:
        config["num_workers"] = int(args.num_workers)
    steps = config["branch_steps"]
    if not steps or len(set(steps)) != len(steps) or any(int(x) <= 0 or int(x) >= int(config["horizon"]) for x in steps):
        raise ValueError("Branch steps must be unique and strictly between 0 and horizon")
    if int(config["branch_repeats"]) <= 0 or int(config["recheck_repeats"]) <= 0:
        raise ValueError("Repeat counts must be positive")
    if int(config["num_workers"]) <= 0 or int(config["num_workers"]) % 2:
        raise ValueError("num_workers must be a positive even number")
    if int(config["num_workers"]) > len(config["npu_masks"]):
        raise ValueError("num_workers exceeds configured NPU masks")
    if args.max_cases is not None and int(args.max_cases) <= 0:
        raise ValueError("--max-cases must be positive")
    return config


def prepare_run(config, requested):
    run_dir = Path(requested).expanduser().resolve() if requested else \
              Path(config["output_root"]).expanduser() / datetime.now().strftime("%Y%m%d%H%M%S")
    ensure_dirs(run_dir)
    path = run_dir / "config.json"
    if path.exists() and canonical(read_json(path)) != canonical(config):
        raise RuntimeError(f"Existing run config differs from requested effective config: {path}")
    if not path.exists():
        atomic_json(path, config)
    return run_dir, path


def run_script(script, config_path, run_dir, config, log_handle, extra=(), mask=None):
    command = [sys.executable, "-u", str(EXPERIMENT_DIR / "scripts" / script),
               "--config", str(config_path), "--run-dir", str(run_dir), *map(str, extra)]
    environment = os.environ.copy()
    if mask is not None:
        environment["ASCEND_RT_VISIBLE_DEVICES"] = str(mask)
    print("Command:", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=str(EXPERIMENT_DIR), env=environment,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True); log_handle.write(line); log_handle.flush()
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def run_validation_script(config_path, run_dir, config, log_handle):
    """Run validation without PIPE so inherited CANN descriptors cannot stall EOF."""
    command = [
        sys.executable, "-u",
        str(EXPERIMENT_DIR / "scripts" / "validate_inputs.py"),
        "--config", str(config_path),
        "--run-dir", str(run_dir),
        "--source-run", str(config["source_run"]),
    ]
    environment = os.environ.copy()
    environment["ASCEND_RT_VISIBLE_DEVICES"] = str(config["npu_masks"][0])
    validation_log = Path(run_dir) / "logs" / "validation.log"
    print("Command:", " ".join(command), flush=True)
    with validation_log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command, cwd=str(EXPERIMENT_DIR), env=environment,
            stdout=output, stderr=subprocess.STDOUT, text=True,
        )
        code = process.wait()
    text = validation_log.read_text(encoding="utf-8", errors="replace")
    print(text, end="" if text.endswith("\n") else "\n", flush=True)
    log_handle.write(text)
    if text and not text.endswith("\n"):
        log_handle.write("\n")
    log_handle.flush()
    marker = "VALIDATION COMPLETE"
    if code or marker not in text:
        raise RuntimeError(
            f"Validation subprocess failed: return_code={code}, "
            f"success_marker_present={marker in text}; see {validation_log}"
        )


def preflight_devices(config, log_handle):
    masks = [str(x) for x in config["npu_masks"][:int(config["num_workers"])]]
    result = subprocess.run(["npu-smi", "info"], text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=60)
    print(result.stdout, flush=True); log_handle.write(result.stdout); log_handle.flush()
    if result.returncode:
        raise RuntimeError("npu-smi info failed")
    rows = re.findall(r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|",
                      result.stdout, flags=re.MULTILINE)
    if config.get("require_idle_devices", True) and rows:
        raise RuntimeError("NPU cards are not idle: " + "; ".join(f"NPU={a} pid={c} {d.strip()}" for a,b,c,d in rows))
    return masks


def run_sharded_stage_workers(stage, script, config_path, run_dir, config, log_handle,
                              common_extra=(), force=False):
    """Run one independent process per NPU and aggregate only after all workers exit."""
    masks = preflight_devices(config, log_handle)
    count = int(config["num_workers"])
    assignments = [
        {"worker_id": worker_id, "mask": masks[worker_id],
         "shard_index": worker_id, "shard_count": count}
        for worker_id in range(count)
    ]
    assignment_roots = {
        "recheck": run_dir / "recheck/workers",
        "build": run_dir / "source_trajectories/workers",
        "reconstruct": run_dir / "reconstruction/workers",
    }
    assignment_root = assignment_roots[stage]
    assignment_root.mkdir(parents=True, exist_ok=True)
    atomic_json(assignment_root / "worker_assignment.json", assignments)
    print(f"[{stage}] launching {count} NPU workers: {assignments}", flush=True)

    processes, output_handles, tail_threads, stop_events = [], [], [], []
    log_lock = threading.Lock()

    def tail_worker(path, offset, assignment, stop_event):
        with path.open("r", encoding="utf-8", errors="replace") as source:
            source.seek(offset)
            while True:
                line = source.readline()
                if line:
                    tagged = (
                        f"[{stage}-worker-{assignment['worker_id']}|mask-{assignment['mask']}] {line}"
                    )
                    print(tagged, end="", flush=True)
                    with log_lock:
                        log_handle.write(tagged)
                        log_handle.flush()
                    continue
                if stop_event.is_set():
                    break
                stop_event.wait(0.2)

    try:
        for assignment in assignments:
            command = [
                sys.executable, "-u", str(EXPERIMENT_DIR / "scripts" / script),
                "--config", str(config_path), "--run-dir", str(run_dir),
                *map(str, common_extra),
                "--worker-id", str(assignment["worker_id"]),
                "--shard-index", str(assignment["shard_index"]),
                "--shard-count", str(assignment["shard_count"]),
            ]
            if force:
                command.append("--force")
            env = os.environ.copy()
            env["ASCEND_RT_VISIBLE_DEVICES"] = assignment["mask"]
            worker_log_path = run_dir / "logs" / f"{stage}_worker_{assignment['worker_id']}.log"
            offset = worker_log_path.stat().st_size if worker_log_path.exists() else 0
            output = worker_log_path.open("a", encoding="utf-8")
            output_handles.append(output)
            print("Command:", " ".join(command), flush=True)
            process = subprocess.Popen(
                command, cwd=str(EXPERIMENT_DIR), env=env,
                stdout=output, stderr=subprocess.STDOUT, text=True,
            )
            processes.append((assignment, process))
            stop_event = threading.Event()
            stop_events.append(stop_event)
            thread = threading.Thread(
                target=tail_worker,
                args=(worker_log_path, offset, assignment, stop_event),
                daemon=True,
            )
            thread.start()
            tail_threads.append(thread)

        remaining = {process.pid: (assignment, process) for assignment, process in processes}
        failures = []
        while remaining:
            for pid, (assignment, process) in list(remaining.items()):
                code = process.poll()
                if code is None:
                    continue
                remaining.pop(pid)
                if code:
                    failures.append((assignment["worker_id"], code))
            if failures:
                for _, process in remaining.values():
                    if process.poll() is None:
                        process.terminate()
                for _, process in remaining.values():
                    process.wait()
                break
            if remaining:
                time.sleep(0.2)
    except BaseException:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            if process.poll() is None:
                process.wait()
        raise
    finally:
        for output in output_handles:
            output.flush()
            output.close()
        for stop_event in stop_events:
            stop_event.set()
        for thread in tail_threads:
            thread.join(timeout=5)

    if failures:
        raise RuntimeError(f"{stage} worker failures: {failures}")

    aggregate_extra = [*common_extra, "--aggregate-only"]
    if force:
        aggregate_extra.append("--force")
    run_script(script, config_path, run_dir, config, log_handle, extra=aggregate_extra)


def run_branch_workers(config_path, run_dir, config, log_handle, force):
    reconstruction = read_json(run_dir / "reconstruction/reconstruction_summary.json")
    if not reconstruction.get("branch_evaluation_allowed"):
        raise RuntimeError("Reconstruction gate forbids branch evaluation")
    masks = preflight_devices(config, log_handle)
    count = int(config["num_workers"]); per_policy = count // 2
    assignments = []
    policies = ("bc_gmm_rnn", "bc_gmm_transformer")
    for worker_id in range(count):
        policy = policies[worker_id % 2]
        shard_index = worker_id // 2
        assignments.append({"worker_id": worker_id, "policy": policy, "mask": masks[worker_id],
                            "shard_index": shard_index, "shard_count": per_policy})
    atomic_json(run_dir / "branch_results/worker_assignment.json", assignments)
    processes, threads, lock = [], [], threading.Lock()
    def pump(assignment, process):
        path = run_dir / "logs" / f"branch_worker_{assignment['worker_id']}.log"
        with path.open("a", encoding="utf-8") as worker_log:
            for line in process.stdout:
                worker_log.write(line); worker_log.flush()
                tagged = f"[worker-{assignment['worker_id']}|{assignment['policy']}|mask-{assignment['mask']}] {line}"
                print(tagged, end="", flush=True)
                with lock:
                    log_handle.write(tagged); log_handle.flush()
    try:
        for assignment in assignments:
            command = [sys.executable, "-u", str(EXPERIMENT_DIR / "scripts/evaluate_branch_rollouts.py"),
                "--config", str(config_path), "--run-dir", str(run_dir),
                "--branch-steps", ",".join(map(str, config["branch_steps"])),
                "--branch-repeats", str(config["branch_repeats"]),
                "--worker-id", str(assignment["worker_id"]), "--policy", assignment["policy"],
                "--shard-index", str(assignment["shard_index"]), "--shard-count", str(assignment["shard_count"])]
            if force: command.append("--force")
            env = os.environ.copy(); env["ASCEND_RT_VISIBLE_DEVICES"] = assignment["mask"]
            process = subprocess.Popen(command, cwd=str(EXPERIMENT_DIR), env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            processes.append((assignment, process))
            thread = threading.Thread(target=pump, args=(assignment, process)); thread.start(); threads.append(thread)
        failures = [(a["worker_id"], p.wait()) for a, p in processes]
        for thread in threads: thread.join()
    except BaseException:
        for _, process in processes:
            if process.poll() is None: process.terminate()
        raise
    failures = [item for item in failures if item[1]]
    if failures:
        raise RuntimeError(f"Branch worker failures: {failures}")
    run_script("evaluate_branch_rollouts.py", config_path, run_dir, config, log_handle,
               extra=("--branch-steps", ",".join(map(str, config["branch_steps"])),
                      "--branch-repeats", config["branch_repeats"], "--aggregate-only"))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG)); parser.add_argument("--run-dir")
    parser.add_argument("--source-run"); parser.add_argument("--max-cases", type=int)
    parser.add_argument("--branch-repeats", type=int); parser.add_argument("--recheck-repeats", type=int)
    parser.add_argument("--branch-steps"); parser.add_argument("--num-workers", type=int)
    parser.add_argument("--force-recheck", action="store_true")
    parser.add_argument("--force-source-trajectories", action="store_true")
    parser.add_argument("--force-reconstruction", action="store_true")
    parser.add_argument("--force-branch-eval", action="store_true")
    parser.add_argument("--force-analysis", action="store_true")
    args = parser.parse_args(); config = effective_config(args)
    run_dir, config_path = prepare_run(config, args.run_dir)
    print("=" * 84); print("Experiment 02 - Local Competence / Branch Takeover")
    print(f"Stage      : {args.stage}\nRun dir    : {run_dir}\nSource run : {config['source_run']}")
    print(f"Steps      : {config['branch_steps']}\nRecheck K  : {config['recheck_repeats']}\nBranch K   : {config['branch_repeats']}")
    print("=" * 84, flush=True)
    with (run_dir / "logs/experiment.log").open("a", encoding="utf-8") as log:
        if args.stage == "all":
            stages = ("validate", "select", "recheck", "build", "reconstruct", "branch", "analyze")
        elif args.stage == "resume":
            stages = ("recheck", "build", "reconstruct", "branch", "analyze")
        else:
            stages = (args.stage,)
        for stage in stages:
            print(f"\n===== STAGE {stage.upper()} =====", flush=True)
            if stage == "validate":
                run_validation_script(config_path, run_dir, config, log)
            elif stage == "select":
                extra = ["--source-run", config["source_run"]]
                if args.max_cases is not None: extra += ["--max-cases", args.max_cases]
                run_script("select_cross_success_cases.py", config_path, run_dir, config, log, extra)
            elif stage == "recheck":
                extra = ["--source-run", config["source_run"], "--repeats", config["recheck_repeats"]]
                run_sharded_stage_workers(
                    "recheck", "recheck_selected_cases.py", config_path, run_dir, config, log,
                    extra, args.force_recheck,
                )
            elif stage == "build":
                extra = ["--source-run", config["source_run"], "--branch-steps", ",".join(map(str, config["branch_steps"]))]
                run_sharded_stage_workers(
                    "build", "build_source_trajectories.py", config_path, run_dir, config, log,
                    extra, args.force_source_trajectories,
                )
            elif stage == "reconstruct":
                extra = ["--branch-steps", ",".join(map(str, config["branch_steps"]))]
                run_sharded_stage_workers(
                    "reconstruct", "validate_branch_reconstruction.py", config_path, run_dir, config, log,
                    extra, args.force_reconstruction,
                )
            elif stage == "branch":
                run_branch_workers(config_path, run_dir, config, log, args.force_branch_eval)
            elif stage == "analyze":
                if args.force_analysis:
                    print("[analyze] force requested; deterministic aggregate outputs will be replaced", flush=True)
                run_script("analyze_local_competence.py", config_path, run_dir, config, log)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
