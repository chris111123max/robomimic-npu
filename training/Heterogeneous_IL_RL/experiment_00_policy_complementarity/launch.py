#!/usr/bin/env python3
"""Unified launcher for build, evaluation, and paired complementarity analysis."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from utils.env_utils import (
    close_environment, create_dataset_environment,
    initialize_observation_utils_from_checkpoint, load_dataset_env_metadata,
)
from utils.policy_loader import checkpoint_metadata, load_policy, release_policy, select_device
from utils.result_utils import POLICY_ORDER, atomic_json, read_json, rebuild_result_tables


DEFAULT_CONFIG = EXPERIMENT_DIR / "config" / "experiment_config.json"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def metadata_difference_paths(left, right, prefix=""):
    """Return human-readable paths that differ between two JSON objects."""
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(metadata_difference_paths(left[key], right[key], path))
        return differences
    return [] if canonical(left) == canonical(right) else [prefix]


def validate_environment_compatibility(policy_name, checkpoint_meta, dataset_meta):
    """Validate policy/environment semantics without requiring byte-identical metadata.

    Checkpoints and datasets can legitimately differ in env_version, rendering
    defaults, or optional kwargs. The common rollout environment always comes
    from the dataset metadata, so only task and action-semantics fields are hard
    compatibility requirements here. Observation and action tensor interfaces
    are checked separately against the instantiated environment.
    """
    if checkpoint_meta.get("env_name") != dataset_meta.get("env_name"):
        raise RuntimeError(
            f"{policy_name} checkpoint belongs to {checkpoint_meta.get('env_name')}, "
            f"not {dataset_meta.get('env_name')}"
        )
    if checkpoint_meta.get("type") != dataset_meta.get("type"):
        raise RuntimeError(
            f"{policy_name} environment type differs: "
            f"checkpoint={checkpoint_meta.get('type')}, dataset={dataset_meta.get('type')}"
        )

    checkpoint_kwargs = checkpoint_meta.get("env_kwargs", {})
    dataset_kwargs = dataset_meta.get("env_kwargs", {})
    semantic_keys = (
        "robots", "env_configuration", "controller_configs",
        "gripper_types", "control_freq",
    )
    for key in semantic_keys:
        if key in checkpoint_kwargs and key in dataset_kwargs:
            if canonical(checkpoint_kwargs[key]) != canonical(dataset_kwargs[key]):
                raise RuntimeError(
                    f"{policy_name} action-semantic metadata differs for env_kwargs.{key}: "
                    f"checkpoint={checkpoint_kwargs[key]!r}, dataset={dataset_kwargs[key]!r}"
                )

    differences = metadata_difference_paths(checkpoint_meta, dataset_meta)
    if differences:
        preview = ", ".join(differences[:20])
        suffix = " ..." if len(differences) > 20 else ""
        print(
            f"  metadata note       : {len(differences)} non-identical field(s): "
            f"{preview}{suffix}"
        )


def prepare_run_directory(config, requested_run_dir):
    if requested_run_dir:
        run_dir = Path(requested_run_dir).expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        run_dir = Path(config["output_root"]).expanduser() / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("logs", "initial_states", "trajectories", "raw_results", "analysis"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    copied_config = run_dir / "config.json"
    if copied_config.exists():
        if canonical(read_json(copied_config)) != canonical(config):
            raise RuntimeError(f"Existing run config differs from requested config: {copied_config}")
    else:
        atomic_json(copied_config, config)
    return run_dir, copied_config


def validate_config(config):
    dataset_path = Path(config["dataset_path"]).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if tuple(config.get("policies", {}).keys()) != POLICY_ORDER:
        raise ValueError(f"policies must appear exactly in this order: {POLICY_ORDER}")
    if int(config["horizon"]) != 700:
        raise ValueError("This experiment requires horizon=700")
    if not bool(config.get("terminate_on_success")):
        raise ValueError("This experiment requires terminate_on_success=true")
    for policy_name in POLICY_ORDER:
        checkpoint = Path(config["policies"][policy_name]["checkpoint_path"]).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found for {policy_name}: {checkpoint}")

    output_root = Path(config["output_root"]).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".write_test_", dir=str(output_root), delete=True):
        pass

    dataset_meta = load_dataset_env_metadata(dataset_path)
    if dataset_meta["env_name"] != config["expected_environment_name"]:
        raise RuntimeError(
            f"Dataset environment is {dataset_meta['env_name']}, expected {config['expected_environment_name']}"
        )
    # Official evaluators load a policy before env.reset(), which initializes
    # ObsUtils as a side effect. Validation creates the common dataset env first,
    # so initialize from the first native checkpoint explicitly.
    initialize_observation_utils_from_checkpoint(
        config["policies"][POLICY_ORDER[0]]["checkpoint_path"]
    )
    env, _ = create_dataset_environment(dataset_path)
    initial_observation = env.reset()
    device = select_device()
    print(f"Dataset              : {dataset_path}")
    print(f"Environment          : {dataset_meta['env_name']}")
    print(f"Horizon              : {config['horizon']}")
    print(f"Device               : {device}")
    print(f"Environment obs keys : {sorted(initial_observation)}")
    print()
    try:
        for policy_name in POLICY_ORDER:
            checkpoint_path = config["policies"][policy_name]["checkpoint_path"]
            metadata = checkpoint_metadata(policy_name, checkpoint_path)
            validate_environment_compatibility(
                policy_name, metadata["environment_metadata"], dataset_meta,
            )
            if metadata["checkpoint_horizon"] != int(config["horizon"]):
                raise RuntimeError(
                    f"{policy_name} checkpoint horizon={metadata['checkpoint_horizon']} differs from experiment horizon={config['horizon']}"
                )
            missing = [key for key in metadata["observation_keys"] if key not in initial_observation]
            if missing:
                raise RuntimeError(f"{policy_name} requires unavailable observations: {missing}")
            for key, expected_shape in metadata["observation_shapes"].items():
                if tuple(initial_observation[key].shape) != tuple(expected_shape):
                    raise RuntimeError(
                        f"{policy_name} observation {key} shape mismatch: env={initial_observation[key].shape}, checkpoint={expected_shape}"
                    )
            if int(env.action_dimension) != metadata["action_dimension"]:
                raise RuntimeError(
                    f"{policy_name} action dimension mismatch: env={env.action_dimension}, checkpoint={metadata['action_dimension']}"
                )
            policy, _, _ = load_policy(policy_name, checkpoint_path, device=device)
            try:
                policy.start_episode()
                action = policy(ob=initial_observation)
                if tuple(action.shape) != (int(env.action_dimension),):
                    raise RuntimeError(f"{policy_name} returned action shape {action.shape}")
                print(f"Policy               : {policy_name}")
                print(f"  algo                : {metadata['algo_name']}")
                print(f"  checkpoint          : {checkpoint_path}")
                print(f"  environment         : {metadata['environment_name']}")
                print(f"  observation keys    : {metadata['observation_keys']}")
                print(f"  policy class        : {policy.policy_class_name}")
                print(f"  frame stack         : {policy.frame_stack}")
                print(f"  action shape        : {tuple(action.shape)}")
            finally:
                release_policy(policy)
    finally:
        close_environment(env)
    print("\nConfiguration, checkpoints, imports, metadata, observation specs, and policy inference: OK")


def write_logged(message, log_handle, lock=None):
    print(message, end="" if message.endswith("\n") else "\n", flush=True)
    if lock is None:
        log_handle.write(message)
        if not message.endswith("\n"):
            log_handle.write("\n")
        log_handle.flush()
    else:
        with lock:
            log_handle.write(message)
            if not message.endswith("\n"):
                log_handle.write("\n")
            log_handle.flush()


def preflight_parallel_devices(devices, log_handle, require_idle=True):
    """Record hardware status and execute a real tensor op through each mask."""
    devices = [str(device) for device in devices]
    if len(devices) != len(POLICY_ORDER) or len(set(devices)) != len(devices):
        raise ValueError(
            f"parallel_evaluation.devices must contain {len(POLICY_ORDER)} unique entries; got {devices}"
        )
    write_logged("=" * 84 + "\nNPU parallel-evaluation preflight\n" + "=" * 84, log_handle)
    try:
        status = subprocess.run(
            ["npu-smi", "info"], cwd=str(EXPERIMENT_DIR), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60, check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("npu-smi was not found in PATH; source the CANN environment first") from exc
    write_logged(status.stdout or "<npu-smi produced no output>", log_handle)
    if status.returncode:
        raise RuntimeError(f"npu-smi info failed with exit code {status.returncode}")
    process_rows = re.findall(
        r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|",
        status.stdout or "", flags=re.MULTILINE,
    )
    if require_idle and process_rows:
        descriptions = [
            f"physical_npu={npu}, chip={chip}, pid={pid}, process={name.strip()}"
            for npu, chip, pid, name in process_rows
        ]
        raise RuntimeError(
            "NPU preflight found active accelerator processes; refusing to oversubscribe all four cards: "
            + "; ".join(descriptions)
        )

    probe = (
        "import torch, torch_npu; "
        "assert hasattr(torch, 'npu') and torch.npu.is_available(), 'NPU unavailable'; "
        "x=torch.ones(32, device='npu:0'); y=(x*x).sum(); "
        "torch.npu.synchronize(); "
        "print('logical_device=npu:0 result=', float(y.cpu().item()))"
    )
    for device in devices:
        environment = os.environ.copy()
        environment["ASCEND_RT_VISIBLE_DEVICES"] = device
        result = subprocess.run(
            [sys.executable, "-c", probe], cwd=str(EXPERIMENT_DIR), env=environment,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=120, check=False,
        )
        write_logged(
            f"[NPU mask {device}] return_code={result.returncode}\n{result.stdout}",
            log_handle,
        )
        if result.returncode:
            raise RuntimeError(f"NPU tensor probe failed for ASCEND_RT_VISIBLE_DEVICES={device}")
    write_logged(f"NPU preflight passed for masks: {devices}\n", log_handle)


def run_parallel_policy_evaluation(
    config_path, run_dir, policy_devices, force_eval, log_handle,
):
    """Run one isolated evaluator per policy and aggregate only after all exit."""
    policy_devices = [(policy, str(device)) for policy, device in policy_devices]
    atomic_json(
        Path(run_dir) / "parallel_assignment.json",
        {
            "mode": "one_policy_per_npu",
            "assignments": [
                {"policy_name": policy, "ascend_rt_visible_devices": device, "worker_device": "npu:0"}
                for policy, device in policy_devices
            ],
        },
    )
    output_lock = threading.Lock()
    processes = []
    threads = []

    def pump_output(policy_name, device, process, worker_log_path):
        with worker_log_path.open("a", encoding="utf-8") as worker_log:
            for line in process.stdout:
                worker_log.write(line)
                worker_log.flush()
                tagged = f"[{policy_name}|NPU-mask-{device}] {line}"
                write_logged(tagged, log_handle, lock=output_lock)

    write_logged("=" * 84 + "\nLaunching four policy workers\n" + "=" * 84, log_handle)
    try:
        for policy_name, device in policy_devices:
            command = [
                sys.executable, "-u",
                str(EXPERIMENT_DIR / "scripts" / "evaluate_policy_bank.py"),
                "--run-dir", str(run_dir), "--config", str(config_path),
                "--policy", policy_name, "--defer-aggregate",
            ]
            if force_eval:
                command.append("--force-eval")
            environment = os.environ.copy()
            environment["ASCEND_RT_VISIBLE_DEVICES"] = device
            write_logged(
                f"Policy worker: {policy_name} -> ASCEND_RT_VISIBLE_DEVICES={device}\n"
                f"Command: {' '.join(command)}",
                log_handle,
            )
            process = subprocess.Popen(
                command, cwd=str(EXPERIMENT_DIR), env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            processes.append((policy_name, device, process))
            thread = threading.Thread(
                target=pump_output,
                args=(policy_name, device, process, run_dir / "logs" / f"eval_{policy_name}.log"),
                daemon=False,
            )
            thread.start()
            threads.append(thread)

        return_codes = []
        for policy_name, device, process in processes:
            return_codes.append((policy_name, device, process.wait()))
        for thread in threads:
            thread.join()
    except BaseException:
        for _, _, process in processes:
            if process.poll() is None:
                process.terminate()
        for thread in threads:
            thread.join(timeout=10)
        raise

    completed, errors = rebuild_result_tables(run_dir)
    expected = len(POLICY_ORDER) * len(read_json(run_dir / "seed_manifest.json")["environment_seeds"])
    write_logged(
        f"Parallel evaluation aggregate: complete={len(completed)}/{expected}, errors={len(errors)}",
        log_handle,
    )
    failed = [(policy, device, code) for policy, device, code in return_codes if code]
    if failed:
        raise RuntimeError(f"Policy worker process failures: {failed}")


def run_validation_subprocess(config_path, run_dir, log_handle, device_mask=None):
    """Validate in a disposable process so the launcher retains no NPU context."""
    command = [
        sys.executable, "-u", str(EXPERIMENT_DIR / "launch.py"),
        "--stage", "validate", "--config", str(config_path),
        "--run-dir", str(run_dir), "--internal-hard-exit-after-validate",
    ]
    environment = os.environ.copy()
    if device_mask is not None:
        environment["ASCEND_RT_VISIBLE_DEVICES"] = str(device_mask)
    write_logged(
        "Validation runs in a disposable subprocess to release its NPU context before evaluation.\n"
        f"Command: {' '.join(command)}",
        log_handle,
    )
    process = subprocess.Popen(
        command, cwd=str(EXPERIMENT_DIR), env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    validation_succeeded = False
    success_marker = "Configuration, checkpoints, imports, metadata, observation specs, and policy inference: OK"
    for line in process.stdout:
        if success_marker in line:
            validation_succeeded = True
        write_logged(f"[validation] {line}", log_handle)
    return_code = process.wait()
    if return_code:
        if validation_succeeded:
            write_logged(
                f"Validation checks passed; ignoring post-success native teardown exit code {return_code}.",
                log_handle,
            )
        else:
            raise subprocess.CalledProcessError(return_code, command)


def run_stage(script_name, config_path, run_dir, num_seeds=None, force_flag=None, log_handle=None):
    command = [
        sys.executable, "-u", str(EXPERIMENT_DIR / "scripts" / script_name),
        "--run-dir", str(run_dir),
    ]
    if script_name != "analyze_complementarity.py":
        command.extend(["--config", str(config_path)])
    if num_seeds is not None and script_name == "build_initial_state_bank.py":
        command.extend(["--num-seeds", str(num_seeds)])
    if force_flag:
        command.append(force_flag)
    print("Command:", " ".join(command), flush=True)
    process = subprocess.Popen(
        command, cwd=str(EXPERIMENT_DIR), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in process.stdout:
        print(line, end="", flush=True)
        log_handle.write(line)
        log_handle.flush()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validate", "build", "eval", "analyze", "all"), default="all")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--num-seeds", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--sequential-eval", action="store_true")
    parser.add_argument(
        "--internal-hard-exit-after-validate", action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = read_json(config_path)
    count = int(args.num_seeds or config["seed_generation"]["num_initial_conditions"])
    if count <= 0:
        raise ValueError("--num-seeds must be positive")
    run_dir, copied_config = prepare_run_directory(config, args.run_dir)
    log_path = run_dir / "logs" / "experiment.log"
    print("=" * 84)
    print("Heterogeneous IL Policy Complementarity")
    print(f"Stage    : {args.stage}")
    print(f"Run dir  : {run_dir}")
    print(f"Num seeds: {count}")
    print("=" * 84, flush=True)
    with log_path.open("a", encoding="utf-8") as log_handle:
        parallel_cfg = config.get("parallel_evaluation", {})
        parallel_enabled = bool(parallel_cfg.get("enabled", False)) and not args.sequential_eval
        devices = [str(device) for device in parallel_cfg.get("devices", [])]
        if args.stage in ("eval", "all") and parallel_enabled:
            if len(devices) != len(POLICY_ORDER) or len(set(devices)) != len(devices):
                raise ValueError(
                    f"parallel_evaluation.devices must contain four unique masks; got {devices}"
                )
            if bool(parallel_cfg.get("preflight_device_check", True)):
                preflight_parallel_devices(
                    devices, log_handle,
                    require_idle=bool(parallel_cfg.get("require_idle_devices", True)),
                )
        if args.stage == "validate":
            validate_config(config)
            if args.internal_hard_exit_after_validate:
                print(
                    "Validation subprocess completed; using immediate process exit to avoid native teardown hooks.",
                    flush=True,
                )
                log_handle.flush()
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)
        elif args.stage in ("build", "eval", "all"):
            validation_mask = devices[0] if parallel_enabled and devices else None
            run_validation_subprocess(copied_config, run_dir, log_handle, validation_mask)
        if args.stage in ("build", "all"):
            run_stage(
                "build_initial_state_bank.py", copied_config, run_dir, count,
                "--force-rebuild" if args.force_rebuild else None, log_handle,
            )
        if args.stage in ("eval", "all"):
            if not (run_dir / "initial_state_manifest.json").is_file():
                raise FileNotFoundError("Initial state bank is missing; run --stage build first")
            if parallel_enabled:
                run_parallel_policy_evaluation(
                    copied_config, run_dir, zip(POLICY_ORDER, devices),
                    args.force_eval, log_handle,
                )
            else:
                run_stage(
                    "evaluate_policy_bank.py", copied_config, run_dir, None,
                    "--force-eval" if args.force_eval else None, log_handle,
                )
        if args.stage in ("analyze", "all"):
            run_stage("analyze_complementarity.py", copied_config, run_dir, log_handle=log_handle)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
