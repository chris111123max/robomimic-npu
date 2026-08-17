#!/usr/bin/env python3
"""Unified launcher for build, evaluation, and paired complementarity analysis."""

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from utils.env_utils import close_environment, create_dataset_environment, load_dataset_env_metadata
from utils.policy_loader import checkpoint_metadata, load_policy, release_policy, select_device
from utils.result_utils import POLICY_ORDER, atomic_json, read_json


DEFAULT_CONFIG = EXPERIMENT_DIR / "config" / "experiment_config.json"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


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
            if metadata["environment_name"] != dataset_meta["env_name"]:
                raise RuntimeError(f"{policy_name} checkpoint belongs to {metadata['environment_name']}")
            if canonical(metadata["environment_metadata"]) != canonical(dataset_meta):
                raise RuntimeError(f"{policy_name} checkpoint environment metadata differs from the dataset metadata")
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
                print(f"  policy class        : {policy.policy.__class__.__name__}")
                print(f"  action shape        : {tuple(action.shape)}")
            finally:
                release_policy(policy)
    finally:
        close_environment(env)
    print("\nConfiguration, checkpoints, imports, metadata, observation specs, and policy inference: OK")


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
        if args.stage in ("validate", "build", "eval", "all"):
            validate_config(config)
        if args.stage in ("build", "all"):
            run_stage(
                "build_initial_state_bank.py", copied_config, run_dir, count,
                "--force-rebuild" if args.force_rebuild else None, log_handle,
            )
        if args.stage in ("eval", "all"):
            if not (run_dir / "initial_state_manifest.json").is_file():
                raise FileNotFoundError("Initial state bank is missing; run --stage build first")
            run_stage(
                "evaluate_policy_bank.py", copied_config, run_dir, None,
                "--force-eval" if args.force_eval else None, log_handle,
            )
        if args.stage in ("analyze", "all"):
            run_stage("analyze_complementarity.py", copied_config, run_dir, log_handle=log_handle)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
