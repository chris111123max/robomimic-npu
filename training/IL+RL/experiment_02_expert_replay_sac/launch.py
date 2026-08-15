#!/usr/bin/env python3
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.json"
WORKSPACE = Path("/data/home/3220251075/lerobot_workspace")


def main():
    parser = argparse.ArgumentParser(description="Launch Expert Replay SAC")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    source_config = args.config.expanduser().resolve()
    with source_config.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    dataset_path = (
        args.dataset_path.expanduser().resolve()
        if args.dataset_path is not None
        else Path(config["paths"]["dataset"]).expanduser()
    )
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    config["paths"]["dataset"] = str(dataset_path)
    config["environment"]["metadata_dataset"] = str(dataset_path)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_root = (
        Path("/tmp/robomimic_ilrl_experiment_02_smoke")
        if args.smoke_test
        else Path(config["paths"]["output_root"]).expanduser()
    )
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    effective_config = run_dir / "config.json"
    with effective_config.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=4, ensure_ascii=False)

    env = os.environ.copy()
    python_paths = [
        str(WORKSPACE / "robomimic"),
        str(WORKSPACE / "robomimic" / "rlkit"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join(path for path in python_paths if path)

    command = [
        sys.executable,
        "-u",
        str(EXPERIMENT_DIR / "train.py"),
        "--config",
        str(effective_config),
        "--run-dir",
        str(run_dir),
    ]
    if args.smoke_test:
        command.append("--smoke-test")

    print("=" * 88)
    print("Experiment       :", config["experiment"]["name"])
    print("Mode             :", "SMOKE TEST" if args.smoke_test else "FORMAL")
    print("Dataset          :", dataset_path)
    print("Expert Split     :", config["expert_replay"]["split"], "only")
    print("Expert / Online  :", config["expert_replay"]["expert_batch_size"], "/", config["expert_replay"]["online_batch_size"])
    print("Parallel Envs    :", config["environment"]["parallel_envs"])
    print("Run Directory    :", run_dir)
    print("NPU visible      :", env.get("ASCEND_RT_VISIBLE_DEVICES", "<unset>"))
    print("Command          :", " ".join(command))
    print("=" * 88, flush=True)
    subprocess.run(command, env=env, check=True)
    print("Experiment 02 completed successfully.")
    print("Run directory:", run_dir)


if __name__ == "__main__":
    main()
