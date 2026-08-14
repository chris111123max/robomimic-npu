#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.json"


def main():
    parser = argparse.ArgumentParser(description="Launch the Pure Online SAC experiment")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny server-side integration test under /tmp instead of formal training",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    dataset_path = Path(config["environment"]["metadata_dataset"]).expanduser()
    train_path = EXPERIMENT_DIR / "train.py"
    rlkit_root = Path("/data/home/3220251075/lerobot_workspace/robomimic/rlkit")

    for path, label in (
        (dataset_path, "environment metadata dataset"),
        (train_path, "experiment train.py"),
        (rlkit_root, "RLKit root"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    env = os.environ.copy()
    workspace = Path("/data/home/3220251075/lerobot_workspace")
    python_paths = [
        str(workspace / "robomimic"),
        str(workspace / "robomimic" / "rlkit"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join(path for path in python_paths if path)

    command = [sys.executable, "-u", str(train_path), "--config", str(config_path)]
    if args.smoke_test:
        command.append("--smoke-test")

    print("=" * 88)
    print("Experiment  :", config["experiment"]["name"])
    print("Mode        :", "SMOKE TEST" if args.smoke_test else "FORMAL TRAINING")
    print("Config      :", config_path)
    print("Dataset     :", dataset_path)
    print("Output Root :", config["experiment"]["output_dir"])
    print("Parallel Env:", config["environment"].get("parallel_envs", 1))
    print("Episodes/Ep :", config["training"]["episodes_per_epoch"])
    print("Eval Every  :", config["evaluation"]["eval_every_n_epochs"], "epoch(s)")
    print("Python      :", sys.executable)
    print("NPU visible :", env.get("ASCEND_RT_VISIBLE_DEVICES", "<not explicitly set>"))
    print("Command     :", " ".join(command))
    print("=" * 88, flush=True)

    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
