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


def resolve_teacher_checkpoint(config):
    configured = Path(config["paths"]["teacher_checkpoint"]).expanduser()
    expected_name = config["teacher"]["checkpoint_filename"]
    if configured.is_file():
        if configured.name != expected_name:
            raise AssertionError(f"Configured Teacher has wrong filename: {configured}")
        return configured.resolve()
    search_root = WORKSPACE / "training_runs"
    matches = sorted(search_root.rglob(expected_name)) if search_root.is_dir() else []
    if len(matches) == 1:
        print("Configured Teacher path moved; resolved unique workspace match:", matches[0])
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            f"BC-GMM Epoch 1850 Teacher not found at {configured} or under {search_root}"
        )
    raise RuntimeError(
        "Multiple BC-GMM Epoch 1850 Teacher checkpoints found; refusing arbitrary choice:\n"
        + "\n".join(str(path) for path in matches)
    )


def create_run_directory(config, smoke_test):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if smoke_test:
        root = Path("/tmp/robomimic_ilrl_experiment_01_smoke")
    else:
        root = Path(config["experiment"]["output_dir"]).expanduser()
    run_dir = root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_command(command, env):
    print("Command:", " ".join(str(item) for item in command), flush=True)
    subprocess.run([str(item) for item in command], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description="IL+RL Experiment 01 launcher")
    parser.add_argument("--stage", choices=("stage1", "stage2", "all"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--actor-checkpoint", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    source_config_path = args.config.expanduser().resolve()
    with source_config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    dataset_path = Path(config["paths"]["dataset"]).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if config["environment"]["parallel_envs"] != 16:
        raise AssertionError("Experiment 01 requires exactly 16 parallel environments")
    if config["environment"]["parallel_start_method"] != "forkserver":
        raise AssertionError("Experiment 01 must reuse verified forkserver environment loading")

    if args.stage in ("stage1", "all"):
        teacher_checkpoint = resolve_teacher_checkpoint(config)
        config["paths"]["teacher_checkpoint"] = str(teacher_checkpoint)
        print("Resolved BC-GMM Epoch 1850 Teacher:", teacher_checkpoint)

    run_dir = create_run_directory(config, smoke_test=args.smoke_test)
    effective_config_path = run_dir / "config.json"
    with effective_config_path.open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=4, ensure_ascii=False)

    env = os.environ.copy()
    workspace_python_paths = [
        str(WORKSPACE / "robomimic"),
        str(WORKSPACE / "robomimic" / "rlkit"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join(path for path in workspace_python_paths if path)
    smoke_args = ["--smoke-test"] if args.smoke_test else []

    print("=" * 88)
    print("Experiment       :", config["experiment"]["name"])
    print("Stage            :", args.stage)
    print("Mode             :", "SMOKE TEST" if args.smoke_test else "FORMAL")
    print("Run Directory    :", run_dir)
    print("Parallel Envs    :", config["environment"]["parallel_envs"])
    print("Start Method     :", config["environment"]["parallel_start_method"])
    print("NPU visible      :", env.get("ASCEND_RT_VISIBLE_DEVICES", "<unset>"))
    print("=" * 88, flush=True)

    best_actor = None
    if args.stage in ("stage1", "all"):
        run_command(
            [
                sys.executable,
                "-u",
                EXPERIMENT_DIR / "pretrain_actor.py",
                "--config",
                effective_config_path,
                "--run-dir",
                run_dir,
                *smoke_args,
            ],
            env,
        )
        best_actor = run_dir / "stage1_actor_pretraining" / "models" / "best_actor.pth"
        if not best_actor.is_file():
            raise FileNotFoundError(f"Stage1 finished without best_actor.pth: {best_actor}")

    if args.stage == "stage2":
        if args.actor_checkpoint is None:
            candidate = run_dir / "stage1_actor_pretraining" / "models" / "best_actor.pth"
            if candidate.is_file():
                best_actor = candidate
            else:
                raise ValueError(
                    "--stage stage2 requires --actor-checkpoint <best_actor.pth>; "
                    "last_actor.pth is never selected implicitly"
                )
        else:
            best_actor = args.actor_checkpoint.expanduser().resolve()

    if args.stage in ("stage2", "all"):
        if best_actor is None or not best_actor.is_file():
            raise FileNotFoundError(f"Stage1 best_actor.pth not found: {best_actor}")
        run_command(
            [
                sys.executable,
                "-u",
                EXPERIMENT_DIR / "train.py",
                "--config",
                effective_config_path,
                "--actor-checkpoint",
                best_actor,
                "--run-dir",
                run_dir,
                *smoke_args,
            ],
            env,
        )

    print("Requested stage completed successfully.")
    print("Run directory:", run_dir)


if __name__ == "__main__":
    main()
