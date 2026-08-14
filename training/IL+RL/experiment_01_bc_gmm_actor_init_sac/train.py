#!/usr/bin/env python3
"""Stage 2: standard Pure SAC with only the Actor weights replaced."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
PURE_SAC_PATH = (
    EXPERIMENT_DIR.parent.parent
    / "Pure RL"
    / "experiment_01_two_arm_transport_sac"
    / "train.py"
)


def _load_pure_sac_module():
    if not PURE_SAC_PATH.is_file():
        raise FileNotFoundError(f"Pure SAC implementation not found: {PURE_SAC_PATH}")
    spec = importlib.util.spec_from_file_location("verified_pure_sac_baseline", PURE_SAC_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PURE = _load_pure_sac_module()
_ORIGINAL_PARALLEL_ENV_WORKER = PURE.parallel_env_worker


def parallel_env_worker(connection, config, env_meta, seed):
    """Importable worker entry used by forkserver in this experiment."""
    return _ORIGINAL_PARALLEL_ENV_WORKER(connection, config, env_meta, seed)


# ParallelEnvPool resolves this name in the verified baseline module. Replacing
# it with an entry defined in this executable keeps forkserver pickling stable.
PURE.parallel_env_worker = parallel_env_worker

# Stage 1 imports these verified environment utilities from this module.
ParallelEnvPool = PURE.ParallelEnvPool
policy_actions = PURE.policy_actions
configure_device = PURE.configure_device
set_random_seeds = PURE.set_random_seeds
initialize_observation_modalities = PURE.initialize_observation_modalities


def make_stage2_config(raw_config, smoke_test):
    """Project the unified experiment config onto the exact Pure SAC schema."""
    config = {
        "experiment": {
            "name": raw_config["experiment"]["name"] + "_stage2",
            "seed": raw_config["experiment"]["seed"],
            "output_dir": raw_config["experiment"]["output_dir"],
        },
        "environment": dict(raw_config["environment"]),
        "network": dict(raw_config["stage2"]["network"]),
        "sac": dict(raw_config["stage2"]["sac"]),
        "training": dict(raw_config["stage2"]["training"]),
        "evaluation": dict(raw_config["stage2"]["evaluation"]),
        "logging": dict(raw_config["logging"]),
        "checkpoint": dict(raw_config["checkpoint"]),
        "device": dict(raw_config["device"]),
        "runtime": {"smoke_test": bool(smoke_test)},
    }
    if smoke_test:
        config["experiment"]["name"] += "_smoke"
        config["environment"]["max_episode_steps"] = 5
        config["training"].update(
            num_epochs=1,
            episodes_per_epoch=config["environment"]["parallel_envs"],
            replay_buffer_size=1000,
            learning_starts=1,
        )
        config["evaluation"].update(
            eval_every_n_epochs=1,
            eval_episodes=config["environment"]["parallel_envs"],
            max_episode_steps=5,
        )
        config["checkpoint"]["save_every_n_epochs"] = 1
    return config


def _load_actor_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    if "actor" not in checkpoint:
        raise KeyError(f"Stage 1 checkpoint has no 'actor' state_dict: {path}")
    return checkpoint, checkpoint["actor"]


def _save_stage2_checkpoint(
    path, networks, trainer, epoch, counters, best_eval_success, config, actor_checkpoint
):
    policy, qf1, qf2, target_qf1, target_qf2 = networks
    checkpoint = {
        "policy": PURE.cpu_tree(policy.state_dict()),
        "qf1": PURE.cpu_tree(qf1.state_dict()),
        "qf2": PURE.cpu_tree(qf2.state_dict()),
        "target_qf1": PURE.cpu_tree(target_qf1.state_dict()),
        "target_qf2": PURE.cpu_tree(target_qf2.state_dict()),
        "policy_optimizer": PURE.cpu_tree(trainer.policy_optimizer.state_dict()),
        "qf1_optimizer": PURE.cpu_tree(trainer.qf1_optimizer.state_dict()),
        "qf2_optimizer": PURE.cpu_tree(trainer.qf2_optimizer.state_dict()),
        "epoch": epoch,
        "training_env_steps_total": counters["training_env_steps"],
        "evaluation_env_steps_total": counters["evaluation_env_steps"],
        "gradient_steps_total": counters["gradient_steps"],
        "best_eval_success": best_eval_success,
        "stage1_actor_checkpoint": str(actor_checkpoint),
        "config": config,
    }
    if trainer.use_automatic_entropy_tuning:
        checkpoint["log_alpha"] = PURE.cpu_tree(trainer.log_alpha)
        checkpoint["alpha_optimizer"] = PURE.cpu_tree(
            trainer.alpha_optimizer.state_dict()
        )
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)
    print("Saved checkpoint:", path)


def run_stage2(raw_config, config_path, actor_checkpoint, run_dir, smoke_test=False):
    actor_checkpoint = actor_checkpoint.expanduser().resolve()
    if not actor_checkpoint.is_file():
        raise FileNotFoundError(f"Stage 1 best_actor checkpoint not found: {actor_checkpoint}")
    stage1_checkpoint, actor_state = _load_actor_checkpoint(actor_checkpoint)
    if int(stage1_checkpoint.get("obs_dim", -1)) != 59:
        raise AssertionError("Stage 1 actor checkpoint obs_dim must be 59")
    if int(stage1_checkpoint.get("action_dim", -1)) != 14:
        raise AssertionError("Stage 1 actor checkpoint action_dim must be 14")

    config = make_stage2_config(raw_config, smoke_test=smoke_test)
    config["runtime"]["stage1_actor_checkpoint"] = str(actor_checkpoint)
    stage2_dir = run_dir / "stage2_rl_training"
    logs_dir = stage2_dir / "logs"
    models_dir = stage2_dir / "models"
    evaluation_dir = stage2_dir / "evaluation"

    original_create_run_directory = PURE.create_run_directory
    original_build_sac = PURE.build_sac
    original_print_summary = PURE.print_startup_summary
    original_save_checkpoint = PURE.save_checkpoint
    original_replay_buffer = PURE.EnvReplayBuffer
    loaded_policy = {"policy": None}
    fresh_replay = {"buffer": None}
    epoch0_done = {"value": False}

    def create_run_directory(_config):
        logs_dir.mkdir(parents=True, exist_ok=False)
        models_dir.mkdir(parents=True, exist_ok=False)
        evaluation_dir.mkdir(parents=True, exist_ok=False)
        return stage2_dir, logs_dir, models_dir

    def build_sac(stage2_config, train_env):
        result = original_build_sac(stage2_config, train_env)
        policy = result[0]
        # Network construction order remains identical to Pure SAC. Only now,
        # after random Actor/Q creation and fresh SAC optimizers, replace Actor.
        policy.load_state_dict(actor_state, strict=True)
        for key, value in actor_state.items():
            if not torch.equal(policy.state_dict()[key].detach().cpu(), value.detach().cpu()):
                raise AssertionError(f"Stage 1 Actor load mismatch at parameter {key}")
        loaded_policy["policy"] = policy
        print("Stage1 Actor Checkpoint:", actor_checkpoint)
        print("Actor state_dict loaded exactly; Q networks remain random.")
        return result

    def create_replay_buffer(*args, **kwargs):
        replay_buffer = original_replay_buffer(*args, **kwargs)
        if int(replay_buffer._size) != 0:
            raise AssertionError("Stage2 replay buffer must be empty at initialization")
        fresh_replay["buffer"] = replay_buffer
        return replay_buffer

    def print_summary(stage2_config, source_path, actual_run_dir, env_meta, env_pool):
        original_print_summary(
            stage2_config, source_path, actual_run_dir, env_meta, env_pool
        )
        if fresh_replay["buffer"] is None or int(fresh_replay["buffer"]._size) != 0:
            raise AssertionError("Epoch0 evaluation must start with an empty replay buffer")
        print("Stage2 Teacher        : absent")
        print("Stage2 Demonstrations : absent")
        print("Stage2 Replay Initial : empty")
        if epoch0_done["value"]:
            return
        counters = {
            "training_env_steps": 0,
            "evaluation_env_steps": 0,
            "gradient_steps": 0,
            "update_budget": 0.0,
        }
        metrics = PURE.evaluate(env_pool, loaded_policy["policy"], stage2_config, counters, 0)
        epoch0_metrics = {
            "Epoch": 0,
            "Epoch0_Eval_Success_Rate": metrics["Eval_Success_Rate"],
            "Epoch0_Return_Mean": metrics["Eval_Return_Mean"],
            "Epoch0_Return_Std": metrics["Eval_Return_Std"],
            "Epoch0_Episode_Length_Mean": metrics["Eval_Episode_Length_Mean"],
            "Epoch0_Eval_Time": metrics["Eval_Time"],
            "Training_Env_Steps_Total": 0,
            "Gradient_Steps_Total": 0,
            "Replay_Buffer_Size": 0,
            "stage1_actor_checkpoint": str(actor_checkpoint),
        }
        with (evaluation_dir / "epoch_000.json").open("w", encoding="utf-8") as stream:
            json.dump(epoch0_metrics, stream, indent=4, ensure_ascii=False)
        with (logs_dir / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(epoch0_metrics, ensure_ascii=False) + "\n")
        print("\nStage2 Epoch 0 Evaluation")
        print(json.dumps(epoch0_metrics, indent=4, ensure_ascii=False))
        epoch0_done["value"] = True

    def save_checkpoint(path, networks, trainer, epoch, counters, best, stage2_config):
        return _save_stage2_checkpoint(
            path, networks, trainer, epoch, counters, best, stage2_config, actor_checkpoint
        )

    PURE.create_run_directory = create_run_directory
    PURE.build_sac = build_sac
    PURE.EnvReplayBuffer = create_replay_buffer
    PURE.print_startup_summary = print_summary
    PURE.save_checkpoint = save_checkpoint
    try:
        PURE.train(config, config_path)
    finally:
        PURE.create_run_directory = original_create_run_directory
        PURE.build_sac = original_build_sac
        PURE.EnvReplayBuffer = original_replay_buffer
        PURE.print_startup_summary = original_print_summary
        PURE.save_checkpoint = original_save_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Stage 2 BC-GMM Actor initialized SAC")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    with args.config.expanduser().resolve().open("r", encoding="utf-8") as stream:
        raw_config = json.load(stream)
    run_stage2(
        raw_config=raw_config,
        config_path=args.config.expanduser().resolve(),
        actor_checkpoint=args.actor_checkpoint,
        run_dir=args.run_dir.expanduser().resolve(),
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
