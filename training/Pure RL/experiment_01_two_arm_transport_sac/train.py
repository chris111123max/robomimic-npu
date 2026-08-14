#!/usr/bin/env python3
import argparse
from collections import OrderedDict
from copy import deepcopy
import datetime
import json
import os
from pathlib import Path
import random
import sys
import time

import gym
import numpy as np
import torch

from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
import rlkit.torch.pytorch_util as ptu

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils


class TeeStream:
    """Mirror terminal output into logs/log.txt."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()


class TwoArmTransportLowDimAdapter:
    """Thin RLKit adapter around robomimic's metadata-created EnvRobosuite."""

    def __init__(
        self,
        env,
        observation_keys,
        observation_shapes,
        obs_dim,
        action_dim,
        max_episode_steps,
        terminate_on_success,
        configured_action_transform,
    ):
        self.env = env
        self.observation_keys = tuple(observation_keys)
        self.observation_shapes = {
            key: tuple(shape) for key, shape in observation_shapes.items()
        }
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.max_episode_steps = int(max_episode_steps)
        self.terminate_on_success = bool(terminate_on_success)
        self.configured_action_transform = configured_action_transform
        self.episode_steps = 0

        if set(self.observation_keys) != set(self.observation_shapes):
            raise ValueError("observation_keys and observation_shapes must contain identical keys")
        if sum(int(np.prod(self.observation_shapes[key])) for key in self.observation_keys) != self.obs_dim:
            raise ValueError("Configured observation shapes do not sum to obs_dim")

        robosuite_env = getattr(self.env, "env", None)
        if robosuite_env is None or not hasattr(robosuite_env, "action_spec"):
            raise AttributeError("Expected metadata-created EnvRobosuite with env.action_spec")
        action_spec = robosuite_env.action_spec
        if callable(action_spec):
            action_spec = action_spec()
        if not isinstance(action_spec, (tuple, list)) or len(action_spec) != 2:
            raise ValueError(f"Unexpected robosuite action_spec: {action_spec}")

        self.action_low = np.asarray(action_spec[0], dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(action_spec[1], dtype=np.float32).reshape(-1)
        if self.action_low.shape != (self.action_dim,) or self.action_high.shape != (self.action_dim,):
            raise AssertionError(
                f"Expected action bounds shape ({self.action_dim},), got "
                f"{self.action_low.shape} and {self.action_high.shape}"
            )
        if np.any(self.action_high <= self.action_low):
            raise ValueError("Every action upper bound must be greater than its lower bound")

        self.action_space = gym.spaces.Box(
            low=self.action_low,
            high=self.action_high,
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self._direct_action = np.allclose(self.action_low, -1.0) and np.allclose(
            self.action_high, 1.0
        )
        if configured_action_transform not in ("none", "linear"):
            raise ValueError("environment.action_transform must be 'none' or 'linear'")
        if configured_action_transform == "linear":
            self._direct_action = False
        self.effective_action_transform = "none" if self._direct_action else "linear"
        if configured_action_transform == "none" and not self._direct_action:
            print(
                "WARNING: action_spec is not [-1, 1]; using explicit linear mapping "
                "from normalized SAC actions to robosuite bounds."
            )

    def _flatten_observation(self, observation):
        flattened = []
        for key in self.observation_keys:
            if key not in observation:
                raise KeyError(f"Required observation key missing: {key}")
            value = np.asarray(observation[key])
            expected_shape = self.observation_shapes[key]
            if value.shape != expected_shape:
                raise AssertionError(
                    f"Observation {key} expected shape {expected_shape}, got {value.shape}"
                )
            flattened.append(value.reshape(-1))
        result = np.concatenate(flattened, axis=0).astype(np.float32, copy=False)
        if result.shape != (self.obs_dim,):
            raise AssertionError(f"Expected flattened observation ({self.obs_dim},), got {result.shape}")
        return result

    @staticmethod
    def _task_success(info, env):
        success = info.get("is_success") if isinstance(info, dict) else None
        if success is None:
            success = env.is_success()
        if isinstance(success, dict):
            success = success.get("task", False)
        return bool(success)

    def _map_action(self, normalized_action):
        normalized_action = np.asarray(normalized_action, dtype=np.float32).reshape(-1)
        if normalized_action.shape != (self.action_dim,):
            raise AssertionError(
                f"Expected policy action ({self.action_dim},), got {normalized_action.shape}"
            )
        normalized_action = np.clip(normalized_action, -1.0, 1.0)
        if self._direct_action:
            return normalized_action
        return self.action_low + 0.5 * (normalized_action + 1.0) * (
            self.action_high - self.action_low
        )

    def reset(self):
        self.episode_steps = 0
        return self._flatten_observation(self.env.reset())

    def step(self, normalized_action):
        env_action = self._map_action(normalized_action)
        raw_next_obs, reward, _ignored_done, info = self.env.step(env_action)
        self.episode_steps += 1

        success = self._task_success(info, self.env)
        terminated = self.terminate_on_success and success
        truncated = self.episode_steps >= self.max_episode_steps and not terminated
        done = terminated or truncated

        info = dict(info)
        info["success"] = success
        info["terminated"] = terminated
        info["time_limit_truncated"] = truncated
        info["env_action"] = env_action
        return self._flatten_observation(raw_next_obs), float(reward), done, info

    def seed(self, seed):
        for candidate in (self.env, getattr(self.env, "env", None)):
            seed_method = getattr(candidate, "seed", None)
            if callable(seed_method):
                try:
                    seed_method(seed)
                    return True
                except TypeError:
                    continue
        return False

    def close(self):
        for candidate in (self.env, getattr(self.env, "env", None)):
            close_method = getattr(candidate, "close", None)
            if callable(close_method):
                close_method()
                return


def load_config(config_path, smoke_test):
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if smoke_test:
        config["experiment"]["name"] += "_smoke"
        config["experiment"]["output_dir"] = "/tmp/robomimic_pure_sac_smoke"
        config["environment"]["max_episode_steps"] = 5
        config["training"]["num_epochs"] = 1
        config["training"]["episodes_per_epoch"] = 1
        config["training"]["replay_buffer_size"] = 1000
        config["training"]["learning_starts"] = 1
        config["evaluation"]["eval_episodes"] = 1
        config["evaluation"]["max_episode_steps"] = 5
        config["checkpoint"]["save_every_n_epochs"] = 1
    config["runtime"] = {"smoke_test": bool(smoke_test)}
    return config


def validate_config(config):
    env_cfg = config["environment"]
    train_cfg = config["training"]
    sac_cfg = config["sac"]
    if env_cfg["obs_dim"] != 59:
        raise ValueError("Pure SAC baseline requires obs_dim=59")
    if env_cfg["action_dim"] != 14:
        raise ValueError("Pure SAC baseline requires action_dim=14")
    if env_cfg["obs_normalization"] is not False:
        raise ValueError("Dataset observation normalization is forbidden for this baseline")
    if train_cfg["batch_size"] <= 0 or train_cfg["learning_starts"] <= 0:
        raise ValueError("batch_size and learning_starts must be positive")
    if train_cfg["updates_per_env_step"] <= 0:
        raise ValueError("updates_per_env_step must be positive")
    if sac_cfg["optimizer"] != "Adam":
        raise ValueError("This experiment currently supports the required Adam optimizer only")


def create_run_directory(config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = Path(config["experiment"]["output_dir"]) / timestamp
    logs_dir = run_dir / "logs"
    models_dir = run_dir / "models"
    logs_dir.mkdir(parents=True, exist_ok=False)
    models_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, logs_dir, models_dir


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    npu = getattr(torch, "npu", None)
    if npu is not None and hasattr(npu, "manual_seed_all"):
        npu.manual_seed_all(seed)


def configure_device(config):
    device_cfg = config["device"]
    ptu.set_gpu_mode(device_cfg["use_accelerator"], gpu_id=device_cfg["gpu_id"])
    ptu.set_device(device_cfg["gpu_id"])
    if device_cfg["require_npu"] and ptu.device.type != "npu":
        raise RuntimeError(f"Ascend NPU is required, but RLKit selected {ptu.device}")
    print("RLKit device:", ptu.device)


def synchronize_device():
    if ptu.device.type == "npu":
        torch.npu.synchronize()
    elif ptu.device.type == "cuda":
        torch.cuda.synchronize()


def create_adapter(
    config,
    env_meta,
    max_episode_steps,
    terminate_on_success,
    seed,
):
    env_cfg = config["environment"]
    base_env = EnvUtils.create_env_from_metadata(
        env_meta=deepcopy(env_meta),
        env_name=None,
        render=env_cfg["render"],
        render_offscreen=env_cfg["render_offscreen"],
        use_image_obs=env_cfg["use_image_obs"],
        use_depth_obs=False,
    )
    adapter = TwoArmTransportLowDimAdapter(
        env=base_env,
        observation_keys=env_cfg["observation_keys"],
        observation_shapes=env_cfg["observation_shapes"],
        obs_dim=env_cfg["obs_dim"],
        action_dim=env_cfg["action_dim"],
        max_episode_steps=max_episode_steps,
        terminate_on_success=terminate_on_success,
        configured_action_transform=env_cfg["action_transform"],
    )
    adapter.seed(seed)
    return adapter


def build_sac(config, train_env):
    env_cfg = config["environment"]
    network_cfg = config["network"]
    sac_cfg = config["sac"]
    obs_dim = env_cfg["obs_dim"]
    action_dim = env_cfg["action_dim"]

    qf_kwargs = dict(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=network_cfg["critic_hidden_sizes"],
    )
    qf1 = FlattenMlp(**qf_kwargs).to(ptu.device)
    qf2 = FlattenMlp(**qf_kwargs).to(ptu.device)
    target_qf1 = FlattenMlp(**qf_kwargs).to(ptu.device)
    target_qf2 = FlattenMlp(**qf_kwargs).to(ptu.device)
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=network_cfg["actor_hidden_sizes"],
    ).to(ptu.device)

    ptu.copy_model_params_from_to(qf1, target_qf1)
    ptu.copy_model_params_from_to(qf2, target_qf2)
    for source, target, label in (
        (qf1, target_qf1, "Q1"),
        (qf2, target_qf2, "Q2"),
    ):
        if not all(torch.equal(a, b) for a, b in zip(source.parameters(), target.parameters())):
            raise AssertionError(f"Target {label} initialization mismatch")

    trainer = SACTrainer(
        env=train_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        discount=sac_cfg["discount"],
        reward_scale=sac_cfg["reward_scale"],
        policy_lr=sac_cfg["policy_lr"],
        qf_lr=sac_cfg["qf_lr"],
        soft_target_tau=sac_cfg["soft_target_tau"],
        target_update_period=sac_cfg["target_update_period"],
        use_automatic_entropy_tuning=sac_cfg["use_automatic_entropy_tuning"],
        target_entropy=sac_cfg["target_entropy"],
    )
    return policy, qf1, qf2, target_qf1, target_qf2, trainer


def policy_action(policy, observation, deterministic):
    with torch.no_grad():
        action, _ = policy.get_action(observation, deterministic=deterministic)
    action = np.asarray(action, dtype=np.float32)
    return action


def run_training_episode(
    epoch,
    episode,
    env,
    policy,
    replay_buffer,
    trainer,
    config,
    counters,
):
    train_cfg = config["training"]
    observation = env.reset()
    episode_return = 0.0
    success = False
    collect_time = 0.0
    update_time = 0.0

    for _ in range(env.max_episode_steps):
        collect_start = time.perf_counter()
        action = policy_action(policy, observation, deterministic=False)
        next_observation, reward, done, info = env.step(action)
        collect_time += time.perf_counter() - collect_start

        success = bool(info["success"])
        terminal = float(success)
        replay_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            env_info={},
        )
        counters["training_env_steps"] += 1
        episode_return += reward
        observation = next_observation

        if replay_buffer.num_steps_can_sample() >= train_cfg["learning_starts"]:
            counters["update_budget"] += float(train_cfg["updates_per_env_step"])
            updates_now = int(counters["update_budget"])
            counters["update_budget"] -= updates_now
            update_start = time.perf_counter()
            for _update in range(updates_now):
                trainer.train(replay_buffer.random_batch(train_cfg["batch_size"]))
                counters["gradient_steps"] += 1
            if updates_now:
                synchronize_device()
            update_time += time.perf_counter() - update_start

        if done:
            break

    result = {
        "Epoch": epoch,
        "Episode": episode,
        "Episode_Return": episode_return,
        "Episode_Length": env.episode_steps,
        "Success": float(success),
        "Training_Env_Steps_Total": counters["training_env_steps"],
        "Gradient_Steps_Total": counters["gradient_steps"],
        "Replay_Buffer_Size": replay_buffer.num_steps_can_sample(),
    }
    print(
        "Train Episode | "
        f"Epoch={epoch:03d} Episode={episode:02d} "
        f"Return={episode_return:.3f} Length={env.episode_steps} "
        f"Success={int(success)} EnvSteps={counters['training_env_steps']} "
        f"GradSteps={counters['gradient_steps']} "
        f"Replay={replay_buffer.num_steps_can_sample()}"
    )
    return result, collect_time, update_time


def evaluate(eval_env, deterministic_policy, config, counters):
    eval_cfg = config["evaluation"]
    returns = []
    lengths = []
    successes = []
    eval_start = time.perf_counter()

    for episode in range(1, eval_cfg["eval_episodes"] + 1):
        observation = eval_env.reset()
        episode_return = 0.0
        success = False
        for _ in range(eval_env.max_episode_steps):
            with torch.no_grad():
                action, _ = deterministic_policy.get_action(observation)
            action = np.asarray(action, dtype=np.float32)
            observation, reward, done, info = eval_env.step(action)
            counters["evaluation_env_steps"] += 1
            episode_return += reward
            success = bool(info["success"])
            if done:
                break
        returns.append(episode_return)
        lengths.append(eval_env.episode_steps)
        successes.append(float(success))
        print(
            "Eval Episode  | "
            f"Episode={episode:02d} Return={episode_return:.3f} "
            f"Length={eval_env.episode_steps} Success={int(success)}"
        )

    synchronize_device()
    eval_time = time.perf_counter() - eval_start
    return {
        "Eval_Return_Mean": float(np.mean(returns)),
        "Eval_Return_Std": float(np.std(returns)),
        "Eval_Success_Rate": float(np.mean(successes)),
        "Eval_Episode_Length_Mean": float(np.mean(lengths)),
        "Eval_Time": eval_time,
    }


def scalar_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return float(np.mean(value))
    if torch.is_tensor(value):
        return float(value.detach().cpu().mean().item())
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return str(value)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def save_checkpoint(path, networks, trainer, epoch, counters, best_eval_success, config):
    policy, qf1, qf2, target_qf1, target_qf2 = networks
    checkpoint = {
        "policy": cpu_tree(policy.state_dict()),
        "qf1": cpu_tree(qf1.state_dict()),
        "qf2": cpu_tree(qf2.state_dict()),
        "target_qf1": cpu_tree(target_qf1.state_dict()),
        "target_qf2": cpu_tree(target_qf2.state_dict()),
        "policy_optimizer": cpu_tree(trainer.policy_optimizer.state_dict()),
        "qf1_optimizer": cpu_tree(trainer.qf1_optimizer.state_dict()),
        "qf2_optimizer": cpu_tree(trainer.qf2_optimizer.state_dict()),
        "epoch": epoch,
        "training_env_steps_total": counters["training_env_steps"],
        "evaluation_env_steps_total": counters["evaluation_env_steps"],
        "gradient_steps_total": counters["gradient_steps"],
        "best_eval_success": best_eval_success,
        "config": config,
    }
    if trainer.use_automatic_entropy_tuning:
        checkpoint["log_alpha"] = cpu_tree(trainer.log_alpha)
        checkpoint["alpha_optimizer"] = cpu_tree(trainer.alpha_optimizer.state_dict())

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)
    print("Saved checkpoint:", path)


def print_startup_summary(config, config_path, run_dir, env_meta, train_env):
    env_cfg = config["environment"]
    train_cfg = config["training"]
    sac_cfg = config["sac"]
    eval_cfg = config["evaluation"]
    print("=" * 88)
    print("Experiment Name       :", config["experiment"]["name"])
    print("Config Path           :", config_path)
    print("Run Directory         :", run_dir)
    print("Dataset Metadata Path :", env_cfg["metadata_dataset"])
    print("Environment Name      :", env_meta["env_name"])
    print("Observation Keys      :", env_cfg["observation_keys"])
    print("Observation Shapes    :", env_cfg["observation_shapes"])
    print("obs_dim               :", env_cfg["obs_dim"])
    print("action_dim            :", env_cfg["action_dim"])
    print("action low            :", train_env.action_low)
    print("action high           :", train_env.action_high)
    print("Action Transform      :", train_env.effective_action_transform)
    print("Actor Architecture    :", [env_cfg["obs_dim"]] + config["network"]["actor_hidden_sizes"] + [env_cfg["action_dim"]])
    print("Critic Architecture   :", [env_cfg["obs_dim"] + env_cfg["action_dim"]] + config["network"]["critic_hidden_sizes"] + [1])
    print("Batch Size            :", train_cfg["batch_size"])
    print("Policy LR             :", sac_cfg["policy_lr"])
    print("Q LR                  :", sac_cfg["qf_lr"])
    print("Gamma                 :", sac_cfg["discount"])
    print("Tau                   :", sac_cfg["soft_target_tau"])
    print("Replay Buffer Size    :", train_cfg["replay_buffer_size"])
    print("Learning Starts       :", train_cfg["learning_starts"])
    print("Updates Per Env Step  :", train_cfg["updates_per_env_step"])
    print("Episodes Per Epoch    :", train_cfg["episodes_per_epoch"])
    print("Max Episode Steps     :", env_cfg["max_episode_steps"])
    print("Num Epochs            :", train_cfg["num_epochs"])
    print("Eval Every N Epochs   :", eval_cfg["eval_every_n_epochs"])
    print("Eval Episodes         :", eval_cfg["eval_episodes"])
    print("Device                :", ptu.device)
    print("Pure Online Data Only : True (HDF5 transitions are never read)")
    print("=" * 88)


def train(config, config_path):
    validate_config(config)
    seed = int(config["experiment"]["seed"])
    configure_device(config)
    set_random_seeds(seed)

    run_dir, logs_dir, models_dir = create_run_directory(config)
    effective_config_path = run_dir / "config.json"
    with effective_config_path.open("w", encoding="utf-8") as output_file:
        json.dump(config, output_file, indent=4, ensure_ascii=False)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = (logs_dir / "log.txt").open("a", encoding="utf-8", buffering=1)
    if config["logging"]["terminal_output_to_txt"]:
        sys.stdout = TeeStream(original_stdout, log_handle)
        sys.stderr = TeeStream(original_stderr, log_handle)

    writer = None
    if config["logging"]["tensorboard"]:
        try:
            from tensorboardX import SummaryWriter

            writer = SummaryWriter(str(logs_dir / "tb"))
        except ImportError:
            print("WARNING: tensorboardX is unavailable; TensorBoard logging disabled.")
    metrics_path = logs_dir / "metrics.jsonl"

    train_env = None
    eval_env = None
    try:
        dataset_path = Path(config["environment"]["metadata_dataset"])
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(dataset_path))
        expected_env_name = config["environment"]["expected_env_name"]
        if env_meta["env_name"] != expected_env_name:
            raise AssertionError(
                f"Expected environment {expected_env_name}, metadata contains {env_meta['env_name']}"
            )

        train_env = create_adapter(
            config,
            env_meta,
            config["environment"]["max_episode_steps"],
            config["environment"]["terminate_on_success"],
            seed,
        )
        eval_env = create_adapter(
            config,
            env_meta,
            config["evaluation"]["max_episode_steps"],
            config["evaluation"]["terminate_on_success"],
            seed + config["evaluation"]["seed_offset"],
        )

        initial_obs = train_env.reset()
        if initial_obs.shape != (59,):
            raise AssertionError(f"Initial observation expected (59,), got {initial_obs.shape}")
        if train_env.action_space.shape != (14,):
            raise AssertionError(f"Action space expected (14,), got {train_env.action_space.shape}")

        policy, qf1, qf2, target_qf1, target_qf2, trainer = build_sac(config, train_env)
        deterministic_policy = MakeDeterministic(policy)
        replay_buffer = EnvReplayBuffer(
            config["training"]["replay_buffer_size"],
            train_env,
        )

        # Static runtime shape assertions before collecting data.
        with torch.no_grad():
            sample_obs = torch.zeros(1, 59, device=ptu.device)
            sample_action = policy(sample_obs, deterministic=True)[0]
            if sample_action.shape != (1, 14):
                raise AssertionError(f"Actor output expected (1,14), got {sample_action.shape}")
            if qf1(sample_obs, sample_action).shape != (1, 1):
                raise AssertionError("Q1 output must have shape (1,1)")
            if qf2(sample_obs, sample_action).shape != (1, 1):
                raise AssertionError("Q2 output must have shape (1,1)")
        synchronize_device()

        print_startup_summary(config, config_path, run_dir, env_meta, train_env)
        print(policy)
        print(qf1)

        counters = {
            "training_env_steps": 0,
            "evaluation_env_steps": 0,
            "gradient_steps": 0,
            "update_budget": 0.0,
        }
        best_eval_success = -1.0
        networks = (policy, qf1, qf2, target_qf1, target_qf2)
        train_cfg = config["training"]
        eval_cfg = config["evaluation"]
        checkpoint_cfg = config["checkpoint"]

        for epoch in range(1, train_cfg["num_epochs"] + 1):
            epoch_start = time.perf_counter()
            episode_results = []
            collect_time = 0.0
            update_time = 0.0

            for episode in range(1, train_cfg["episodes_per_epoch"] + 1):
                result, episode_collect_time, episode_update_time = run_training_episode(
                    epoch=epoch,
                    episode=episode,
                    env=train_env,
                    policy=policy,
                    replay_buffer=replay_buffer,
                    trainer=trainer,
                    config=config,
                    counters=counters,
                )
                episode_results.append(result)
                collect_time += episode_collect_time
                update_time += episode_update_time

            epoch_metrics = OrderedDict(
                Epoch=epoch,
                Train_Return_Mean=float(np.mean([item["Episode_Return"] for item in episode_results])),
                Train_Return_Std=float(np.std([item["Episode_Return"] for item in episode_results])),
                Train_Success_Rate=float(np.mean([item["Success"] for item in episode_results])),
                Train_Episode_Length_Mean=float(np.mean([item["Episode_Length"] for item in episode_results])),
            )

            eval_time = 0.0
            if epoch % eval_cfg["eval_every_n_epochs"] == 0:
                eval_metrics = evaluate(eval_env, deterministic_policy, config, counters)
                eval_time = eval_metrics["Eval_Time"]
                epoch_metrics.update(eval_metrics)

            diagnostics = {
                key: scalar_value(value) for key, value in trainer.get_diagnostics().items()
            }
            epoch_metrics.update(diagnostics)
            epoch_metrics.update(
                Training_Env_Steps_Total=counters["training_env_steps"],
                Evaluation_Env_Steps_Total=counters["evaluation_env_steps"],
                Gradient_Steps_Total=counters["gradient_steps"],
                Replay_Buffer_Size=replay_buffer.num_steps_can_sample(),
                Time_Collect=collect_time,
                Time_Update=update_time,
                Time_Eval=eval_time,
                Time_Epoch=time.perf_counter() - epoch_start,
            )

            print("\nEpoch Summary")
            print(json.dumps(epoch_metrics, indent=4, ensure_ascii=False))

            if config["logging"]["metrics_jsonl"]:
                with metrics_path.open("a", encoding="utf-8") as metrics_file:
                    metrics_file.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
            if writer is not None:
                for key, value in epoch_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(key, value, epoch)
                writer.flush()

            eval_success = epoch_metrics.get("Eval_Success_Rate", -1.0)
            if checkpoint_cfg["save_last"]:
                save_checkpoint(
                    models_dir / "last.pth",
                    networks,
                    trainer,
                    epoch,
                    counters,
                    max(best_eval_success, eval_success),
                    config,
                )
            if checkpoint_cfg["save_best_success"] and eval_success > best_eval_success:
                best_eval_success = eval_success
                save_checkpoint(
                    models_dir / "best_success.pth",
                    networks,
                    trainer,
                    epoch,
                    counters,
                    best_eval_success,
                    config,
                )
            save_period = checkpoint_cfg["save_every_n_epochs"]
            if save_period and epoch % save_period == 0:
                save_checkpoint(
                    models_dir / f"model_epoch_{epoch:02d}.pth",
                    networks,
                    trainer,
                    epoch,
                    counters,
                    best_eval_success,
                    config,
                )

            trainer.end_epoch(epoch)

        print("Training finished successfully.")
        print("Run directory:", run_dir)
    finally:
        if writer is not None:
            writer.close()
        if train_env is not None:
            train_env.close()
        if eval_env is not None:
            eval_env.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


def main():
    parser = argparse.ArgumentParser(description="Pure Online SAC for TwoArmTransport")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path, smoke_test=args.smoke_test)
    train(config, config_path)


if __name__ == "__main__":
    main()
