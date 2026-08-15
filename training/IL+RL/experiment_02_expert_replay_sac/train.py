#!/usr/bin/env python3
"""Experiment 02: Pure SAC with a fixed 50:50 expert / online replay batch."""

import argparse
from collections import OrderedDict
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer


EXPERIMENT_DIR = Path(__file__).resolve().parent
PURE_EXPERIMENT_DIR = (
    EXPERIMENT_DIR.parent.parent / "Pure RL" / "experiment_01_two_arm_transport_sac"
)
PURE_TRAIN_PATH = PURE_EXPERIMENT_DIR / "train.py"
PURE_CONFIG_PATH = PURE_EXPERIMENT_DIR / "config.json"
MIXED_FIELDS = (
    "observations",
    "actions",
    "rewards",
    "terminals",
    "next_observations",
)


def _load_pure_sac_module():
    if not PURE_TRAIN_PATH.is_file():
        raise FileNotFoundError(f"Pure SAC implementation not found: {PURE_TRAIN_PATH}")
    spec = importlib.util.spec_from_file_location(
        "verified_pure_sac_for_expert_replay", PURE_TRAIN_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PURE = _load_pure_sac_module()
_ORIGINAL_PARALLEL_ENV_WORKER = PURE.parallel_env_worker


def parallel_env_worker(connection, config, env_meta, seed):
    """Importable forkserver entry that delegates to the verified worker."""
    return _ORIGINAL_PARALLEL_ENV_WORKER(connection, config, env_meta, seed)


PURE.parallel_env_worker = parallel_env_worker


class FixedExpertReplayBuffer(SimpleReplayBuffer):
    """Exact-capacity CPU replay buffer that becomes immutable after bulk load."""

    def __init__(self, capacity, observation_dim, action_dim):
        super().__init__(capacity, observation_dim, action_dim, env_info_sizes={})
        # RLKit defaults these arrays to float64. Expert data stays on CPU, but
        # float32 is sufficient and matches the NPU training input pipeline.
        self._observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, observation_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, 1), dtype=np.float32)
        self._sealed = False

    def bulk_load(self, observations, actions, rewards, terminals, next_observations):
        if self._sealed:
            raise RuntimeError("Fixed Expert Buffer is already sealed")
        count = observations.shape[0]
        expected = self._max_replay_buffer_size
        if count != expected:
            raise AssertionError(f"Expert load expected {expected} transitions, got {count}")
        self._observations[:] = observations
        self._actions[:] = actions
        self._rewards[:] = rewards.reshape(-1, 1)
        self._terminals[:] = terminals.reshape(-1, 1)
        self._next_obs[:] = next_observations
        self._size = count
        self._top = 0
        self._sealed = True

    def add_sample(self, *args, **kwargs):
        if self._sealed:
            raise RuntimeError("Fixed Expert Buffer is immutable during RL")
        return super().add_sample(*args, **kwargs)


class MixedBatchSampler:
    """Sample, concatenate, and jointly shuffle one expert and one online batch."""

    def __init__(self, expert_buffer, expert_batch_size, online_batch_size, seed):
        self.expert_buffer = expert_buffer
        self.expert_batch_size = int(expert_batch_size)
        self.online_batch_size = int(online_batch_size)
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)
        self.expert_samples_total = 0
        self.online_samples_total = 0

    @property
    def batch_size(self):
        return self.expert_batch_size + self.online_batch_size

    @property
    def actual_expert_ratio(self):
        total = self.expert_samples_total + self.online_samples_total
        return self.expert_samples_total / total if total else 0.0

    def reset_counters(self, reset_rng=False):
        self.expert_samples_total = 0
        self.online_samples_total = 0
        if reset_rng:
            self.rng = np.random.RandomState(self.seed)

    def random_batch(self, online_buffer):
        expert = self.expert_buffer.random_batch(self.expert_batch_size)
        online = online_buffer.random_batch(self.online_batch_size)
        mixed = {}
        for field in MIXED_FIELDS:
            if field not in expert or field not in online:
                raise KeyError(f"Mixed replay field missing: {field}")
            mixed[field] = np.concatenate((expert[field], online[field]), axis=0)
        permutation = self.rng.permutation(self.batch_size)
        mixed = {field: value[permutation] for field, value in mixed.items()}
        self.expert_samples_total += self.expert_batch_size
        self.online_samples_total += self.online_batch_size
        validate_mixed_shapes(mixed, self.batch_size)
        return mixed


def validate_mixed_shapes(batch, batch_size, obs_dim=59, action_dim=14):
    expected = {
        "observations": (batch_size, obs_dim),
        "actions": (batch_size, action_dim),
        "rewards": (batch_size, 1),
        "terminals": (batch_size, 1),
        "next_observations": (batch_size, obs_dim),
    }
    for field, shape in expected.items():
        if batch[field].shape != shape:
            raise AssertionError(f"{field} expected {shape}, got {batch[field].shape}")
        if not np.isfinite(batch[field]).all():
            raise FloatingPointError(f"Non-finite values in mixed batch field {field}")


def _decode_mask(values):
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _flatten(group, keys, shapes, demo_key, group_name):
    arrays = []
    length = None
    for key in keys:
        if key not in group:
            raise KeyError(f"{demo_key}/{group_name}/{key} is missing")
        value = np.asarray(group[key][()], dtype=np.float32)
        if tuple(value.shape[1:]) != tuple(shapes[key]):
            raise AssertionError(
                f"{demo_key}/{group_name}/{key} expected {shapes[key]}, got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Non-finite observation in {demo_key}/{group_name}/{key}")
        length = value.shape[0] if length is None else length
        if value.shape[0] != length:
            raise AssertionError(f"Observation lengths disagree in {demo_key}/{group_name}")
        arrays.append(value.reshape(value.shape[0], -1))
    result = np.concatenate(arrays, axis=1)
    if result.shape[1] != sum(int(np.prod(shapes[key])) for key in keys):
        raise AssertionError(f"Flatten width mismatch in {demo_key}/{group_name}")
    return result


def _split_statistics(lengths, actions, rewards, dones, source_dtypes):
    return {
        "demo_count": len(lengths),
        "transition_count": int(actions.shape[0]),
        "demo_length_min": int(min(lengths)),
        "demo_length_max": int(max(lengths)),
        "demo_length_mean": float(np.mean(lengths)),
        "action_shape": list(actions.shape),
        "action_source_dtype": source_dtypes["actions"],
        "action_buffer_dtype": str(actions.dtype),
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
        "action_mean": float(actions.mean()),
        "action_std": float(actions.std()),
        "reward_shape": list(rewards.shape),
        "reward_source_dtype": source_dtypes["rewards"],
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_unique": np.unique(rewards).tolist(),
        "done_shape": list(dones.shape),
        "done_source_dtype": source_dtypes["dones"],
        "done_unique": np.unique(dones).tolist(),
        "done_sum": int(dones.sum()),
        "terminal_frequency": float(dones.mean()),
        "reward_equals_done": bool(np.array_equal(rewards, dones)),
    }


def load_expert_buffer(dataset_path, config):
    env_cfg = config["environment"]
    expert_cfg = config["expert_replay"]
    keys = list(env_cfg["observation_keys"])
    shapes = env_cfg["observation_shapes"]
    train_name = expert_cfg["split"]
    valid_name = expert_cfg["validation_split"]
    split_payload = {}

    with h5py.File(dataset_path, "r") as dataset:
        if set(("data", "mask")) - set(dataset.keys()):
            raise KeyError("HDF5 must contain top-level data and mask groups")
        for split in (train_name, valid_name):
            if split not in dataset["mask"]:
                raise KeyError(f"HDF5 mask/{split} is missing")
        train_demos = _decode_mask(dataset["mask"][train_name][()])
        valid_demos = _decode_mask(dataset["mask"][valid_name][()])
        overlap = sorted(set(train_demos) & set(valid_demos))
        if overlap:
            raise AssertionError(f"Train/validation masks overlap: {overlap[:5]}")
        if expert_cfg["use_validation"] is not False:
            raise AssertionError("Validation data is forbidden in Expert Buffer")
        all_demos = set(dataset["data"].keys())
        if set(train_demos) | set(valid_demos) != all_demos:
            raise AssertionError("mask/train and mask/valid must cover every demonstration")
        top_level_groups = list(dataset.keys())
        mask_names = list(dataset["mask"].keys())
        first_demo = dataset[f"data/{train_demos[0]}"]
        demo_fields = list(first_demo.keys())
        data_total_attribute = int(dataset["data"].attrs.get("total", -1))
        num_samples_attribute_count = sum(
            "num_samples" in dataset[f"data/{demo_key}"].attrs for demo_key in all_demos
        )

        env_args_raw = dataset["data"].attrs.get("env_args")
        if env_args_raw is None:
            raise KeyError("HDF5 data.env_args metadata is missing")
        if isinstance(env_args_raw, bytes):
            env_args_raw = env_args_raw.decode("utf-8")
        env_metadata = json.loads(env_args_raw)
        if env_metadata["env_name"] != env_cfg["expected_env_name"]:
            raise AssertionError("Expert dataset environment does not match Online environment")
        env_kwargs = env_metadata["env_kwargs"]
        if bool(env_kwargs.get("reward_shaping")) != bool(
            expert_cfg["expected_reward_shaping"]
        ):
            raise AssertionError("Expert and Online reward shaping definitions are incompatible")

        for split_name, demos, load_observations in (
            (train_name, train_demos, True),
            (valid_name, valid_demos, False),
        ):
            observations, next_observations = [], []
            actions, rewards, dones, lengths = [], [], [], []
            native_next_obs = True
            source_dtypes = None
            for demo_key in demos:
                demo_path = f"data/{demo_key}"
                if demo_path not in dataset:
                    raise KeyError(f"Mask references missing demonstration: {demo_path}")
                demo = dataset[demo_path]
                for required in ("obs", "actions", "rewards", "dones"):
                    if required not in demo:
                        raise KeyError(f"{demo_path}/{required} is missing")
                obs = _flatten(demo["obs"], keys, shapes, demo_key, "obs")
                raw_action = np.asarray(demo["actions"][()])
                raw_reward = np.asarray(demo["rewards"][()])
                raw_done = np.asarray(demo["dones"][()])
                current_dtypes = {
                    "actions": str(raw_action.dtype),
                    "rewards": str(raw_reward.dtype),
                    "dones": str(raw_done.dtype),
                }
                if source_dtypes is None:
                    source_dtypes = current_dtypes
                elif current_dtypes != source_dtypes:
                    raise AssertionError(f"HDF5 dtypes vary within split {split_name}")
                action = raw_action.astype(np.float32, copy=False)
                reward = raw_reward.astype(np.float32, copy=False)
                done = raw_done.astype(np.uint8, copy=False)
                if "next_obs" in demo:
                    next_obs = _flatten(demo["next_obs"], keys, shapes, demo_key, "next_obs")
                else:
                    native_next_obs = False
                    if obs.shape[0] < 2:
                        raise RuntimeError(f"Cannot reconstruct next_obs for {demo_key}")
                    next_obs = obs[1:]
                    obs = obs[:-1]
                    action, reward, done = action[:-1], reward[:-1], done[:-1]
                count = action.shape[0]
                if action.shape != (count, env_cfg["action_dim"]):
                    raise AssertionError(f"{demo_key} actions have shape {action.shape}")
                if reward.shape != (count,) or done.shape != (count,):
                    raise AssertionError(f"{demo_key} reward/done shape mismatch")
                if obs.shape != (count, env_cfg["obs_dim"]):
                    raise AssertionError(f"{demo_key} observations have shape {obs.shape}")
                if next_obs.shape != obs.shape:
                    raise AssertionError(f"{demo_key} next_obs shape mismatch")
                if not all(np.isfinite(value).all() for value in (action, reward, done)):
                    raise FloatingPointError(f"Non-finite transition in {demo_key}")
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                lengths.append(count)
                if load_observations:
                    observations.append(obs)
                    next_observations.append(next_obs)
            action_array = np.concatenate(actions)
            reward_array = np.concatenate(rewards)
            done_array = np.concatenate(dones)
            split_payload[split_name] = {
                "statistics": _split_statistics(
                    lengths, action_array, reward_array, done_array, source_dtypes
                ),
                "native_next_obs": native_next_obs,
                "demos": demos,
            }
            if load_observations:
                split_payload[split_name].update(
                    observations=np.concatenate(observations),
                    next_observations=np.concatenate(next_observations),
                    actions=action_array,
                    rewards=reward_array,
                    dones=done_array,
                )

    train = split_payload[train_name]
    stats = train["statistics"]
    loaded_transition_total = (
        stats["transition_count"]
        + split_payload[valid_name]["statistics"]["transition_count"]
    )
    if data_total_attribute >= 0 and loaded_transition_total != data_total_attribute:
        raise AssertionError(
            f"HDF5 data.total={data_total_attribute}, but train+valid={loaded_transition_total}"
        )
    if stats["action_min"] < -1.000001 or stats["action_max"] > 1.000001:
        raise AssertionError(
            f"Expert actions exceed [-1,1]; clipping is forbidden: "
            f"[{stats['action_min']}, {stats['action_max']}]"
        )
    if set(stats["done_unique"]) - {0, 1}:
        raise AssertionError(f"Illegal HDF5 done values: {stats['done_unique']}")
    if not stats["reward_equals_done"]:
        raise AssertionError(
            "Sparse reward is not identical to success done; Expert terminal semantics "
            "cannot be proven compatible with Pure SAC"
        )
    if expert_cfg["terminal_rule"] != "stored_hdf5_dones":
        raise AssertionError("Experiment 02 requires stored HDF5 dones as terminals")

    count = stats["transition_count"]
    expert_buffer = FixedExpertReplayBuffer(
        count, env_cfg["obs_dim"], env_cfg["action_dim"]
    )
    expert_buffer.bulk_load(
        train["observations"],
        train["actions"],
        train["rewards"],
        train["dones"],
        train["next_observations"],
    )
    if expert_buffer.num_steps_can_sample() != count or not expert_buffer._sealed:
        raise AssertionError("Fixed Expert Buffer construction failed")
    try:
        expert_buffer.add_sample()
    except RuntimeError:
        pass
    else:
        raise AssertionError("Fixed Expert Buffer accepted a write after sealing")

    info = {
        "dataset_path": str(dataset_path),
        "top_level_groups": top_level_groups,
        "mask_names": mask_names,
        "demo_fields": demo_fields,
        "states_present": "states" in demo_fields,
        "data_total_attribute": data_total_attribute,
        "demos_with_num_samples_attribute": num_samples_attribute_count,
        "total_demo_count": len(train_demos) + len(valid_demos),
        "train_split": train_name,
        "validation_split": valid_name,
        "train_validation_overlap": 0,
        "validation_used_in_replay": False,
        "observation_keys": keys,
        "observation_shapes": shapes,
        "flattened_observation_dim": env_cfg["obs_dim"],
        "next_obs_source": "stored HDF5 next_obs" if train["native_next_obs"] else "within-demo obs[t+1]",
        "terminal_rule": "stored HDF5 dones; no forced demo-boundary terminal",
        "reward_rule": "stored HDF5 rewards",
        "reward_compatibility": {
            "same_metadata_file_used_by_online_environment": True,
            "env_name": env_metadata["env_name"],
            "env_version": env_metadata.get("env_version"),
            "reward_shaping": env_kwargs.get("reward_shaping"),
            "robots": env_kwargs.get("robots"),
            "env_configuration": env_kwargs.get("env_configuration"),
            "control_freq": env_kwargs.get("control_freq"),
            "controller_configs": env_kwargs.get("controller_configs"),
        },
        "train": train["statistics"],
        "validation": split_payload[valid_name]["statistics"],
        "expert_buffer_capacity": count,
        "expert_buffer_size": expert_buffer.num_steps_can_sample(),
        "expert_buffer_immutable": True,
    }
    return expert_buffer, info


def make_runtime_config(raw_config, smoke_test):
    config = json.loads(json.dumps(raw_config))
    config["runtime"] = {"smoke_test": bool(smoke_test)}
    if smoke_test:
        smoke = config["smoke_test"]
        config["experiment"]["name"] += "_smoke"
        config["environment"]["max_episode_steps"] = smoke["max_episode_steps"]
        config["training"].update(
            num_epochs=smoke["num_epochs"],
            episodes_per_epoch=config["environment"]["parallel_envs"],
            replay_buffer_size=config["training"]["learning_starts"],
            learning_starts=smoke["learning_starts_override"],
        )
        config["evaluation"].update(
            eval_every_n_epochs=1,
            eval_episodes=config["environment"]["parallel_envs"],
            max_episode_steps=smoke["max_episode_steps"],
        )
    return config


def validate_pure_config_parity(raw_config):
    with PURE_CONFIG_PATH.open("r", encoding="utf-8") as stream:
        pure = json.load(stream)
    for key in (
        "environment",
        "network",
        "sac",
        "training",
        "evaluation",
        "logging",
        "checkpoint",
        "device",
    ):
        experiment_value = raw_config[key]
        pure_value = pure[key]
        if key == "environment":
            # --dataset-path is an intentional development override. Every
            # other environment parameter must remain identical.
            experiment_value = dict(experiment_value)
            pure_value = dict(pure_value)
            experiment_value.pop("metadata_dataset")
            pure_value.pop("metadata_dataset")
        if experiment_value != pure_value:
            raise AssertionError(f"Experiment 02 drifted from Pure SAC config section: {key}")
    expert = raw_config["expert_replay"]
    if expert["expert_batch_size"] + expert["online_batch_size"] != raw_config["training"]["batch_size"]:
        raise AssertionError("Expert + online batch sizes must equal SAC batch_size")
    if expert["sampling_ratio"] != expert["expert_batch_size"] / raw_config["training"]["batch_size"]:
        raise AssertionError("Configured expert sampling ratio is inconsistent")
    if raw_config["paths"]["dataset"] != raw_config["environment"]["metadata_dataset"]:
        raise AssertionError("Expert and Online environment must use the same HDF5 metadata")


def _eligible_updates(before_steps, after_steps, learning_starts):
    eligible_before = max(0, int(before_steps) - int(learning_starts) + 1)
    eligible_after = max(0, int(after_steps) - int(learning_starts) + 1)
    return eligible_after - eligible_before


def validate_smoke_sampling(expert_buffer, config, seed):
    expert_config = config["expert_replay"]
    obs_dim = config["environment"]["obs_dim"]
    action_dim = config["environment"]["action_dim"]
    learning_starts = config["training"]["learning_starts"]
    batch_checks = config["smoke_test"]["mixed_batch_checks"]
    online = SimpleReplayBuffer(learning_starts, obs_dim, action_dim, env_info_sizes={})
    online._observations[:] = 0.25
    online._actions[:] = 0.5
    online._rewards[:] = 0.0
    online._terminals[:] = 0
    online._next_obs[:] = 0.75
    online._size = learning_starts

    # Prove that one shared permutation preserves cross-field transition
    # correspondence rather than shuffling every field independently.
    tag_count = max(expert_config["expert_batch_size"], expert_config["online_batch_size"]) * 2
    tagged_expert = FixedExpertReplayBuffer(tag_count, obs_dim, action_dim)
    tag = np.arange(tag_count, dtype=np.float32)
    tag_obs = np.zeros((tag_count, obs_dim), dtype=np.float32)
    tag_action = np.zeros((tag_count, action_dim), dtype=np.float32)
    tag_next = np.zeros((tag_count, obs_dim), dtype=np.float32)
    tag_obs[:, 0] = tag
    tag_action[:, 0] = tag
    tag_next[:, 0] = tag + 1000
    tagged_expert.bulk_load(tag_obs, tag_action, tag[:, None], (tag % 2)[:, None], tag_next)
    tagged_online = SimpleReplayBuffer(tag_count, obs_dim, action_dim, env_info_sizes={})
    tagged_online._observations[:] = tag_obs + 10000
    tagged_online._actions[:] = tag_action + 10000
    tagged_online._rewards[:] = tag[:, None] + 10000
    tagged_online._terminals[:] = (tag % 2)[:, None]
    tagged_online._next_obs[:] = tag_next + 10000
    tagged_online._size = tag_count
    tagged_sampler = MixedBatchSampler(
        tagged_expert,
        expert_config["expert_batch_size"],
        expert_config["online_batch_size"],
        seed + 1,
    )
    tagged = tagged_sampler.random_batch(tagged_online)
    identifiers = tagged["observations"][:, 0]
    if not np.array_equal(identifiers, tagged["actions"][:, 0]):
        raise AssertionError("Joint shuffle broke observation/action correspondence")
    if not np.array_equal(identifiers, tagged["rewards"][:, 0]):
        raise AssertionError("Joint shuffle broke observation/reward correspondence")
    if not np.array_equal(identifiers + 1000, tagged["next_observations"][:, 0]):
        raise AssertionError("Joint shuffle broke observation/next_observation correspondence")

    sampler = MixedBatchSampler(
        expert_buffer,
        expert_config["expert_batch_size"],
        expert_config["online_batch_size"],
        seed,
    )
    first = sampler.random_batch(online)
    validate_mixed_shapes(first, sampler.batch_size, obs_dim, action_dim)
    for _ in range(batch_checks - 1):
        sampler.random_batch(online)
    expected_expert = batch_checks * expert_config["expert_batch_size"]
    expected_online = batch_checks * expert_config["online_batch_size"]
    if sampler.expert_samples_total != expected_expert or sampler.online_samples_total != expected_online:
        raise AssertionError("100-batch 50:50 sampler count validation failed")
    if sampler.actual_expert_ratio != 0.5:
        raise AssertionError("Actual Expert Sampling Ratio must be exactly 0.5")
    if _eligible_updates(0, learning_starts - 1, learning_starts) != 0:
        raise AssertionError("Expert data incorrectly bypasses online learning_starts")
    if _eligible_updates(learning_starts - 1, learning_starts, learning_starts) != 1:
        raise AssertionError("Online transition 1000 must enable exactly one update")
    if _eligible_updates(learning_starts, learning_starts + batch_checks, learning_starts) != batch_checks:
        raise AssertionError("Update scheduler is not 1 online step : 1 update")


def _save_checkpoint(path, networks, trainer, epoch, counters, best, config, info, sampler):
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
        "best_eval_success": best,
        "config": config,
        "expert_replay": {
            "dataset_path": info["dataset_path"],
            "split": info["train_split"],
            "train_demo_count": info["train"]["demo_count"],
            "transition_count": info["train"]["transition_count"],
            "sampling_ratio": config["expert_replay"]["sampling_ratio"],
            "expert_samples_total": sampler.expert_samples_total,
            "online_samples_total": sampler.online_samples_total,
        },
    }
    if trainer.use_automatic_entropy_tuning:
        checkpoint["log_alpha"] = PURE.cpu_tree(trainer.log_alpha)
        checkpoint["alpha_optimizer"] = PURE.cpu_tree(trainer.alpha_optimizer.state_dict())
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)
    print("Saved checkpoint:", path)


def run_experiment(raw_config, config_path, run_dir, smoke_test=False):
    validate_pure_config_parity(raw_config)
    dataset_path = Path(raw_config["paths"]["dataset"]).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if raw_config["expert_replay"]["enabled"] is not True:
        raise AssertionError("Experiment 02 requires Expert Replay")

    logs_dir = run_dir / "logs"
    models_dir = run_dir / "models"
    info_dir = run_dir / "expert_buffer_info"
    for path in (logs_dir, models_dir, info_dir):
        path.mkdir(parents=True, exist_ok=False)

    expert_buffer, expert_info = load_expert_buffer(dataset_path, raw_config)
    with (info_dir / "hdf5_schema_and_statistics.json").open("w", encoding="utf-8") as stream:
        json.dump(expert_info, stream, indent=4, ensure_ascii=False)

    if smoke_test:
        validate_smoke_sampling(
            expert_buffer, raw_config, raw_config["experiment"]["seed"] + 22000
        )
        print("CPU smoke checks: 100 mixed batches = 5000 expert + 5000 online; ratio=0.5")
        print("CPU smoke checks: online size 999 -> 0 updates; 1000 -> 1 update")

    config = make_runtime_config(raw_config, smoke_test)
    expert_cfg = config["expert_replay"]
    sampler = MixedBatchSampler(
        expert_buffer,
        expert_cfg["expert_batch_size"],
        expert_cfg["online_batch_size"],
        config["experiment"]["seed"] + 23000,
    )
    state = {"online_buffer": None, "trainer": None}

    original_create_run_directory = PURE.create_run_directory
    original_env_replay_buffer = PURE.EnvReplayBuffer
    original_build_sac = PURE.build_sac
    original_perform_updates = PURE.perform_sac_updates
    original_print_summary = PURE.print_startup_summary
    original_save_checkpoint = PURE.save_checkpoint

    def create_run_directory(_config):
        return run_dir, logs_dir, models_dir

    def create_online_buffer(*args, **kwargs):
        buffer = original_env_replay_buffer(*args, **kwargs)
        if buffer.num_steps_can_sample() != 0:
            raise AssertionError("Online Replay Buffer must start empty")
        state["online_buffer"] = buffer
        return buffer

    def build_sac(runtime_config, train_env):
        result = original_build_sac(runtime_config, train_env)
        trainer = result[-1]
        state["trainer"] = trainer
        original_diagnostics = trainer.get_diagnostics

        def diagnostics():
            values = OrderedDict(original_diagnostics())
            online_size = (
                state["online_buffer"].num_steps_can_sample()
                if state["online_buffer"] is not None
                else 0
            )
            values.update(
                Expert_Buffer_Size=expert_buffer.num_steps_can_sample(),
                Online_Buffer_Size=online_size,
                Expert_Batch_Size=sampler.expert_batch_size,
                Online_Batch_Size=sampler.online_batch_size,
                Expert_Samples_Total=sampler.expert_samples_total,
                Online_Samples_Total=sampler.online_samples_total,
                Actual_Expert_Sampling_Ratio=sampler.actual_expert_ratio,
                Expert_Train_Demo_Count=expert_info["train"]["demo_count"],
                Expert_Transition_Count=expert_info["train"]["transition_count"],
            )
            return values

        trainer.get_diagnostics = diagnostics
        return result

    def perform_updates(new_transitions, online_buffer, trainer, runtime_config, counters):
        train_cfg = runtime_config["training"]
        before_steps = counters["training_env_steps"] - int(new_transitions)
        after_steps = counters["training_env_steps"]
        learning_starts = int(train_cfg["learning_starts"])
        if online_buffer.num_steps_can_sample() < learning_starts:
            updates_now = 0
        else:
            newly_eligible = _eligible_updates(before_steps, after_steps, learning_starts)
            counters["update_budget"] += newly_eligible * float(train_cfg["updates_per_env_step"])
            updates_now = int(counters["update_budget"])
            counters["update_budget"] -= updates_now
        update_start = time.perf_counter()
        for _ in range(updates_now):
            trainer.train(sampler.random_batch(online_buffer))
            counters["gradient_steps"] += 1
        if updates_now:
            PURE.synchronize_device()
        return time.perf_counter() - update_start

    def print_summary(runtime_config, source_path, actual_run_dir, env_meta, env_pool):
        original_print_summary(runtime_config, source_path, actual_run_dir, env_meta, env_pool)
        print("Expert Split              : TRAIN ONLY")
        print("Validation Used In Replay : False")
        print("Train Demo Count          :", expert_info["train"]["demo_count"])
        print("Train Transition Count    :", expert_info["train"]["transition_count"])
        print("Validation Demo Count     :", expert_info["validation"]["demo_count"])
        print("Expert Buffer Size        :", expert_buffer.num_steps_can_sample())
        print("Expert Buffer Device      : CPU / NumPy")
        print("Expert Buffer Immutable   :", expert_buffer._sealed)
        print("Expert Batch Size         :", sampler.expert_batch_size)
        print("Online Batch Size         :", sampler.online_batch_size)
        print("Expert Sampling Ratio     :", f"{expert_cfg['sampling_ratio']:.4f}")
        print("Expert next_obs           :", expert_info["next_obs_source"])
        print("Expert terminal rule      :", expert_info["terminal_rule"])
        print("Expert reward rule        :", expert_info["reward_rule"])
        print("Actor / Q Initialization  : random (identical to Pure SAC)")
        print("BC / Teacher / IL Loss    : absent")

    def save_checkpoint(path, networks, trainer, epoch, counters, best, runtime_config):
        return _save_checkpoint(
            path, networks, trainer, epoch, counters, best, runtime_config, expert_info, sampler
        )

    PURE.create_run_directory = create_run_directory
    PURE.EnvReplayBuffer = create_online_buffer
    PURE.build_sac = build_sac
    PURE.perform_sac_updates = perform_updates
    PURE.print_startup_summary = print_summary
    PURE.save_checkpoint = save_checkpoint
    try:
        PURE.train(config, config_path)
    finally:
        PURE.create_run_directory = original_create_run_directory
        PURE.EnvReplayBuffer = original_env_replay_buffer
        PURE.build_sac = original_build_sac
        PURE.perform_sac_updates = original_perform_updates
        PURE.print_startup_summary = original_print_summary
        PURE.save_checkpoint = original_save_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Experiment 02 Expert Replay SAC")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    run_experiment(
        config,
        config_path,
        args.run_dir.expanduser().resolve(),
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
