"""Strict read-only loader for the audited Stage1 rollout schema."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

POLICIES = ("bc_rnn", "bc_transformer", "bc_gmm")
CANONICAL_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                  "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos", "object")
REQUIRED = ("actions", "rewards", "dones", "terminated", "truncated",
            "episode_id", "initial_seed", "episode_success", "episode_length")


def _scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _scalar(value.item())
    return value


def _constant(group, name):
    if name in group.attrs:
        return _scalar(group.attrs[name])
    if name in group:
        values = np.asarray(group[name])
        if values.size == 0 or not np.all(values == values.reshape(-1)[0]):
            raise RuntimeError(f"{group.name}/{name} must be a nonempty constant episode field")
        return _scalar(values.reshape(-1)[0])
    raise RuntimeError(f"{group.name}: missing required {name}")


def monte_carlo_returns(rewards, gamma):
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(rewards)):
        raise RuntimeError("Rewards contain NaN or Inf")
    result = np.empty_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + float(gamma) * running
        result[index] = running
    return result


@dataclass(frozen=True)
class Episode:
    policy: str
    seed: int
    episode_id: int
    success: bool
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    returns: np.ndarray

    @property
    def length(self): return int(self.action.shape[0])


class Stage2NewDataset:
    """Fully validated episodes plus a transition-uniform sampling view."""
    def __init__(self, policy, path, seeds, gamma):
        if policy not in POLICIES: raise ValueError(f"Unknown policy {policy!r}")
        self.policy, self.path, self.seeds = policy, str(Path(path).resolve()), tuple(map(int, seeds))
        self.episodes = []
        with h5py.File(self.path, "r") as handle:
            if str(_scalar(handle.attrs.get("policy_id", ""))) != policy:
                raise RuntimeError(f"{path}: root policy_id does not equal {policy}")
            if "episodes" not in handle: raise RuntimeError(f"{path}: missing /episodes group")
            raw_keys = handle.attrs.get("canonical_observation_keys")
            if raw_keys is None: raise RuntimeError(f"{path}: missing canonical_observation_keys")
            keys = tuple(json.loads(_scalar(raw_keys)))
            if keys != CANONICAL_KEYS:
                raise RuntimeError(f"{path}: canonical observation order differs from Stage1 contract: {keys}")
            by_seed = {}
            for group_name, group in handle["episodes"].items():
                seed = int(_constant(group, "initial_seed"))
                if seed in by_seed: raise RuntimeError(f"{path}: duplicate initial_seed {seed}")
                by_seed[seed] = group_name
            missing = sorted(set(self.seeds) - set(by_seed))
            if missing: raise RuntimeError(f"{path}: missing requested explicit initial_seed values {missing}")
            for seed in self.seeds:
                self.episodes.append(self._read_episode(handle["episodes"][by_seed[seed]], seed, gamma))
        if not self.episodes: raise RuntimeError(f"{path}: no requested episodes")
        self.state = np.concatenate([x.state for x in self.episodes]).astype(np.float32, copy=False)
        self.action = np.concatenate([x.action for x in self.episodes]).astype(np.float32, copy=False)
        self.returns = np.concatenate([x.returns for x in self.episodes]).astype(np.float32, copy=False).reshape(-1, 1)
        self.success = np.concatenate([np.full(x.length, x.success, dtype=bool) for x in self.episodes])
        self.transition_seed = np.concatenate([np.full(x.length, x.seed, dtype=np.int64) for x in self.episodes])
        self.transition_episode_id = np.concatenate([np.full(x.length, x.episode_id, dtype=np.int64) for x in self.episodes])
        self.timestep = np.concatenate([np.arange(x.length, dtype=np.int64) for x in self.episodes])
        self.obs_dim, self.action_dim = self.state.shape[1], self.action.shape[1]
        if (self.obs_dim, self.action_dim) != (59, 14):
            raise RuntimeError(f"{path}: expected dimensions (59, 14), got {(self.obs_dim, self.action_dim)}")
        if not all(np.all(np.isfinite(x)) for x in (self.state, self.action, self.returns)):
            raise RuntimeError(f"{path}: non-finite flattened state/action/return")

    def _read_episode(self, group, seed, gamma):
        missing = [name for name in REQUIRED if name not in group and name not in group.attrs]
        if missing: raise RuntimeError(f"{group.name}: missing required fields {missing}")
        if "obs" not in group or "next_obs" not in group: raise RuntimeError(f"{group.name}: missing obs or next_obs")
        if set(group["obs"].keys()) != set(CANONICAL_KEYS) or set(group["next_obs"].keys()) != set(CANONICAL_KEYS):
            raise RuntimeError(f"{group.name}: observation keys do not match canonical contract")
        action = np.asarray(group["actions"], dtype=np.float32)
        reward = np.asarray(group["rewards"], dtype=np.float32).reshape(-1)
        length = len(action)
        if length <= 0 or action.shape != (length, 14): raise RuntimeError(f"{group.name}: invalid action shape {action.shape}")
        state_parts = []
        for key in CANONICAL_KEYS:
            value = np.asarray(group["obs"][key], dtype=np.float32)
            next_value = np.asarray(group["next_obs"][key], dtype=np.float32)
            if value.shape[0] != length or next_value.shape != value.shape:
                raise RuntimeError(f"{group.name}: invalid obs/next_obs length for {key}")
            if not np.all(np.isfinite(value)) or not np.all(np.isfinite(next_value)):
                raise RuntimeError(f"{group.name}: non-finite observation {key}")
            state_parts.append(value.reshape(length, -1))
        state = np.concatenate(state_parts, axis=1)
        flags = {name: np.asarray(group[name], dtype=bool).reshape(-1) for name in ("dones", "terminated", "truncated")}
        if any(len(value) != length for value in (*flags.values(), reward)):
            raise RuntimeError(f"{group.name}: transition field length mismatch")
        if not np.array_equal(flags["dones"], flags["terminated"] | flags["truncated"]):
            raise RuntimeError(f"{group.name}: dones must equal terminated OR truncated")
        if np.any(flags["dones"][:-1]) or not flags["dones"][-1] or np.any(flags["terminated"] & flags["truncated"]):
            raise RuntimeError(f"{group.name}: invalid finite-episode boundary flags")
        if int(_constant(group, "initial_seed")) != seed: raise RuntimeError(f"{group.name}: seed lookup mismatch")
        if int(_constant(group, "episode_length")) != length: raise RuntimeError(f"{group.name}: episode_length mismatch")
        episode_id, success = int(_constant(group, "episode_id")), bool(_constant(group, "episode_success"))
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(reward)):
            raise RuntimeError(f"{group.name}: non-finite action or reward")
        returns = monte_carlo_returns(reward, gamma)
        return Episode(self.policy, seed, episode_id, success, state, action, reward, returns)

    @property
    def transition_count(self): return int(self.state.shape[0])

    def batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return {"state": self.state[indices], "action": self.action[indices], "return": self.returns[indices],
                "success": self.success[indices], "policy": np.full(len(indices), self.policy),
                "seed": self.transition_seed[indices], "episode_id": self.transition_episode_id[indices], "timestep": self.timestep[indices]}


def load_split_datasets(dataset_root, train_seeds, val_seeds, gamma):
    root = Path(dataset_root)
    paths = {policy: root / policy / "transitions.hdf5" for policy in POLICIES}
    for path in paths.values():
        if not path.is_file(): raise FileNotFoundError(path)
    train = {policy: Stage2NewDataset(policy, paths[policy], train_seeds, gamma) for policy in POLICIES}
    val = {policy: Stage2NewDataset(policy, paths[policy], val_seeds, gamma) for policy in POLICIES}
    overlap = set(train_seeds) & set(val_seeds)
    if overlap: raise RuntimeError(f"Train/validation seed leakage: {sorted(overlap)}")
    return train, val


def split_manifest(datasets, split):
    rows = []
    for policy, dataset in datasets.items():
        for episode in dataset.episodes:
            rows.append({"policy": policy, "seed": episode.seed, "episode_id": episode.episode_id,
                         "success": episode.success, "episode_length": episode.length, "split": split})
    return rows


def data_audit(train, val):
    report = {"dimensions": {"obs_dim": 59, "action_dim": 14}, "policies": {}}
    for policy in POLICIES:
        result = {}
        for label, dataset in (("train", train[policy]), ("validation", val[policy])):
            result[label] = {"episodes": len(dataset.episodes), "transitions": dataset.transition_count,
                "success_episodes": sum(x.success for x in dataset.episodes), "failure_episodes": sum(not x.success for x in dataset.episodes),
                "episode_length": _stats(np.asarray([x.length for x in dataset.episodes])), "reward": _stats(np.concatenate([x.reward for x in dataset.episodes])),
                "return": _stats(dataset.returns), "obs": _stats(dataset.state), "action": _stats(dataset.action)}
        report["policies"][policy] = result
    return report


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std())}
