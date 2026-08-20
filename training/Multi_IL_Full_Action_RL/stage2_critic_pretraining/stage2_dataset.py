"""Read-only Stage 1 episode loading and frozen-RNN target caching."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from validate_frozen_rnn_target import ObservationHistory, decode, episode_lookup


POLICIES = ("bc_rnn", "bc_transformer", "bc_gmm")


def canonical_keys(handle):
    value = decode(handle.attrs.get("canonical_observation_keys"))
    if value is None:
        raise RuntimeError(f"missing canonical_observation_keys: {handle.filename}")
    keys = json.loads(value) if isinstance(value, str) else list(value)
    if not keys:
        raise RuntimeError(f"empty canonical_observation_keys: {handle.filename}")
    return list(keys)


def flatten_observation(group, keys):
    arrays = []
    length = None
    for key in keys:
        if key not in group:
            raise RuntimeError(f"missing observation key {key!r}: {group.name}")
        value = np.asarray(group[key], dtype=np.float32)
        if length is None:
            length = value.shape[0]
        elif value.shape[0] != length:
            raise RuntimeError(f"observation length mismatch: {group.name}/{key}")
        arrays.append(value.reshape(value.shape[0], -1))
    return np.concatenate(arrays, axis=1)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=str(path.parent))
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def cache_rnn_targets(policy, dataset_paths, seeds, cache_root, frame_stack):
    """Replay every requested trajectory once and persist T+1 target actions."""
    import torch
    from common import seed_everything

    cache_root = Path(cache_root)
    summary = {}
    for source in POLICIES:
        source_dir = cache_root / source
        source_dir.mkdir(parents=True, exist_ok=True)
        created = reused = transitions = 0
        with h5py.File(dataset_paths[source], "r") as handle:
            if str(decode(handle.attrs.get("policy_id", ""))) != source:
                raise RuntimeError(f"dataset policy mismatch for {source}")
            lookup = episode_lookup(handle)
            for seed in seeds:
                if seed not in lookup:
                    raise RuntimeError(f"{source} missing seed {seed}")
                group = lookup[seed]
                output = source_dir / f"seed_{seed}.npz"
                length = int(group["actions"].shape[0])
                if output.is_file():
                    with np.load(output, allow_pickle=False) as cached:
                        valid = (
                            "target_actions" in cached
                            and cached["target_actions"].shape[0] == length + 1
                            and int(cached["seed"]) == seed
                        )
                    if valid:
                        reused += 1
                        transitions += length
                        continue
                    raise RuntimeError(f"invalid existing RNN target cache: {output}")
                observations = group["obs"]
                next_observations = group["next_obs"]
                behavior_actions = group["actions"]
                keys = sorted(observations.keys())
                policy.start_episode()
                seed_everything(seed)
                history = ObservationHistory(frame_stack)
                actions = []
                for timestep in range(length):
                    observation = {
                        key: np.asarray(observations[key][timestep]).copy() for key in keys
                    }
                    previous = None if timestep == 0 else np.asarray(behavior_actions[timestep - 1])
                    policy_observation = history.append(
                        observation, timestep, previous, int(behavior_actions.shape[1])
                    )
                    with torch.no_grad():
                        actions.append(np.asarray(policy(ob=policy_observation), dtype=np.float32))
                final_observation = {
                    key: np.asarray(next_observations[key][length - 1]).copy() for key in keys
                }
                final_policy_observation = history.append(
                    final_observation, length, np.asarray(behavior_actions[length - 1]),
                    int(behavior_actions.shape[1]),
                )
                with torch.no_grad():
                    actions.append(np.asarray(policy(ob=final_policy_observation), dtype=np.float32))
                target_actions = np.asarray(actions, dtype=np.float32)
                if not np.all(np.isfinite(target_actions)):
                    raise RuntimeError(f"frozen RNN emitted NaN/Inf for {source} seed={seed}")
                atomic_npz(
                    output,
                    target_actions=target_actions,
                    seed=np.asarray(seed, dtype=np.int64),
                    source=np.asarray(source),
                )
                created += 1
                transitions += length
        summary[source] = {
            "episodes": len(seeds),
            "transitions": transitions,
            "created": created,
            "reused": reused,
        }
    return summary


@dataclass
class EpisodeRecord:
    source: str
    seed: int
    success: bool
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    next_target_action: np.ndarray
    bootstrap_mask: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    @property
    def length(self):
        return int(self.action.shape[0])


class Stage2PolicyDataset:
    """In-memory episode representation; source HDF5 remains read-only."""

    def __init__(self, source, dataset_path, seeds, cache_root):
        self.source = source
        self.path = str(Path(dataset_path).resolve())
        self.seeds = list(seeds)
        self.episodes = []
        self.reward_values = set()
        self.terminated_count = 0
        self.truncated_count = 0
        self.truncated_only_count = 0
        with h5py.File(self.path, "r") as handle:
            content_policy = str(decode(handle.attrs.get("policy_id", "")))
            if content_policy != source:
                raise RuntimeError(f"expected {source}, HDF5 contains {content_policy}")
            keys = canonical_keys(handle)
            lookup = episode_lookup(handle)
            if any(seed not in lookup for seed in self.seeds):
                missing = [seed for seed in self.seeds if seed not in lookup]
                raise RuntimeError(f"{source} missing seeds {missing}")
            for seed in self.seeds:
                group = lookup[seed]
                state = flatten_observation(group["obs"], keys)
                next_state = flatten_observation(group["next_obs"], keys)
                action = np.asarray(group["actions"], dtype=np.float32)
                reward = np.asarray(group["rewards"], dtype=np.float32).reshape(-1, 1)
                terminated = np.asarray(group["terminated"], dtype=np.bool_).reshape(-1)
                truncated = np.asarray(group["truncated"], dtype=np.bool_).reshape(-1)
                length = action.shape[0]
                if not all(value.shape[0] == length for value in (
                    state, next_state, reward, terminated, truncated
                )):
                    raise RuntimeError(f"transition length mismatch: {source} seed={seed}")
                cache_path = Path(cache_root) / source / f"seed_{seed}.npz"
                with np.load(cache_path, allow_pickle=False) as cache:
                    target_actions = np.asarray(cache["target_actions"], dtype=np.float32)
                if target_actions.shape != (length + 1, action.shape[1]):
                    raise RuntimeError(f"target cache shape mismatch: {cache_path}")
                if "success" in group.attrs:
                    success = bool(group.attrs["success"])
                elif "episode_success" in group:
                    success = bool(group["episode_success"][0])
                else:
                    raise RuntimeError(f"missing trajectory outcome: {source} seed={seed}")
                self.reward_values.update(float(value) for value in np.unique(reward))
                self.terminated_count += int(terminated.sum())
                self.truncated_count += int(truncated.sum())
                self.truncated_only_count += int(np.count_nonzero(truncated & ~terminated))
                self.episodes.append(EpisodeRecord(
                    source=source,
                    seed=seed,
                    success=success,
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    next_target_action=target_actions[1:],
                    bootstrap_mask=(~terminated).astype(np.float32).reshape(-1, 1),
                    terminated=terminated,
                    truncated=truncated,
                ))
        self.state_dim = self.episodes[0].state.shape[1]
        self.action_dim = self.episodes[0].action.shape[1]

    def batch(self, episode_indices, timesteps):
        rows = [
            (self.episodes[int(episode_index)], int(timestep))
            for episode_index, timestep in zip(episode_indices, timesteps)
        ]
        return {
            "state": np.stack([episode.state[t] for episode, t in rows]),
            "action": np.stack([episode.action[t] for episode, t in rows]),
            "reward": np.stack([episode.reward[t] for episode, t in rows]),
            "next_state": np.stack([episode.next_state[t] for episode, t in rows]),
            "next_target_action": np.stack([
                episode.next_target_action[t] for episode, t in rows
            ]),
            "bootstrap_mask": np.stack([episode.bootstrap_mask[t] for episode, t in rows]),
        }

    @property
    def transition_count(self):
        return sum(episode.length for episode in self.episodes)
