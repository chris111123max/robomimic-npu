"""Transition and boundary-safe recurrent replay for Stage3-v3."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import h5py
import numpy as np

from stage3_v5_actor import CANONICAL_KEYS


CORE = ("observations", "actions", "rewards", "next_observations", "terminals")


def _flat(group):
    length = len(next(iter(group.values())))
    return np.concatenate([
        np.asarray(group[key], np.float32).reshape(length, -1) for key in CANONICAL_KEYS
    ], axis=1)


class OfflineDemonstrations:
    def __init__(self, path, seed=0):
        self.path = str(Path(path).resolve())
        self.rng = np.random.default_rng(int(seed))
        self.episodes = []
        rows = {key: [] for key in CORE}
        with h5py.File(self.path, "r") as handle:
            if "data" not in handle:
                raise RuntimeError("Stage3-v3 requires robomimic /data/demo_* demonstrations")
            for name in sorted(handle["data"]):
                demo = handle["data"][name]
                for key in ("obs", "next_obs", "actions", "rewards", "dones"):
                    if key not in demo:
                        raise RuntimeError(f"{demo.name} misses {key}")
                obs, nxt = _flat(demo["obs"]), _flat(demo["next_obs"])
                actions = np.asarray(demo["actions"], np.float32)
                rewards = np.asarray(demo["rewards"], np.float32).reshape(-1, 1)
                terminals = np.asarray(demo["dones"], np.float32).reshape(-1, 1)
                length = len(actions)
                if obs.shape != (length, 59) or nxt.shape != (length, 59) or actions.shape != (length, 14):
                    raise RuntimeError(f"{demo.name} shape contract failed")
                episode = dict(observations=obs, actions=actions, rewards=rewards,
                               next_observations=nxt, terminals=terminals,
                               episode_steps=np.arange(length, dtype=np.int64))
                self.episodes.append(episode)
                for key in CORE:
                    rows[key].append(episode[key])
        self.data = {key: np.concatenate(value) for key, value in rows.items()}
        self.size = len(self.data["actions"])
        if not self.episodes or not all(np.isfinite(value).all() for value in self.data.values()):
            raise RuntimeError("Offline demonstrations are empty or non-finite")

    def sample(self, count):
        indices = self.rng.integers(self.size, size=int(count))
        return {key: value[indices].copy() for key, value in self.data.items()}

    def sample_sequences(self, count, length):
        eligible = [episode for episode in self.episodes if len(episode["actions"]) >= int(length)]
        if not eligible:
            raise RuntimeError("No offline episode is long enough for recurrent sampling")
        result = {key: [] for key in CORE + ("episode_steps",)}
        for _ in range(int(count)):
            episode = eligible[int(self.rng.integers(len(eligible)))]
            start = int(self.rng.integers(len(episode["actions"]) - int(length) + 1))
            for key in result:
                result[key].append(episode[key][start:start + int(length)])
        return {key: np.stack(value) for key, value in result.items()}


class OnlineTransitionReplay:
    def __init__(self, capacity, seed=0):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(int(seed))
        self.data = {
            "observations": np.empty((self.capacity, 59), np.float32),
            "actions": np.empty((self.capacity, 14), np.float32),
            "rewards": np.empty((self.capacity, 1), np.float32),
            "next_observations": np.empty((self.capacity, 59), np.float32),
            "terminals": np.empty((self.capacity, 1), np.float32),
        }
        self.top = 0
        self.size = 0
        self.insertions = 0

    def add(self, observation, action, reward, next_observation, terminal):
        values = (np.asarray(observation, np.float32), np.asarray(action, np.float32),
                  np.asarray(next_observation, np.float32))
        if values[0].shape != (59,) or values[1].shape != (14,) or values[2].shape != (59,):
            raise ValueError("Online transition shape mismatch")
        if not all(np.isfinite(value).all() for value in values):
            raise FloatingPointError("Non-finite online transition")
        index = self.top
        self.data["observations"][index] = values[0]
        self.data["actions"][index] = values[1]
        self.data["rewards"][index, 0] = float(reward)
        self.data["next_observations"][index] = values[2]
        self.data["terminals"][index, 0] = float(bool(terminal))
        self.top = (self.top + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.insertions += 1

    def sample(self, count):
        if not self.size:
            raise RuntimeError("Online transition replay is empty")
        indices = self.rng.integers(self.size, size=int(count))
        return {key: value[indices].copy() for key, value in self.data.items()}

    def save(self, path):
        stored = self.capacity if self.size == self.capacity else self.size
        np.savez_compressed(path, capacity=self.capacity, top=self.top, size=self.size,
                            insertions=self.insertions,
                            rng_state=np.asarray([self.rng.bit_generator.state], dtype=object),
                            **{key: value[:stored] for key, value in self.data.items()})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=True) as payload:
            replay = cls(int(payload["capacity"]))
            replay.top, replay.size = int(payload["top"]), int(payload["size"])
            replay.insertions = int(payload["insertions"])
            for key in replay.data:
                replay.data[key][:len(payload[key])] = payload[key]
            replay.rng.bit_generator.state = payload["rng_state"].item()
        return replay


class OnlineSequenceReplay:
    """Completed episodes only; partial and cross-episode sequences are unsampleable."""
    def __init__(self, capacity_transitions, seed=0):
        self.capacity = int(capacity_transitions)
        self.rng = np.random.default_rng(int(seed))
        self.episodes = deque()
        self.current = {}
        self.transitions = 0

    def add(self, env_id, observation, action, reward, next_observation, terminal,
            episode_step):
        env_id = int(env_id)
        episode = self.current.setdefault(env_id, {key: [] for key in CORE + ("episode_steps",)})
        values = {
            "observations": np.asarray(observation, np.float32),
            "actions": np.asarray(action, np.float32),
            "rewards": np.asarray([reward], np.float32),
            "next_observations": np.asarray(next_observation, np.float32),
            "terminals": np.asarray([terminal], np.float32),
            "episode_steps": np.asarray(episode_step, np.int64),
        }
        for key, value in values.items():
            episode[key].append(value)

    def finish(self, env_id):
        episode = self.current.pop(int(env_id), None)
        if episode:
            self.add_episode({key: np.asarray(value) for key, value in episode.items()})

    def abort(self, env_id):
        self.current.pop(int(env_id), None)

    def add_episode(self, episode):
        length = len(episode["actions"])
        if not length:
            return
        saved = {key: np.asarray(value).copy() for key, value in episode.items()}
        self.episodes.append(saved)
        self.transitions += length
        while self.transitions > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.popleft()
            self.transitions -= len(removed["actions"])

    def can_sample(self, length):
        return any(len(episode["actions"]) >= int(length) for episode in self._all_episodes())

    def _all_episodes(self):
        # Keep partial episodes as lists. Converting every growing episode to a
        # full array on every gradient update would dominate collection time.
        return list(self.episodes) + list(self.current.values())

    def sample_sequences(self, count, length):
        eligible = [episode for episode in self._all_episodes()
                    if len(episode["actions"]) >= int(length)]
        if not eligible:
            raise RuntimeError("No completed online episode is long enough")
        result = {key: [] for key in CORE + ("episode_steps",)}
        for _ in range(int(count)):
            episode = eligible[int(self.rng.integers(len(eligible)))]
            start = int(self.rng.integers(len(episode["actions"]) - int(length) + 1))
            for key in result:
                result[key].append(episode[key][start:start + int(length)])
        return {key: np.stack(value) for key, value in result.items()}

    def save(self, path):
        np.save(path, {"capacity": self.capacity, "transitions": self.transitions,
                       "rng_state": self.rng.bit_generator.state,
                       "episodes": list(self.episodes), "current": self.current},
                allow_pickle=True)

    @classmethod
    def load(cls, path):
        payload = np.load(path, allow_pickle=True).item()
        replay = cls(payload["capacity"])
        replay.transitions = int(payload["transitions"])
        replay.episodes = deque(payload["episodes"])
        replay.current = payload.get("current", {})
        replay.rng.bit_generator.state = payload["rng_state"]
        return replay


def symmetric_transition_batch(offline, online, batch_size=256):
    if int(batch_size) != 256:
        raise ValueError("Stage3-v3 Critic batch must be 256")
    left, right = offline.sample(128), online.sample(128)
    return {key: np.concatenate((left[key], right[key]), axis=0) for key in CORE}


def symmetric_sequence_batch(offline, online, count, length):
    if int(count) % 2:
        raise ValueError("Actor sequence batch must be even")
    half = int(count) // 2
    left = offline.sample_sequences(half, length)
    right = online.sample_sequences(half, length)
    result = {key: np.concatenate((left[key], right[key]), axis=0)
              for key in CORE + ("episode_steps",)}
    result["is_offline"] = np.concatenate((np.ones(half, np.float32),
                                            np.zeros(half, np.float32)))
    return result


def final_transition(sequence_batch):
    """Use the final transition after burn-in; source remains exactly 50/50."""
    return {key: value[:, -1].copy() for key, value in sequence_batch.items()
            if key in CORE}
