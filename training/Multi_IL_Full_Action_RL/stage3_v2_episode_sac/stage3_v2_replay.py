"""Stage3-v2 replay with episode-source metadata and exact 50/50 sampling."""
from __future__ import annotations

from pathlib import Path
import numpy as np


CORE_FIELDS = (
    "observations", "actions", "rewards", "next_observations", "terminals"
)
TRAIN_FIELDS = CORE_FIELDS + ("action_rnn", "is_online", "behavior_source")
IDENTITY_FIELDS = ("env_id", "episode_id", "episode_seed", "behavior_phase")


class EpisodeReplay:
    def __init__(self, capacity, obs_dim=59, action_dim=14, seed=0):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        if self.capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self.data = {
            "observations": np.empty((self.capacity, self.obs_dim), np.float32),
            "actions": np.empty((self.capacity, self.action_dim), np.float32),
            "rewards": np.empty((self.capacity, 1), np.float32),
            "next_observations": np.empty((self.capacity, self.obs_dim), np.float32),
            "terminals": np.empty((self.capacity, 1), np.float32),
            "action_rnn": np.zeros((self.capacity, self.action_dim), np.float32),
            "is_online": np.ones((self.capacity, 1), np.float32),
            "behavior_source": np.empty((self.capacity, 1), np.float32),
        }
        self.data.update({
            key: np.full((self.capacity, 1), -1, np.int64)
            for key in IDENTITY_FIELDS
        })
        self.top = 0
        self.size = 0
        self.insertions = 0
        self.rng = np.random.default_rng(seed)

    def add(self, observation, action, reward, next_observation, terminal, metadata):
        obs = np.asarray(observation, np.float32)
        act = np.asarray(action, np.float32)
        nxt = np.asarray(next_observation, np.float32)
        source = int(metadata["behavior_source"])
        if obs.shape != (self.obs_dim,) or nxt.shape != (self.obs_dim,):
            raise ValueError("Stage3-v2 replay observation shape mismatch")
        if act.shape != (self.action_dim,):
            raise ValueError("Stage3-v2 replay action shape mismatch")
        if source not in (0, 1):
            raise ValueError("behavior_source must be 0 (RNN) or 1 (RL)")
        teacher = np.asarray(metadata.get("action_rnn", np.zeros(self.action_dim)), np.float32)
        if teacher.shape != (self.action_dim,):
            raise ValueError("action_rnn shape mismatch")
        if source == 0 and not np.allclose(act, teacher, rtol=0, atol=1e-6):
            raise RuntimeError("RNN episode replay action differs from its frozen-RNN action")
        if not all(np.isfinite(value).all() for value in (obs, act, nxt, teacher)):
            raise FloatingPointError("Non-finite Stage3-v2 replay transition")
        index = self.top
        self.data["observations"][index] = obs
        self.data["actions"][index] = act
        self.data["rewards"][index, 0] = float(reward)
        self.data["next_observations"][index] = nxt
        self.data["terminals"][index, 0] = float(bool(terminal))
        self.data["action_rnn"][index] = teacher
        self.data["is_online"][index, 0] = 1.0
        self.data["behavior_source"][index, 0] = float(source)
        for key in IDENTITY_FIELDS:
            self.data[key][index, 0] = int(metadata[key])
        self.top = (self.top + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.insertions += 1

    def sample(self, count):
        if self.size <= 0:
            raise RuntimeError("Cannot sample empty Stage3-v2 online replay")
        indices = self.rng.integers(self.size, size=int(count))
        return {key: value[indices].copy() for key, value in self.data.items()}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stored = self.capacity if self.size == self.capacity else self.size
        np.savez_compressed(
            path,
            capacity=self.capacity,
            top=self.top,
            size=self.size,
            insertions=self.insertions,
            rng_state=np.asarray([self.rng.bit_generator.state], dtype=object),
            **{key: value[:stored] for key, value in self.data.items()},
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=True) as payload:
            replay = cls(
                int(payload["capacity"]),
                payload["observations"].shape[1],
                payload["actions"].shape[1],
            )
            replay.top = int(payload["top"])
            replay.size = int(payload["size"])
            replay.insertions = int(payload["insertions"])
            for key in replay.data:
                replay.data[key][:len(payload[key])] = payload[key]
            replay.rng.bit_generator.state = payload["rng_state"].item()
        return replay


class SymmetricEpisodeSampler:
    """Exactly 128 offline + 128 online for the required batch size 256."""
    def __init__(self, offline, online, seed=0):
        self.offline = offline
        self.online = online
        self.rng = np.random.default_rng(seed)
        self.counts = {"offline": 0, "online": 0}

    @staticmethod
    def _offline_batch(batch, count):
        required = set(CORE_FIELDS) | {"action_rnn"}
        missing = required - set(batch)
        if missing:
            raise RuntimeError(f"Offline replay misses Stage3-v2 fields: {sorted(missing)}")
        result = {key: batch[key] for key in CORE_FIELDS}
        result["action_rnn"] = batch["action_rnn"]
        result["is_online"] = np.zeros((count, 1), np.float32)
        result["behavior_source"] = np.full((count, 1), -1.0, np.float32)
        return result

    def sample(self, batch_size):
        if int(batch_size) != 256:
            raise ValueError("Stage3-v2 contract requires batch_size=256")
        offline_n = online_n = 128
        left = self._offline_batch(self.offline.sample(offline_n), offline_n)
        right_raw = self.online.sample(online_n)
        right = {key: right_raw[key] for key in TRAIN_FIELDS}
        merged = {
            key: np.concatenate((left[key], right[key]), axis=0)
            for key in TRAIN_FIELDS
        }
        order = self.rng.permutation(batch_size)
        self.counts["offline"] += offline_n
        self.counts["online"] += online_n
        return {key: value[order] for key, value in merged.items()}

    def fractions(self):
        total = sum(self.counts.values())
        return {key: value / total if total else 0.0 for key, value in self.counts.items()}
