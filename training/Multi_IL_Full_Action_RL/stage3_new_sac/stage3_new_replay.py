"""Independent offline/online buffers and exact symmetric batch composition."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

FIELDS = (
    "observations",
    "actions",
    "rewards",
    "next_observations",
    "terminals",
)
HANDOFF_FIELDS = (
    "action_exec",
    "action_rl",
    "action_rnn",
    "rnn_next_actions",
    "selected_source",
    "q_select_rl",
    "q_select_rnn",
    "q_select_margin",
    "is_online",
)
IDENTITY_FIELDS = ("env_id", "episode_id", "episode_seed")


class TransitionBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int = 59,
        action_dim: int = 14,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        if self.capacity <= 0:
            raise ValueError("Replay capacity must be positive")

        self.data = {
            "observations": np.empty(
                (self.capacity, self.obs_dim), dtype=np.float32
            ),
            "actions": np.empty(
                (self.capacity, self.action_dim), dtype=np.float32
            ),
            "rewards": np.empty((self.capacity, 1), dtype=np.float32),
            "next_observations": np.empty(
                (self.capacity, self.obs_dim), dtype=np.float32
            ),
            "terminals": np.empty((self.capacity, 1), dtype=np.float32),
            "action_exec": np.zeros(
                (self.capacity, self.action_dim), dtype=np.float32
            ),
            "action_rl": np.zeros(
                (self.capacity, self.action_dim), dtype=np.float32
            ),
            "action_rnn": np.zeros(
                (self.capacity, self.action_dim), dtype=np.float32
            ),
            "rnn_next_actions": np.zeros(
                (self.capacity, self.action_dim), dtype=np.float32
            ),
            "selected_source": np.full(
                (self.capacity, 1), -1.0, dtype=np.float32
            ),
            "q_select_rl": np.zeros((self.capacity, 1), dtype=np.float32),
            "q_select_rnn": np.zeros((self.capacity, 1), dtype=np.float32),
            "q_select_margin": np.zeros(
                (self.capacity, 1), dtype=np.float32
            ),
            "is_online": np.ones((self.capacity, 1), dtype=np.float32),
        }
        # Identity metadata is for trajectory auditing only; it is deliberately
        # not mixed into the SAC state. Keep exact integer types.
        self.data.update(
            {
                key: np.full((self.capacity, 1), -1, dtype=np.int64)
                for key in IDENTITY_FIELDS
            }
        )

        self.top = 0
        self.size = 0
        self.insertions = 0
        self.rng = np.random.default_rng(seed)

    def add(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        metadata: Optional[Dict] = None,
    ) -> None:
        observation = np.asarray(observation, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_observation = np.asarray(next_observation, dtype=np.float32)

        if observation.shape != (self.obs_dim,):
            raise ValueError(
                f"observation shape {observation.shape}, expected "
                f"({self.obs_dim},)"
            )
        if next_observation.shape != (self.obs_dim,):
            raise ValueError(
                f"next_observation shape {next_observation.shape}, expected "
                f"({self.obs_dim},)"
            )
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"action shape {action.shape}, expected ({self.action_dim},)"
            )
        if (
            not np.isfinite(observation).all()
            or not np.isfinite(next_observation).all()
            or not np.isfinite(action).all()
            or not np.isfinite(float(reward))
        ):
            raise FloatingPointError("Non-finite transition passed to replay")

        index = self.top
        self.data["observations"][index] = observation
        self.data["actions"][index] = action
        self.data["rewards"][index] = np.asarray([reward], dtype=np.float32)
        self.data["next_observations"][index] = next_observation
        self.data["terminals"][index] = np.asarray(
            [terminal], dtype=np.float32
        )

        if metadata:
            selected = int(
                np.asarray(metadata["selected_source"]).reshape(-1)[0]
            )
            if selected not in (0, 1):
                raise RuntimeError(
                    "selected_source must be 0 (RNN) or 1 (RL)"
                )

            action_exec = np.asarray(
                metadata["action_exec"], dtype=np.float32
            ).reshape(-1)
            expected = np.asarray(
                metadata["action_rl"]
                if selected == 1
                else metadata["action_rnn"],
                dtype=np.float32,
            ).reshape(-1)

            if (
                action_exec.shape != (self.action_dim,)
                or expected.shape != (self.action_dim,)
                or not np.allclose(
                    action, action_exec, rtol=0, atol=1e-6
                )
                or not np.allclose(
                    action, expected, rtol=0, atol=1e-6
                )
            ):
                raise RuntimeError(
                    "Handoff replay executed-action/source integrity failure"
                )

            for key in HANDOFF_FIELDS:
                if key in metadata:
                    self.data[key][index] = np.asarray(
                        metadata[key], dtype=np.float32
                    )

            for key in IDENTITY_FIELDS:
                if key in metadata:
                    value = int(np.asarray(metadata[key]).reshape(-1)[0])
                    self.data[key][index, 0] = value

        self.top = (self.top + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.insertions += 1

    def add_batch(self, transitions: Iterable[Dict]) -> int:
        """Insert a vector-rollout batch without changing per-transition rules."""
        count = 0
        for item in transitions:
            self.add(
                item["observation"],
                item["action"],
                item["reward"],
                item["next_observation"],
                item["terminal"],
                item.get("metadata"),
            )
            count += 1
        return count

    def sample(self, count: int):
        count = int(count)
        if self.size == 0:
            raise RuntimeError("Cannot sample an empty online replay")
        if count < 0:
            raise ValueError("sample count must be non-negative")
        idx = self.rng.integers(self.size, size=count)
        return {key: value[idx].copy() for key, value in self.data.items()}

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stored = self.capacity if self.size == self.capacity else self.size
        np.savez_compressed(
            path,
            capacity=self.capacity,
            top=self.top,
            size=self.size,
            insertions=self.insertions,
            rng_state=np.asarray(
                [self.rng.bit_generator.state], dtype=object
            ),
            **{key: value[:stored] for key, value in self.data.items()},
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=True) as payload:
            obj = cls(
                int(payload["capacity"]),
                payload["observations"].shape[1],
                payload["actions"].shape[1],
            )
            obj.top = int(payload["top"])
            obj.size = int(payload["size"])
            obj.insertions = int(payload["insertions"])
            for key in obj.data:
                if key in payload:
                    obj.data[key][: len(payload[key])] = payload[key]
            obj.rng.bit_generator.state = payload["rng_state"].item()
        return obj


class SymmetricSampler:
    """Exactly 50/50 offline / online when batch_size is even."""

    def __init__(self, offline, online, seed: int = 0):
        self.offline = offline
        self.online = online
        self.turn = 0
        self.rng = np.random.default_rng(seed)
        self.counts = {"offline": 0, "online": 0}

    def sample(self, batch_size: int):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        base, remainder = divmod(batch_size, 2)
        offline_n = base + (remainder if self.turn == 0 else 0)
        online_n = batch_size - offline_n
        if remainder:
            self.turn = 1 - self.turn

        left = self.offline.sample(offline_n)
        right = self.online.sample(online_n)

        # Identity metadata intentionally remains online-audit-only. Training
        # batches contain exactly the fields needed by Stage3SAC.
        keys = list(FIELDS) + [
            key
            for key in HANDOFF_FIELDS
            if key in left and key in right
        ]
        result = {
            key: np.concatenate((left[key], right[key]), axis=0)
            for key in keys
        }
        order = self.rng.permutation(batch_size)

        self.counts["offline"] += offline_n
        self.counts["online"] += online_n
        return {key: value[order] for key, value in result.items()}

    def fractions(self):
        total = sum(self.counts.values())
        return {
            key: (value / total if total else 0.0)
            for key, value in self.counts.items()
        }
