"""Historical full-prefix sampler for Stage2.3.

This intentionally preserves the Stage2.2 contract used by the 20260922
Stage3 run: recurrent state is rebuilt from episode step zero and the
learning_mask selects the supervised block.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE2_2 = HERE.parent / "stage2_2_history_aware_critic"
if str(STAGE2_2) not in sys.path:
    sys.path.insert(0, str(STAGE2_2))

from sequence_dataset import POLICIES, previous_actions  # noqa: E402


class _SourceAllocation:
    def _init_allocation(self, datasets, seed, balanced):
        self.datasets = datasets
        self.rng = np.random.default_rng(seed)
        self.balanced = bool(balanced)
        self.cursor = 0
        self.counts = {p: 0 for p in POLICIES}

    def allocation(self, count):
        if not self.balanced:
            return {"bc_rnn": int(count), "bc_transformer": 0, "bc_gmm": 0}
        base, rem = divmod(int(count), 3)
        out = {p: base for p in POLICIES}
        for i in range(rem):
            out[POLICIES[(self.cursor + i) % 3]] += 1
        self.cursor = (self.cursor + rem) % 3
        return out

    def proportions(self):
        total = sum(self.counts.values())
        return {
            p: self.counts[p] / total if total else 0.0
            for p in POLICIES
        }


class SequenceSampler(_SourceAllocation):
    def __init__(self, datasets, burn_in, learning, horizon, seed, balanced):
        del burn_in
        self.learning = int(learning)
        self.horizon = float(horizon)
        self._init_allocation(datasets, seed, balanced)
        self.valid = {
            p: [
                (ei, start)
                for ei, episode in enumerate(datasets[p].episodes)
                for start in range(max(1, episode.length - self.learning + 1))
            ]
            for p in datasets
        }
        if any(not rows for rows in self.valid.values()):
            raise RuntimeError("a source has no episode")

    def sample(self, count):
        rows = []
        for policy, n in self.allocation(count).items():
            ids = self.rng.integers(len(self.valid[policy]), size=n)
            rows.extend(
                (policy, *self.valid[policy][int(i)]) for i in ids
            )
            self.counts[policy] += n
        self.rng.shuffle(rows)

        specs, max_stop = [], 0
        for policy, episode_index, start in rows:
            episode = self.datasets[policy].episodes[episode_index]
            stop = min(episode.length, start + self.learning)
            max_stop = max(max_stop, stop)
            specs.append((policy, episode, start, stop))

        b = len(specs)
        observations = np.zeros((b, max_stop, 59), np.float32)
        previous = np.zeros((b, max_stop, 14), np.float32)
        actions = np.zeros((b, max_stop, 14), np.float32)
        progress = np.zeros((b, max_stop, 1), np.float32)
        returns = np.zeros((b, max_stop, 1), np.float32)
        valid = np.zeros((b, max_stop, 1), bool)
        learning = np.zeros((b, max_stop, 1), bool)
        metadata = {
            k: []
            for k in (
                "policy", "seed", "episode_id", "start", "stop",
                "prefix_length", "episode_length",
            )
        }

        for row, (policy, episode, start, stop) in enumerate(specs):
            prev = previous_actions(episode.actions)
            observations[row, :stop] = episode.observations[:stop]
            previous[row, :stop] = prev[:stop]
            actions[row, :stop] = episode.actions[:stop]
            progress[row, :stop, 0] = (
                np.arange(stop, dtype=np.float32) / self.horizon
            )
            returns[row, :stop, 0] = episode.returns[:stop]
            valid[row, :stop] = True
            learning[row, start:stop] = True
            for key, value in (
                ("policy", policy),
                ("seed", episode.seed),
                ("episode_id", episode.episode_id),
                ("start", start),
                ("stop", stop),
                ("prefix_length", start),
                ("episode_length", episode.length),
            ):
                metadata[key].append(value)

        result = {
            "observations": observations,
            "previous_actions": previous,
            "actions": actions,
            "progress": progress,
            "returns": returns,
            "valid_mask": valid,
            "learning_mask": learning,
        }
        result.update({k: np.asarray(v) for k, v in metadata.items()})
        return result
