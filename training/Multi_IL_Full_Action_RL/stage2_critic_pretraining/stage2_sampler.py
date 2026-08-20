"""Episode- and policy-balanced offline samplers."""

from __future__ import annotations

import numpy as np


POLICIES = ("bc_rnn", "bc_transformer", "bc_gmm")


class EpisodeBalancedSampler:
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size):
        episodes = self.rng.integers(0, len(self.dataset.episodes), size=int(batch_size))
        timesteps = np.asarray([
            self.rng.integers(0, self.dataset.episodes[index].length)
            for index in episodes
        ], dtype=np.int64)
        return self.dataset.batch(episodes, timesteps)


class MultiPolicyBalancedSampler:
    def __init__(self, datasets, seed):
        if set(datasets) != set(POLICIES):
            raise ValueError(f"Expected datasets for {POLICIES}, got {sorted(datasets)}")
        self.samplers = {
            policy: EpisodeBalancedSampler(datasets[policy], seed + index + 1)
            for index, policy in enumerate(POLICIES)
        }
        self.rng = np.random.default_rng(seed)
        self.batch_index = 0

    def allocation(self, batch_size):
        base, remainder = divmod(int(batch_size), len(POLICIES))
        counts = {policy: base for policy in POLICIES}
        for offset in range(remainder):
            policy = POLICIES[(self.batch_index + offset) % len(POLICIES)]
            counts[policy] += 1
        self.batch_index = (self.batch_index + remainder) % len(POLICIES)
        return counts

    def sample(self, batch_size):
        counts = self.allocation(batch_size)
        parts = [self.samplers[policy].sample(counts[policy]) for policy in POLICIES]
        result = {
            key: np.concatenate([part[key] for part in parts], axis=0)
            for key in parts[0]
        }
        permutation = self.rng.permutation(int(batch_size))
        return {key: value[permutation] for key, value in result.items()}

