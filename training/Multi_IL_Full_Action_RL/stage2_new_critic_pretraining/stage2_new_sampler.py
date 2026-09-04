"""Transition-uniform samplers; Multi-Q is exactly policy balanced over time."""
from __future__ import annotations
import numpy as np
from stage2_new_dataset import POLICIES

class RNNTransitionSampler:
    def __init__(self, dataset, seed): self.dataset, self.rng = dataset, np.random.default_rng(seed)
    def sample(self, batch_size): return self.dataset.batch(self.rng.integers(self.dataset.transition_count, size=int(batch_size)))

class MultiPolicyBalancedSampler:
    def __init__(self, datasets, seed):
        if set(datasets) != set(POLICIES): raise ValueError(f"Expected policies {POLICIES}")
        self.datasets, self.rng, self.batch_index = datasets, np.random.default_rng(seed), 0
        self.counts = {policy: 0 for policy in POLICIES}
    def allocation(self, batch_size):
        base, remainder = divmod(int(batch_size), len(POLICIES)); result = {policy: base for policy in POLICIES}
        for offset in range(remainder): result[POLICIES[(self.batch_index + offset) % len(POLICIES)]] += 1
        self.batch_index = (self.batch_index + remainder) % len(POLICIES); return result
    def sample(self, batch_size):
        allocation = self.allocation(batch_size); parts = []
        for policy in POLICIES:
            count = allocation[policy]; self.counts[policy] += count
            parts.append(self.datasets[policy].batch(self.rng.integers(self.datasets[policy].transition_count, size=count)))
        result = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
        order = self.rng.permutation(int(batch_size)); return {key: value[order] for key, value in result.items()}
    def proportions(self):
        total = sum(self.counts.values()); return {key: (self.counts[key] / total if total else 0.0) for key in POLICIES}
