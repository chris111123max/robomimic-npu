#!/usr/bin/env python3
"""Small device-independent stochastic-GMM/Q arithmetic regression."""
from __future__ import annotations

import json

import numpy as np
import torch
from torch import distributions as D

from stage3_v4_gmm_math import single_expected_q, sequence_expected_q
from stage3_v4_boundary import aligned_start


class ToyQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0
        self.q1_calls = 0

    def q1(self, states, actions):
        self.q1_calls += 1
        return ((actions - 0.3).square().sum(-1, keepdim=True)
                + 0.01 * states.sum(-1, keepdim=True))

    def forward(self, states, actions):
        self.forward_calls += 1
        first = self.q1(states, actions)
        return first, first + 0.4


def distribution(mean, raw_std, logits):
    normal = D.Normal(mean, torch.nn.functional.softplus(raw_std) + 1e-4)
    return D.MixtureSameFamily(D.Categorical(logits=logits), D.Independent(normal, 1))


def main():
    torch.manual_seed(57)
    means = torch.nn.Parameter(torch.randn(3, 5, 14) * 0.2)
    raw_std = torch.nn.Parameter(torch.randn(3, 5, 14) * 0.1 - 1)
    logits = torch.nn.Parameter(torch.randn(3, 5) * 0.1)
    critic = ToyQ()
    scale, offset = torch.ones(14), torch.zeros(14)
    states = torch.randn(3, 59)
    eps = torch.ones(3, 5, 1, 14)
    policy = distribution(means, raw_std, logits)
    expected, q1, q2, _, _ = single_expected_q(
        critic, states, policy, scale, offset, twin_min=False, epsilon=eps)
    expected.sum().backward()
    gradients = {"mean": means.grad.norm().item(), "std": raw_std.grad.norm().item(),
                 "logits": logits.grad.norm().item()}
    actor_batched = critic.q1_calls == 1 and q1.shape == (3, 5, 1) and q2 is None

    with torch.no_grad():
        critic.forward_calls = 0
        target, _, _, _, _ = single_expected_q(
            critic, states, policy, scale, offset, twin_min=True, epsilon=eps)
        target_no_grad = not target.requires_grad and critic.forward_calls == 1
    critic.q1_calls = 0
    sequence_eps = torch.ones(3, 2, 5, 1, 14)
    sequence_expected, sequence_q, _, _ = sequence_expected_q(
        critic, states[:, None].expand(-1, 2, -1), [policy, policy], scale, offset,
        epsilon=sequence_eps)
    sequence_batched = (critic.q1_calls == 1 and sequence_q.shape == (3, 2, 5, 1)
                        and torch.allclose(sequence_expected[:, 0], expected))
    multi_eps = torch.ones(3, 5, 3, 14)
    multi_expected, multi_q, _, _, _ = single_expected_q(
        critic, states, policy, scale, offset, samples=3,
        twin_min=False, epsilon=multi_eps)
    multi_sample_supported = (multi_q.shape == (3, 5, 3)
                              and torch.allclose(multi_expected, expected))
    episode = {"episode_steps": np.arange(30), "actions": np.zeros((30, 14))}
    starts = [aligned_start(episode, 10, 10, np.random.default_rng(index))
              for index in range(10)]
    corrupt = {"episode_steps": np.r_[np.arange(9), 11],
               "actions": np.zeros((10, 14))}
    try:
        aligned_start(corrupt, 10, 10, np.random.default_rng(1))
        corrupt_rejected = False
    except RuntimeError:
        corrupt_rejected = True
    checks = {"mean_gradient": gradients["mean"] > 0,
              "std_gradient": gradients["std"] > 0,
              "logit_gradient": gradients["logits"] > 0,
              "actor_one_q_call_for_modes": actor_batched,
              "target_one_twin_call_and_no_grad": target_no_grad,
              "sequence_one_q_call_for_time_and_modes": sequence_batched}
    checks["future_multiple_samples_supported"] = multi_sample_supported
    checks["boundary_windows_aligned"] = all(start in (0, 10, 20) for start in starts)
    checks["corrupt_episode_rejected"] = corrupt_rejected
    status = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps({"status": status, "checks": checks, "gradients": gradients}, indent=2))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
