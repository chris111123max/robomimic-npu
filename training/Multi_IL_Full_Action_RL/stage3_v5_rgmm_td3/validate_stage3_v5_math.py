#!/usr/bin/env python3
"""Small device-independent low-noise component-mean Q regression."""
from __future__ import annotations

import json

import numpy as np
import torch
from torch import distributions as D

from stage3_v5_gmm_math import (single_component_mean_q, sequence_component_mean_q,
                                 single_expected_q)
from stage3_v5_boundary import aligned_start


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


class CrossingTwinQ(ToyQ):
    def forward(self, states, actions):
        self.forward_calls += 1
        first = self.q1(states, actions)
        second = 1.0 - first
        return first, second


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
    policy = distribution(means, raw_std, logits)
    expected, q1, q2, _, _ = single_component_mean_q(
        critic, states, policy, scale, offset, twin_min=False)
    expected.sum().backward()
    gradients = {"mean": means.grad.norm().item(), "std": (raw_std.grad.norm().item()
                                                            if raw_std.grad is not None else 0.0),
                 "logits": logits.grad.norm().item()}
    actor_batched = critic.q1_calls == 1 and q1.shape == (3, 5) and q2 is None

    with torch.no_grad():
        critic.forward_calls = 0
        target, _, _, _, _ = single_component_mean_q(
            critic, states, policy, scale, offset, twin_min=True)
        target_no_grad = not target.requires_grad and critic.forward_calls == 1
        crossing = CrossingTwinQ()
        nontrivial_scale = torch.full((14,), 1.5)
        nontrivial_offset = torch.full((14,), -0.2)
        crossing_target, crossing_q1, crossing_q2, crossing_params, crossing_actions = (
            single_component_mean_q(crossing, states, policy,
                                    nontrivial_scale, nontrivial_offset, twin_min=True))
        manual_target = (crossing_params["probs"] * torch.minimum(
            crossing_q1, crossing_q2)).sum(-1)
        target_per_mode_min_and_denorm = (
            crossing.forward_calls == 1
            and torch.allclose(crossing_target, manual_target)
            and torch.allclose(crossing_actions,
                               means * nontrivial_scale + nontrivial_offset))
    critic.q1_calls = 0
    sequence_expected, sequence_q, _, _ = sequence_component_mean_q(
        critic, states[:, None].expand(-1, 2, -1), [policy, policy], scale, offset)
    sequence_batched = (critic.q1_calls == 1 and sequence_q.shape == (3, 2, 5)
                        and torch.allclose(sequence_expected[:, 0], expected))
    with torch.no_grad():
        sampled_diagnostic, _, _, _, _ = single_expected_q(
            critic, states, policy, scale, offset, samples=1,
            twin_min=False, epsilon=torch.ones(3, 5, 1, 14))
    diagnostic_only = not sampled_diagnostic.requires_grad and not torch.allclose(
        sampled_diagnostic, expected)
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
              "std_head_no_rl_gradient": gradients["std"] == 0,
              "logit_gradient": gradients["logits"] > 0,
              "actor_one_q_call_for_modes": actor_batched,
              "target_one_twin_call_and_no_grad": target_no_grad,
              "target_min_per_mode_and_action_denorm": target_per_mode_min_and_denorm,
              "sequence_one_q_call_for_time_and_modes": sequence_batched}
    checks["sampled_learned_std_diagnostic_only"] = diagnostic_only
    checks["boundary_windows_aligned"] = all(start in (0, 10, 20) for start in starts)
    checks["corrupt_episode_rejected"] = corrupt_rejected
    status = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps({"status": status, "checks": checks, "gradients": gradients}, indent=2))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
