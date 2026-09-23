"""Regression tests for the Stage2.2 recurrent Critic integration."""
import json
import sys
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
STAGE2_2 = HERE.parent / "stage2_2_history_aware_critic"
for path in (HERE, STAGE2_2):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from history_critic import build_critic
from stage3_v5_history_critic import (component_mean_q, encode_replay_contexts,
                                      previous_actions, sampled_q)


class TestHistoryCriticIntegration(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((STAGE2_2 / "stage2_2_config.json").read_text())
        torch.manual_seed(9)
        self.critic = build_critic(self.config, torch.device("cpu"))

    def test_previous_action_and_successor_tokens(self):
        actions = torch.arange(2 * 4 * 14, dtype=torch.float32).reshape(2, 4, 14)
        steps = torch.tensor([[0, 1, 2, 3], [7, 8, 9, 10]])
        previous = previous_actions(actions, steps)
        self.assertTrue(torch.equal(previous[:, 1:], actions[:, :-1]))
        self.assertTrue(torch.equal(previous[:, 0], torch.zeros_like(previous[:, 0])))
        observations = torch.randn(2, 4, 59)
        successor = torch.randn_like(observations)
        current = encode_replay_contexts(
            self.critic, observations, actions, steps, 700)
        target = encode_replay_contexts(
            self.critic, observations, actions, steps, 700,
            next_observations=successor)
        self.assertEqual(current[0].shape, (2, 4, 96))
        self.assertEqual(target[1].shape, (2, 4, 96))

    def test_component_and_sampled_q_keep_actor_gradient(self):
        batch, time_steps, modes, action_dim = 2, 3, 5, 14
        observations = torch.randn(batch, time_steps, 59)
        actions = torch.randn(batch, time_steps, action_dim)
        steps = torch.arange(time_steps).repeat(batch, 1)
        contexts = encode_replay_contexts(
            self.critic, observations, actions, steps, 700)
        means = torch.randn(batch, time_steps, modes, action_dim,
                            requires_grad=True)
        scales = torch.full_like(means, 0.1)
        logits = torch.randn(batch, time_steps, modes)
        distribution = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=logits),
            torch.distributions.Independent(
                torch.distributions.Normal(means, scales), 1))
        expected, q1, _, _, _ = component_mean_q(
            self.critic, contexts, distribution,
            torch.ones(action_dim), torch.zeros(action_dim), twin_min=False)
        (-expected.mean()).backward()
        self.assertEqual(expected.shape, (batch, time_steps))
        self.assertEqual(q1.shape, (batch, time_steps, modes))
        self.assertTrue(torch.isfinite(means.grad).all())
        sampled, *_ = sampled_q(
            self.critic, contexts, distribution,
            torch.ones(action_dim), torch.zeros(action_dim), samples=3,
            twin_min=False)
        self.assertEqual(sampled.shape, (batch, time_steps))


if __name__ == "__main__":
    unittest.main()
