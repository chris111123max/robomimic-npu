"""Regression tests for Stage3-v5 integration of the Stage2.2 horizon-10 Critic."""
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

    def test_previous_action_window_contract(self):
        actions = torch.arange(2 * 10 * 14, dtype=torch.float32).reshape(2, 10, 14)
        steps = torch.stack((torch.arange(10), torch.arange(31, 41)))
        previous = previous_actions(actions, steps)
        self.assertTrue(torch.equal(previous[:, 0], torch.zeros_like(previous[:, 0])))
        self.assertTrue(torch.equal(previous[:, 1:], actions[:, :-1]))

    def test_current_and_successor_are_both_ten_step_zero_state_windows(self):
        actions = torch.arange(2 * 10 * 14, dtype=torch.float32).reshape(2, 10, 14)
        steps = torch.stack((torch.arange(10), torch.arange(31, 41)))
        observations = torch.randn(2, 10, 59)
        successor = torch.randn_like(observations)
        current = encode_replay_contexts(
            self.critic, observations, actions, steps, 700)
        target = encode_replay_contexts(
            self.critic, observations, actions, steps, 700,
            next_observations=successor)
        self.assertEqual(current[0].shape, (2, 10, 96))
        self.assertEqual(target[1].shape, (2, 10, 96))

        # Explicit successor reference:
        # [o_(s+1),0], [o_(s+2),a_(s+1)], ..., [o_(t+1),a_t].
        target_prev = torch.zeros_like(actions)
        target_prev[:, 1:] = actions[:, 1:]
        progress = (steps + 1).float().unsqueeze(-1) / 700.0
        q1_ref, _ = self.critic.q1.encode_history(
            successor, target_prev, progress)
        torch.testing.assert_close(target[0], q1_ref, rtol=1e-5, atol=1e-6)

    def test_more_than_ten_steps_is_rejected(self):
        observations = torch.randn(1, 11, 59)
        actions = torch.randn(1, 11, 14)
        steps = torch.arange(11).reshape(1, -1)
        with self.assertRaises(ValueError):
            encode_replay_contexts(
                self.critic, observations, actions, steps, 700)

    def test_component_and_sampled_q_keep_actor_gradient(self):
        batch, time_steps, modes, action_dim = 2, 10, 5, 14
        observations = torch.randn(batch, time_steps, 59)
        actions = torch.randn(batch, time_steps, action_dim)
        steps = torch.arange(time_steps).repeat(batch, 1)
        contexts = encode_replay_contexts(
            self.critic, observations, actions, steps, 700)
        final_contexts = (contexts[0][:, -1], contexts[1][:, -1])
        means = torch.randn(batch, modes, action_dim, requires_grad=True)
        scales = torch.full_like(means, 0.1)
        logits = torch.randn(batch, modes)
        distribution = torch.distributions.MixtureSameFamily(
            torch.distributions.Categorical(logits=logits),
            torch.distributions.Independent(
                torch.distributions.Normal(means, scales), 1))
        expected, q1, _, _, _ = component_mean_q(
            self.critic, final_contexts, distribution,
            torch.ones(action_dim), torch.zeros(action_dim), twin_min=False)
        (-expected.mean()).backward()
        self.assertEqual(expected.shape, (batch,))
        self.assertEqual(q1.shape, (batch, modes))
        self.assertTrue(torch.isfinite(means.grad).all())
        sampled, *_ = sampled_q(
            self.critic, final_contexts, distribution,
            torch.ones(action_dim), torch.zeros(action_dim), samples=3,
            twin_min=False)
        self.assertEqual(sampled.shape, (batch,))


if __name__ == "__main__":
    unittest.main()
