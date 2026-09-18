import copy
import json
import unittest
from pathlib import Path

from stage3_v5_schedule import CriticHandoff, TrainingState


CONFIG = json.loads(Path(__file__).with_name("stage3_v5_config.json").read_text())


def metrics(step=100000, **overrides):
    result = {"env_steps": step, "completed_episodes": 150, "success_episodes": 30,
              "failure_episodes": 30, "spearman_q_return": .70, "success_failure_auc": .80,
              "delta_q": .1, "twin_q_disagreement_median": .10,
              "twin_q_disagreement_p95": .25, "td_plateau": True,
              "td_error_worsening": False, "q_scale_stable": True,
              "ood_stress_pass": True, "finite": True}
    result.update(overrides); return result


class TestV5Handoff(unittest.TestCase):
    def test_no_ready_before_100k(self):
        handoff = CriticHandoff(copy.deepcopy(CONFIG))
        record = handoff.submit_readiness(metrics(99999))
        self.assertFalse(record["overall_pass"]); self.assertEqual(handoff.state.state, TrainingState.CRITIC_ONLY)

    def test_three_passes_then_ready_and_warmup_formula(self):
        handoff = CriticHandoff(copy.deepcopy(CONFIG))
        for step in (100000, 110000, 120000): handoff.submit_readiness(metrics(step))
        self.assertEqual(handoff.state.state, TrainingState.ACTOR_WARMUP)
        self.assertEqual(handoff.state.critic_ready_step, 120000)
        self.assertEqual(handoff.state.actor_warmup_steps, 120000)

    def test_fail_resets_counter_and_timeout(self):
        handoff = CriticHandoff(copy.deepcopy(CONFIG))
        handoff.submit_readiness(metrics()); handoff.submit_readiness(metrics(110000, success_failure_auc=.79))
        self.assertEqual(handoff.state.consecutive_readiness_passes, 0)
        self.assertTrue(handoff.fail_if_timed_out(300000))
        self.assertEqual(handoff.state.state, TrainingState.CRITIC_NOT_READY)

    def test_step_based_learning_rates_and_evaluation(self):
        handoff = CriticHandoff(copy.deepcopy(CONFIG))
        for step in (100000, 110000, 120000): handoff.submit_readiness(metrics(step))
        start = handoff.state.critic_ready_step
        schedule = handoff.schedule(start, .0003)
        self.assertEqual(schedule["actor_lr"], 0.0)
        end = start + handoff.state.actor_warmup_steps
        schedule = handoff.schedule(end, .0003)
        self.assertAlmostEqual(schedule["actor_lr"], 2e-6)
        self.assertAlmostEqual(schedule["critic_lr"], .000075)
        self.assertTrue(handoff.evaluation_due(end))
        self.assertFalse(handoff.evaluation_due(end + 1))


if __name__ == "__main__":
    unittest.main()
