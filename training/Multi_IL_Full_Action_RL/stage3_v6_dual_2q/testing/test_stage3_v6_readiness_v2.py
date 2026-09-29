#!/usr/bin/env python3
"""CPU-only structural/unit checks for production V6 Readiness V2."""
from __future__ import annotations

import json
import pickle
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

V6 = Path(__file__).resolve().parents[1]
V5 = V6.parent / "stage3_v5_rgmm_td3"
for folder in (V6, V5):
    sys.path.insert(0, str(folder))

from stage3_v5_schedule import TrainingState
from stage3_v6_readiness import (
    CriticHandoffV2, HandoffStateV2, REFERENCE_SOURCE,
    assess_v2, qmean_spearman_from_episodes, replay_metrics_v2,
)
from stage3_v6_target import TargetSelector2Q

CONFIG = json.loads((V6 / "stage3_v6_config.json").read_text())
RULES = CONFIG["critic_readiness_v2"]


def metrics(step=120000, e=1.0):
    return {
        "env_steps": step, "completed_episodes": 150,
        "success_episodes": 30, "failure_episodes": 120,
        "diagnostic_available": True, "fixed_diagnostic_sha256": "fixed",
        "qmean_mc_spearman": 0.75,
        "td_e1_mae": e, "td_e2_mae": e, "td_emax_mae": e,
        "diagnostic_target_finite": True, "diagnostic_arrays_finite": True,
        "training_numeric_finite": True,
        "qmean_mean": 0.0, "qmean_std": 1.0,
        "qmean_reference_mean": 0.0, "qmean_reference_std": 1.0,
    }


def history(values=(1.0, 1.0), steps=(100000, 110000)):
    result = []
    for step, value in zip(steps, values):
        row = metrics(step, value)
        result.append(row)
    return result


class ReadinessV2Tests(unittest.TestCase):
    def check(self, current=None, earlier=None):
        return assess_v2(current or metrics(), earlier if earlier is not None else history(), RULES)

    def test_data_ready_boundaries(self):
        for total, success, failure in ((149, 30, 119), (150, 29, 121), (150, 121, 29)):
            row = metrics(); row.update(completed_episodes=total,
                                        success_episodes=success, failure_episodes=failure)
            self.assertFalse(self.check(row)["data_ready"])
        self.assertTrue(self.check()["data_ready"])

    def test_rank_uses_qmean_not_qmin(self):
        ep = {"rewards": np.asarray([-1., -1., 3.]), "actions": np.zeros((3, 1))}
        q1 = np.asarray([.9, -10., -11.]); q2 = np.asarray([1.1, 14., 17.])
        score, _, finite = qmean_spearman_from_episodes([(q1, q2)], [ep], 1.0)
        self.assertTrue(finite)
        self.assertGreaterEqual(score, .70)
        qmin = np.minimum(q1, q2)
        qmin_score, _, _ = qmean_spearman_from_episodes(
            [(qmin, qmin)], [ep], 1.0)
        self.assertLess(qmin_score, .70)
        self.assertTrue(self.check()["rank_ready"])
        row = metrics(); row["qmean_mc_spearman"] = .69
        self.assertFalse(self.check(row)["rank_ready"])

    def test_td_history_and_boundaries(self):
        for prior in ([], history()[:1]):
            result = self.check(earlier=prior)
            self.assertFalse(result["td_history_ready"])
            self.assertFalse(result["td_healthy"])
        self.assertTrue(self.check()["td_history_ready"])
        self.assertTrue(self.check()["td_healthy"])
        self.assertTrue(self.check(metrics(e=1.4), history((1., 1.5)))["td_healthy"])
        bad = self.check(metrics(e=1.8225), history((1., 1.35)))
        self.assertTrue(bad["td_sustained_worsening"])
        self.assertFalse(bad["td_healthy"])
        self.assertTrue(self.check(metrics(e=.5), history((1., .7)))["td_healthy"])
        self.assertFalse(self.check(metrics(step=110000), history()[:1])["overall_ready_v2"])

    def test_scale_mean_and_std_edges(self):
        for offset, expected in ((.49, True), (.50, True), (.5001, False)):
            row = metrics(); row["qmean_mean"] = offset
            self.assertEqual(self.check(row)["q_scale_safe"], expected)
        for ratio, expected in ((1.49, True), (1.50, True), (1.51, False),
                                (1/1.49, True), (1/1.51, False)):
            row = metrics(); row["qmean_std"] = ratio
            self.assertEqual(self.check(row)["q_scale_safe"], expected)
        row = metrics(); row["qmean_std"] = 1e-7
        self.assertTrue(self.check(row)["numeric_catastrophic"])

    def test_nonfinite_and_diagnostics_only(self):
        for name in ("Q1 NaN", "Q2 Inf"):
            row = metrics(); row["diagnostic_arrays_finite"] = False
            self.assertFalse(self.check(row)["numeric_safe"], name)
            self.assertTrue(self.check(row)["numeric_catastrophic"], name)
        row = metrics(); row["diagnostic_target_finite"] = False
        self.assertFalse(self.check(row)["numeric_safe"])
        row = metrics(e=float("inf"))
        self.assertFalse(self.check(row)["td_healthy"])
        self.assertTrue(self.check(row)["numeric_catastrophic"])
        row = metrics(); row["training_numeric_finite"] = False
        self.assertFalse(self.check(row)["numeric_safe"])
        row = metrics(); row["twin_q_disagreement_p95"] = 1.0
        row["td_plateau"] = False; row["ood_stress_pass"] = False
        self.assertTrue(self.check(row)["overall_ready_v2"])

    def test_streak_warning_and_resume(self):
        handoff = CriticHandoffV2(CONFIG)
        self.assertFalse(handoff.readiness_due(99000))
        self.assertTrue(handoff.readiness_due(100000))
        handoff.state.readiness_history = history()
        first = handoff.submit_readiness(metrics())
        self.assertEqual(first["readiness_pass_streak"], 1)
        self.assertEqual(handoff.state.state, TrainingState.CRITIC_ONLY)
        # Simulate checkpoint/resume while streak=1, then verify the next check opens.
        resumed_state = HandoffStateV2.restore(handoff.state.serialize())
        self.assertEqual(resumed_state.consecutive_readiness_passes, 1)
        handoff = CriticHandoffV2(CONFIG, resumed_state)
        second = handoff.submit_readiness(metrics(step=130000))
        self.assertEqual(second["readiness_pass_streak"], 2)
        self.assertEqual(handoff.state.state, TrainingState.ACTOR_WARMUP)
        self.assertEqual(handoff.state.critic_ready_step, 130000)
        restored = HandoffStateV2.restore(handoff.state.serialize())
        self.assertEqual(restored.consecutive_readiness_passes, 2)
        self.assertEqual(restored.critic_ready_step, 130000)
        self.assertEqual(restored.state, TrainingState.ACTOR_WARMUP)
        other = CriticHandoffV2(CONFIG)
        other.state.readiness_history = history()
        fail = metrics(); fail["qmean_mc_spearman"] = .69
        self.assertEqual(other.submit_readiness(fail)["readiness_pass_streak"], 0)
        self.assertFalse(other.warning_if_timed_out(299999))
        self.assertTrue(other.warning_if_timed_out(300000))
        self.assertFalse(other.warning_if_timed_out(310000))
        self.assertEqual(other.state.state, TrainingState.CRITIC_ONLY)

    def test_training_metric_latch_and_shared_formula(self):
        for mode in ("mean2q", "random2q"):
            cfg = dict(CONFIG, critic_target_mode=mode)
            h = CriticHandoffV2(cfg)
            h.note_training_metrics({"critic_loss_q1": .1, "critic_loss_q2": .2,
                                     "critic_grad_norm": 1.0})
            self.assertTrue(h.state.training_numeric_finite_latch)
            h.note_training_metrics({"critic_grad_norm": float("inf")})
            self.assertFalse(h.state.training_numeric_finite_latch)
            row = metrics()
            row["training_numeric_finite"] = h.state.training_numeric_finite_latch
            self.assertFalse(assess_v2(row, history(), h.rules)["overall_ready_v2"])

    def test_selector_isolation_and_initial_reference_contract(self):
        class Online:
            rng = np.random.default_rng(4)
            fixed_critic_diagnostic_set = {
                "seed": 1, "episodes": [{"observations": np.zeros((3, 59)),
                                          "actions": np.zeros((3, 14)),
                                          "rewards": np.array([0., 0., 1.]),
                                          "episode_steps": np.arange(3), "success": True}],
                "sequences": {"observations": np.zeros((2, 10, 59)),
                              "actions": np.zeros((2, 10, 14)),
                              "rewards": np.zeros((2, 10, 1)),
                              "episode_steps": np.zeros((2, 10), int)},
            }
        class Agent:
            target_selector = TargetSelector2Q("random2q", 123)
            def target_selector_state_dict(self):
                return self.target_selector.state_dict()
            def initial_qmean_on_sequences(self, sequences):
                return np.array([0., 2.])
            def q_values_for_episodes(self, episodes):
                return [(np.array([0., 1., 2.]), np.array([0., 1., 2.]))]
            def fixed_td_diagnostic(self, sequences):
                return {"q1": np.array([0., 2.]), "q2": np.array([0., 2.]),
                        "td_target": np.array([0., 2.])}
        agent, online, state = Agent(), Online(), HandoffStateV2()
        selector_before = pickle.dumps(agent.target_selector_state_dict())
        legacy = {"env_steps": 120000, "completed_episodes": 150,
                  "success_episodes": 30, "failure_episodes": 120,
                  "diagnostic_available": True, "td_mae": 0.0,
                  "twin_q_disagreement_p95": 1.0, "td_plateau": False}
        with mock.patch("stage3_v6_readiness._replay_metrics", return_value=legacy):
            out = replay_metrics_v2(agent, online, 150, 30, CONFIG, state, 120000)
        self.assertEqual(selector_before, pickle.dumps(agent.target_selector_state_dict()))
        self.assertEqual(out["qmean_reference_source"], REFERENCE_SOURCE)
        self.assertEqual(out["qmean_reference_mean"], 1.0)
        self.assertEqual(out["qmean_reference_std"], 1.0)
        self.assertEqual(out["qmean_mc_spearman"], 1.0)
        self.assertEqual(state.fixed_diagnostic_sha256, out["fixed_diagnostic_sha256"])
        restored = HandoffStateV2.restore(state.serialize())
        self.assertEqual(restored.fixed_diagnostic_sha256, state.fixed_diagnostic_sha256)


if __name__ == "__main__":
    unittest.main(verbosity=2)
