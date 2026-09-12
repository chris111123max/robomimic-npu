"""Pure metadata checks for safe reuse of an already completed Phase-0 gate."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stage3_v3_phase0_reuse import validate_phase0_reuse


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class Phase0ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.pair = Path(self.temporary.name) / "old_pair"
        shared = self.pair / "shared"
        shared.mkdir(parents=True)
        self.config = {
            "stage": "stage3-v3-rgmm-td3",
            "obs_dim": 59, "action_dim": 14,
            "actor_source_contract": {"checkpoint_sha256": "bc-sha"},
            "actor_gate": {"competence_episodes": 2,
                           "competence_min_successes": 1,
                           "equivalence_tolerance": 1e-5},
            "bc_rnn_checkpoint_sha256": "bc-sha",
            "expert_dataset_sha256": "dataset-sha",
            "training_seed": 7, "evaluation_seeds": [20000, 20001],
            "horizon": 700, "sim_error_handling": {"evaluation_retry_count": 1},
            "policy_delay": 8,
        }
        _write(shared / "config_resolved.json", self.config)
        _write(shared / "pair_fairness.json", {
            "stage": "stage3-v3", "actor_hashes_identical": True,
            "actor_hash": "actor-hash",
        })
        _write(shared / "transfer_validation.json", {
            "checkpoint_sha256": "bc-sha", "source_actor_hash": "actor-hash",
            "new_actor_hash": "actor-hash", "equivalence_pass": True,
            "missing_keys": [], "unexpected_keys": [], "max_abs_diff": 0.0,
        })
        _write(shared / "step0_competence.json", {
            "evaluation_seeds": [20000, 20001], "valid_episodes": 2,
            "sim_error_episodes": 0, "success_count": 1, "success_rate": 0.5,
            "minimum_successes": 1, "competence_pass": True, "env_steps": 0,
            "episodes": [
                {"seed": 20000, "success": True, "sim_error": False},
                {"seed": 20001, "success": False, "sim_error": False},
            ],
        })
        _write(shared / "phase0_gate.json", {
            "env_steps": 0, "eval_success_count": 1, "eval_success_rate": 0.5,
            "equivalence_pass": True, "competence_pass": True,
            "warmup_pass": False, "gate_open": False, "latched": True,
        })

    def test_training_only_change_can_reuse(self):
        target = dict(self.config, policy_delay=4)
        result = validate_phase0_reuse(self.pair, target, "actor-hash")
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(len(result["artifact_sha256"]), 3)

    def test_evaluation_change_is_rejected(self):
        target = dict(self.config, evaluation_seeds=[20000, 20002])
        with self.assertRaisesRegex(RuntimeError, "evaluation_seeds"):
            validate_phase0_reuse(self.pair, target, "actor-hash")

    def test_actor_mismatch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Actor hash"):
            validate_phase0_reuse(self.pair, self.config, "different")

    def test_failed_gate_is_rejected(self):
        path = self.pair / "shared" / "phase0_gate.json"
        gate = json.loads(path.read_text(encoding="utf-8"))
        gate["competence_pass"] = False
        _write(path, gate)
        with self.assertRaisesRegex(RuntimeError, "inconsistent gate"):
            validate_phase0_reuse(self.pair, self.config, "actor-hash")

    def test_nonfinite_transfer_difference_is_rejected(self):
        path = self.pair / "shared" / "transfer_validation.json"
        transfer = json.loads(path.read_text(encoding="utf-8"))
        transfer["max_abs_diff"] = float("nan")
        _write(path, transfer)
        with self.assertRaisesRegex(RuntimeError, "transfer equivalence"):
            validate_phase0_reuse(self.pair, self.config, "actor-hash")


if __name__ == "__main__":
    unittest.main()
