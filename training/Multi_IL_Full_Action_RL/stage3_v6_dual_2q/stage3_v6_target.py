"""Stage3-v6 two-Q target selection with an isolated RNG stream."""
from __future__ import annotations

import copy
import numpy as np

TARGET_MODES = ("mean2q", "random2q")


class TargetSelector2Q:
    """One selector draw per Critic update without touching global RNG state."""

    def __init__(self, mode, seed):
        if mode not in TARGET_MODES:
            raise ValueError(f"unsupported Stage3-v6 target mode: {mode!r}")
        self.mode = str(mode)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.q1_updates = 0
        self.q2_updates = 0

    def begin_update(self):
        if self.mode == "mean2q":
            return None
        selected = int(self._rng.integers(0, 2))
        if selected == 0:
            self.q1_updates += 1
        else:
            self.q2_updates += 1
        return selected

    def combine(self, q1_expected, q2_expected, selected=None):
        if self.mode == "mean2q":
            if selected is not None:
                raise ValueError("mean2q must not receive a selector index")
            return 0.5 * (q1_expected + q2_expected)
        if selected not in (0, 1):
            raise ValueError("random2q requires selected Q index 0 or 1")
        return q1_expected if int(selected) == 0 else q2_expected

    def diagnostic_expectation(self, q1_expected, q2_expected):
        """Deterministic E_j[Q_j] used by diagnostics; consumes no RNG."""
        return 0.5 * (q1_expected + q2_expected)

    def state_dict(self):
        return {
            "mode": self.mode,
            "seed": self.seed,
            "q1_updates": int(self.q1_updates),
            "q2_updates": int(self.q2_updates),
            "bit_generator_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        if state.get("mode") != self.mode or int(state.get("seed", -1)) != self.seed:
            raise RuntimeError("Stage3-v6 target selector resume contract mismatch")
        self.q1_updates = int(state.get("q1_updates", 0))
        self.q2_updates = int(state.get("q2_updates", 0))
        self._rng.bit_generator.state = copy.deepcopy(state["bit_generator_state"])

    def audit(self):
        total = int(self.q1_updates + self.q2_updates)
        return {
            "mode": self.mode,
            "rng_stream": "dedicated_numpy_generator_not_global_training_rng",
            "seed": self.seed,
            "granularity": "one_target_q_for_entire_critic_update_minibatch",
            "q1_updates": int(self.q1_updates),
            "q2_updates": int(self.q2_updates),
            "random_updates": total if self.mode == "random2q" else 0,
        }
