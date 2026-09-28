#!/usr/bin/env python3
"""CPU structural checks for the Stage3-v6 target selector."""
from __future__ import annotations

import copy
import numpy as np

from stage3_v6_target import TargetSelector2Q


def main():
    q1 = np.asarray([1.0, 2.0, 3.0])
    q2 = np.asarray([3.0, 4.0, 5.0])

    mean = TargetSelector2Q("mean2q", 123)
    assert mean.begin_update() is None
    assert np.array_equal(mean.combine(q1, q2), np.asarray([2.0, 3.0, 4.0]))
    assert mean.audit()["random_updates"] == 0

    # Dedicated Generator must not perturb numpy's legacy global RNG stream.
    np.random.seed(991)
    expected_global = np.random.random(4)
    np.random.seed(991)
    random_selector = TargetSelector2Q("random2q", 456)
    sequence = [random_selector.begin_update() for _ in range(100)]
    observed_global = np.random.random(4)
    assert np.array_equal(expected_global, observed_global)
    assert set(sequence) <= {0, 1}
    assert (
        random_selector.q1_updates + random_selector.q2_updates == 100
    )

    # Resume must continue the exact selector sequence.
    state = copy.deepcopy(random_selector.state_dict())
    continuation = [random_selector.begin_update() for _ in range(50)]
    restored = TargetSelector2Q("random2q", 456)
    restored.load_state_dict(state)
    restored_continuation = [restored.begin_update() for _ in range(50)]
    assert continuation == restored_continuation

    # Diagnostics use the selector expectation without advancing counts.
    before = restored.audit()
    value = restored.diagnostic_expectation(q1, q2)
    after = restored.audit()
    assert np.array_equal(value, np.asarray([2.0, 3.0, 4.0]))
    assert before == after

    print({
        "status": "PASS",
        "mean2q": "(Q1+Q2)/2",
        "random2q": "one Q per Critic update",
        "global_numpy_rng_untouched": True,
        "resume_exact": True,
    })


if __name__ == "__main__":
    main()
