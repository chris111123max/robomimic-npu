"""Temporal-history reconstruction helpers."""

import numpy as np

from utils.policy_loader import warm_history
from utils.rng_utils import restore_rng_state, seed_policy_rng


def reconstruct_before_action(policy, observations, branch_step, rng_state=None, trial_seed=None):
    """Return policy state immediately before consuming obs_t (never double-feeds obs_t)."""
    history_length = warm_history(policy, observations, int(branch_step))
    if rng_state is not None:
        exact_rng, reason = restore_rng_state(rng_state)
    else:
        seed_policy_rng(int(trial_seed))
        exact_rng, reason = False, "formal branch uses an explicit trial seed"
    return {"history_length": history_length, "exact_rng_restored": exact_rng,
            "rng_limitation": reason}


def action_matches(original, reconstructed, atol):
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    return bool(np.allclose(original, reconstructed, rtol=0.0, atol=float(atol))), \
           float(np.max(np.abs(original - reconstructed)))
