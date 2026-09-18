"""Proof-enforcing sampling of one complete zero-hidden RNN horizon."""
from __future__ import annotations

import numpy as np


def aligned_start(episode, length, horizon, rng):
    steps = episode["episode_steps"]
    if len(steps) != len(episode["actions"]) or int(np.asarray(steps[0])) != 0:
        raise RuntimeError("Episode timestep contract does not start at reset")
    windows = (len(steps) - length) // horizon + 1
    if windows <= 0:
        raise RuntimeError("Episode has no complete boundary-aligned window")
    start = int(rng.integers(windows)) * horizon
    window_steps = np.asarray(steps[start:start + length], dtype=np.int64).reshape(-1)
    if not np.array_equal(window_steps, np.arange(start, start + length)):
        raise RuntimeError("Recurrent window crossed a reset or episode boundary")
    return start
