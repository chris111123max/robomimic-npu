"""Boundary-aligned Actor windows; Critic replay retains the v3 contract."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parents[1] / "stage3_v3_rgmm_td3"
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))
from stage3_v3_replay import (CORE, OfflineDemonstrations, OnlineSequenceReplay,
                              final_transition, symmetric_sequence_batch)
from stage3_v4_boundary import aligned_start


def _sample_aligned(source, count, length, horizon):
    episodes = source.episodes if isinstance(source, OfflineDemonstrations) else source._all_episodes()
    eligible = [episode for episode in episodes if len(episode["actions"]) >= length]
    if not eligible:
        raise RuntimeError("No complete boundary-aligned Actor window available")
    count = int(count)
    # Draw the same distribution as the reference implementation: episodes
    # are uniform, then complete horizon-aligned starts are uniform within the
    # selected episode. Vectorized draws remove one RNG call per sample.
    episode_ids = source.rng.integers(len(eligible), size=count)
    window_counts = np.asarray(
        [(len(episode["actions"]) - int(length)) // int(horizon) + 1
         for episode in eligible], dtype=np.int64)
    start_ids = source.rng.integers(
        window_counts[episode_ids], size=count) * int(horizon)
    result = {}
    for key in CORE + ("episode_steps",):
        result[key] = np.stack([
            np.asarray(eligible[int(episode_id)][key][int(start):int(start) + int(length)])
            for episode_id, start in zip(episode_ids, start_ids)
        ])
    expected_steps = start_ids[:, None] + np.arange(int(length), dtype=np.int64)[None, :]
    if not np.array_equal(result["episode_steps"], expected_steps):
        raise RuntimeError("Aligned replay sampler produced a non-boundary window")
    return result


def aligned_sequence_batch(offline, online, count, length, horizon=10):
    if int(count) <= 0 or int(count) % 2 or int(length) != int(horizon):
        raise ValueError("Aligned Actor batch requires even count and one RNN horizon")
    half = int(count) // 2
    left = _sample_aligned(offline, half, int(length), int(horizon))
    right = _sample_aligned(online, half, int(length), int(horizon))
    batch = {key: np.concatenate((left[key], right[key]), axis=0)
             for key in CORE + ("episode_steps",)}
    batch["is_offline"] = np.concatenate((np.ones(half, np.float32),
                                           np.zeros(half, np.float32)))
    if not np.all(batch["episode_steps"][:, 0] % horizon == 0):
        raise RuntimeError("Actor batch starts outside a hidden-state reset boundary")
    return batch
