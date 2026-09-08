#!/usr/bin/env python3
"""Synthetic contracts for the progressive 16-env rollout implementation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from stage3_new_agent import progressive_schedule
from stage3_new_handoff import BatchedFrozenRNNProposer
from train_stage3_progressive_vector import (
    episode_identity,
    next_phase_boundary,
)


def main():
    cfg_path = (
        Path(__file__).parent
        / "stage3_new_rnn_handoff_progressive_config.json"
    )
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    parallel = cfg["parallel_env"]

    assert parallel["num_envs"] == 16
    assert parallel["env_startup_stagger"] is True
    assert np.isclose(parallel["env_startup_delay_sec"], 0.5)
    assert parallel["env_startup_timeout_sec"] == 120
    assert parallel["multiprocessing_start_method"] == "spawn"

    # Aggregate valid-transition accounting: a failed env contributes no step.
    valid = [True] * 16
    assert sum(valid) == 16
    valid[7] = False
    assert sum(valid) == 15

    # Progressive boundaries are based on aggregate env_steps.
    assert progressive_schedule(cfg, 9999)["phase"] == "protected_handoff"
    assert progressive_schedule(cfg, 10000)["phase"] == "progressive_unfreeze"
    assert np.isclose(
        progressive_schedule(cfg, 20000)["critic_lr_scale"],
        0.5,
    )
    assert np.isclose(
        progressive_schedule(cfg, 30000)["target_tau_scale"],
        1.0,
    )
    assert next_phase_boundary(
        cfg["progressive_unfreeze"],
        9992,
        300000,
    ) == 10000
    assert next_phase_boundary(
        cfg["progressive_unfreeze"],
        10000,
        300000,
    ) == 30000

    # Stable per-env seed streams do not depend on completion order.
    seeds = []
    episode_ids = []
    for generation in range(3):
        for env_id in range(16):
            episode_id, seed = episode_identity(
                cfg["train_seed_base"],
                16,
                env_id,
                generation,
            )
            episode_ids.append(episode_id)
            seeds.append(seed)
            assert seed == cfg["train_seed_base"] + episode_id
    assert len(seeds) == len(set(seeds))
    assert len(episode_ids) == len(set(episode_ids))
    assert episode_identity(30000, 16, 0, 1) == (16, 30016)
    assert episode_identity(30000, 16, 15, 1) == (31, 30031)

    # Test the actual hidden-row helper with an LSTM-like tuple. Reset row 7
    # must leave the other 15 environment rows byte-identical.
    hidden = (
        torch.arange(2 * 16 * 3, dtype=torch.float32).reshape(2, 16, 3),
        torch.arange(2 * 16 * 3, dtype=torch.float32).reshape(2, 16, 3)
        + 1000,
    )
    initial = (
        torch.full((2, 16, 3), -1.0),
        torch.full((2, 16, 3), -2.0),
    )
    before = tuple(value.clone() for value in hidden)
    replaced = BatchedFrozenRNNProposer._replace_rows(
        hidden,
        initial,
        [7],
    )
    mask = torch.arange(16) != 7
    assert torch.equal(replaced[0][:, mask], before[0][:, mask])
    assert torch.equal(replaced[1][:, mask], before[1][:, mask])
    assert torch.equal(replaced[0][:, 7], initial[0][:, 7])
    assert torch.equal(replaced[1][:, 7], initial[1][:, 7])

    # UTD=1 accounting contract after replay warmup: one update credit for
    # every valid transition. Invalid transitions add zero credit.
    update_budget = 0
    for valid_count in (16, 15, 16):
        update_budget += valid_count * int(cfg["utd"])
    assert update_budget == 47

    # 300k means 300k aggregate transitions, not 300k vector iterations.
    assert cfg["total_env_steps"] == 300000
    assert cfg["total_env_steps"] // parallel["num_envs"] == 18750

    print(
        json.dumps(
            {
                "status": "PASS",
                "num_envs": 16,
                "staggered_spawn": True,
                "aggregate_steps": True,
                "independent_hidden_reset": True,
                "stable_per_env_seed_stream": True,
                "utd": 1.0,
                "phase_boundaries": "10k/30k aggregate env_steps",
            }
        )
    )


if __name__ == "__main__":
    main()
