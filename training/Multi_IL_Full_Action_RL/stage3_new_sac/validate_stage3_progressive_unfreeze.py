#!/usr/bin/env python3
"""Synthetic sanity checks for protected handoff and progressive unfreezing."""
from __future__ import annotations

import json

import numpy as np
import torch

from stage3_new_agent import (
    Stage3SAC,
    build_actor,
    build_critic,
    progressive_schedule,
    state_hash,
)


def config():
    return {
        "hidden_dims": [256, 256],
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "alpha_lr": 3e-4,
        "critic_weight_decay": 1e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "target_entropy": -14.0,
        "alpha_init": 0.01,
        "progressive_unfreeze": {
            "enabled": True,
            "protected_until_env_steps": 10000,
            "unfreeze_end_env_steps": 30000,
        },
        "cql": {
            "enabled": True,
            "lambda": 0.1,
            "num_random_actions": 10,
            "num_policy_actions": 1,
            "apply_to_expert": True,
            "apply_to_online": True,
            "detach_policy_actions": True,
        },
        "anchor": {"enabled": False},
        "handoff": {
            "enabled": True,
            "lambda_handoff": 1.0,
        },
    }


def batch(n=16):
    rng = np.random.default_rng(19)
    return {
        "observations": rng.normal(size=(n, 59)).astype(np.float32),
        "actions": rng.uniform(-1, 1, size=(n, 14)).astype(np.float32),
        "rewards": rng.integers(0, 2, size=(n, 1)).astype(np.float32),
        "next_observations": rng.normal(size=(n, 59)).astype(np.float32),
        "terminals": np.zeros((n, 1), np.float32),
        "action_rl": rng.uniform(-1, 1, size=(n, 14)).astype(np.float32),
        "action_rnn": rng.uniform(-1, 1, size=(n, 14)).astype(np.float32),
        "rnn_next_actions": rng.uniform(
            -1, 1, size=(n, 14)
        ).astype(np.float32),
        # Force RNN-selected online samples so handoff has a gradient.
        "selected_source": np.zeros((n, 1), np.float32),
        "is_online": np.ones((n, 1), np.float32),
    }


def main():
    cfg = config()

    schedules = {
        step: progressive_schedule(cfg, step)
        for step in (0, 9999, 10000, 20000, 30000)
    }
    assert schedules[0]["phase"] == "protected_handoff"
    assert schedules[9999]["phase"] == "protected_handoff"
    assert schedules[0]["critic_lr_effective"] == 0
    assert schedules[0]["target_tau_effective"] == 0

    assert schedules[10000]["phase"] == "progressive_unfreeze"
    assert schedules[10000]["critic_lr_effective"] == 0
    assert schedules[10000]["target_tau_effective"] == 0
    assert schedules[10000]["actor_sac_enabled"]
    assert schedules[10000]["alpha_tuning_enabled"]
    assert not schedules[10000]["critic_update_enabled"]

    assert np.isclose(
        schedules[20000]["critic_lr_effective"],
        0.5 * cfg["critic_lr"],
    )
    assert np.isclose(
        schedules[20000]["target_tau_effective"],
        0.5 * cfg["tau"],
    )

    assert schedules[30000]["phase"] == "full_sac"
    assert np.isclose(
        schedules[30000]["critic_lr_effective"],
        cfg["critic_lr"],
    )
    assert np.isclose(
        schedules[30000]["target_tau_effective"],
        cfg["tau"],
    )

    torch.manual_seed(23)
    agent = Stage3SAC(
        build_actor(cfg, "cpu"),
        build_critic(
            59,
            14,
            [256, 256],
            "relu",
            True,
            "cpu",
        ),
        cfg,
        "cpu",
    )

    critic0 = state_hash(agent.critic)
    target0 = state_hash(agent.target)
    actor0 = state_hash(agent.actor)
    alpha0 = float(agent.alpha.item())

    protected = agent.update(batch(), env_steps=5000)
    assert protected["phase"] == "protected_handoff"
    assert protected["actor_sac_loss"] == 0
    assert protected["cql_loss_raw"] == 0
    assert state_hash(agent.critic) == critic0
    assert state_hash(agent.target) == target0
    assert float(agent.alpha.item()) == alpha0
    assert state_hash(agent.actor) != actor0
    assert any(
        parameter.grad is not None
        for parameter in agent.actor.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in agent.critic.parameters()
    )

    boundary = agent.update(batch(), env_steps=10000)
    assert boundary["critic_lr_effective"] == 0
    assert boundary["target_tau_effective"] == 0
    # At exactly 10k, SAC Actor + alpha resume, but Critic/tau remain zero.
    assert state_hash(agent.critic) == critic0
    assert state_hash(agent.target) == target0
    assert not np.isclose(float(agent.alpha.item()), alpha0)

    midpoint = agent.update(batch(), env_steps=20000)
    assert np.isclose(
        midpoint["critic_lr_effective"],
        1.5e-4,
    )
    assert np.isclose(
        midpoint["target_tau_effective"],
        0.0025,
    )
    assert state_hash(agent.critic) != critic0

    full = agent.update(batch(), env_steps=30000)
    assert np.isclose(full["critic_lr_effective"], 3e-4)
    assert np.isclose(full["target_tau_effective"], 0.005)

    try:
        bad = config()
        bad["progressive_unfreeze"]["unfreeze_end_env_steps"] = 10000
        progressive_schedule(bad, 10000)
    except ValueError:
        invalid_schedule_rejected = True
    else:
        invalid_schedule_rejected = False
    assert invalid_schedule_rejected

    print(
        json.dumps(
            {
                "status": "PASS",
                "phase_a_critic_frozen": True,
                "phase_a_target_frozen": True,
                "phase_a_actor_changed": True,
                "phase_a_alpha_fixed": True,
                "phase_b_actor_alpha_resume_at_10k": True,
                "schedule_10k_20k_30k": True,
                "invalid_schedule_rejected": True,
            }
        )
    )


if __name__ == "__main__":
    main()
