#!/usr/bin/env python3
"""Pure Stage3-V7 logic regression. No simulator, no checkpoint loading, no NPU."""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RL = HERE.parents[1]
for name in ("stage3_v5_rgmm_td3", "stage3_v6_dual_2q", "stage3_v7_pirlnav_schedule"):
    sys.path.insert(0, str(RL / name))

from stage3_v7_schedule import (
    CriticHandoffV7, HandoffStateV7, TrainingStateV7, validate_v7_config
)


def cfg():
    return {
        "critic_lr": 3e-4, "actor_lr": 2e-6,
        "critic_target_mode": "random2q", "policy_delay": 4, "utd": .25,
        "critic_readiness_v7": dict(
            min_online_steps=100000, check_interval_steps=10000,
            min_completed_episodes=150, min_success_episodes=30,
            min_failure_episodes=30, min_spearman=.7,
            consecutive_passes=2),
        "v7_schedule": dict(
            critic_decay_env_steps=200000,
            critic_lr_final=7.5e-5, actor_warmup_env_steps=100000),
        "actor_warmup": dict(target_actor_lr=2e-6),
        "evaluation": dict(interval_steps=20000)
    }


def metrics(step, rank=.8, finite=True, success=50, failure=50):
    return dict(env_steps=step, completed_episodes=200, success_episodes=success,
                failure_episodes=failure, diagnostic_available=True,
                qmean_mc_spearman=rank,
                qmean_mean=0.1, qmean_std=.2,
                td_e1_mae=.3, td_e2_mae=.31, td_emax_mae=.31,
                last_critic_loss=.4, last_critic_grad_norm=1.,
                diagnostic_arrays_finite=finite,
                diagnostic_target_finite=finite,
                training_numeric_finite=finite)


def close(x, y):
    assert math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-12), (x, y)


def run():
    c = cfg()
    validate_v7_config(c)
    h = CriticHandoffV7(c)
    assert h.readiness_due(100000)
    assert not h.readiness_due(100001)
    assert not h.submit_readiness(metrics(100000, rank=.5))["overall_ready_v7"]
    assert not h.submit_readiness(metrics(120000))["critic_ready"]
    first = h.submit_readiness(metrics(130000))
    assert first["critic_ready"] and first["transition"] == "CRITIC_ONLY_TO_CRITIC_DECAY"
    assert h.state.critic_ready_step == 130000
    assert h.state.actor_unlock_step == 330000
    assert h.state.joint_rl_start_step == 430000
    for t, expected in [(130000, 3e-4), (230000, 1.875e-4), (329999, 7.5e-5)]:
        row = h.schedule(t, 3e-4)
        assert not row["actor_enabled"] and row["actor_lr"] == 0.
        if t != 329999:
            close(row["critic_lr"], expected)
        else:
            assert 7.5e-5 < row["critic_lr"] < 7.51e-5
    assert not h.evaluation_due(329999)
    row = h.schedule(330000, 3e-4)
    assert h.state.state == TrainingStateV7.ACTOR_WARMUP
    assert row["actor_enabled"] and row["actor_lr"] == 0.
    close(row["critic_lr"], 7.5e-5)
    assert h.evaluation_due(330000)
    assert not h.evaluation_due(330001)
    row = h.schedule(380000, 3e-4)
    close(row["actor_lr"], 1e-6)
    row = h.schedule(430000, 3e-4)
    assert h.state.state == TrainingStateV7.JOINT_RL
    close(row["actor_lr"], 2e-6)
    assert h.evaluation_due(430000)

    # Serialize after actor-unlock/joint; restoring never re-runs the gate.
    packed = h.state.serialize()
    restored = CriticHandoffV7(c, HandoffStateV7.restore(packed))
    assert restored.state.state == TrainingStateV7.JOINT_RL
    assert restored.state.critic_ready_step == 130000
    assert restored.state.next_evaluation_step == h.state.next_evaluation_step
    close(restored.schedule(450000, 3e-4)["actor_lr"], 2e-6)

    # Old 100K V6 state is imported as CRITIC_ONLY without inherited readiness passes.
    imported = HandoffStateV7.from_v6_100k(
        {"state": "CRITIC_ONLY", "critic_ready": False,
         "readiness_history": [{"env_steps": 100000}],
         "consecutive_readiness_passes": 1,
         "fixed_diagnostic_sha256": "original-digest"})
    assert imported.consecutive_readiness_passes == 0
    assert imported.critic_ready_step is None
    assert imported.fixed_diagnostic_sha256 == "original-digest"
    try:
        HandoffStateV7.from_v6_100k(
            {"state": "ACTOR_WARMUP", "critic_ready": True})
        raise AssertionError("Incorrectly accepted already-unlocked V6")
    except RuntimeError:
        pass

    # No 300K critic-only stop; bad data/rank simply remains critic-only.
    waiting = CriticHandoffV7(c)
    assert not waiting.warning_if_timed_out(310000)
    assert not waiting.submit_readiness(metrics(310000, rank=.2))["critic_ready"]
    assert waiting.state.state == TrainingStateV7.CRITIC_ONLY
    bad = waiting.submit_readiness(metrics(320000, finite=False))
    assert bad["numeric_catastrophic"] and not bad["critic_ready"]
    assert waiting.state.state == TrainingStateV7.CRITIC_ONLY

    # Drift/TD plateau and member agreement are intentionally not V7 hard gates.
    c2 = cfg()
    c2["critic_readiness_v7"]["consecutive_passes"] = 2
    assert "td_plateau" not in c2["critic_readiness_v7"]
    print("PASS: dynamic readiness, 200K frozen decay, 100K Actor warmup, "
          "V6 handoff, V7 resume, no 300K timeout")


if __name__ == "__main__":
    run()
