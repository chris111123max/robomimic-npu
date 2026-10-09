"""Stage3-V7: simple dynamic readiness, PIRLNav-inspired *sequential* handoff.

Readiness starts a frozen-Actor Critic LR decay, NOT an immediate Actor unfreeze.
All counters refer to aggregate environment transitions, not PPO updates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from stage3_v5_schedule import CriticHandoff, HandoffState
from stage3_v6_readiness import HandoffStateV2, CriticHandoffV2, replay_metrics_v2, fixed_set_digest


class TrainingStateV7(str, Enum):
    CRITIC_ONLY = "CRITIC_ONLY"
    CRITIC_DECAY = "CRITIC_DECAY"
    ACTOR_WARMUP = "ACTOR_WARMUP"
    JOINT_RL = "JOINT_RL"


@dataclass
class HandoffStateV7(HandoffStateV2):
    state: TrainingStateV7 = TrainingStateV7.CRITIC_ONLY
    critic_decay_start_step: int | None = None
    actor_unlock_step: int | None = None

    @classmethod
    def restore(cls, payload):
        value = dict(payload)
        value["state"] = TrainingStateV7(value["state"])
        return cls(**value)

    @classmethod
    def from_v6_100k(cls, payload):
        # 100K V6 is expected to be CRITIC_ONLY with no Actor updates.
        if payload.get("state") != "CRITIC_ONLY" or payload.get("critic_ready"):
            raise RuntimeError("V6 100K checkpoint has already crossed readiness")
        fields = cls.__dataclass_fields__
        kwargs = {key: value for key, value in payload.items()
                  if key in fields and key not in ("state", "critic_ready",
                                                 "critic_ready_step", "actor_warmup_steps",
                                                 "joint_rl_start_step",
                                                 "next_evaluation_step",
                                                 "consecutive_readiness_passes")}
        return cls(**kwargs)


class CriticHandoffV7(CriticHandoffV2):
    def __init__(self, config, state=None):
        # CriticHandoffV2 imposes V2 rules; V7 deliberately owns only V7 rules.
        CriticHandoff.__init__(self, config, state or HandoffStateV7())
        self.rules = config["critic_readiness_v7"]

    def readiness_due(self, env_steps):
        return (self.state.state == TrainingStateV7.CRITIC_ONLY
                and int(env_steps) >= int(self.rules["min_online_steps"])
                and int(env_steps) % int(self.rules["check_interval_steps"]) == 0)

    def submit_readiness(self, metrics):
        if self.state.state != TrainingStateV7.CRITIC_ONLY:
            raise RuntimeError("Readiness only allowed in CRITIC_ONLY")
        r = self.rules
        enough = (int(metrics["env_steps"]) >= int(r["min_online_steps"])
                  and int(metrics.get("completed_episodes", 0)) >= int(r["min_completed_episodes"])
                  and int(metrics.get("success_episodes", 0)) >= int(r["min_success_episodes"])
                  and int(metrics.get("failure_episodes", 0)) >= int(r["min_failure_episodes"]))
        rank = metrics.get("qmean_mc_spearman")
        rank_ok = bool(rank is not None and math.isfinite(float(rank))
                       and float(rank) >= float(r["min_spearman"]))
        # Finite gate, not an undocumented Q-scale or TD plateau gate.
        mandatory = ("qmean_mean", "qmean_std", "td_e1_mae", "td_e2_mae",
                     "td_emax_mae", "last_critic_loss", "last_critic_grad_norm")
        finite_values = all(metrics.get(key) is not None
                            and math.isfinite(float(metrics[key])) for key in mandatory)
        numeric_safe = (bool(metrics.get("diagnostic_available", False))
                        and bool(metrics.get("diagnostic_arrays_finite", False))
                        and bool(metrics.get("diagnostic_target_finite", False))
                        and bool(metrics.get("training_numeric_finite", False))
                        and finite_values)
        passed = enough and rank_ok and numeric_safe
        self.state.consecutive_readiness_passes = (
            self.state.consecutive_readiness_passes + 1 if passed else 0)
        record = dict(metrics)
        record.update({
            "readiness_version": 7, "data_ready": bool(enough),
            "rank_ready": bool(rank_ok), "numeric_safe": bool(numeric_safe),
            "overall_ready_v7": bool(passed),
            "readiness_pass_streak": self.state.consecutive_readiness_passes,
            "critic_ready": False,
            # Nonfinite evidence should stop; lack of data/score is only NOT_READY.
            "numeric_catastrophic": bool(
                metrics.get("diagnostic_available", False) and
                not numeric_safe and (
                    not metrics.get("diagnostic_arrays_finite", True)
                    or not metrics.get("diagnostic_target_finite", True)
                    or not metrics.get("training_numeric_finite", True)
                    or any(metrics.get(k) is not None
                           and not math.isfinite(float(metrics[k])) for k in mandatory)
                )),
            "training_state": self.state.state.value,
            "diagnostic_only": [
                "td_plateau", "td_healthy", "q_scale_safe",
                "twin_q_disagreement_median", "success_failure_auc",
                "ood_stress_pass",
            ],
        })
        self.state.readiness_history.append(record)
        self.state.training_numeric_finite_latch = True
        self.state.training_numeric_samples = 0
        if (passed and self.state.consecutive_readiness_passes >=
                int(r["consecutive_passes"])):
            step = int(metrics["env_steps"])
            self.state.critic_ready = True
            self.state.critic_ready_step = step
            self.state.critic_decay_start_step = step
            self.state.actor_unlock_step = step + int(
                self.config["v7_schedule"]["critic_decay_env_steps"])
            self.state.actor_warmup_steps = int(
                self.config["v7_schedule"]["actor_warmup_env_steps"])
            self.state.joint_rl_start_step = (
                self.state.actor_unlock_step + self.state.actor_warmup_steps)
            self.state.next_evaluation_step = self.state.actor_unlock_step
            self.state.state = TrainingStateV7.CRITIC_DECAY
            record.update(critic_ready=True,
                          training_state=self.state.state.value,
                          transition="CRITIC_ONLY_TO_CRITIC_DECAY",
                          critic_decay_start_step=step,
                          actor_unlock_step=self.state.actor_unlock_step,
                          joint_rl_start_step=self.state.joint_rl_start_step)
        return record

    def schedule(self, env_steps, critic_lr_ready):
        state = self.state
        c0 = float(critic_lr_ready)
        c1 = float(self.config["v7_schedule"]["critic_lr_final"])
        if state.state == TrainingStateV7.CRITIC_ONLY:
            return dict(actor_enabled=False, actor_lr=0., critic_lr=c0,
                        warmup_progress=0., critic_decay_progress=0.)
        if state.critic_decay_start_step is None or state.actor_unlock_step is None:
            raise RuntimeError("Handoff transition metadata missing")
        decay = int(self.config["v7_schedule"]["critic_decay_env_steps"])
        elapsed = max(0, int(env_steps) - state.critic_decay_start_step)
        progress = min(1., elapsed / decay)
        critic_lr = c0 + progress * (c1 - c0)
        if int(env_steps) < state.actor_unlock_step:
            state.state = TrainingStateV7.CRITIC_DECAY
            return dict(actor_enabled=False, actor_lr=0., critic_lr=critic_lr,
                        warmup_progress=0., critic_decay_progress=progress)
        if state.joint_rl_start_step is None:
            raise RuntimeError("Joint training start not set")
        warmup = int(self.config["v7_schedule"]["actor_warmup_env_steps"])
        p = min(1., max(0., int(env_steps) - state.actor_unlock_step) / warmup)
        state.state = (TrainingStateV7.JOINT_RL if p >= 1.
                       else TrainingStateV7.ACTOR_WARMUP)
        return dict(actor_enabled=True,
                    actor_lr=float(self.config["actor_warmup"]["target_actor_lr"]) * p,
                    critic_lr=c1, warmup_progress=p, critic_decay_progress=1.)

    def warning_if_timed_out(self, env_steps):
        # No 300K critic-only hard stop or artificial readiness timeout.
        return False

    def evaluation_due(self, env_steps):
        s = self.state
        if (s.state not in (TrainingStateV7.ACTOR_WARMUP, TrainingStateV7.JOINT_RL)
                or s.next_evaluation_step is None
                or int(env_steps) < s.next_evaluation_step):
            return False
        s.next_evaluation_step += int(self.config["evaluation"]["interval_steps"])
        return True


def validate_v7_config(config):
    r = config["critic_readiness_v7"]
    s = config["v7_schedule"]
    if config.get("critic_target_mode") != "random2q":
        raise RuntimeError("V7 only supports random2q")
    if (int(r["min_online_steps"]) != 100000
            or int(r["check_interval_steps"]) != 10000
            or int(r["consecutive_passes"]) != 2):
        raise RuntimeError("V7 dynamic readiness contract unexpected")
    if (int(s["critic_decay_env_steps"]) != 200000
            or int(s["actor_warmup_env_steps"]) != 100000
            or float(s["critic_lr_final"]) != .000075):
        raise RuntimeError("V7 schedule contract unexpected")
    if float(config["critic_lr"]) != .0003 or float(config["actor_lr"]) != .000002:
        raise RuntimeError("Optimizer LR contract unexpected")
    if int(config["policy_delay"]) != 4 or float(config["utd"]) != .25:
        raise RuntimeError("Training optimizer schedule changed")
