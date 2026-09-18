"""PIRLNav-style Critic gate and gradual TD3 handoff, independent of model math."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TrainingState(str, Enum):
    CRITIC_ONLY = "CRITIC_ONLY"
    ACTOR_WARMUP = "ACTOR_WARMUP"
    JOINT_RL = "JOINT_RL"
    CRITIC_NOT_READY = "CRITIC_NOT_READY"


@dataclass
class HandoffState:
    state: TrainingState = TrainingState.CRITIC_ONLY
    critic_ready: bool = False
    critic_ready_step: int | None = None
    actor_warmup_steps: int | None = None
    joint_rl_start_step: int | None = None
    next_evaluation_step: int | None = None
    consecutive_readiness_passes: int = 0
    readiness_history: list[dict[str, Any]] = field(default_factory=list)

    def serialize(self):
        value = asdict(self)
        value["state"] = self.state.value
        return value

    @classmethod
    def restore(cls, value):
        value = dict(value)
        value["state"] = TrainingState(value["state"])
        return cls(**value)


class CriticHandoff:
    """Owns all step-based state transitions; no optimizer update counts enter it."""
    def __init__(self, config, state=None):
        self.config = config
        self.rules = config["critic_readiness"]
        self.state = state or HandoffState()

    def readiness_due(self, online_env_steps):
        return (self.state.state == TrainingState.CRITIC_ONLY
                and online_env_steps >= self.rules["min_online_steps"]
                and online_env_steps % self.rules["check_interval_steps"] == 0)

    def _flags(self, metrics):
        r = self.rules
        flags = {
            "insufficient_online_steps": metrics["env_steps"] < r["min_online_steps"],
            "insufficient_episodes": metrics["completed_episodes"] < r["min_completed_episodes"],
            "insufficient_success_episodes": metrics["success_episodes"] < r["min_success_episodes"],
            "insufficient_failure_episodes": metrics["failure_episodes"] < r["min_failure_episodes"],
            "spearman_low": metrics["spearman_q_return"] < r["min_spearman"],
            "auc_low": metrics["success_failure_auc"] < r["min_auc"],
            "q_separation_nonpositive": metrics["delta_q"] <= 0,
            "twin_q_disagreement_high": (metrics["twin_q_disagreement_median"] > r["max_twin_median"]
                                           or metrics["twin_q_disagreement_p95"] > r["max_twin_p95"]),
            "td_not_plateau": not metrics["td_plateau"],
            "td_error_worsening": bool(metrics["td_error_worsening"]),
            "q_scale_unstable": not metrics["q_scale_stable"],
            "ood_q_explosion": not metrics["ood_stress_pass"],
            "non_finite": not bool(metrics["finite"]),
        }
        return flags

    def submit_readiness(self, metrics):
        if self.state.state != TrainingState.CRITIC_ONLY:
            raise RuntimeError("Readiness checks are valid only in CRITIC_ONLY")
        flags = self._flags(metrics)
        passed = not any(flags.values())
        self.state.consecutive_readiness_passes = (self.state.consecutive_readiness_passes + 1
                                                   if passed else 0)
        record = dict(metrics, individual_gate_flags=flags, overall_pass=passed,
                      consecutive_pass_count=self.state.consecutive_readiness_passes,
                      critic_ready=False)
        self.state.readiness_history.append(record)
        if self.state.consecutive_readiness_passes >= self.rules["consecutive_passes"]:
            step = int(metrics["env_steps"])
            self.state.critic_ready = True
            self.state.critic_ready_step = step
            self.state.actor_warmup_steps = (int(self.config.get("smoke_warmup_steps", 0))
                                             if self.config.get("run_type") == "SMOKE" else 0)
            if not self.state.actor_warmup_steps:
                self.state.actor_warmup_steps = min(300000, max(100000, step))
            self.state.joint_rl_start_step = step + self.state.actor_warmup_steps
            self.state.next_evaluation_step = self.state.joint_rl_start_step
            self.state.state = TrainingState.ACTOR_WARMUP
            record["critic_ready"] = True
            record["transition"] = "CRITIC_ONLY_TO_ACTOR_WARMUP"
        return record

    def fail_if_timed_out(self, online_env_steps):
        if (self.state.state == TrainingState.CRITIC_ONLY
                and online_env_steps >= self.rules["max_critic_only_steps"]):
            self.state.state = TrainingState.CRITIC_NOT_READY
            return True
        return False

    def schedule(self, online_env_steps, critic_lr_ready):
        s = self.state
        if s.state == TrainingState.CRITIC_ONLY:
            return {"actor_enabled": False, "actor_lr": 0.0, "critic_lr": critic_lr_ready,
                    "warmup_progress": 0.0, "actor_warmup_env_steps": 0}
        if s.state == TrainingState.CRITIC_NOT_READY:
            return {"actor_enabled": False, "actor_lr": 0.0, "critic_lr": critic_lr_ready,
                    "warmup_progress": 0.0, "actor_warmup_env_steps": 0}
        elapsed = max(0, int(online_env_steps) - s.critic_ready_step)
        progress = min(1.0, elapsed / s.actor_warmup_steps)
        if s.state == TrainingState.ACTOR_WARMUP and progress >= 1.0:
            s.state = TrainingState.JOINT_RL
        return {"actor_enabled": True,
                "actor_lr": self.config["actor_warmup"]["target_actor_lr"] * progress,
                "critic_lr": critic_lr_ready * (1.0 - 0.75 * progress),
                "warmup_progress": progress, "actor_warmup_env_steps": elapsed}

    def evaluation_due(self, online_env_steps):
        s = self.state
        if s.state != TrainingState.JOINT_RL or online_env_steps < s.next_evaluation_step:
            return False
        s.next_evaluation_step += self.config["evaluation"]["interval_steps"]
        return True
