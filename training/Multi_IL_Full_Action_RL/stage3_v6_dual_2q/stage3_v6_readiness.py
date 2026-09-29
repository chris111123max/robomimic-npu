"""Stage3-v6-only Critic Readiness V2 diagnostics and handoff.

The V5 diagnostic fields remain observable, but cannot affect this gate.
"""
from __future__ import annotations

import hashlib
import math
import pickle
from dataclasses import dataclass

import numpy as np

from stage3_v5_diagnostics import isolated_training_rng
from stage3_v5_readiness import _replay_metrics, correlation, discounted_returns
from stage3_v5_schedule import CriticHandoff, HandoffState, TrainingState

REFERENCE_SOURCE = "stage2_2_initial_critic_on_frozen_readiness_set"
EPS = 1e-6


def fixed_set_digest(fixed):
    """Bind both TD windows and complete ranking episodes to one identity."""
    digest = hashlib.sha256()
    digest.update(str(fixed["seed"]).encode())
    for key, array in sorted(fixed["sequences"].items()):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    for episode in fixed["episodes"]:
        for key in ("observations", "actions", "rewards", "episode_steps"):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(episode[key]).tobytes())
        digest.update(bytes((bool(episode.get("success", False)),)))
    return digest.hexdigest()


def qmean_spearman_from_episodes(outputs, episodes, gamma):
    values, returns = [], []
    for episode, (q1, q2) in zip(episodes, outputs):
        values.extend((0.5 * (np.asarray(q1) + np.asarray(q2))).tolist())
        returns.extend(discounted_returns(episode, gamma).tolist())
    finite = bool(np.isfinite(values).all() and np.isfinite(returns).all())
    if not finite:
        return float("nan"), float("nan"), False
    spearman, pearson = correlation(values, returns)
    return spearman, pearson, bool(math.isfinite(spearman) and math.isfinite(pearson))


def scale_values(mu, sigma, previous_mu, previous_sigma, reference_mu, reference_sigma):
    values = (mu, sigma, previous_mu, previous_sigma, reference_mu, reference_sigma)
    if not all(math.isfinite(float(v)) for v in values):
        return None
    if min(sigma, previous_sigma, reference_sigma) <= EPS:
        return None
    d_prev = abs(mu - previous_mu) / max(previous_sigma, EPS)
    d_reference = abs(mu - reference_mu) / max(reference_sigma, EPS)
    r_prev = max(sigma / previous_sigma, previous_sigma / sigma)
    r_reference = max(sigma / reference_sigma, reference_sigma / sigma)
    return {
        "qmean_mean_shift_prev_z": float(d_prev),
        "qmean_mean_shift_reference_z": float(d_reference),
        "qmean_mean_shift_max_z": float(max(d_prev, d_reference)),
        "qmean_std_ratio_prev": float(r_prev),
        "qmean_std_ratio_reference": float(r_reference),
        "qmean_std_ratio_max": float(max(r_prev, r_reference)),
    }


def assess_v2(metrics, history, rules):
    """Pure, mode-independent V2 gate calculation."""
    step = int(metrics["env_steps"])
    data_ready = (step >= int(rules["min_online_steps"])
                  and metrics["completed_episodes"] >= rules["data"]["min_completed_episodes"]
                  and metrics["success_episodes"] >= rules["data"]["min_success_episodes"]
                  and metrics["failure_episodes"] >= rules["data"]["min_failure_episodes"])
    available = bool(metrics.get("diagnostic_available", False))
    rank = metrics.get("qmean_mc_spearman")
    rank_ready = bool(available and rank is not None and math.isfinite(rank)
                      and rank >= rules["rank"]["min_spearman"])
    identity = metrics.get("fixed_diagnostic_sha256")
    interval = int(rules["check_interval_steps"])
    prior = [row for row in history if row.get("diagnostic_available")
             and row.get("fixed_diagnostic_sha256") == identity]
    last_two = prior[-2:]
    td_history_ready = bool(available and len(last_two) == 2
                            and [row["env_steps"] for row in last_two]
                            == [step - 2 * interval, step - interval])
    e1, e2, emax = (metrics.get(key) for key in
                    ("td_e1_mae", "td_e2_mae", "td_emax_mae"))
    target_finite = bool(metrics.get("diagnostic_target_finite", False))
    td_finite = bool(available and target_finite
                     and all(value is not None and math.isfinite(value)
                             for value in (e1, e2, emax)))
    r1 = r2 = None
    worsening = False
    if td_history_ready and td_finite:
        old, mid = (float(row["td_emax_mae"]) for row in last_two)
        if all(math.isfinite(value) and value >= 0 for value in (old, mid)):
            r1 = (mid - old) / max(abs(old), EPS)
            r2 = (emax - mid) / max(abs(mid), EPS)
            threshold = float(rules["td_health"]["sustained_worsening_fraction"])
            # A tiny comparison tolerance makes the specified exact 35% boundary
            # stable under ordinary binary floating point arithmetic.
            worsening = bool(r1 >= threshold - 1e-12 and r2 >= threshold - 1e-12)
        else:
            td_finite = False
    td_healthy = bool(td_history_ready and td_finite and not worsening)
    previous = prior[-1] if prior and prior[-1]["env_steps"] == step - interval else None
    scale = None
    if available and previous is not None:
        scale = scale_values(metrics["qmean_mean"], metrics["qmean_std"],
                             previous["qmean_mean"], previous["qmean_std"],
                             metrics["qmean_reference_mean"],
                             metrics["qmean_reference_std"])
    scale_safe = bool(scale is not None
                      and scale["qmean_mean_shift_max_z"] <= rules["numeric"]["qmean_mean_shift_z_max"]
                      and scale["qmean_std_ratio_max"] <= rules["numeric"]["qmean_std_ratio_max"])
    numeric_finite = bool(metrics.get("diagnostic_arrays_finite", False)
                          and td_finite and metrics.get("training_numeric_finite", True)
                          and (scale is not None))
    numeric_safe = bool(numeric_finite and scale_safe)
    current_and_reference = (metrics.get("qmean_mean"), metrics.get("qmean_std"),
                             metrics.get("qmean_reference_mean"),
                             metrics.get("qmean_reference_std"))
    valid_base_scale = bool(available and all(
        value is not None and math.isfinite(value) for value in current_and_reference)
        and min(metrics["qmean_std"], metrics["qmean_reference_std"]) > EPS)
    numeric_catastrophic = bool(available and (
        not metrics.get("diagnostic_arrays_finite", False)
        or not metrics.get("training_numeric_finite", True)
        or not td_finite or not valid_base_scale
        or (previous is not None and not scale_safe)))
    result = {
        "data_ready": bool(data_ready), "rank_ready": rank_ready,
        "td_relative_change_prev1": r2, "td_relative_change_prev2": r1,
        "td_sustained_worsening": worsening, "td_history_ready": td_history_ready,
        "td_healthy": td_healthy, "numeric_finite": numeric_finite,
        "q_scale_safe": scale_safe, "numeric_safe": numeric_safe,
        "numeric_catastrophic": numeric_catastrophic,
        "overall_ready_v2": bool(data_ready and rank_ready and td_healthy and numeric_safe),
    }
    result.update(scale or {key: None for key in (
        "qmean_mean_shift_prev_z", "qmean_mean_shift_reference_z",
        "qmean_mean_shift_max_z", "qmean_std_ratio_prev",
        "qmean_std_ratio_reference", "qmean_std_ratio_max")})
    return result


@dataclass
class HandoffStateV2(HandoffState):
    qmean_reference_mean: float | None = None
    qmean_reference_std: float | None = None
    qmean_reference_source: str | None = None
    fixed_diagnostic_sha256: str | None = None
    critic_only_warning_emitted: bool = False
    training_numeric_finite_latch: bool = True
    training_numeric_samples: int = 0
    last_critic_loss: float | None = None
    last_critic_grad_norm: float | None = None


class CriticHandoffV2(CriticHandoff):
    def __init__(self, config, state=None):
        super().__init__(config, state or HandoffStateV2())
        self.rules = config["critic_readiness_v2"]
        if self.rules.get("version") != 2:
            raise RuntimeError("Stage3-v6 requires Critic Readiness V2")

    def note_training_metrics(self, metrics):
        state = self.state
        values = []
        for key in ("critic_loss_q1", "critic_loss_q2", "critic_grad_norm"):
            if key in metrics and metrics[key] is not None:
                values.append(float(metrics[key]))
        if values:
            state.training_numeric_samples += 1
            state.training_numeric_finite_latch &= all(map(math.isfinite, values))
            if "critic_loss_q1" in metrics and "critic_loss_q2" in metrics:
                state.last_critic_loss = float(metrics["critic_loss_q1"] + metrics["critic_loss_q2"])
            if "critic_grad_norm" in metrics:
                state.last_critic_grad_norm = float(metrics["critic_grad_norm"])

    def submit_readiness(self, metrics):
        if self.state.state != TrainingState.CRITIC_ONLY:
            raise RuntimeError("V2 readiness is valid only in CRITIC_ONLY")
        assessed = assess_v2(metrics, self.state.readiness_history, self.rules)
        passed = assessed["overall_ready_v2"]
        self.state.consecutive_readiness_passes = (self.state.consecutive_readiness_passes + 1
                                                    if passed else 0)
        record = dict(metrics, **assessed)
        record.update({"readiness_pass_streak": self.state.consecutive_readiness_passes,
                       "critic_ready": False, "training_state": self.state.state.value,
                       "readiness_version": 2,
                       "diagnostic_only": ["twin_q_disagreement_mean", "twin_q_disagreement_median",
                                           "twin_q_disagreement_p95", "twin_q_disagreement_p99",
                                           "success_failure_auc", "pearson_q_return",
                                           "td_plateau", "td_mae", "ood_stress_pass"]})
        self.state.readiness_history.append(record)
        self.state.training_numeric_finite_latch = True
        self.state.training_numeric_samples = 0
        if passed and self.state.consecutive_readiness_passes >= self.rules["consecutive_passes"]:
            step = int(metrics["env_steps"])
            self.state.critic_ready = True
            self.state.critic_ready_step = step
            self.state.actor_warmup_steps = (max(1, int(self.config.get("smoke_warmup_steps", 4)))
                                             if self.config.get("run_type") == "SMOKE"
                                             else min(300000, max(100000, step)))
            self.state.joint_rl_start_step = step + self.state.actor_warmup_steps
            self.state.next_evaluation_step = self.state.joint_rl_start_step
            self.state.state = TrainingState.ACTOR_WARMUP
            record["critic_ready"] = True
            record["transition"] = "CRITIC_ONLY_TO_ACTOR_WARMUP"
            record["training_state"] = self.state.state.value
        return record

    def warning_if_timed_out(self, env_steps):
        if (self.state.state == TrainingState.CRITIC_ONLY
                and not self.state.critic_only_warning_emitted
                and env_steps >= self.rules["critic_only_warning_steps"]):
            self.state.critic_only_warning_emitted = True
            return True
        return False


def replay_metrics_v2(agent, online, episodes, successes, config, state,
                      env_steps, offline=None):
    """Preserve V5 explanations while computing independent V2 hard metrics."""
    rules = config["critic_readiness_v2"]
    data_count_ready = (episodes >= rules["data"]["min_completed_episodes"]
                        and successes >= rules["data"]["min_success_episodes"]
                        and episodes - successes >= rules["data"]["min_failure_episodes"])
    with isolated_training_rng(online, offline):
        before_selector = pickle.dumps(agent.target_selector_state_dict())
        if data_count_ready:
            legacy = _replay_metrics(agent, online, episodes, successes, config,
                                     state.readiness_history, 256, env_steps)
        else:
            legacy = {"env_steps": int(env_steps), "completed_episodes": int(episodes),
                      "success_episodes": int(successes),
                      "failure_episodes": int(episodes - successes),
                      "diagnostic_available": False}
        fixed = online.fixed_critic_diagnostic_set
        if fixed is None:
            result = dict(legacy, qmean_mc_spearman=None, qmean_mc_pearson=None,
                          td_e1_mae=None, td_e2_mae=None, td_emember_mae=None,
                          td_emax_mae=None, td_eqmean_mae=None,
                          qmean_mean=None, qmean_std=None,
                          qmean_reference_mean=state.qmean_reference_mean,
                          qmean_reference_std=state.qmean_reference_std,
                          qmean_reference_source=state.qmean_reference_source,
                          fixed_diagnostic_sha256=state.fixed_diagnostic_sha256,
                          diagnostic_arrays_finite=False, diagnostic_target_finite=False)
        else:
            digest = fixed_set_digest(fixed)
            if state.fixed_diagnostic_sha256 not in (None, digest):
                raise RuntimeError("Frozen readiness set changed across checks/resume")
            state.fixed_diagnostic_sha256 = digest
            if state.qmean_reference_mean is None:
                initial = np.asarray(agent.initial_qmean_on_sequences(fixed["sequences"]), float)
                if not np.isfinite(initial).all():
                    raise FloatingPointError("Nonfinite Stage2 reference Qmean")
                state.qmean_reference_mean = float(initial.mean())
                state.qmean_reference_std = float(initial.std())
                state.qmean_reference_source = REFERENCE_SOURCE
            outputs = agent.q_values_for_episodes(fixed["episodes"])
            spearman, pearson, episode_finite = qmean_spearman_from_episodes(
                outputs, fixed["episodes"], config["gamma"])
            diagnostic = agent.fixed_td_diagnostic(fixed["sequences"])
            q1 = np.asarray(diagnostic["q1"], float)
            q2 = np.asarray(diagnostic["q2"], float)
            target = np.asarray(diagnostic["td_target"], float)
            qmean = 0.5 * (q1 + q2)
            finite = bool(episode_finite and all(np.isfinite(v).all()
                                                for v in (q1, q2, target, qmean)))
            e1 = float(np.abs(q1 - target).mean())
            e2 = float(np.abs(q2 - target).mean())
            result = dict(legacy,
                          qmean_mc_spearman=spearman, qmean_mc_pearson=pearson,
                          td_e1_mae=e1, td_e2_mae=e2,
                          td_emember_mae=0.5 * (e1 + e2),
                          td_emax_mae=max(e1, e2),
                          td_eqmean_mae=float(np.abs(qmean - target).mean()),
                          qmean_mean=float(qmean.mean()), qmean_std=float(qmean.std()),
                          qmean_reference_mean=state.qmean_reference_mean,
                          qmean_reference_std=state.qmean_reference_std,
                          qmean_reference_source=state.qmean_reference_source,
                          fixed_diagnostic_sha256=digest,
                          diagnostic_arrays_finite=finite,
                          diagnostic_target_finite=bool(np.isfinite(target).all()),
                          twin_q_disagreement_p99=float(np.percentile(
                              np.abs(q1-q2)/(np.abs(q1)+np.abs(q2)+1e-8), 99)))
            result["qmean_min"] = float(qmean.min())
            result["qmean_max"] = float(qmean.max())
        result["training_numeric_finite"] = bool(state.training_numeric_finite_latch)
        result["training_numeric_samples"] = int(state.training_numeric_samples)
        result["last_critic_loss"] = state.last_critic_loss
        result["last_critic_grad_norm"] = state.last_critic_grad_norm
        if pickle.dumps(agent.target_selector_state_dict()) != before_selector:
            raise RuntimeError("V2 readiness consumed random-one selector RNG")
        return result
