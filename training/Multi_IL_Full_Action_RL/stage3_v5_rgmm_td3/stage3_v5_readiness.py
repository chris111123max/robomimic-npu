"""Replay-only Critic readiness diagnostics.  These never step an environment."""
from __future__ import annotations

import math
import numpy as np


def _rank(x):
    order = np.argsort(x, kind="mergesort"); ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x), dtype=float)
    values, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    for index, count in enumerate(counts):
        if count > 1: ranks[inverse == index] = ranks[inverse == index].mean()
    return ranks


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0: return 0.0, 0.0
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1]), float(np.corrcoef(x, y)[0, 1])


def auc(labels, scores):
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    pos, neg = labels.sum(), len(labels) - labels.sum()
    if not pos or not neg: return 0.0
    ranks = _rank(scores) + 1.0
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def discounted_returns(episode, gamma):
    rewards = np.asarray(episode["rewards"], float).reshape(-1)
    out = np.empty_like(rewards); running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running; out[index] = running
    return out


def stability(history, rules):
    if len(history) < 3: return False, False
    recent = history[-3:]
    td = [row["td_mae"] for row in recent]
    improvement = (td[0] - td[-1]) / max(abs(td[0]), 1e-12)
    worsening = (td[-2] - td[-1]) / max(abs(td[-2]), 1e-12) < -rules["td_worsening_limit"]
    td_plateau = improvement < rules["td_plateau_relative_improvement"]
    q_means = [abs(row["qmin_mean"]) for row in recent]
    q_stds = [row["qmin_std"] for row in recent]
    def stable(values, bound):
        return all(abs(b - a) / max(abs(a), 1e-12) <= bound for a, b in zip(values, values[1:]))
    return td_plateau, bool(worsening), stable(q_means, rules["q_mean_relative_change"]) and stable(q_stds, rules["q_std_relative_change"])


def replay_metrics(agent, online, episodes, successes, config, history, sample_count=1024, env_steps=None):
    """Ask the agent for Q/TD arrays on full replay episodes and a fixed replay sample.

    The model-facing methods keep recurrent/normalization details inside the TD3
    implementation; this module only owns the gate statistics.
    """
    completed = list(online.episodes)
    if not completed: raise RuntimeError("No completed online episode for readiness check")
    values, returns, labels = [], [], []
    for episode in completed:
        q1, q2 = agent.q_values_for_episode(episode)
        values.extend(np.minimum(q1, q2)); returns.extend(discounted_returns(episode, config["gamma"]))
        labels.append(int(bool(episode.get("success", False))))
    episode_q = [float(np.mean(np.minimum(*agent.q_values_for_episode(ep)))) for ep in completed]
    success_q = [score for score, label in zip(episode_q, labels) if label]
    failure_q = [score for score, label in zip(episode_q, labels) if not label]
    diagnostic = agent.fixed_td_diagnostic(online, sample_count)
    q1, q2 = np.asarray(diagnostic["q1"]), np.asarray(diagnostic["q2"])
    qmin = np.minimum(q1, q2); disagreement = np.abs(q1-q2)/(np.abs(q1)+np.abs(q2)+1e-8)
    td_plateau, td_worsening, q_scale_stable = stability(history + [{"td_mae": diagnostic["td_mae"], "qmin_mean": qmin.mean(), "qmin_std": qmin.std()}], config["critic_readiness"])
    ood = agent.ood_action_stress(online, config["critic_readiness"]["ood_radii"])
    spearman, pearson = correlation(values, returns)
    finite = all(np.isfinite(value).all() for value in (q1, q2, qmin, np.asarray(values), np.asarray(returns)))
    return {"env_steps": int(online.transitions if env_steps is None else env_steps), "completed_episodes": int(episodes), "success_episodes": int(successes), "failure_episodes": int(episodes-successes), "spearman_q_return": spearman, "pearson_q_return": pearson, "success_failure_auc": auc(labels, episode_q), "mean_q_success": float(np.mean(success_q)) if success_q else 0.0, "mean_q_failure": float(np.mean(failure_q)) if failure_q else 0.0, "delta_q": (float(np.mean(success_q))-float(np.mean(failure_q))) if success_q and failure_q else 0.0, "twin_q_disagreement_mean": float(disagreement.mean()), "twin_q_disagreement_median": float(np.median(disagreement)), "twin_q_disagreement_p95": float(np.percentile(disagreement,95)), "td_mae": float(diagnostic["td_mae"]), "td_mse": float(diagnostic["td_mse"]), "q1_mean": float(q1.mean()), "q1_std": float(q1.std()), "q1_min": float(q1.min()), "q1_max": float(q1.max()), "q2_mean": float(q2.mean()), "q2_std": float(q2.std()), "q2_min": float(q2.min()), "q2_max": float(q2.max()), "qmin_mean": float(qmin.mean()), "qmin_std": float(qmin.std()), "qmin_min": float(qmin.min()), "qmin_max": float(qmin.max()), "td_target_mean": float(np.mean(diagnostic["td_target"])), "td_target_std": float(np.std(diagnostic["td_target"])), "td_target_min": float(np.min(diagnostic["td_target"])), "td_target_max": float(np.max(diagnostic["td_target"])), "ood_excess_q_mean": ood["mean"], "ood_excess_q_p95": ood["p95"], "ood_excess_q_max": ood["max"], "ood_stress_pass": ood["max"] <= config["critic_readiness"]["ood_max_excess_q"], "td_plateau": td_plateau, "td_error_worsening": td_worsening, "q_scale_stable": q_scale_stable, "finite": finite}
