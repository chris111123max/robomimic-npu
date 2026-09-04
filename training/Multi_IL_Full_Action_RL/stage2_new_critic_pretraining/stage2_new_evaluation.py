"""Actor-free return-regression and action-gradient diagnostics."""
from __future__ import annotations
import math
import numpy as np
import torch
from stage2_new_dataset import POLICIES

def _safe_float(value): return None if not np.isfinite(value) else float(value)
def _safe_corr(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0: return None
    return _safe_float(np.corrcoef(x, y)[0, 1])
def _ranks(values):
    order = np.argsort(values, kind="mergesort"); ranks = np.empty(len(values), dtype=float); ranks[order] = np.arange(len(values), dtype=float)
    sorted_values = values[order]; start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            ranks[order[start:end]] = (start + end - 1) / 2.0; start = end
    return ranks
def _auc(scores, labels):
    labels = np.asarray(labels, dtype=bool)
    pos, neg = int(labels.sum()), int((~labels).sum())
    if not pos or not neg: return {"value": None, "reason": "validation subset has one class"}
    ranks = _ranks(np.asarray(scores, dtype=float)); return {"value": float((ranks[labels].sum() - pos * (pos - 1) / 2.0) / (pos * neg)), "reason": None}
def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values): return {key: None for key in ("mean", "std", "median", "p90", "p95")}
    return {"mean": float(values.mean()), "std": float(values.std()), "median": float(np.median(values)), "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95))}

@torch.no_grad()
def evaluate_critic(critic, datasets, device, batch_size=4096):
    critic.eval(); results = {}
    for policy in POLICIES:
        data = datasets[policy]; q1_rows, q2_rows = [], []
        for start in range(0, data.transition_count, int(batch_size)):
            end = min(start + int(batch_size), data.transition_count)
            state = torch.as_tensor(data.state[start:end], device=device); action = torch.as_tensor(data.action[start:end], device=device)
            q1, q2 = critic(state, action); q1_rows.append(q1.cpu().numpy().reshape(-1)); q2_rows.append(q2.cpu().numpy().reshape(-1))
        q1, q2, target = np.concatenate(q1_rows), np.concatenate(q2_rows), data.returns.reshape(-1)
        qmin = np.minimum(q1, q2); success, failure = qmin[data.success], qmin[~data.success]
        results[policy] = {"q1_mse": float(np.mean((q1-target)**2)), "q2_mse": float(np.mean((q2-target)**2)),
            "twin_mean_mse": float((np.mean((q1-target)**2)+np.mean((q2-target)**2))/2),
            "q1_mae": float(np.mean(np.abs(q1-target))), "q2_mae": float(np.mean(np.abs(q2-target))),
            "twin_mean_mae": float((np.mean(np.abs(q1-target))+np.mean(np.abs(q2-target)))/2),
            "mean_success_q": _safe_float(np.mean(success)) if len(success) else None,
            "mean_failure_q": _safe_float(np.mean(failure)) if len(failure) else None,
            "q_gap": _safe_float(np.mean(success)-np.mean(failure)) if len(success) and len(failure) else None,
            "roc_auc": _auc(qmin, data.success), "pearson_qmin_return": _safe_corr(qmin, target),
            "spearman_qmin_return": _safe_corr(_ranks(qmin), _ranks(target))}
    results["balanced_aggregate"] = {key: float(np.mean([results[p][key] for p in POLICIES])) for key in ("q1_mse", "q2_mse", "twin_mean_mse", "q1_mae", "q2_mae", "twin_mean_mae")}
    return results

def make_probes(datasets, max_per_probe=256):
    """Deterministically take source-order transitions and persist their identifiers."""
    choices = {"rnn_success": [("bc_rnn", datasets["bc_rnn"].success)],
        "other_failure": [(p, ~datasets[p].success) for p in ("bc_transformer", "bc_gmm")],
        "mixed_balanced": [(p, np.ones(datasets[p].transition_count, dtype=bool)) for p in POLICIES]}
    probes, manifest = {}, {}
    for name, groups in choices.items():
        per_group = max(1, int(max_per_probe) // len(groups)); indices = []
        for policy, mask in groups:
            dataset = datasets[policy]; selected = np.flatnonzero(mask)[:per_group]
            indices.extend((policy, int(i)) for i in selected)
        rows = []
        for policy, index in indices:
            data = datasets[policy]; rows.append({"policy": policy, "index": index, "seed": int(data.transition_seed[index]), "episode_id": int(data.transition_episode_id[index]), "timestep": int(data.timestep[index]), "success": bool(data.success[index])})
        probes[name] = rows; manifest[name] = {"count": len(rows), "rows": rows}
    return probes, manifest

def probe_arrays(probe_rows, datasets):
    return (np.stack([datasets[row["policy"]].state[row["index"]] for row in probe_rows]),
            np.stack([datasets[row["policy"]].action[row["index"]] for row in probe_rows])) if probe_rows else (np.empty((0,59), np.float32), np.empty((0,14), np.float32))

def action_gradient_diagnostics(critic, states, actions, device):
    if not len(states): return {"q1_norm": _summary([]), "q2_norm": _summary([]), "q1_gradients": np.empty((0,14)), "q2_gradients": np.empty((0,14))}
    critic.eval(); action = torch.as_tensor(actions, device=device).detach().clone().requires_grad_(True); state = torch.as_tensor(states, device=device)
    q1, q2 = critic(state, action)
    g1 = torch.autograd.grad(q1.sum(), action, retain_graph=True)[0].detach().cpu().numpy(); g2 = torch.autograd.grad(q2.sum(), action)[0].detach().cpu().numpy()
    return {"q1_norm": _summary(np.linalg.norm(g1, axis=1)), "q2_norm": _summary(np.linalg.norm(g2, axis=1)), "q1_gradients": g1, "q2_gradients": g2}

def gradient_cosine(left, right):
    result = {}
    for head in ("q1_gradients", "q2_gradients"):
        a, b = left[head], right[head]
        if len(a) != len(b) or not len(a): result[head.replace("_gradients", "_cosine")] = None; continue
        values = np.sum(a*b, axis=1) / np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1), 1e-12)
        result[head.replace("_gradients", "_cosine")] = _summary(values)
    return result
