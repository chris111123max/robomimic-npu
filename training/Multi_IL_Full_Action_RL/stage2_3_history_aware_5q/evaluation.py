"""Held-out evaluation and fixed best-checkpoint rule for Stage2.3 five-Q."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STAGE2_2 = HERE.parent / "stage2_2_history_aware_critic"
if str(STAGE2_2) not in sys.path:
    sys.path.insert(0, str(STAGE2_2))

from sequence_dataset import POLICIES, previous_actions  # noqa: E402


def ranks(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        out[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return out


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def auc(scores, labels):
    labels = np.asarray(labels, bool)
    p, n = labels.sum(), (~labels).sum()
    if not p or not n:
        return None
    return float(
        (ranks(scores)[labels].sum() - p * (p - 1) / 2) / (p * n)
    )


def _mean_defined(values):
    values = [float(v) for v in values if v is not None]
    return None if not values else float(np.mean(values))


def metrics(qs, target, success, progress):
    qs = np.asarray(qs, np.float64)
    target = np.asarray(target, np.float64)
    qmean = qs.mean(axis=0)
    error = qmean - target
    per_q_mse = np.mean((qs - target[None, :]) ** 2, axis=1)
    per_q_mae = np.mean(np.abs(qs - target[None, :]), axis=1)
    disagreement = qs.std(axis=0)

    result = {
        "ensemble_mean_mse": float(np.mean(error ** 2)),
        "ensemble_mean_mae": float(np.mean(np.abs(error))),
        "spearman": corr(ranks(qmean), ranks(target)),
        "pearson": corr(qmean, target),
        "auc": auc(qmean, success),
        "q_mean": float(qmean.mean()),
        "q_std": float(qmean.std()),
        "q_min": float(qmean.min()),
        "q_max": float(qmean.max()),
        "ensemble_disagreement_mean_std": float(disagreement.mean()),
        "ensemble_disagreement_p95_std": float(np.percentile(disagreement, 95)),
        "per_q_mse": [float(x) for x in per_q_mse],
        "per_q_mae": [float(x) for x in per_q_mae],
        "success_q_mean": (
            float(qmean[success].mean()) if success.any() else None
        ),
        "failure_q_mean": (
            float(qmean[~success].mean()) if (~success).any() else None
        ),
    }

    result["outcome_slices"] = {}
    for name, mask in (("success", success), ("failure", ~success)):
        if mask.any():
            err = error[mask]
            result["outcome_slices"][name] = {
                "count": int(mask.sum()),
                "q_mean": float(qmean[mask].mean()),
                "q_std": float(qmean[mask].std()),
                "target_mean": float(target[mask].mean()),
                "mae": float(np.mean(np.abs(err))),
                "mse": float(np.mean(err ** 2)),
                "spearman": corr(ranks(qmean[mask]), ranks(target[mask])),
            }

    slices = {
        "early": progress < 1 / 3,
        "middle": (progress >= 1 / 3) & (progress < 2 / 3),
        "late": progress >= 2 / 3,
    }
    result["progress_slices"] = {}
    for name, mask in slices.items():
        if mask.any():
            err = error[mask]
            result["progress_slices"][name] = {
                "count": int(mask.sum()),
                "mae": float(np.mean(np.abs(err))),
                "mse": float(np.mean(err ** 2)),
                "q_mean": float(qmean[mask].mean()),
                "target_mean": float(target[mask].mean()),
            }
    return result


@torch.no_grad()
def evaluate(model, datasets, device, horizon=700):
    was_training = model.training
    model.eval()
    output = {}

    for policy in POLICIES:
        q_parts = [[] for _ in range(5)]
        targets, labels, progress = [], [], []

        for episode in datasets[policy].episodes:
            observations = torch.as_tensor(
                episode.observations[None], device=device
            )
            previous = torch.as_tensor(
                previous_actions(episode.actions)[None], device=device
            )
            prog = (
                torch.arange(
                    episode.length, device=device, dtype=torch.float32
                )[None, :, None]
                / float(horizon)
            )
            actions = torch.as_tensor(episode.actions[None], device=device)
            values = model.forward_sequence(
                observations, previous, prog, actions
            )
            if len(values) != 5:
                raise RuntimeError("Stage2.3 evaluator expected five Q outputs")
            for index, value in enumerate(values):
                q_parts[index].append(value.cpu().numpy().ravel())
            targets.append(episode.returns)
            labels.append(np.full(episode.length, episode.success))
            progress.append(
                np.arange(episode.length, dtype=np.float64) / float(horizon)
            )

        qs = np.stack(
            [np.concatenate(parts) for parts in q_parts], axis=0
        )
        output[policy] = metrics(
            qs,
            np.concatenate(targets),
            np.concatenate(labels).astype(bool),
            np.concatenate(progress),
        )

    aggregate_keys = (
        "ensemble_mean_mse",
        "ensemble_mean_mae",
        "spearman",
        "pearson",
        "auc",
        "ensemble_disagreement_mean_std",
    )
    output["balanced_aggregate"] = {
        key: _mean_defined([output[p][key] for p in POLICIES])
        for key in aggregate_keys
    }
    model.train(was_training)
    return output


SELECTION_DIRECTIONS = {
    "spearman": "max",
    "pearson": "max",
    "auc": "max",
    "ensemble_mean_mae": "min",
    "ensemble_mean_mse": "min",
}


def select_best_checkpoint(records, label):
    """Equal-rank aggregate over five predeclared validation metrics."""
    if not records:
        raise RuntimeError("no validation records to select from")
    scope = "bc_rnn" if label == "rnn_q" else "balanced_aggregate"
    n = len(records)
    scored = [
        {
            "step": int(record["step"]),
            "checkpoint": record["checkpoint"],
            "metrics": {
                key: record["validation"][scope][key]
                for key in SELECTION_DIRECTIONS
            },
            "metric_ranks": {},
        }
        for record in records
    ]

    for key, direction in SELECTION_DIRECTIONS.items():
        values = np.asarray(
            [row["metrics"][key] for row in scored], dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"non-finite or undefined selection metric {scope}.{key}"
            )
        raw = ranks(values if direction == "min" else -values)
        normalized = raw / float(max(1, n - 1))
        for row, rank in zip(scored, normalized):
            row["metric_ranks"][key] = float(rank)

    for row in scored:
        row["composite_rank_score"] = float(
            np.mean(list(row["metric_ranks"].values()))
        )

    best = min(
        scored,
        key=lambda row: (
            row["composite_rank_score"],
            row["metrics"]["ensemble_mean_mse"],
            -row["metrics"]["spearman"],
            row["step"],
        ),
    )
    return {
        "scope": scope,
        "rule": (
            "equal normalized rank aggregation: "
            "spearman max, pearson max, auc max, "
            "ensemble_mean_mae min, ensemble_mean_mse min"
        ),
        "eligible_checkpoint_count": int(n),
        "selected_step": int(best["step"]),
        "selected_checkpoint": best["checkpoint"],
        "selected_composite_rank_score": best["composite_rank_score"],
        "rows": scored,
    }
