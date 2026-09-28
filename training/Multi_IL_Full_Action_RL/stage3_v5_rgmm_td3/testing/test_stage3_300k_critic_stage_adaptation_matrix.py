#!/usr/bin/env python3
"""Read-only 300K Stage3-v5 Critic distribution-adaptation matrix.

Evaluate the Stage3 online Twin-Q checkpoints at 0/100K/200K/300K on four
complete-episode datasets:

  old   : Stage1 offline rollout source(s) actually configured for the group.
  early : complete online episodes generated in (0, 100K].
  mid   : complete online episodes generated in (100K, 200K].
  late  : complete online episodes generated in (200K, 300K].

The experiment is designed to distinguish two hypotheses:

1. harmful Critic degradation: newer Critics get worse even on newer online data;
2. distribution adaptation: old-distribution metrics may fall while metrics on
   newly covered online states improve.

Important contracts:
- no environment is created;
- no optimizer / Actor / Critic update is performed;
- no checkpoint is modified;
- finite Monte-Carlo returns are recomputed from complete episodes only;
- Stage3 online episodes are assigned to time stages using episode_metrics.jsonl,
  not file modification time;
- the first completed episode per env after each 100K/200K boundary is excluded
  conservatively because it may have started before that boundary;
- history encoding reuses the exact sliding horizon-10 zero-state Stage3
  readiness implementation.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STAGE3 = HERE.parent
for directory in (HERE, STAGE3):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from stage3_v5_agent import strict_stage2_load  # noqa: E402
from stage3_v5_replay import (  # noqa: E402
    OnlineSequenceReplay,
    Stage1OfflineSequenceReplay,
)
from stage3_v5_readiness import auc, correlation, discounted_returns  # noqa: E402
from test_stage2_stage3_readiness_compare import (  # noqa: E402
    q_values_for_episode,
    resolve_device,
    sync,
)


CHECKPOINTS = {
    "step0": "step0_transfer.pth",
    "100k": "step_0100000.pth",
    "200k": "step_0200000.pth",
    "300k": "step_0300000.pth",
}
STAGE_SPECS = {
    "early": (0, 100_000, "step_0100000.sequences.npy"),
    "mid": (100_000, 200_000, "step_0200000.sequences.npy"),
    "late": (200_000, 300_000, "step_0300000.sequences.npy"),
}
SOURCE_SPECS = {
    "bc_rnn": "rnn",
    "bc_transformer": "transformer",
    "bc_gmm": "gmm",
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-run-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=("multi_q", "rnn_q"),
        default=("multi_q", "rnn_q"),
    )
    parser.add_argument(
        "--output-dir",
        help="Default: <run>/testing/critic_stage_adaptation_matrix",
    )
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
    return rows


def release(model, device):
    del model
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def bool_last(array):
    value = np.asarray(array).reshape(-1)
    return bool(value[-1]) if len(value) else False


def episode_signature(episode):
    rewards = np.asarray(episode["rewards"], dtype=np.float64).reshape(-1)
    return {
        "length": int(len(episode["actions"])),
        "return": float(rewards.sum()),
        "success": bool(episode.get("success", False)),
        "terminated": bool_last(episode.get("terminated", [False])),
        "truncated": bool_last(episode.get("truncated", [False])),
    }


def log_signature(row):
    return {
        "length": int(row["length"]),
        "return": float(row["return"]),
        "success": bool(row["success"]),
        "terminated": bool(row["terminated"]),
        "truncated": bool(row["truncated"]),
    }


def signatures_match(episode, row, atol=1e-5):
    left = episode_signature(episode)
    right = log_signature(row)
    return (
        left["length"] == right["length"]
        and left["success"] == right["success"]
        and left["terminated"] == right["terminated"]
        and left["truncated"] == right["truncated"]
        and math.isclose(
            left["return"], right["return"], rel_tol=0.0, abs_tol=atol
        )
    )


def validate_episode(episode, source):
    required = (
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "episode_steps",
    )
    missing = [key for key in required if key not in episode]
    if missing:
        raise RuntimeError(f"{source}: episode missing {missing}")

    n = len(episode["actions"])
    if n <= 0:
        raise RuntimeError(f"{source}: empty episode")
    obs = np.asarray(episode["observations"])
    nxt = np.asarray(episode["next_observations"])
    actions = np.asarray(episode["actions"])
    rewards = np.asarray(episode["rewards"])
    steps = np.asarray(episode["episode_steps"]).reshape(-1)
    if obs.shape != (n, 59) or nxt.shape != (n, 59):
        raise RuntimeError(f"{source}: observation shape mismatch")
    if actions.shape != (n, 14):
        raise RuntimeError(f"{source}: action shape mismatch")
    if rewards.reshape(-1).shape != (n,) or steps.shape != (n,):
        raise RuntimeError(f"{source}: reward/step shape mismatch")
    if not np.array_equal(steps, np.arange(n, dtype=steps.dtype)):
        raise RuntimeError(f"{source}: episode_steps are not 0..T-1")
    for name, value in (
        ("observations", obs),
        ("next_observations", nxt),
        ("actions", actions),
        ("rewards", rewards),
    ):
        if not np.isfinite(value).all():
            raise RuntimeError(f"{source}: non-finite {name}")


def align_replay_to_logs(episodes, rows, source):
    """Find the unique contiguous log interval represented by a replay deque.

    OnlineSequenceReplay evicts whole oldest episodes, so a saved replay is a
    contiguous suffix of the completed-episode stream. We still search and
    validate instead of assuming the suffix position.
    """
    episodes = list(episodes)
    if not episodes:
        raise RuntimeError(f"{source}: replay contains no completed episodes")
    if len(episodes) > len(rows):
        raise RuntimeError(
            f"{source}: replay episodes={len(episodes)} > log rows={len(rows)}"
        )

    max_start = len(rows) - len(episodes)
    matches = []
    for start in range(max_start + 1):
        # Cheap end-point checks first.
        if not signatures_match(episodes[0], rows[start]):
            continue
        if not signatures_match(episodes[-1], rows[start + len(episodes) - 1]):
            continue
        ok = True
        for offset, episode in enumerate(episodes):
            if not signatures_match(episode, rows[start + offset]):
                ok = False
                break
        if ok:
            matches.append(start)

    if len(matches) != 1:
        raise RuntimeError(
            f"{source}: expected one replay/log alignment, found {matches}"
        )

    start = matches[0]
    aligned_rows = rows[start:start + len(episodes)]
    return [
        {"episode": episode, "log": row, "log_index": start + index}
        for index, (episode, row) in enumerate(zip(episodes, aligned_rows))
    ], start


def stage_dataset(group_dir, name, lower, upper, replay_name):
    replay_path = group_dir / "checkpoints" / replay_name
    log_path = group_dir / "episode_metrics.jsonl"
    if not replay_path.exists():
        raise FileNotFoundError(replay_path)
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    all_rows = read_jsonl(log_path)
    rows = [row for row in all_rows if int(row["env_steps"]) <= int(upper)]
    if not rows:
        raise RuntimeError(f"{name}: no episode logs through {upper}")

    replay = OnlineSequenceReplay.load(replay_path)
    episodes = list(replay.episodes)
    aligned, alignment_start = align_replay_to_logs(
        episodes, rows, f"{group_dir.name}:{name}"
    )

    excluded_first_after_boundary = set()
    if lower > 0:
        first_by_env = {}
        for row in rows:
            step = int(row["env_steps"])
            if lower < step <= upper:
                env_id = int(row["env_id"])
                if env_id not in first_by_env:
                    first_by_env[env_id] = int(row["episode_id"])
        excluded_first_after_boundary = {
            (env_id, episode_id)
            for env_id, episode_id in first_by_env.items()
        }

    selected = []
    selected_rows = []
    excluded_cross_boundary = []
    for item in aligned:
        row = item["log"]
        step = int(row["env_steps"])
        if not (lower < step <= upper):
            continue
        identity = (int(row["env_id"]), int(row["episode_id"]))
        if identity in excluded_first_after_boundary:
            excluded_cross_boundary.append({
                "env_id": identity[0],
                "episode_id": identity[1],
                "completion_env_steps": step,
            })
            continue
        validate_episode(item["episode"], f"{group_dir.name}:{name}")
        selected.append(item["episode"])
        selected_rows.append(row)

    if not selected:
        raise RuntimeError(f"{group_dir.name}:{name}: empty pure-stage dataset")

    transition_count = int(sum(len(ep["actions"]) for ep in selected))
    success_count = int(sum(bool(ep.get("success", False)) for ep in selected))
    return selected, {
        "name": name,
        "kind": "stage3_online_complete_episodes",
        "interval": {
            "lower_exclusive": int(lower),
            "upper_inclusive": int(upper),
        },
        "replay_path": str(replay_path.resolve()),
        "episode_log_path": str(log_path.resolve()),
        "replay_completed_episodes": int(len(episodes)),
        "replay_transitions": int(replay.transitions),
        "log_rows_through_upper": int(len(rows)),
        "replay_log_alignment_start_index": int(alignment_start),
        "selected_episodes": int(len(selected)),
        "selected_transitions": transition_count,
        "selected_success_episodes": success_count,
        "selected_failure_episodes": int(len(selected) - success_count),
        "excluded_first_completion_after_boundary": excluded_cross_boundary,
        "selected_completion_env_steps_min": int(
            min(int(row["env_steps"]) for row in selected_rows)
        ),
        "selected_completion_env_steps_max": int(
            max(int(row["env_steps"]) for row in selected_rows)
        ),
        "complete_episode_mc_valid": True,
    }


def old_dataset(config, group):
    offline_sources = config.get("offline_sources", {})
    keys = (
        ("bc_rnn",)
        if group == "rnn_q"
        else ("bc_rnn", "bc_transformer", "bc_gmm")
    )
    episodes = []
    source_rows = []
    for key in keys:
        path_value = offline_sources.get(key)
        if not path_value:
            raise RuntimeError(f"{group}: missing configured offline source {key}")
        path = Path(path_value).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        source = Stage1OfflineSequenceReplay(
            path, SOURCE_SPECS[key], seed=0
        )
        for episode in source.episodes:
            validate_episode(episode, f"{group}:old:{key}")
        episodes.extend(source.episodes)
        source_rows.append({
            "key": key,
            "path": str(path),
            "episodes": int(len(source.episodes)),
            "transitions": int(source.transition_count),
        })

    success_count = int(
        sum(bool(ep.get("success", False)) for ep in episodes)
    )
    return episodes, {
        "name": "old",
        "kind": "configured_stage1_offline_complete_episodes",
        "sources": source_rows,
        "selected_episodes": int(len(episodes)),
        "selected_transitions": int(
            sum(len(ep["actions"]) for ep in episodes)
        ),
        "selected_success_episodes": success_count,
        "selected_failure_episodes": int(len(episodes) - success_count),
        "complete_episode_mc_valid": True,
    }


def load_stage3_critic(stage2_path, stage3_path, device):
    critic, stage2_payload = strict_stage2_load(stage2_path, device)
    payload = torch.load(stage3_path, map_location="cpu")
    if payload.get("stage") != "stage3-v5":
        raise RuntimeError(f"Not Stage3-v5: {stage3_path}")
    config = payload.get("config", {})
    recurrent = config.get("recurrent_replay", {})
    if int(recurrent.get("critic_context_length", -1)) != 10:
        raise RuntimeError(f"{stage3_path}: Critic context is not 10")
    critic.load_state_dict(payload["q1_q2"], strict=True)
    critic.eval()
    return critic, payload, stage2_payload


def safe_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    if len(np.unique(labels)) < 2:
        return None
    return float(auc(labels, scores))


def evaluate_dataset(critic, episodes, device, gamma):
    q1_parts = []
    q2_parts = []
    qmin_parts = []
    return_parts = []
    episode_qmin = []
    labels = []
    episode_lengths = []

    critic.eval()
    with torch.no_grad():
        for episode in episodes:
            q1, q2 = q_values_for_episode(
                critic, episode, device, horizon=700, context_length=10
            )
            qmin = np.minimum(q1, q2)
            mc = discounted_returns(episode, gamma)
            if qmin.shape != mc.shape:
                raise RuntimeError(
                    f"Q/MC shape mismatch {qmin.shape} vs {mc.shape}"
                )
            q1_parts.append(q1.astype(np.float64, copy=False))
            q2_parts.append(q2.astype(np.float64, copy=False))
            qmin_parts.append(qmin.astype(np.float64, copy=False))
            return_parts.append(mc.astype(np.float64, copy=False))
            episode_qmin.append(float(np.mean(qmin)))
            labels.append(int(bool(episode.get("success", False))))
            episode_lengths.append(int(len(qmin)))

    q1 = np.concatenate(q1_parts)
    q2 = np.concatenate(q2_parts)
    qmin = np.concatenate(qmin_parts)
    mc = np.concatenate(return_parts)
    labels = np.asarray(labels, dtype=np.int64)
    episode_qmin = np.asarray(episode_qmin, dtype=np.float64)
    errors = qmin - mc
    disagreement_abs = np.abs(q1 - q2)
    disagreement_relative = disagreement_abs / (
        np.abs(q1) + np.abs(q2) + 1e-8
    )
    spearman, pearson = correlation(qmin, mc)

    success_scores = episode_qmin[labels == 1]
    failure_scores = episode_qmin[labels == 0]

    zero_mask = np.isclose(mc, 0.0, atol=1e-12)
    positive_mask = mc > 0.0

    def subset_error(mask):
        if not np.any(mask):
            return None
        value = errors[mask]
        return {
            "count": int(mask.sum()),
            "mae": float(np.mean(np.abs(value))),
            "bias_q_minus_mc": float(np.mean(value)),
            "rmse": float(np.sqrt(np.mean(value * value))),
        }

    return {
        "episodes": int(len(episodes)),
        "transitions": int(len(qmin)),
        "success_episodes": int(labels.sum()),
        "failure_episodes": int(len(labels) - labels.sum()),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "finite": bool(
            np.isfinite(q1).all()
            and np.isfinite(q2).all()
            and np.isfinite(mc).all()
        ),
        "spearman_qmin_mc": float(spearman),
        "pearson_qmin_mc": float(pearson),
        "mae_qmin_mc": float(np.mean(np.abs(errors))),
        "rmse_qmin_mc": float(np.sqrt(np.mean(errors * errors))),
        "bias_qmin_minus_mc": float(np.mean(errors)),
        "success_failure_auc_episode_mean_qmin": safe_auc(
            labels, episode_qmin
        ),
        "mean_qmin_success_episode": (
            float(success_scores.mean()) if len(success_scores) else None
        ),
        "mean_qmin_failure_episode": (
            float(failure_scores.mean()) if len(failure_scores) else None
        ),
        "delta_qmin_success_minus_failure": (
            float(success_scores.mean() - failure_scores.mean())
            if len(success_scores) and len(failure_scores)
            else None
        ),
        "q1_mean": float(q1.mean()),
        "q1_std": float(q1.std()),
        "q2_mean": float(q2.mean()),
        "q2_std": float(q2.std()),
        "qmin_mean": float(qmin.mean()),
        "qmin_std": float(qmin.std()),
        "mc_mean": float(mc.mean()),
        "mc_std": float(mc.std()),
        "mc_min": float(mc.min()),
        "mc_max": float(mc.max()),
        "twin_abs_disagreement_mean": float(disagreement_abs.mean()),
        "twin_abs_disagreement_median": float(
            np.median(disagreement_abs)
        ),
        "twin_relative_disagreement_mean": float(
            disagreement_relative.mean()
        ),
        "twin_relative_disagreement_median": float(
            np.median(disagreement_relative)
        ),
        "twin_relative_disagreement_p95": float(
            np.percentile(disagreement_relative, 95)
        ),
        "zero_mc_error": subset_error(zero_mask),
        "positive_mc_error": subset_error(positive_mask),
    }


def checkpoint_paths(group_dir):
    paths = {
        name: group_dir / "checkpoints" / filename
        for name, filename in CHECKPOINTS.items()
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{group_dir.name}:{name}: {path}")
    return paths


def flatten_rows(group, matrix):
    rows = []
    for checkpoint_name in CHECKPOINTS:
        for dataset_name in ("old", "early", "mid", "late"):
            entry = matrix[checkpoint_name][dataset_name]
            m = entry["metrics"]
            rows.append({
                "group": group,
                "checkpoint": checkpoint_name,
                "checkpoint_env_steps": entry["checkpoint_env_steps"],
                "critic_updates": entry["critic_updates"],
                "actor_updates": entry["actor_updates"],
                "dataset": dataset_name,
                "episodes": m["episodes"],
                "transitions": m["transitions"],
                "spearman_qmin_mc": m["spearman_qmin_mc"],
                "pearson_qmin_mc": m["pearson_qmin_mc"],
                "mae_qmin_mc": m["mae_qmin_mc"],
                "rmse_qmin_mc": m["rmse_qmin_mc"],
                "bias_qmin_minus_mc": m["bias_qmin_minus_mc"],
                "auc_episode_mean_qmin": (
                    m["success_failure_auc_episode_mean_qmin"]
                ),
                "delta_qmin_success_minus_failure": (
                    m["delta_qmin_success_minus_failure"]
                ),
                "qmin_mean": m["qmin_mean"],
                "qmin_std": m["qmin_std"],
                "mc_mean": m["mc_mean"],
                "mc_std": m["mc_std"],
                "twin_relative_disagreement_median": (
                    m["twin_relative_disagreement_median"]
                ),
                "twin_relative_disagreement_p95": (
                    m["twin_relative_disagreement_p95"]
                ),
                "finite": m["finite"],
            })
    return rows


def descriptive_deltas(matrix):
    datasets = ("old", "early", "mid", "late")
    result = {
        "step0_to_300k_by_dataset": {},
        "adjacent_checkpoint_spearman_delta_by_dataset": {},
        "matched_stage_vs_step0": {},
    }
    for dataset in datasets:
        result["step0_to_300k_by_dataset"][dataset] = {
            "spearman_delta": float(
                matrix["300k"][dataset]["metrics"]["spearman_qmin_mc"]
                - matrix["step0"][dataset]["metrics"]["spearman_qmin_mc"]
            ),
            "mae_delta": float(
                matrix["300k"][dataset]["metrics"]["mae_qmin_mc"]
                - matrix["step0"][dataset]["metrics"]["mae_qmin_mc"]
            ),
        }
        result["adjacent_checkpoint_spearman_delta_by_dataset"][dataset] = {
            "step0_to_100k": float(
                matrix["100k"][dataset]["metrics"]["spearman_qmin_mc"]
                - matrix["step0"][dataset]["metrics"]["spearman_qmin_mc"]
            ),
            "100k_to_200k": float(
                matrix["200k"][dataset]["metrics"]["spearman_qmin_mc"]
                - matrix["100k"][dataset]["metrics"]["spearman_qmin_mc"]
            ),
            "200k_to_300k": float(
                matrix["300k"][dataset]["metrics"]["spearman_qmin_mc"]
                - matrix["200k"][dataset]["metrics"]["spearman_qmin_mc"]
            ),
        }

    matching = {
        "early": "100k",
        "mid": "200k",
        "late": "300k",
    }
    for dataset, checkpoint in matching.items():
        result["matched_stage_vs_step0"][dataset] = {
            "checkpoint": checkpoint,
            "spearman_delta": float(
                matrix[checkpoint][dataset]["metrics"]["spearman_qmin_mc"]
                - matrix["step0"][dataset]["metrics"]["spearman_qmin_mc"]
            ),
            "mae_delta": float(
                matrix[checkpoint][dataset]["metrics"]["mae_qmin_mc"]
                - matrix["step0"][dataset]["metrics"]["mae_qmin_mc"]
            ),
        }
    return result


def evaluate_group(run_dir, group, shared, device):
    group_dir = run_dir / group
    if not group_dir.exists():
        raise FileNotFoundError(group_dir)

    sources_manifest = shared["stage2_sources"]
    if group not in sources_manifest:
        raise RuntimeError(f"stage2_source_manifest misses {group}")
    stage2_path = Path(
        sources_manifest[group]["checkpoint"]
    ).resolve()
    if not stage2_path.exists():
        raise FileNotFoundError(stage2_path)

    datasets = {}
    dataset_manifest = {}

    datasets["old"], dataset_manifest["old"] = old_dataset(
        shared["config"], group
    )
    for name, (lower, upper, replay_name) in STAGE_SPECS.items():
        datasets[name], dataset_manifest[name] = stage_dataset(
            group_dir, name, lower, upper, replay_name
        )

    checkpoints = checkpoint_paths(group_dir)
    matrix = {}

    for checkpoint_name, checkpoint_path in checkpoints.items():
        critic, payload, stage2_payload = load_stage3_critic(
            stage2_path, checkpoint_path, device
        )
        sync(device)

        actor_updates = int(payload.get("actor_updates", -1))
        if actor_updates != 0:
            release(critic, device)
            raise RuntimeError(
                f"{group}:{checkpoint_name}: actor_updates={actor_updates}; "
                "fixed-policy interpretation is invalid"
            )

        gamma = float(payload["config"]["gamma"])
        if not math.isclose(gamma, 0.99, rel_tol=0.0, abs_tol=1e-12):
            release(critic, device)
            raise RuntimeError(
                f"{group}:{checkpoint_name}: unexpected gamma={gamma}"
            )

        matrix[checkpoint_name] = {}
        for dataset_name in ("old", "early", "mid", "late"):
            print(
                f"[EVAL] group={group} checkpoint={checkpoint_name} "
                f"dataset={dataset_name} "
                f"episodes={len(datasets[dataset_name])}",
                flush=True,
            )
            metrics = evaluate_dataset(
                critic, datasets[dataset_name], device, gamma
            )
            matrix[checkpoint_name][dataset_name] = {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_env_steps": int(payload.get("env_steps", -1)),
                "critic_updates": int(payload.get("updates", -1)),
                "actor_updates": actor_updates,
                "stage2_source_checkpoint": str(stage2_path),
                "stage2_source_step": int(
                    stage2_payload.get(
                        "checkpoint_step",
                        stage2_payload.get("step", -1),
                    )
                ),
                "metrics": metrics,
            }
            if not metrics["finite"]:
                release(critic, device)
                raise FloatingPointError(
                    f"{group}:{checkpoint_name}:{dataset_name}: "
                    "non-finite evaluation"
                )

        sync(device)
        release(critic, device)

    checkpoint_meta = {
        name: {
            "path": str(path.resolve()),
            "env_steps": int(
                torch.load(path, map_location="cpu").get("env_steps", -1)
            ),
            "critic_updates": int(
                torch.load(path, map_location="cpu").get("updates", -1)
            ),
            "actor_updates": int(
                torch.load(path, map_location="cpu").get("actor_updates", -1)
            ),
        }
        for name, path in checkpoints.items()
    }

    return {
        "group": group,
        "stage2_source_checkpoint": str(stage2_path),
        "checkpoint_metadata": checkpoint_meta,
        "dataset_manifest": dataset_manifest,
        "matrix": matrix,
        "descriptive_deltas": descriptive_deltas(matrix),
        "fixed_policy_actor_updates_zero": all(
            row["actor_updates"] == 0
            for row in checkpoint_meta.values()
        ),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_matrix(group_result):
    print(f"\n=== {group_result['group']} : Spearman(Qmin, finite MC) ===")
    header = ["checkpoint", "old", "early", "mid", "late"]
    print("\t".join(header))
    for checkpoint in CHECKPOINTS:
        cells = [checkpoint]
        for dataset in ("old", "early", "mid", "late"):
            value = group_result["matrix"][checkpoint][dataset][
                "metrics"
            ]["spearman_qmin_mc"]
            cells.append(f"{value:.6f}")
        print("\t".join(cells))

    print(f"\n=== {group_result['group']} : MAE(Qmin, finite MC) ===")
    print("\t".join(header))
    for checkpoint in CHECKPOINTS:
        cells = [checkpoint]
        for dataset in ("old", "early", "mid", "late"):
            value = group_result["matrix"][checkpoint][dataset][
                "metrics"
            ]["mae_qmin_mc"]
            cells.append(f"{value:.6f}")
        print("\t".join(cells))


def main():
    args = arguments()
    run_dir = Path(args.stage3_run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    shared_dir = run_dir / "shared"
    config_path = shared_dir / "config_resolved.json"
    sources_path = shared_dir / "stage2_source_manifest.json"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not sources_path.exists():
        raise FileNotFoundError(sources_path)

    config = read_json(config_path)
    stage2_sources = read_json(sources_path)
    device = resolve_device(args.device)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / "testing" / "critic_stage_adaptation_matrix"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "status": "RUNNING",
        "experiment": "stage3_300k_critic_stage_adaptation_matrix",
        "read_only": True,
        "stage3_run_dir": str(run_dir),
        "device": str(device),
        "history_contract": "sliding_horizon_10_zero_state",
        "finite_mc_complete_episodes_only": True,
        "stage_boundaries_env_steps": [100_000, 200_000, 300_000],
        "boundary_policy": (
            "exclude first completed episode per env after 100K/200K "
            "to conservatively remove cross-boundary episodes"
        ),
        "shared_config": str(config_path.resolve()),
        "stage2_source_manifest": str(sources_path.resolve()),
        "groups": {},
        "safety": {
            "environment_steps_performed": 0,
            "optimizer_steps_performed": 0,
            "actor_updates_performed": 0,
            "critic_updates_performed": 0,
            "rollout_started": False,
            "training_checkpoint_writes": 0,
            "production_source_modified": False,
        },
    }

    csv_rows = []
    for group in args.groups:
        result = evaluate_group(
            run_dir,
            group,
            {"config": config, "stage2_sources": stage2_sources},
            device,
        )
        report["groups"][group] = result
        csv_rows.extend(flatten_rows(group, result["matrix"]))
        print_matrix(result)

        # Save partial progress after each group.
        (output_dir / "critic_stage_adaptation_matrix.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        write_csv(
            output_dir / "critic_stage_adaptation_matrix.csv",
            csv_rows,
        )

    report["status"] = "PASS"
    report["validity"] = {
        "all_requested_groups_evaluated": (
            set(report["groups"]) == set(args.groups)
        ),
        "all_actor_updates_zero": all(
            value["fixed_policy_actor_updates_zero"]
            for value in report["groups"].values()
        ),
        "all_matrix_cells_finite": all(
            cell["metrics"]["finite"]
            for group in report["groups"].values()
            for checkpoint in group["matrix"].values()
            for cell in checkpoint.values()
        ),
        "no_training_or_rollout": True,
    }
    if not all(report["validity"].values()):
        report["status"] = "INCONCLUSIVE"

    json_path = output_dir / "critic_stage_adaptation_matrix.json"
    csv_path = output_dir / "critic_stage_adaptation_matrix.csv"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    write_csv(csv_path, csv_rows)

    print(f"\n[STATUS] {report['status']}", flush=True)
    print(f"[JSON] {json_path}", flush=True)
    print(f"[CSV] {csv_path}", flush=True)


if __name__ == "__main__":
    main()
