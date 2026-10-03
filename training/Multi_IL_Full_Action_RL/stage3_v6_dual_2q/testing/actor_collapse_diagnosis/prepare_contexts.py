#!/usr/bin/env python3
"""Select immutable, horizon-aligned Actor/Critic diagnostic contexts (no simulator)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
V5 = ROOT / "stage3_v5_rgmm_td3"
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))
from stage3_v5_replay import Stage1OfflineSequenceReplay

LENGTH = 10
SEED = 20260929


def candidates(episodes, success):
    result = []
    for episode_index, episode in enumerate(episodes):
        if bool(episode.get("success", False)) != success:
            continue
        for start in range(0, len(episode["actions"]) - LENGTH + 1, LENGTH):
            result.append((episode_index, start))
    return result


def select(rng, rows, count):
    if len(rows) < count:
        raise RuntimeError(f"Only {len(rows)} eligible windows for {count} requested")
    indices = rng.choice(len(rows), size=count, replace=False)
    return [rows[int(i)] for i in indices]


def pack(episodes, selections, source):
    obs, actions, steps, metadata = [], [], [], []
    for episode_index, start in selections:
        episode = episodes[int(episode_index)]
        stop = int(start) + LENGTH
        o = np.asarray(episode["observations"][start:stop], np.float32)
        a = np.asarray(episode["actions"][start:stop], np.float32)
        s = np.asarray(episode["episode_steps"][start:stop], np.int64)
        if o.shape != (LENGTH, 59) or a.shape != (LENGTH, 14):
            raise RuntimeError("Invalid context shape")
        if start % LENGTH or not np.array_equal(s, np.arange(start, stop)):
            raise RuntimeError("Context violates Actor reset-boundary contract")
        if not (np.isfinite(o).all() and np.isfinite(a).all()):
            raise RuntimeError("Non-finite context")
        obs.append(o)
        actions.append(a)
        steps.append(s)
        metadata.append({
            "source": source,
            "episode_index": int(episode_index),
            "start": int(start),
            "success": bool(episode.get("success", False)),
            "diagnostic_episode_id": (
                int(episode["diagnostic_episode_id"])
                if "diagnostic_episode_id" in episode else None
            ),
        })
    return (np.stack(obs), np.stack(actions), np.stack(steps), metadata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    output_dir = args.output_dir.resolve()
    context_path = output_dir / "fixed_contexts.npz"
    index_path = output_dir / "fixed_context_indices.json"
    if context_path.exists() or index_path.exists():
        raise FileExistsError("Fixed context outputs already exist; refusing overwrite")
    config = json.loads((run / "shared/config_resolved.json").read_text())
    replay_path = run / "mean2q/multi_q/checkpoints/last.sequences.npy"
    payload = np.load(replay_path, allow_pickle=True).item()
    frozen = payload["fixed_critic_diagnostic_set"]
    if not isinstance(frozen, dict) or len(frozen["episodes"]) < 2:
        raise RuntimeError("Frozen readiness episode set is unavailable")
    episodes = frozen["episodes"]
    rng = np.random.default_rng(SEED)
    success_rows = select(rng, candidates(episodes, True), 128)
    failure_rows = select(rng, candidates(episodes, False), 128)
    a3_rows = success_rows + failure_rows
    a3_obs, a3_actions, a3_steps, a3_meta = pack(
        episodes, a3_rows, "frozen_online"
    )
    a3_success = np.asarray([row["success"] for row in a3_meta], np.bool_)

    bc_source = Stage1OfflineSequenceReplay(
        config["offline_sources"]["bc_rnn"], "rnn", seed=SEED
    )
    bc_rows = select(rng, candidates(bc_source.episodes, True), 64)
    bc_obs, bc_actions, bc_steps, bc_meta = pack(
        bc_source.episodes, bc_rows, "bc_rnn_success"
    )
    online_rows = success_rows[:64] + failure_rows[:64]
    online_obs, online_actions, online_steps, online_meta = pack(
        episodes, online_rows, "frozen_online"
    )
    a2_obs = np.concatenate((bc_obs, online_obs), axis=0)
    a2_actions = np.concatenate((bc_actions, online_actions), axis=0)
    a2_steps = np.concatenate((bc_steps, online_steps), axis=0)
    a2_group = np.concatenate((
        np.zeros(64, np.int8),
        np.ones(64, np.int8),
        np.full(64, 2, np.int8),
    ))
    if not np.all(a2_steps[:, 0] % LENGTH == 0):
        raise RuntimeError("A2 context is not Actor boundary-aligned")
    if not np.all(a3_steps[:, 0] % LENGTH == 0):
        raise RuntimeError("A3 context is not Actor boundary-aligned")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        context_path,
        a2_observations=a2_obs,
        a2_actions=a2_actions,
        a2_episode_steps=a2_steps,
        a2_group=a2_group,
        a3_observations=a3_obs,
        a3_actions=a3_actions,
        a3_episode_steps=a3_steps,
        a3_success=a3_success,
    )
    index = {
        "rng_seed": SEED,
        "horizon": LENGTH,
        "frozen_replay_path": str(replay_path),
        "frozen_source": frozen.get("source"),
        "frozen_seed": int(frozen["seed"]),
        "frozen_episode_count": len(episodes),
        "frozen_success_episode_count": int(frozen["success_episode_count"]),
        "frozen_failure_episode_count": int(frozen["failure_episode_count"]),
        "a2_groups": {"bc_rnn_success": 64, "online_success": 64, "online_failure": 64},
        "a3_groups": {"online_success": 128, "online_failure": 128},
        "a2_indices": bc_meta + online_meta,
        "a3_indices": a3_meta,
    }
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "contexts": str(context_path),
        "indices": str(index_path),
        "a2_count": len(a2_group),
        "a3_count": len(a3_success),
        "a3_success_count": int(a3_success.sum()),
        "a3_failure_count": int((~a3_success).sum()),
    }, indent=2))


if __name__ == "__main__":
    main()
