#!/usr/bin/env python3
"""Controlled stochastic recheck of all selected initial conditions."""

import argparse
import copy
import sys
import time
import traceback
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from utils.env_utils import (close_environment, create_environment, initialize_observation_utils,
                             exp00_environment_stream_seed, restore_initial_state,
                             seed_environment_stream, task_succeeded)
from utils.exp00_reader import RNN, TRANSFORMER
from utils.policy_loader import load_policy, release_policy, select_device
from utils.result_utils import atomic_csv, read_csv, read_json, stable_seed
from utils.rng_utils import seed_policy_rng
from utils.state_utils import load_exp00_state, load_trajectory, save_trajectory, trajectory_length


FIELDS = ["initial_state_id", "environment_seed", "direction", "policy_name", "trial_index",
          "policy_rng_seed", "success", "episode_length", "episode_return", "success_step",
          "termination_reason", "trajectory_path", "wall_time_seconds"]
ERROR_FIELDS = ["initial_state_id", "environment_seed", "direction", "policy_name", "trial_index",
                "policy_rng_seed", "termination_reason", "error_type", "error_message", "traceback"]


def rollout(policy, env, observation, horizon, terminate_on_success):
    policy.start_episode()
    observations, actions, rewards, dones, successes = [], [], [], [], []
    total_return, success_step, reason = 0.0, None, "horizon"
    for step in range(horizon):
        observations.append({k: np.asarray(v).copy() for k, v in observation.items()})
        action = np.asarray(policy(ob=observation)).copy()
        observation, reward, done, _ = env.step(action)
        success = task_succeeded(env)
        actions.append(action); rewards.append(float(reward)); dones.append(bool(done)); successes.append(success)
        total_return += float(reward)
        if success and success_step is None:
            success_step = step + 1
        if success and terminate_on_success:
            reason = "success"; break
        if done:
            reason = "environment_done"; break
        observation = copy.deepcopy(observation)
    return {"success": success_step is not None, "episode_length": len(actions),
            "episode_return": total_return, "success_step": success_step,
            "termination_reason": reason, "observations": observations, "actions": actions,
            "rewards": rewards, "dones": dones, "successes": successes}


def _key(row):
    return int(row["initial_state_id"]), row["policy_name"], int(row["trial_index"])


def _valid_rows(path, run_dir):
    if not path.exists():
        return []
    rows = []
    for row in read_csv(path):
        try:
            if trajectory_length(run_dir / row["trajectory_path"]) == int(row["episode_length"]):
                rows.append(row)
        except Exception:
            pass
    return rows


def _case_summary(config, run_dir, cases, rows, repeats):
    by_case = []
    for case in cases:
        sid = int(case["initial_state_id"])
        relevant = [row for row in rows if int(row["initial_state_id"]) == sid]
        counts = {
            policy: sum(int(row["success"]) for row in relevant if row["policy_name"] == policy)
            for policy in (RNN, TRANSFORMER)
        }
        source, target = case["source_policy"], case["target_policy"]
        stable = (
            (int(repeats) - counts[source]) >= int(config["stable_failure_min_count"])
            and counts[target] >= int(config["stable_success_min_count"])
        )
        by_case.append({
            **case,
            "rnn_success_count": counts[RNN],
            "transformer_success_count": counts[TRANSFORMER],
            "recheck_repeats": int(repeats),
            "stable_cross_success": int(stable),
            "unstable_cross_success": int(not stable),
        })
        print(
            f"[recheck] case {sid:06d} {RNN}={counts[RNN]}/{repeats} "
            f"{TRANSFORMER}={counts[TRANSFORMER]}/{repeats} "
            f"stable_cross_success={str(stable).lower()}",
            flush=True,
        )
    atomic_csv(run_dir / "recheck/case_summary.csv", list(by_case[0]), by_case)


def aggregate(config_path, run_dir, repeats, force=False):
    config, run_dir = read_json(config_path), Path(run_dir)
    cases = sorted(read_csv(run_dir / "selected_cases.csv"), key=lambda row: int(row["initial_state_id"]))
    raw_path, error_path = run_dir / "recheck/raw_trials.csv", run_dir / "recheck/errors.csv"
    candidates = [] if force else _valid_rows(raw_path, run_dir)
    for path in sorted((run_dir / "recheck/workers").glob("worker_*_trials.csv")):
        candidates.extend(_valid_rows(path, run_dir))
    merged = {_key(row): row for row in candidates}
    rows = sorted(merged.values(), key=_key)
    expected = len(cases) * 2 * int(repeats)
    if len(rows) != expected:
        raise RuntimeError(f"Recheck aggregation is incomplete: {len(rows)}/{expected} valid trials")

    error_candidates = [] if force or not error_path.exists() else read_csv(error_path)
    for path in sorted((run_dir / "recheck/workers").glob("worker_*_errors.csv")):
        error_candidates.extend(read_csv(path))
    errors = {_key(row): row for row in error_candidates if _key(row) not in merged}
    atomic_csv(raw_path, FIELDS, rows)
    atomic_csv(error_path, ERROR_FIELDS, sorted(errors.values(), key=_key))
    _case_summary(config, run_dir, cases, rows, repeats)
    if errors:
        raise RuntimeError(f"Recheck has {len(errors)} runtime errors")
    print(f"[recheck] aggregate complete trials={len(rows)}, cases={len(cases)}", flush=True)


def recheck(config_path, run_dir, source_run, repeats, force=False,
            worker_id=0, shard_index=0, shard_count=1):
    config, run_dir = read_json(config_path), Path(run_dir)
    cases = sorted(read_csv(run_dir / "selected_cases.csv"), key=lambda row: int(row["initial_state_id"]))
    if not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("shard-index must satisfy 0 <= shard-index < shard-count")
    assigned_cases = cases[int(shard_index)::int(shard_count)]
    source_manifest = read_json(Path(source_run) / "initial_state_manifest.json")
    seed_manifest = read_json(Path(source_run) / "seed_manifest.json")
    entries = {int(row["initial_state_id"]): row for row in source_manifest["states"]}
    raw_path, error_path = run_dir / "recheck/raw_trials.csv", run_dir / "recheck/errors.csv"
    worker_dir = run_dir / "recheck/workers"
    worker_path = worker_dir / f"worker_{int(worker_id)}_trials.csv"
    worker_error_path = worker_dir / f"worker_{int(worker_id)}_errors.csv"
    baseline = [] if force else _valid_rows(raw_path, run_dir)
    local_rows = [] if force else _valid_rows(worker_path, run_dir)
    completed = {_key(row) for row in baseline + local_rows}
    errors = [] if force or not worker_error_path.exists() else read_csv(worker_error_path)
    if force:
        atomic_csv(worker_path, FIELDS, [])
        atomic_csv(worker_error_path, ERROR_FIELDS, [])
    pending = sum(
        (int(case["initial_state_id"]), policy, trial) not in completed
        for case in assigned_cases for policy in (RNN, TRANSFORMER) for trial in range(int(repeats))
    )
    print(
        f"[recheck worker={worker_id}] shard={shard_index}/{shard_count} "
        f"cases={len(assigned_cases)} pending_trials={pending}",
        flush=True,
    )
    if pending == 0:
        return
    initialize_observation_utils(config["policies"][RNN]["checkpoint_path"])
    env, _ = create_environment(config["dataset_path"])
    device = select_device()
    try:
        for policy_name in (RNN, TRANSFORMER):
            policy, _, _ = load_policy(policy_name, config["policies"][policy_name]["checkpoint_path"], device)
            try:
                for case in assigned_cases:
                    sid = int(case["initial_state_id"])
                    entry = entries[sid]
                    state, saved_obs = load_exp00_state(Path(source_run) / entry["state_file"])
                    for trial in range(int(repeats)):
                        key = sid, policy_name, trial
                        if key in completed:
                            continue
                        policy_seed = stable_seed(config["seed_base"], "recheck", sid, policy_name, trial)
                        started = time.monotonic()
                        try:
                            seed_environment_stream(exp00_environment_stream_seed(seed_manifest["meta_seed"], sid))
                            observation = restore_initial_state(env, state, saved_obs, entry)
                            seed_policy_rng(policy_seed)
                            stats = rollout(policy, env, observation, int(config["horizon"]),
                                            bool(config["terminate_on_success"]))
                            trajectory_path = run_dir / "recheck/trajectories" / policy_name / \
                                              f"state_{sid:06d}_trial_{trial:02d}.npz"
                            save_trajectory(trajectory_path, stats["observations"], stats["actions"],
                                            stats["rewards"], stats["dones"], stats["successes"],
                                            bool(config["compressed_trajectories"]))
                            row = {"initial_state_id": sid, "environment_seed": int(case["environment_seed"]),
                                   "direction": case["direction"], "policy_name": policy_name,
                                   "trial_index": trial, "policy_rng_seed": policy_seed,
                                   "success": int(stats["success"]), "episode_length": stats["episode_length"],
                                   "episode_return": stats["episode_return"], "success_step": stats["success_step"],
                                   "termination_reason": stats["termination_reason"],
                                   "trajectory_path": str(trajectory_path.relative_to(run_dir)),
                                   "wall_time_seconds": time.monotonic() - started}
                            local_rows = [item for item in local_rows if _key(item) != key] + [row]
                            completed.add(key)
                            atomic_csv(worker_path, FIELDS, sorted(local_rows, key=_key))
                            errors = [r for r in errors if not (int(r["initial_state_id"]) == sid and
                                      r["policy_name"] == policy_name and int(r["trial_index"]) == trial)]
                            atomic_csv(worker_error_path, ERROR_FIELDS, errors)
                            print(
                                f"[recheck worker={worker_id}] case={sid:06d} policy={policy_name} "
                                f"trial={trial} success={int(stats['success'])} "
                                f"length={stats['episode_length']}",
                                flush=True,
                            )
                        except Exception as exc:
                            errors = [r for r in errors if not (int(r["initial_state_id"]) == sid and
                                      r["policy_name"] == policy_name and int(r["trial_index"]) == trial)]
                            errors.append({"initial_state_id": sid, "environment_seed": case["environment_seed"],
                                           "direction": case["direction"], "policy_name": policy_name,
                                           "trial_index": trial, "policy_rng_seed": policy_seed,
                                           "termination_reason": "error", "error_type": type(exc).__name__,
                                           "error_message": str(exc), "traceback": traceback.format_exc()})
                            atomic_csv(worker_error_path, ERROR_FIELDS, errors)
                            raise
            finally:
                release_policy(policy)
    finally:
        close_environment(env)

    if errors:
        raise RuntimeError(f"Recheck worker {worker_id} has {len(errors)} runtime errors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-run", required=True); parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--worker-id", type=int); parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1); parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.aggregate_only:
        aggregate(args.config, args.run_dir, args.repeats, args.force)
    else:
        worker_id = 0 if args.worker_id is None else args.worker_id
        recheck(args.config, args.run_dir, args.source_run, args.repeats, args.force,
                worker_id, args.shard_index, args.shard_count)
        if args.worker_id is None:
            aggregate(args.config, args.run_dir, args.repeats, args.force)


if __name__ == "__main__":
    main()
