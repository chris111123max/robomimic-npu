#!/usr/bin/env python3
"""Four-worker branch evaluation with one persistent policy per worker."""

import argparse
import copy
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from utils.env_utils import (close_environment, create_environment, initialize_observation_utils,
                             restore_branch_state, seed_environment_stream, task_succeeded)
from utils.exp00_reader import RNN, TRANSFORMER
from utils.history_utils import reconstruct_before_action
from utils.policy_loader import load_policy, release_policy, select_device
from utils.result_utils import (append_jsonl, atomic_csv, bool_value, read_csv, read_json,
                                read_jsonl, stable_seed)
from utils.state_utils import load_branch_state, load_trajectory, save_trajectory


RESULT_FIELDS = ["initial_state_id", "environment_seed", "direction", "source_policy", "target_policy",
                 "branch_step", "evaluated_policy", "role", "trial_index", "policy_rng_seed", "success",
                 "episode_return_from_branch", "episode_length_from_branch", "global_end_step",
                 "success_step_global", "termination_reason", "source_state_hash", "restored_state_hash",
                 "history_length", "checkpoint_path", "wall_time_seconds", "stable_cross_success_flag",
                 "trajectory_path", "worker_id", "worker_visible_device"]
ERROR_FIELDS = ["initial_state_id", "environment_seed", "direction", "source_policy", "target_policy",
                "branch_step", "evaluated_policy", "role", "trial_index", "policy_rng_seed",
                "termination_reason", "source_state_hash", "checkpoint_path", "wall_time_seconds",
                "worker_id", "worker_visible_device", "error_type", "error_message", "traceback"]


def trial_key(row):
    return (int(row["initial_state_id"]), row["direction"], int(row["branch_step"]),
            row["evaluated_policy"], int(row["trial_index"]))


def rollout_branch(policy, env, observation, remaining, branch_step, save_states):
    observations, actions, rewards, dones, successes = [], [], [], [], []
    simulator_states = [] if save_states else None
    total_return, success_step, reason = 0.0, None, "horizon"
    for local_step in range(remaining):
        observations.append({key: np.asarray(value).copy() for key, value in observation.items()})
        if simulator_states is not None:
            simulator_states.append(np.asarray(env.get_state()["states"]).copy())
        action = np.asarray(policy(ob=observation)).copy()
        observation, reward, done, _ = env.step(action)
        success = task_succeeded(env)
        actions.append(action); rewards.append(float(reward)); dones.append(bool(done)); successes.append(success)
        total_return += float(reward)
        if success:
            success_step = branch_step + local_step + 1; reason = "success"; break
        if done:
            reason = "environment_done"; break
        observation = copy.deepcopy(observation)
    return {"success": success_step is not None, "episode_return": total_return,
            "episode_length": len(actions), "success_step_global": success_step,
            "termination_reason": reason, "observations": observations, "actions": actions,
            "rewards": rewards, "dones": dones, "successes": successes,
            "simulator_states": simulator_states}


def build_tasks(config, run_dir, policy_name, branch_steps, repeats):
    cases = read_csv(run_dir / "recheck/case_summary.csv")
    source_rows = read_csv(run_dir / "source_trajectories/source_trajectory_summary.csv")
    source_by_id = {int(row["initial_state_id"]): row for row in source_rows}
    tasks = []
    for case in cases:
        sid = int(case["initial_state_id"]); source = source_by_id.get(sid)
        if not source or source["status"] != "complete":
            continue
        role = "source_continuation" if policy_name == case["source_policy"] else \
               "target_takeover" if policy_name == case["target_policy"] else None
        if role is None:
            continue
        for step in branch_steps:
            branch_path = run_dir / "branch_states" / case["direction"].lower() / \
                          f"state_{sid:06d}_step_{step:03d}.npz"
            if not branch_path.exists():
                continue
            for trial in range(repeats):
                tasks.append({**case, "branch_step": step, "evaluated_policy": policy_name,
                              "role": role, "trial_index": trial, "branch_path": str(branch_path),
                              "source_trajectory_path": source["trajectory_path"]})
    return tasks


def worker(config_path, run_dir, policy_name, worker_id, shard_index, shard_count,
           branch_steps, repeats, force):
    config, run_dir = read_json(config_path), Path(run_dir)
    log_path = run_dir / "branch_results/workers" / f"worker_{worker_id}.jsonl"
    if force and log_path.exists():
        log_path.unlink()
    prior = read_jsonl(log_path)
    def valid_prior(row):
        if row.get("status") != "complete":
            return False
        try:
            trajectory = load_trajectory(run_dir / row["trajectory_path"])
            return len(trajectory["actions"]) == int(row["episode_length_from_branch"])
        except Exception:
            return False
    completed = {trial_key(row) for row in prior if valid_prior(row)}
    tasks = build_tasks(config, run_dir, policy_name, branch_steps, repeats)
    tasks = [task for index, task in enumerate(tasks) if index % shard_count == shard_index]
    initialize_observation_utils(config["policies"][policy_name]["checkpoint_path"])
    env, _ = create_environment(config["dataset_path"]); device = select_device()
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "<unset>")
    policy, _, _ = load_policy(policy_name, config["policies"][policy_name]["checkpoint_path"], device)
    try:
        for ordinal, task in enumerate(tasks, 1):
            key = trial_key(task)
            if key in completed:
                continue
            started = time.monotonic(); sid = int(task["initial_state_id"]); step = int(task["branch_step"])
            seed = stable_seed(config["seed_base"], "branch", sid, task["direction"], step,
                               policy_name, int(task["trial_index"]))
            branch = load_branch_state(task["branch_path"])
            base = {k: task[k] for k in ("initial_state_id", "environment_seed", "direction",
                    "source_policy", "target_policy", "branch_step", "evaluated_policy", "role", "trial_index")}
            base.update({"policy_rng_seed": seed, "source_state_hash": branch["metadata"]["state_hash"],
                         "checkpoint_path": config["policies"][policy_name]["checkpoint_path"],
                         "worker_id": worker_id, "worker_visible_device": visible})
            try:
                seed_environment_stream(stable_seed(config["seed_base"], "branch_environment", sid, step))
                observation, restored_hash = restore_branch_state(env, branch)
                source_trajectory = load_trajectory(run_dir / task["source_trajectory_path"])
                history = reconstruct_before_action(policy, source_trajectory["observations"], step,
                                                    trial_seed=seed)
                stats = rollout_branch(policy, env, observation, int(config["horizon"]) - step,
                                       step, bool(config["save_branch_sim_states"]))
                trajectory_path = run_dir / "branch_results/trajectories" / policy_name / \
                                  f"state_{sid:06d}_step_{step:03d}_trial_{int(task['trial_index']):02d}.npz"
                extra = {}
                if stats["simulator_states"] is not None:
                    extra["simulator_states"] = np.asarray(stats["simulator_states"])
                save_trajectory(trajectory_path, stats["observations"], stats["actions"], stats["rewards"],
                                stats["dones"], stats["successes"],
                                bool(config["compressed_trajectories"]), **extra)
                record = {**base, "status": "complete", "success": int(stats["success"]),
                          "episode_return_from_branch": stats["episode_return"],
                          "episode_length_from_branch": stats["episode_length"],
                          "global_end_step": step + stats["episode_length"],
                          "success_step_global": stats["success_step_global"],
                          "termination_reason": stats["termination_reason"],
                          "restored_state_hash": restored_hash, "history_length": history["history_length"],
                          "wall_time_seconds": time.monotonic() - started,
                          "stable_cross_success_flag": int(bool_value(task["stable_cross_success"])),
                          "trajectory_path": str(trajectory_path.relative_to(run_dir))}
                append_jsonl(log_path, record); completed.add(key)
                print(f"[branch worker={worker_id} {ordinal}/{len(tasks)}] case={sid:06d} "
                      f"step={step} policy={policy_name} success={record['success']}", flush=True)
            except Exception as exc:
                append_jsonl(log_path, {**base, "status": "error", "termination_reason": "error",
                    "wall_time_seconds": time.monotonic() - started, "error_type": type(exc).__name__,
                    "error_message": str(exc), "traceback": traceback.format_exc()})
                print(f"[branch worker={worker_id}] ERROR case={sid:06d} step={step}: {exc}", flush=True)
    finally:
        release_policy(policy); close_environment(env)


def aggregate(run_dir):
    run_dir = Path(run_dir); config = read_json(run_dir / "config.json"); latest = {}
    for path in sorted((run_dir / "branch_results/workers").glob("worker_*.jsonl")):
        for row in read_jsonl(path):
            latest[trial_key(row)] = row
    complete = sorted([r for r in latest.values() if r.get("status") == "complete"], key=trial_key)
    errors = sorted([r for r in latest.values() if r.get("status") == "error"], key=trial_key)
    expected = sum(len(build_tasks(config, run_dir, policy, config["branch_steps"], int(config["branch_repeats"])))
                   for policy in (RNN, TRANSFORMER))
    if len(complete) + len(errors) != expected:
        raise RuntimeError(f"Branch aggregate is incomplete: complete={len(complete)}, errors={len(errors)}, expected={expected}")
    atomic_csv(run_dir / "branch_results/raw_trials.csv", RESULT_FIELDS, complete)
    atomic_csv(run_dir / "branch_results/errors.csv", ERROR_FIELDS, errors)
    groups = {}
    for row in complete:
        key = (int(row["initial_state_id"]), row["direction"], int(row["branch_step"]))
        groups.setdefault(key, []).append(row)
    summaries = []
    for (sid, direction, step), rows in sorted(groups.items()):
        source_rows = [r for r in rows if r["role"] == "source_continuation"]
        target_rows = [r for r in rows if r["role"] == "target_takeover"]
        if not source_rows or not target_rows:
            continue
        source_rate = sum(int(r["success"]) for r in source_rows) / len(source_rows)
        target_rate = sum(int(r["success"]) for r in target_rows) / len(target_rows)
        summaries.append({"initial_state_id": sid, "direction": direction, "branch_step": step,
                          "source_policy": source_rows[0]["source_policy"], "target_policy": source_rows[0]["target_policy"],
                          "source_trials": len(source_rows), "target_trials": len(target_rows),
                          "source_success_rate": source_rate, "target_success_rate": target_rate,
                          "local_competence_gap": target_rate - source_rate})
    if summaries:
        atomic_csv(run_dir / "branch_results/by_branch_state.csv", list(summaries[0]), summaries)
    print(f"[branch] aggregate complete trials={len(complete)}, errors={len(errors)}, branch states={len(summaries)}")
    for row in summaries:
        print(f"[branch] case {int(row['initial_state_id']):06d} direction={row['direction']} "
              f"step={row['branch_step']} source={row['source_success_rate']:.3f} "
              f"target={row['target_success_rate']:.3f} gap={row['local_competence_gap']:+.3f}")
    if errors:
        raise RuntimeError(f"Branch evaluation has {len(errors)} runtime errors; errors are not failures")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True); parser.add_argument("--branch-steps", required=True)
    parser.add_argument("--branch-repeats", type=int, required=True); parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-id", type=int); parser.add_argument("--policy", choices=(RNN, TRANSFORMER))
    parser.add_argument("--shard-index", type=int); parser.add_argument("--shard-count", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args(); steps = [int(x) for x in args.branch_steps.split(",") if x]
    if args.aggregate_only:
        aggregate(args.run_dir)
    else:
        worker(args.config, args.run_dir, args.policy, args.worker_id, args.shard_index,
               args.shard_count, steps, args.branch_repeats, args.force)


if __name__ == "__main__":
    main()
