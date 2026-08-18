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
from utils.state_utils import load_exp00_state, load_trajectory, save_trajectory


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


def recheck(config_path, run_dir, source_run, repeats, force=False):
    config, run_dir = read_json(config_path), Path(run_dir)
    cases = read_csv(run_dir / "selected_cases.csv")
    source_manifest = read_json(Path(source_run) / "initial_state_manifest.json")
    seed_manifest = read_json(Path(source_run) / "seed_manifest.json")
    entries = {int(row["initial_state_id"]): row for row in source_manifest["states"]}
    raw_path, error_path = run_dir / "recheck/raw_trials.csv", run_dir / "recheck/errors.csv"
    existing = [] if force or not raw_path.exists() else read_csv(raw_path)
    def valid_existing(row):
        try:
            trajectory = load_trajectory(run_dir / row["trajectory_path"])
            return len(trajectory["actions"]) == int(row["episode_length"])
        except Exception:
            return False
    existing = [row for row in existing if valid_existing(row)]
    completed = {(int(r["initial_state_id"]), r["policy_name"], int(r["trial_index"])) for r in existing}
    errors = [] if force or not error_path.exists() else read_csv(error_path)
    initialize_observation_utils(config["policies"][RNN]["checkpoint_path"])
    env, _ = create_environment(config["dataset_path"])
    device = select_device()
    try:
        for policy_name in (RNN, TRANSFORMER):
            policy, _, _ = load_policy(policy_name, config["policies"][policy_name]["checkpoint_path"], device)
            try:
                for case in cases:
                    sid = int(case["initial_state_id"])
                    entry = entries[sid]
                    state, saved_obs = load_exp00_state(Path(source_run) / entry["state_file"])
                    for trial in range(int(repeats)):
                        key = (sid, policy_name, trial)
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
                            existing.append(row); completed.add(key)
                            atomic_csv(raw_path, FIELDS, sorted(existing, key=lambda r: (int(r["initial_state_id"]), r["policy_name"], int(r["trial_index"]))))
                            errors = [r for r in errors if not (int(r["initial_state_id"]) == sid and
                                      r["policy_name"] == policy_name and int(r["trial_index"]) == trial)]
                            atomic_csv(error_path, ERROR_FIELDS, errors)
                        except Exception as exc:
                            errors = [r for r in errors if not (int(r["initial_state_id"]) == sid and
                                      r["policy_name"] == policy_name and int(r["trial_index"]) == trial)]
                            errors.append({"initial_state_id": sid, "environment_seed": case["environment_seed"],
                                           "direction": case["direction"], "policy_name": policy_name,
                                           "trial_index": trial, "policy_rng_seed": policy_seed,
                                           "termination_reason": "error", "error_type": type(exc).__name__,
                                           "error_message": str(exc), "traceback": traceback.format_exc()})
                            atomic_csv(error_path, ERROR_FIELDS, errors)
                            raise
            finally:
                release_policy(policy)
    finally:
        close_environment(env)

    by_case = []
    for case in cases:
        sid = int(case["initial_state_id"])
        relevant = [r for r in existing if int(r["initial_state_id"]) == sid]
        counts = {p: sum(int(r["success"]) for r in relevant if r["policy_name"] == p)
                  for p in (RNN, TRANSFORMER)}
        source, target = case["source_policy"], case["target_policy"]
        stable = ((int(repeats) - counts[source]) >= int(config["stable_failure_min_count"]) and
                  counts[target] >= int(config["stable_success_min_count"]))
        by_case.append({**case, "rnn_success_count": counts[RNN],
                        "transformer_success_count": counts[TRANSFORMER],
                        "recheck_repeats": int(repeats), "stable_cross_success": int(stable),
                        "unstable_cross_success": int(not stable)})
        print(f"[recheck] case {sid:06d} {RNN}={counts[RNN]}/{repeats} "
              f"{TRANSFORMER}={counts[TRANSFORMER]}/{repeats} stable_cross_success={str(stable).lower()}")
    atomic_csv(run_dir / "recheck/case_summary.csv", list(by_case[0]), by_case)
    if errors:
        raise RuntimeError(f"Recheck has {len(errors)} runtime errors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-run", required=True); parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); recheck(args.config, args.run_dir, args.source_run, args.repeats, args.force)


if __name__ == "__main__":
    main()
