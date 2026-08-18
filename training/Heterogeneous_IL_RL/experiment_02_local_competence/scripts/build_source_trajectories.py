#!/usr/bin/env python3
"""Materialize one reproducible real source-policy failure trajectory per case."""

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
from utils.result_utils import atomic_csv, bool_value, read_csv, read_json, stable_seed
from utils.rng_utils import capture_rng_state, seed_policy_rng
from utils.state_utils import (copy_observation, load_exp00_state, observation_hash,
                               branch_file_valid, load_trajectory, save_branch_state,
                               save_trajectory, simulator_state_hash,
                               state_vector_hash)


SUMMARY_FIELDS = ["initial_state_id", "environment_seed", "direction", "source_policy", "target_policy",
                  "status", "source_trial_origin", "source_trial_index", "policy_rng_seed",
                  "episode_length", "episode_return", "branch_states_saved", "trajectory_path",
                  "wall_time_seconds", "error_message"]


def run_source(policy, env, observation, horizon, branch_steps):
    policy.start_episode()
    observations, actions, rewards, dones, successes, branches = [], [], [], [], [], {}
    total_return, success_step, reason = 0.0, None, "horizon"
    for step in range(horizon):
        observations.append(copy_observation(observation))
        branch_pending = None
        if step in branch_steps:
            state = env.get_state()
            branch_pending = {"state": state, "observation": copy_observation(observation),
                              "rng_state": capture_rng_state()}
        action = np.asarray(policy(ob=observation)).copy()
        if branch_pending is not None:
            branch_pending["original_action"] = action.copy()
            branches[step] = branch_pending
        observation, reward, done, _ = env.step(action)
        success = task_succeeded(env)
        actions.append(action); rewards.append(float(reward)); dones.append(bool(done)); successes.append(success)
        total_return += float(reward)
        if success and success_step is None:
            success_step = step + 1
        if success:
            reason = "success"; break
        if done:
            reason = "environment_done"; break
        observation = copy.deepcopy(observation)
    return {"success": success_step is not None, "success_step": success_step,
            "episode_length": len(actions), "episode_return": total_return,
            "termination_reason": reason, "observations": observations, "actions": actions,
            "rewards": rewards, "dones": dones, "successes": successes, "branches": branches}


def build(config_path, run_dir, source_run, branch_steps, force=False):
    config, run_dir = read_json(config_path), Path(run_dir)
    cases = read_csv(run_dir / "recheck/case_summary.csv")
    rechecks = read_csv(run_dir / "recheck/raw_trials.csv")
    source_manifest = read_json(Path(source_run) / "initial_state_manifest.json")
    seed_manifest = read_json(Path(source_run) / "seed_manifest.json")
    entries = {int(row["initial_state_id"]): row for row in source_manifest["states"]}
    summary_path = run_dir / "source_trajectories/source_trajectory_summary.csv"
    summary = [] if force or not summary_path.exists() else read_csv(summary_path)
    def valid_complete(row):
        if row["status"] == "source_failure_not_reproduced":
            return True
        if row["status"] != "complete":
            return False
        try:
            trajectory = load_trajectory(run_dir / row["trajectory_path"])
            if len(trajectory["actions"]) != int(row["episode_length"]):
                return False
            for step in branch_steps:
                if step < int(row["episode_length"]):
                    path = run_dir / "branch_states" / row["direction"].lower() / f"state_{int(row['initial_state_id']):06d}_step_{step:03d}.npz"
                    if not branch_file_valid(path, step):
                        return False
            return True
        except Exception:
            return False
    summary = [row for row in summary if valid_complete(row)]
    done = {int(row["initial_state_id"]) for row in summary}
    initialize_observation_utils(config["policies"][RNN]["checkpoint_path"])
    env, _ = create_environment(config["dataset_path"]); device = select_device()
    try:
        for policy_name in (RNN, TRANSFORMER):
            policy, _, _ = load_policy(policy_name, config["policies"][policy_name]["checkpoint_path"], device)
            try:
                for case in [c for c in cases if c["source_policy"] == policy_name]:
                    sid = int(case["initial_state_id"])
                    if sid in done:
                        continue
                    started = time.monotonic(); entry = entries[sid]
                    state, saved_obs = load_exp00_state(Path(source_run) / entry["state_file"])
                    failed_trials = sorted(
                        [r for r in rechecks if int(r["initial_state_id"]) == sid and
                         r["policy_name"] == policy_name and not bool_value(r["success"])],
                        key=lambda row: int(row["trial_index"]))
                    attempts = [("recheck", int(r["trial_index"]), int(r["policy_rng_seed"])) for r in failed_trials]
                    attempts += [("failure_search", index,
                                  stable_seed(config["seed_base"], "source_failure", sid, policy_name, index))
                                 for index in range(int(config["max_source_failure_attempts"]))]
                    selected = None
                    try:
                        for origin, trial_index, policy_seed in attempts:
                            seed_environment_stream(exp00_environment_stream_seed(seed_manifest["meta_seed"], sid))
                            observation = restore_initial_state(env, state, saved_obs, entry)
                            seed_policy_rng(policy_seed)
                            stats = run_source(policy, env, observation, int(config["horizon"]), set(branch_steps))
                            if not stats["success"]:
                                selected = (origin, trial_index, policy_seed, stats)
                                break
                        if selected is None:
                            row = {**case, "status": "source_failure_not_reproduced", "source_trial_origin": "",
                                   "source_trial_index": "", "policy_rng_seed": "", "episode_length": "",
                                   "episode_return": "", "branch_states_saved": 0, "trajectory_path": "",
                                   "wall_time_seconds": time.monotonic() - started, "error_message": ""}
                        else:
                            origin, trial_index, policy_seed, stats = selected
                            direction_dir = case["direction"].lower()
                            trajectory_path = run_dir / "source_trajectories" / direction_dir / f"state_{sid:06d}.npz"
                            branch_paths = [str((run_dir / "branch_states" / direction_dir /
                                            f"state_{sid:06d}_step_{step:03d}.npz").relative_to(run_dir))
                                            for step in sorted(stats["branches"])]
                            save_trajectory(trajectory_path, stats["observations"], stats["actions"], stats["rewards"],
                                            stats["dones"], stats["successes"],
                                            bool(config["compressed_trajectories"]),
                                            branch_steps=np.asarray(sorted(stats["branches"]), dtype=np.int64),
                                            branch_state_paths=np.asarray(branch_paths),
                                            source_policy_rng_seed=np.asarray(policy_seed, dtype=np.int64),
                                            termination_reason=np.asarray(stats["termination_reason"]))
                            saved_count = 0
                            for step, branch in sorted(stats["branches"].items()):
                                branch_path = run_dir / "branch_states" / direction_dir / \
                                              f"state_{sid:06d}_step_{step:03d}.npz"
                                metadata = {"initial_state_id": sid, "environment_seed": int(case["environment_seed"]),
                                            "direction": case["direction"], "source_policy": policy_name,
                                            "target_policy": case["target_policy"], "branch_step": step,
                                            "history_length": step,
                                            "state_hash": simulator_state_hash(branch["state"]),
                                            "state_vector_hash": state_vector_hash(branch["state"]["states"]),
                                            "obs_hash": observation_hash(branch["observation"]),
                                            "source_policy_rng_seed": policy_seed}
                                save_branch_state(branch_path, branch["state"], branch["observation"],
                                                  branch["rng_state"], branch["original_action"], step, metadata)
                                saved_count += 1
                            row = {**case, "status": "complete", "source_trial_origin": origin,
                                   "source_trial_index": trial_index, "policy_rng_seed": policy_seed,
                                   "episode_length": stats["episode_length"], "episode_return": stats["episode_return"],
                                   "branch_states_saved": saved_count,
                                   "trajectory_path": str(trajectory_path.relative_to(run_dir)),
                                   "wall_time_seconds": time.monotonic() - started, "error_message": ""}
                            print(f"[build] case {sid:06d} source policy={policy_name} failure trajectory "
                                  f"found at {origin} trial={trial_index}; branch states={saved_count}")
                    except Exception as exc:
                        row = {**case, "status": "error", "source_trial_origin": "", "source_trial_index": "",
                               "policy_rng_seed": "", "episode_length": "", "episode_return": "",
                               "branch_states_saved": 0, "trajectory_path": "",
                               "wall_time_seconds": time.monotonic() - started,
                               "error_message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}
                    summary = [r for r in summary if int(r["initial_state_id"]) != sid] + [row]
                    atomic_csv(summary_path, SUMMARY_FIELDS, sorted(summary, key=lambda r: int(r["initial_state_id"])))
                    if row["status"] == "error":
                        raise RuntimeError(row["error_message"])
            finally:
                release_policy(policy)
    finally:
        close_environment(env)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True); parser.add_argument("--source-run", required=True)
    parser.add_argument("--branch-steps", required=True); parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); build(args.config, args.run_dir, args.source_run,
                                      [int(x) for x in args.branch_steps.split(",") if x], args.force)


if __name__ == "__main__":
    main()
