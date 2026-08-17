#!/usr/bin/env python3
"""Evaluate each native IL policy on every persisted initial condition."""

import argparse
import copy
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from utils.env_utils import (
    close_environment, create_dataset_environment,
    deterministic_environment_stream_seed, restore_and_verify,
    seed_environment, task_succeeded,
)
from utils.policy_loader import load_policy, release_policy, select_device, set_policy_sampling_seed
from utils.result_utils import (
    POLICY_ORDER, atomic_json, atomic_npz, collect_episode_records, load_state_bank_file, read_json,
    rebuild_result_tables, validate_trajectory,
)


def save_trajectory(path, observations, actions, rewards, dones, successes, sim_states, compressed):
    observation_names = sorted(observations[0]) if observations else []
    arrays = {
        "observation_names": np.asarray(observation_names),
        "actions": np.asarray(actions),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "environment_dones": np.asarray(dones, dtype=np.bool_),
        "task_success": np.asarray(successes, dtype=np.bool_),
        "step_indices": np.arange(len(actions), dtype=np.int64),
    }
    for key in observation_names:
        arrays["obs__" + key] = np.stack([np.asarray(obs[key]) for obs in observations], axis=0)
    if sim_states is not None:
        arrays["simulator_states"] = np.stack(sim_states, axis=0)
    atomic_npz(path, compressed=compressed, **arrays)


def completed_and_valid(result_path, trajectory_path, entry):
    if not result_path.exists() or not trajectory_path.exists():
        return False
    try:
        result = read_json(result_path)
    except Exception:
        return False
    return (
        result.get("status") == "complete"
        and result.get("state_hash") == entry["state_hash"]
        and validate_trajectory(trajectory_path, result.get("episode_length"))
    )


def rollout(policy, env, initial_observation, horizon, terminate_on_success, save_sim_states):
    policy.start_episode()
    observation = initial_observation
    observations, actions, rewards, dones, successes = [], [], [], [], []
    simulator_states = [] if save_sim_states else None
    total_return = 0.0
    success_step = None
    termination_reason = "horizon"
    for step in range(horizon):
        if simulator_states is not None:
            simulator_states.append(np.asarray(env.get_state()["states"]).copy())
        observations.append({key: np.asarray(value).copy() for key, value in observation.items()})
        action = np.asarray(policy(ob=observation)).copy()
        next_observation, reward, environment_done, _ = env.step(action)
        success = task_succeeded(env)
        actions.append(action)
        rewards.append(float(reward))
        dones.append(bool(environment_done))
        successes.append(success)
        total_return += float(reward)
        if success and success_step is None:
            success_step = step + 1
        if success and terminate_on_success:
            termination_reason = "success"
            break
        if environment_done:
            termination_reason = "environment_done"
            break
        observation = copy.deepcopy(next_observation)
    return {
        "success": success_step is not None,
        "episode_return": total_return,
        "episode_length": len(actions),
        "success_step": success_step,
        "termination_reason": termination_reason,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "successes": successes,
        "simulator_states": simulator_states,
    }


def evaluate(config_path, run_dir, force_eval=False, policy_names=None, defer_aggregate=False):
    config = read_json(config_path)
    run_dir = Path(run_dir)
    seed_manifest = read_json(run_dir / "seed_manifest.json")
    state_manifest = read_json(run_dir / "initial_state_manifest.json")
    entries = sorted(state_manifest["states"], key=lambda row: int(row["initial_state_id"]))
    if len(entries) != len(seed_manifest["environment_seeds"]):
        raise RuntimeError("Initial-state manifest and seed manifest have different lengths")
    horizon = int(config["horizon"])
    trajectory_cfg = config["trajectory"]
    policy_names = list(POLICY_ORDER if policy_names is None else policy_names)
    invalid = [name for name in policy_names if name not in POLICY_ORDER]
    if invalid or not policy_names or len(set(policy_names)) != len(policy_names):
        raise ValueError(f"Invalid or duplicate policy selection: {policy_names}")
    device = select_device()
    visible_device = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "<unset>")
    print(f"Evaluation device: {device} (ASCEND_RT_VISIBLE_DEVICES={visible_device})", flush=True)
    print(f"Selected policies: {policy_names}", flush=True)

    env, _ = create_dataset_environment(config["dataset_path"])
    try:
        for policy_name in policy_names:
            checkpoint_path = config["policies"][policy_name]["checkpoint_path"]
            print(f"Loading policy {policy_name}: {checkpoint_path}", flush=True)
            policy, _, _ = load_policy(policy_name, checkpoint_path, device=device)
            try:
                for ordinal, entry in enumerate(entries, start=1):
                    state_id = int(entry["initial_state_id"])
                    trajectory_path = run_dir / "trajectories" / policy_name / f"state_{state_id:06d}.npz"
                    result_path = run_dir / "raw_results" / "episodes" / policy_name / f"state_{state_id:06d}.json"
                    if not force_eval and completed_and_valid(result_path, trajectory_path, entry):
                        print(f"[eval {policy_name} {ordinal:03d}/{len(entries):03d}] valid, skipping", flush=True)
                        continue
                    started = time.monotonic()
                    base = {
                        "initial_state_id": state_id,
                        "environment_seed": int(entry["environment_seed"]),
                        "policy_name": policy_name,
                        "checkpoint_path": checkpoint_path,
                        "state_hash": entry["state_hash"],
                        "worker_visible_device": visible_device,
                    }
                    try:
                        # Seed before reset_to: robosuite's XML restoration performs an
                        # internal reset, so every policy must enter that reset with the
                        # same per-initial-condition environment RNG stream.
                        policy_seed = set_policy_sampling_seed(
                            seed_manifest["meta_seed"], policy_name, state_id,
                        )
                        seed_environment(
                            env,
                            deterministic_environment_stream_seed(
                                seed_manifest["meta_seed"], state_id,
                            ),
                        )
                        state_path = run_dir / entry["state_file"]
                        state, saved_observation = load_state_bank_file(state_path)
                        initial_observation = restore_and_verify(env, state, saved_observation, entry)
                        stats = rollout(
                            policy, env, initial_observation, horizon,
                            bool(config["terminate_on_success"]),
                            bool(trajectory_cfg.get("save_sim_states", False)),
                        )
                        save_trajectory(
                            trajectory_path, stats["observations"], stats["actions"], stats["rewards"],
                            stats["dones"], stats["successes"], stats["simulator_states"],
                            bool(trajectory_cfg.get("compressed", True)),
                        )
                        result = dict(base)
                        result.update({
                            "status": "complete",
                            "success": bool(stats["success"]),
                            "episode_return": float(stats["episode_return"]),
                            "episode_length": int(stats["episode_length"]),
                            "success_step": stats["success_step"],
                            "termination_reason": stats["termination_reason"],
                            "observation_hash": entry["observation_hash"],
                            "policy_sampling_seed": int(policy_seed),
                            "trajectory_path": str(trajectory_path.relative_to(run_dir)),
                            "wall_time_seconds": time.monotonic() - started,
                        })
                        atomic_json(result_path, result)
                        print(
                            f"[eval {policy_name} {ordinal:03d}/{len(entries):03d}] "
                            f"success={int(result['success'])} length={result['episode_length']}", flush=True,
                        )
                    except Exception as exc:
                        error = dict(base)
                        error.update({
                            "status": "error", "termination_reason": "error",
                            "error_type": type(exc).__name__, "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                            "wall_time_seconds": time.monotonic() - started,
                        })
                        atomic_json(result_path, error)
                        print(f"[eval {policy_name} {ordinal:03d}/{len(entries):03d}] ERROR: {exc}", flush=True)
                    finally:
                        if not defer_aggregate:
                            rebuild_result_tables(run_dir)
            finally:
                release_policy(policy)
    finally:
        close_environment(env)
    if defer_aggregate:
        records = [row for row in collect_episode_records(run_dir) if row.get("policy_name") in policy_names]
        completed = [row for row in records if row.get("status") == "complete"]
        errors = [row for row in records if row.get("status") == "error"]
    else:
        completed, errors = rebuild_result_tables(run_dir)
        completed = [row for row in completed if row.get("policy_name") in policy_names]
        errors = [row for row in errors if row.get("policy_name") in policy_names]
    expected = len(entries) * len(policy_names)
    print(f"Worker records: complete={len(completed)}/{expected}, errors={len(errors)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--policy", action="append", choices=POLICY_ORDER, dest="policies")
    parser.add_argument("--defer-aggregate", action="store_true")
    args = parser.parse_args()
    evaluate(
        args.config, args.run_dir, args.force_eval,
        policy_names=args.policies, defer_aggregate=args.defer_aggregate,
    )


if __name__ == "__main__":
    main()
