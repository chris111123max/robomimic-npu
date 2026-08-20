#!/usr/bin/env python3
"""Collect same-initial-state rollouts from heterogeneous robomimic IL policies."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import socket
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # allow --help and static inspection outside the rollout environment
    h5py = None

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (SCHEMA_VERSION, VALID_POLICY_IDS, atomic_json, canonical_json,
                    compare_observations, configured_low_dim_keys, discounted_returns,
                    extract_canonical_observation, git_commit, infer_algo_type,
                    observation_hash, read_json, seed_everything, sha256_array,
                    simulator_state_hash, try_seed_environment)


def utc_now():
    return datetime.now(timezone.utc).astimezone().isoformat()


def select_device(name):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        return torch.device("npu:0")
    return torch.device("cpu")


def close_environment(env):
    raw = getattr(env, "unwrapped", env)
    suite_env = getattr(raw, "env", None)
    close = getattr(suite_env, "close", None)
    if callable(close):
        close()


def release_policy(policy):
    del policy
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available() and hasattr(npu, "empty_cache"):
            npu.empty_cache()
    except ImportError:
        pass


def load_seed_list(args):
    if args.seed_list:
        payload = read_json(args.seed_list)
        seeds = payload["seeds"] if isinstance(payload, dict) else payload
    else:
        seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.num_episodes)))
    seeds = [int(item) for item in seeds]
    if args.num_episodes is not None and args.seed_list:
        seeds = seeds[:int(args.num_episodes)]
    if not seeds:
        raise ValueError("Seed list is empty")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed list contains duplicates")
    return seeds


def checkpoint_details(spec, checkpoint, config):
    path = Path(spec["checkpoint"])
    shape_meta = checkpoint["shape_metadata"]
    return {
        "policy_id": spec["policy_id"],
        "checkpoint": str(path),
        "checkpoint_name": path.name,
        "checkpoint_size_bytes": path.stat().st_size,
        "checkpoint_mtime": path.stat().st_mtime,
        "algo_name": checkpoint["algo_name"],
        "algo_type": infer_algo_type(config),
        "observation_keys": configured_low_dim_keys(config),
        "shape_metadata": {
            "ac_dim": int(shape_meta["ac_dim"]),
            "all_shapes": {key: list(value) for key, value in shape_meta["all_shapes"].items()},
        },
        "frame_stack": int(config.train.frame_stack),
        "observation_normalization": checkpoint.get("obs_normalization_stats") is not None,
        "action_normalization": checkpoint.get("action_normalization_stats") is not None,
        "known_success_rate": spec.get("known_success_rate"),
    }


def inspect_checkpoints(policy_specs):
    import robomimic.utils.file_utils as FileUtils
    inspected = []
    reference_env_meta = None
    reference_keys = None
    reference_ac_dim = None
    for spec in policy_specs:
        path = Path(spec["checkpoint"])
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(path))
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint)
        details = checkpoint_details(spec, checkpoint, config)
        env_meta = checkpoint["env_metadata"]
        if reference_env_meta is None:
            reference_env_meta = env_meta
            reference_keys = details["observation_keys"]
            reference_ac_dim = details["shape_metadata"]["ac_dim"]
        else:
            if canonical_json(env_meta) != canonical_json(reference_env_meta):
                raise RuntimeError(f"{spec['policy_id']} checkpoint environment metadata differs")
            if details["observation_keys"] != reference_keys:
                raise RuntimeError(f"{spec['policy_id']} checkpoint observation keys differ")
            if details["shape_metadata"]["ac_dim"] != reference_ac_dim:
                raise RuntimeError(f"{spec['policy_id']} checkpoint action dimension differs")
        inspected.append((spec, checkpoint, config, details))
    return inspected, reference_env_meta, reference_keys, reference_ac_dim


def write_text_dataset(group, name, value):
    group.create_dataset(name, data=str(value), dtype=h5py.string_dtype("utf-8"))


def create_initial_states(inspected, seeds, output_path):
    """Reset once per seed, then persist exact simulator states shared by all policies."""
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    _, checkpoint, config, _ = inspected[0]
    ObsUtils.initialize_obs_utils_with_config(config)
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=checkpoint, render=False,
                                           render_offscreen=False, verbose=False)
    canonical_keys = configured_low_dim_keys(config)
    summaries, shapes = [], None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(output_path, "w") as handle:
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["canonical_observation_keys"] = json.dumps(canonical_keys)
            seeds_group = handle.create_group("seeds")
            for episode_index, seed in enumerate(seeds):
                seed_everything(seed)
                seeded_apis = try_seed_environment(env, seed)
                env.reset()
                state = copy.deepcopy(env.get_state())
                policy_observation = env.reset_to(copy.deepcopy(state))
                canonical_observation = extract_canonical_observation(policy_observation, canonical_keys)
                current_shapes = {key: list(value.shape) for key, value in canonical_observation.items()}
                if shapes is None:
                    shapes = current_shapes
                elif current_shapes != shapes:
                    raise RuntimeError(f"Canonical observation schema changed at seed {seed}")
                restored = env.get_state()
                if not np.array_equal(np.asarray(state["states"]), np.asarray(restored["states"])):
                    raise RuntimeError(f"Reference reset_to state mismatch at seed {seed}")
                state = restored
                group = seeds_group.create_group(f"episode_{episode_index:06d}")
                group.attrs["episode_index"] = episode_index
                group.attrs["initial_seed"] = seed
                group.attrs["state_hash"] = simulator_state_hash(state)
                group.attrs["state_vector_hash"] = sha256_array(state["states"])
                group.attrs["observation_hash"] = observation_hash(canonical_observation)
                group.attrs["environment_seed_apis"] = json.dumps(seeded_apis)
                group.create_dataset("states", data=np.asarray(state["states"]))
                write_text_dataset(group, "model", state.get("model", ""))
                write_text_dataset(group, "ep_meta", state.get("ep_meta", ""))
                obs_group = group.create_group("observation")
                for key, value in canonical_observation.items():
                    obs_group.create_dataset(key, data=value)
                summaries.append({
                    "episode_index": episode_index,
                    "initial_seed": seed,
                    "state_hash": group.attrs["state_hash"],
                    "state_vector_hash": group.attrs["state_vector_hash"],
                    "observation_hash": group.attrs["observation_hash"],
                    "environment_seed_apis": seeded_apis,
                })
            handle.attrs["canonical_observation_shapes"] = json.dumps(shapes, sort_keys=True)
    finally:
        close_environment(env)
    return summaries, shapes


def _decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_scalar(value.item())
    return str(value)


def load_initial_state(handle, episode_index):
    group = handle[f"seeds/episode_{episode_index:06d}"]
    state = {
        "states": np.asarray(group["states"]).copy(),
        "model": _decode_scalar(group["model"][()]),
    }
    ep_meta = _decode_scalar(group["ep_meta"][()])
    if ep_meta:
        state["ep_meta"] = ep_meta
    observation = {key: np.asarray(value).copy() for key, value in group["observation"].items()}
    metadata = {key: group.attrs[key] for key in (
        "initial_seed", "state_hash", "state_vector_hash", "observation_hash"
    )}
    return state, observation, metadata


def rollout_episode(policy, env, initial_state, initial_observation, canonical_keys,
                    canonical_shapes, action_dim, seed, horizon, terminate_on_success,
                    observation_atol):
    seed_everything(seed)
    policy.start_episode()
    policy_observation = env.reset_to(copy.deepcopy(initial_state))
    restored = env.get_state()
    if not np.array_equal(np.asarray(initial_state["states"]), np.asarray(restored["states"])):
        raise RuntimeError(f"reset_to state-vector mismatch for seed {seed}")
    canonical_observation = extract_canonical_observation(
        policy_observation, canonical_keys, canonical_shapes)
    compare_observations(initial_observation, canonical_observation, observation_atol)
    # reset_to can consume framework RNG through environment setup, and wrapper
    # internals differ between feed-forward, RNN, and Transformer checkpoints.
    # Re-seed immediately before the first policy call so policy stochasticity
    # is reproducible and independent of reset implementation details.
    seed_everything(seed)

    trajectory = {
        "obs": {key: [] for key in canonical_keys},
        "next_obs": {key: [] for key in canonical_keys},
        "actions": [], "rewards": [], "dones": [], "terminated": [], "truncated": [],
    }
    success = False
    termination_reason = "horizon"
    try:
        for timestep in range(int(horizon)):
            action = np.asarray(policy(ob=policy_observation)).copy()
            if action.shape != (int(action_dim),):
                raise RuntimeError(f"Action shape {action.shape} != ({action_dim},)")
            if not np.all(np.isfinite(action)):
                raise RuntimeError("Policy emitted NaN or Inf action")
            next_policy_observation, reward, env_done, _ = env.step(action)
            next_canonical = extract_canonical_observation(
                next_policy_observation, canonical_keys, canonical_shapes)
            success = success or bool(env.is_success()["task"])
            stop_for_success = bool(terminate_on_success and success)
            horizon_reached = timestep + 1 >= int(horizon)
            terminated = bool(env_done)
            truncated = bool((stop_for_success or horizon_reached) and not terminated)
            done = bool(terminated or truncated)
            for key in canonical_keys:
                trajectory["obs"][key].append(canonical_observation[key].copy())
                trajectory["next_obs"][key].append(next_canonical[key].copy())
            trajectory["actions"].append(action)
            trajectory["rewards"].append(float(reward))
            trajectory["dones"].append(done)
            trajectory["terminated"].append(terminated)
            trajectory["truncated"].append(truncated)
            policy_observation = next_policy_observation
            canonical_observation = next_canonical
            if terminated:
                termination_reason = "environment_done"
                break
            if stop_for_success:
                termination_reason = "success_collector_stop"
                break
            if horizon_reached:
                termination_reason = "horizon"
                break
    except env.rollout_exceptions as exception:
        raise RuntimeError(
            f"Environment rollout exception for seed {seed}: "
            f"{type(exception).__name__}: {exception}"
        ) from exception

    for section in ("obs", "next_obs"):
        trajectory[section] = {key: np.asarray(values) for key, values in trajectory[section].items()}
    trajectory["actions"] = np.asarray(trajectory["actions"])
    trajectory["rewards"] = np.asarray(trajectory["rewards"], dtype=np.float64)
    for key in ("dones", "terminated", "truncated"):
        trajectory[key] = np.asarray(trajectory[key], dtype=np.bool_)
    return trajectory, {
        "success": bool(success),
        "episode_return": float(np.sum(trajectory["rewards"])),
        "episode_length": int(len(trajectory["actions"])),
        "termination_reason": termination_reason,
        "rollout_exception": None,
    }


def write_episode(data_group, episode_index, policy_id, seed, trajectory, summary,
                  initial_state, initial_metadata, mc_gamma):
    group = data_group.create_group(f"episode_{episode_index:06d}")
    length = int(summary["episode_length"])
    for section in ("obs", "next_obs"):
        sub = group.create_group(section)
        for key, value in trajectory[section].items():
            sub.create_dataset(key, data=value, compression="gzip", compression_opts=1)
    for key in ("actions", "rewards", "dones", "terminated", "truncated"):
        group.create_dataset(key, data=trajectory[key])
    string_dtype = h5py.string_dtype("utf-8")
    group.create_dataset("policy_id", data=np.asarray([policy_id] * length, dtype=object), dtype=string_dtype)
    group.create_dataset("episode_id", data=np.full(length, episode_index, dtype=np.int64))
    group.create_dataset("initial_seed", data=np.full(length, seed, dtype=np.int64))
    group.create_dataset("timestep", data=np.arange(length, dtype=np.int64))
    group.create_dataset("episode_success", data=np.full(length, summary["success"], dtype=np.bool_))
    group.create_dataset("episode_return", data=np.full(length, summary["episode_return"], dtype=np.float64))
    group.create_dataset("episode_length", data=np.full(length, length, dtype=np.int64))
    if mc_gamma is not None:
        group.create_dataset("mc_return", data=discounted_returns(trajectory["rewards"], mc_gamma))
    initial = group.create_group("initial_state")
    initial.create_dataset("states", data=np.asarray(initial_state["states"]))
    write_text_dataset(initial, "model", initial_state.get("model", ""))
    write_text_dataset(initial, "ep_meta", initial_state.get("ep_meta", ""))
    group.attrs["num_samples"] = length
    group.attrs["policy_id"] = policy_id
    group.attrs["episode_id"] = episode_index
    group.attrs["initial_seed"] = seed
    group.attrs["success"] = summary["success"]
    group.attrs["episode_return"] = summary["episode_return"]
    group.attrs["episode_length"] = length
    group.attrs["termination_reason"] = summary["termination_reason"]
    group.attrs["initial_state_hash"] = initial_metadata["state_hash"]
    group.attrs["initial_state_vector_hash"] = initial_metadata["state_vector_hash"]
    group.attrs["initial_observation_hash"] = initial_metadata["observation_hash"]


def collect_policy(item, initial_states_path, seeds, data_root, run_dir, device,
                   canonical_keys, canonical_shapes, horizon, terminate_on_success,
                   observation_atol, mc_gamma):
    import robomimic.utils.file_utils as FileUtils
    spec, inspected_checkpoint, _, details = item
    policy_id = spec["policy_id"]
    policy_dir = data_root / policy_id
    policy_dir.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    policy, env = None, None
    episodes, total_transitions = [], 0
    dataset_path = policy_dir / "transitions.hdf5"
    try:
        policy, checkpoint = FileUtils.policy_from_checkpoint(
            ckpt_dict=inspected_checkpoint, device=device, verbose=False)
        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=checkpoint, render=False, render_offscreen=False, verbose=False)
        action_dim = int(env.action_dimension)
        if action_dim != int(details["shape_metadata"]["ac_dim"]):
            raise RuntimeError(f"{policy_id}: checkpoint and environment action dimensions differ")
        with h5py.File(initial_states_path, "r") as initial_handle, h5py.File(dataset_path, "w") as output:
            output.attrs["schema_version"] = SCHEMA_VERSION
            output.attrs["policy_id"] = policy_id
            output.attrs["checkpoint"] = spec["checkpoint"]
            output.attrs["environment_metadata"] = json.dumps(checkpoint["env_metadata"], sort_keys=True)
            output.attrs["canonical_observation_keys"] = json.dumps(canonical_keys)
            output.attrs["canonical_observation_shapes"] = json.dumps(canonical_shapes, sort_keys=True)
            output.attrs["action_shape"] = json.dumps([action_dim])
            output.attrs["observation_representation"] = "raw_canonical_low_dim_current_frame"
            output.attrs["policy_input_representation"] = (
                "official_checkpoint_wrapper; may include frame-stack and normalization"
            )
            output.attrs["success_and_failure_saved"] = True
            data_group = output.create_group("episodes")
            for episode_index, seed in enumerate(seeds):
                initial_state, initial_observation, initial_metadata = load_initial_state(
                    initial_handle, episode_index)
                trajectory, summary = rollout_episode(
                    policy, env, initial_state, initial_observation, canonical_keys,
                    canonical_shapes, action_dim, seed, horizon, terminate_on_success,
                    observation_atol)
                summary.update({
                    "episode_id": episode_index,
                    "policy_id": policy_id,
                    "initial_seed": seed,
                    "policy_seed": seed,
                    "checkpoint": spec["checkpoint"],
                })
                write_episode(data_group, episode_index, policy_id, seed, trajectory,
                              summary, initial_state, initial_metadata, mc_gamma)
                total_transitions += summary["episode_length"]
                episodes.append(summary)
                print(
                    f"Episode | policy={policy_id} seed={seed} "
                    f"length={summary['episode_length']} return={summary['episode_return']:.6f} "
                    f"success={int(summary['success'])} reason={summary['termination_reason']}",
                    flush=True,
                )
            output.attrs["num_episodes"] = len(episodes)
            output.attrs["total_transitions"] = total_transitions
            output.attrs["mc_return_gamma"] = "none" if mc_gamma is None else float(mc_gamma)
    finally:
        if env is not None:
            close_environment(env)
        if policy is not None:
            release_policy(policy)
    success_count = sum(int(row["success"]) for row in episodes)
    metadata = {
        **details,
        "schema_version": SCHEMA_VERSION,
        "device": str(device),
        "canonical_observation_keys": canonical_keys,
        "canonical_observation_shapes": canonical_shapes,
        "action_shape": [action_dim],
        "horizon": horizon,
        "terminate_on_success": terminate_on_success,
        "done_definition": "terminated OR truncated",
        "terminated_definition": "raw environment done",
        "truncated_definition": "collector success-stop or horizon without raw environment done",
        "mc_return_gamma": mc_gamma,
        "start_time": started,
        "end_time": utc_now(),
        "num_episodes": len(episodes),
        "success_episodes": success_count,
        "failure_episodes": len(episodes) - success_count,
        "success_rate": success_count / len(episodes),
        "total_transitions": total_transitions,
        "mean_episode_length": float(np.mean([row["episode_length"] for row in episodes])),
        "mean_episode_return": float(np.mean([row["episode_return"] for row in episodes])),
        "success_filter_applied": False,
    }
    atomic_json(policy_dir / "episodes.json", episodes)
    atomic_json(policy_dir / "metadata.json", metadata)
    atomic_json(run_dir / f"{policy_id}_metadata.json", metadata)
    return episodes, metadata


def same_seed_summary(policy_episodes, seeds):
    lookup = {
        policy_id: {int(row["initial_seed"]): bool(row["success"]) for row in rows}
        for policy_id, rows in policy_episodes.items()
    }
    patterns = Counter()
    rows = []
    for seed in seeds:
        values = [int(lookup[policy_id][seed]) for policy_id in VALID_POLICY_IDS]
        pattern = "".join(map(str, values))
        patterns[pattern] += 1
        rows.append({
            "initial_seed": seed,
            "bc_gmm_success": values[0],
            "bc_rnn_success": values[1],
            "bc_transformer_success": values[2],
            "success_pattern": pattern,
        })
    return rows, {format(index, "03b"): patterns.get(format(index, "03b"), 0) for index in range(8)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "configs/stage1_three_policies.json"))
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--seed-list")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root")
    parser.add_argument("--run-root")
    parser.add_argument("--mc-gamma", type=float)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if h5py is None:
        raise RuntimeError("h5py is required for Stage 1 collection; activate robosuite_npu")
    config = read_json(args.config)
    policy_specs = config["policies"]
    ids = [row["policy_id"] for row in policy_specs]
    if tuple(ids) != VALID_POLICY_IDS:
        raise RuntimeError(f"Policy order must be {VALID_POLICY_IDS}, got {ids}")
    if args.num_episodes is None and not args.seed_list:
        args.num_episodes = int(config["collection"]["default_num_episodes"])
    seeds = load_seed_list(args)
    horizon = int(args.horizon or config["collection"]["horizon"])
    terminate_on_success = bool(config["collection"]["terminate_on_success"])
    observation_atol = float(config["collection"].get("initial_observation_atol", 1e-6))
    mc_gamma = args.mc_gamma
    if mc_gamma is None:
        mc_gamma = config["collection"].get("mc_return_gamma")
    if mc_gamma is not None and not 0.0 <= float(mc_gamma) <= 1.0:
        raise ValueError("MC-return gamma must be in [0, 1]")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    data_base = Path(args.output_root or config["paths"]["dataset_root"])
    run_base = Path(args.run_root or config["paths"]["run_root"])
    data_root, run_dir = data_base / run_id, run_base / run_id
    if data_root.exists() or run_dir.exists():
        raise FileExistsError(f"Run already exists: data={data_root}, metadata={run_dir}")
    data_root.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    started = utc_now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "stage": "stage1_rollout_collection",
        "smoke_test": args.smoke_test,
        "run_id": run_id,
        "start_time": started,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "command": shlex.join(sys.argv),
        "config_path": str(Path(args.config).resolve()),
        "git_commit": git_commit(REPO_ROOT),
        "dataset_directory": str(data_root),
        "run_directory": str(run_dir),
        "seed_list": seeds,
        "horizon": horizon,
    }
    atomic_json(run_dir / "run_manifest.json", manifest)
    atomic_json(data_root / "seed_list.json", {"seeds": seeds})
    atomic_json(run_dir / "seed_list.json", {"seeds": seeds})
    try:
        inspected, env_meta, canonical_keys, action_dim = inspect_checkpoints(policy_specs)
        device = select_device(args.device)
        print("=" * 88)
        print("Multi-IL + Full-Action RL | Stage 1 rollout collection")
        print("Run ID       :", run_id)
        print("Device       :", device)
        print("Episodes     :", len(seeds), "per policy")
        print("Horizon      :", horizon)
        print("Dataset root :", data_root)
        print("Run metadata :", run_dir)
        for _, _, _, details in inspected:
            print(
                f"Policy        : {details['policy_id']} | {details['algo_type']} | "
                f"action_dim={details['shape_metadata']['ac_dim']} | "
                f"obs={details['observation_keys']}"
            )
            print("Checkpoint    :", details["checkpoint"])
        print("=" * 88, flush=True)
        initial_states_path = data_root / "initial_states.hdf5"
        initial_summaries, canonical_shapes = create_initial_states(
            inspected, seeds, initial_states_path)
        atomic_json(run_dir / "initial_state_manifest.json", {
            "canonical_observation_keys": canonical_keys,
            "canonical_observation_shapes": canonical_shapes,
            "states": initial_summaries,
            "guarantee": "All policies reset_to the same persisted simulator state vector and model XML.",
        })
        policy_episodes, policy_metadata = {}, {}
        for item in inspected:
            episodes, metadata = collect_policy(
                item, initial_states_path, seeds, data_root, run_dir, device,
                canonical_keys, canonical_shapes, horizon,
                terminate_on_success, observation_atol, mc_gamma)
            policy_id = item[0]["policy_id"]
            policy_episodes[policy_id] = episodes
            policy_metadata[policy_id] = metadata
        outcome_rows, patterns = same_seed_summary(policy_episodes, seeds)
        atomic_json(data_root / "same_seed_outcomes.json", outcome_rows)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "start_time": started,
            "end_time": utc_now(),
            "num_seeds": len(seeds),
            "num_policies": len(policy_specs),
            "total_episodes": len(seeds) * len(policy_specs),
            "total_transitions": sum(row["total_transitions"] for row in policy_metadata.values()),
            "environment_metadata": env_meta,
            "canonical_observation_keys": canonical_keys,
            "canonical_observation_shapes": canonical_shapes,
            "action_shape": [action_dim],
            "same_seed_success_patterns": patterns,
            "policies": policy_metadata,
            "runtime_errors": 0,
        }
        atomic_json(data_root / "collection_summary.json", summary)
        atomic_json(run_dir / "collection_summary.json", summary)
        manifest.update({"status": "complete", "end_time": summary["end_time"]})
        atomic_json(run_dir / "run_manifest.json", manifest)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        print("COLLECTION COMPLETE", flush=True)
    except BaseException as exception:
        manifest.update({
            "status": "failed", "end_time": utc_now(),
            "error": f"{type(exception).__name__}: {exception}",
        })
        atomic_json(run_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
