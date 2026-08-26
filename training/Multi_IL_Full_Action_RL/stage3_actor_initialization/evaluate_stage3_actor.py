#!/usr/bin/env python3
"""Evaluate one Stage 3 actor checkpoint in the exact Stage 1 environment."""

import argparse
import copy
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
STAGE1_DIR = REPO_ROOT / "training" / "Multi_IL_Full_Action_RL" / "stage1_rollout_collection"
for path in (THIS_DIR, STAGE1_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from actor_network import load_actor_checkpoint, stochastic_action_and_log_prob  # noqa: E402
from common import atomic_json, extract_canonical_observation  # noqa: E402
from collect_multi_il_rollouts import progress_observation_schema  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def select_device(name):
    if name.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU requested but torch_npu cannot be imported") from exc
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but unavailable")
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


def decode_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_scalar(value.item())
    return value


def load_initial_states(path, requested_seeds):
    requested = set(requested_seeds)
    states = {}
    with h5py.File(path, "r") as handle:
        for group in handle["seeds"].values():
            seed = int(group.attrs["initial_seed"])
            if seed not in requested:
                continue
            state = {"states": np.asarray(group["states"])}
            if "model" in group:
                state["model"] = decode_scalar(group["model"][()])
            if "ep_meta" in group:
                state["ep_meta"] = decode_scalar(group["ep_meta"][()])
            states[seed] = state
    missing = sorted(requested - set(states))
    if missing:
        raise RuntimeError(f"Initial-state file is missing requested seeds: {missing}")
    return states


def env_success(env):
    success = env.is_success()
    if isinstance(success, dict):
        return bool(success.get("task", any(bool(value) for value in success.values())))
    return bool(success)


def close_env(env):
    target = getattr(env, "env", env)
    close = getattr(target, "close", None)
    if callable(close):
        close()


def extract_transport_progress(canonical_observation, progress_schema):
    """Read Stage 1-mapped Transport flags without assuming object-vector indices."""
    result = {}
    for name in ("trash_in_trash_bin", "payload_in_target_bin"):
        descriptor = progress_schema["fields"].get(name)
        if descriptor is None:
            raise RuntimeError(f"Stage 1 progress schema is missing {name!r}")
        key = descriptor["canonical_key"]
        index = int(descriptor["flat_index"])
        if key not in canonical_observation:
            raise RuntimeError(f"Progress source key {key!r} is absent from canonical observation")
        values = np.asarray(canonical_observation[key]).reshape(-1)
        if index < 0 or index >= values.size:
            raise RuntimeError(f"Progress index is out of range: {name} index={index}, width={values.size}")
        value = float(values[index])
        if value not in (0.0, 1.0):
            raise RuntimeError(f"Progress observable is not boolean-valued: {name}={value}")
        result[name] = bool(value)
    return result


def partial_progress_score(success, ever_trash, ever_payload):
    if success:
        return 3
    if ever_trash and ever_payload:
        return 2
    if ever_trash or ever_payload:
        return 1
    return 0


def build_report(args, checkpoint_path, device, source_dataset, initial_states_path,
                 horizon, terminate_on_success, progress_schema, results):
    completed = len(results)
    successes = sum(int(row["success"]) for row in results)

    def count(field):
        return sum(int(row[field]) for row in results)

    trash_ever = count("ever_trash_in_bin")
    trash_final = count("final_trash_in_bin")
    payload_ever = count("ever_payload_in_bin")
    payload_final = count("final_payload_in_bin")
    both_ever = sum(int(row["ever_trash_in_bin"] and row["ever_payload_in_bin"]) for row in results)
    both_final = sum(int(row["final_trash_in_bin"] and row["final_payload_in_bin"]) for row in results)

    def completion(count_value):
        return {
            "count": count_value,
            "rate": count_value / completed if completed else None,
        }

    failures = [row for row in results if not row["success"]]
    failure_progress = {
        "none": sum(int(not row["ever_trash_in_bin"] and not row["ever_payload_in_bin"]) for row in failures),
        "trash_only_ever": sum(int(row["ever_trash_in_bin"] and not row["ever_payload_in_bin"]) for row in failures),
        "payload_only_ever": sum(int(not row["ever_trash_in_bin"] and row["ever_payload_in_bin"]) for row in failures),
        "both_ever_but_failed": sum(int(row["ever_trash_in_bin"] and row["ever_payload_in_bin"]) for row in failures),
    }
    histogram = {
        str(score): sum(int(row["partial_progress_score"] == score) for row in results)
        for score in range(4)
    }
    return {
        "stage": "stage3_actor_initialization_candidate_evaluation",
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "deterministic": args.deterministic,
        "seed_start": args.seed_start,
        "num_seeds": args.num_seeds,
        "requested_episodes": args.num_seeds,
        "completed_episodes": completed,
        "completed_seeds": [int(row["initial_seed"]) for row in results],
        "evaluation_complete": completed == args.num_seeds,
        "successes": successes,
        "success_rate": successes / completed if completed else None,
        "mean_episode_return": float(np.mean([row["episode_return"] for row in results])) if completed else None,
        "mean_episode_length": float(np.mean([row["episode_length"] for row in results])) if completed else None,
        "subtask_completion": {
            "trash": {
                "ever_completed_count": trash_ever,
                "ever_completed_rate": completion(trash_ever)["rate"],
                "completed_at_end_count": trash_final,
                "completed_at_end_rate": completion(trash_final)["rate"],
            },
            "payload": {
                "ever_completed_count": payload_ever,
                "ever_completed_rate": completion(payload_ever)["rate"],
                "completed_at_end_count": payload_final,
                "completed_at_end_rate": completion(payload_final)["rate"],
            },
            "both": {
                "ever_completed_count": both_ever,
                "ever_completed_rate": completion(both_ever)["rate"],
                "completed_at_end_count": both_final,
                "completed_at_end_rate": completion(both_final)["rate"],
            },
        },
        "failure_progress": failure_progress,
        "progress_histogram": histogram,
        "mean_partial_progress_score": (
            float(np.mean([row["partial_progress_score"] for row in results])) if completed else None
        ),
        "horizon": horizon,
        "terminate_on_success": terminate_on_success,
        "source_dataset": str(source_dataset),
        "initial_states": str(initial_states_path),
        "progress_source": "Stage 1 progress_observation_schema derived from active robosuite object observables",
        "progress_observation_schema": progress_schema,
        "episodes": results,
    }


def print_final_summary(report, output):
    completed = report["completed_episodes"]
    requested = report["requested_episodes"]
    subtasks = report["subtask_completion"]
    histogram = report["progress_histogram"]
    failures = report["failure_progress"]
    print("=" * 80)
    print("Stage 3 Actor Evaluation Summary")
    print("=" * 80)
    print(f"Completed: {completed} / {requested}")
    print(f"Success: {report['successes']} / {completed} = {report['success_rate']:.4f}")
    print("Subtask completion:")
    print(f"Trash ever completed: {subtasks['trash']['ever_completed_count']} / {completed}")
    print(f"Trash completed at end: {subtasks['trash']['completed_at_end_count']} / {completed}")
    print(f"Payload ever completed: {subtasks['payload']['ever_completed_count']} / {completed}")
    print(f"Payload completed at end: {subtasks['payload']['completed_at_end_count']} / {completed}")
    print(f"Both ever completed: {subtasks['both']['ever_completed_count']} / {completed}")
    print(f"Both completed at end: {subtasks['both']['completed_at_end_count']} / {completed}")
    print("Partial progress:")
    for score in range(4):
        print(f"  score {score}: {histogram[str(score)]}")
    print("Failure progress:")
    print(f"  none: {failures['none']}")
    print(f"  trash only ever: {failures['trash_only_ever']}")
    print(f"  payload only ever: {failures['payload_only_ever']}")
    print(f"  both ever but failed: {failures['both_ever_but_failed']}")
    print(f"Mean progress score: {report['mean_partial_progress_score']:.6f}")
    print(f"Success rate: {report['success_rate']:.6f}")
    print(f"Evaluation JSON: {output}")
    print("=" * 80)


def main():
    args = parse_args()
    if args.num_seeds <= 0:
        raise ValueError("--num-seeds must be positive")
    device = select_device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    actor, payload = load_actor_checkpoint(checkpoint_path, device=device)
    actor.eval()
    observation_keys = list(payload["observation_keys"])
    observation_shapes = payload["observation_shapes"]
    source_dataset = Path(payload["source_dataset"])
    initial_states_path = source_dataset.parent.parent / "initial_states.hdf5"
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    initial_states = load_initial_states(initial_states_path, seeds)

    teacher_checkpoint = payload.get("teacher_checkpoint")
    if not teacher_checkpoint:
        raise RuntimeError("Actor checkpoint lacks teacher checkpoint metadata needed to recreate Stage 1 env")
    checkpoint_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=teacher_checkpoint)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint_dict)
    ObsUtils.initialize_obs_utils_with_config(config)
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=checkpoint_dict, render=False, render_offscreen=False, verbose=False)
    horizon = int(payload.get("evaluation_horizon", 700))
    terminate_on_success = bool(payload.get("terminate_on_success", True))
    progress_schema = progress_observation_schema(env, observation_keys, observation_shapes)
    output = Path(args.output) if args.output else checkpoint_path.parents[1] / f"evaluation_{checkpoint_path.stem}.json"
    results = []
    atomic_json(output, build_report(
        args, checkpoint_path, device, source_dataset, initial_states_path,
        horizon, terminate_on_success, progress_schema, results,
    ))
    print("=" * 80)
    print("Stage 3 actor environment evaluation")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device} | deterministic: {args.deterministic}")
    print(f"Seeds: {args.seed_start}..{args.seed_start + args.num_seeds - 1}")
    print(f"Initial states: {initial_states_path}")
    print("=" * 80)
    try:
        for position, seed in enumerate(seeds, start=1):
            seed_everything(seed)
            observation = env.reset_to(copy.deepcopy(initial_states[seed]))
            seed_everything(seed)
            episode_return = 0.0
            raw_done = False
            success = env_success(env)
            steps = 0
            ever_trash = False
            ever_payload = False
            final_trash = False
            final_payload = False
            for step in range(horizon):
                canonical = extract_canonical_observation(observation, observation_keys, observation_shapes)
                flat = np.concatenate([canonical[key].reshape(-1) for key in observation_keys]).astype(np.float32)
                state_tensor = torch.as_tensor(flat[None], device=device)
                with torch.no_grad():
                    if args.deterministic:
                        action = actor(state_tensor, deterministic=True, return_log_prob=False)[0]
                    else:
                        action = stochastic_action_and_log_prob(actor, state_tensor)[0]
                action_np = action[0].cpu().numpy()
                if not np.isfinite(action_np).all() or np.any(action_np < -1.0001) or np.any(action_np > 1.0001):
                    raise RuntimeError(f"Invalid actor action for seed {seed}: range [{action_np.min()}, {action_np.max()}]")
                observation, reward, raw_done, _ = env.step(action_np)
                episode_return += float(reward)
                steps = step + 1
                next_canonical = extract_canonical_observation(
                    observation, observation_keys, observation_shapes
                )
                progress = extract_transport_progress(next_canonical, progress_schema)
                final_trash = progress["trash_in_trash_bin"]
                final_payload = progress["payload_in_target_bin"]
                ever_trash = ever_trash or final_trash
                ever_payload = ever_payload or final_payload
                success = env_success(env)
                if raw_done or (terminate_on_success and success):
                    break
            truncated = bool(not raw_done and (success or steps >= horizon))
            score = partial_progress_score(success, ever_trash, ever_payload)
            row = {
                "initial_seed": seed,
                "success": int(success),
                "episode_return": episode_return,
                "episode_length": steps,
                "terminated": bool(raw_done),
                "truncated": truncated,
                "subtasks": {
                    "trash": {
                        "ever_completed": bool(ever_trash),
                        "completed_at_end": bool(final_trash),
                    },
                    "payload": {
                        "ever_completed": bool(ever_payload),
                        "completed_at_end": bool(final_payload),
                    },
                },
                "ever_trash_in_bin": bool(ever_trash),
                "ever_payload_in_bin": bool(ever_payload),
                "final_trash_in_bin": bool(final_trash),
                "final_payload_in_bin": bool(final_payload),
                "partial_progress_score": score,
            }
            results.append(row)
            current_report = build_report(
                args, checkpoint_path, device, source_dataset, initial_states_path,
                horizon, terminate_on_success, progress_schema, results,
            )
            atomic_json(output, current_report)
            print(
                f"[{position:03d}/{len(seeds):03d}] seed={seed} success={int(success)} "
                f"progress={score} trash_ever={int(ever_trash)} payload_ever={int(ever_payload)} "
                f"trash_final={int(final_trash)} payload_final={int(final_payload)} "
                f"return={episode_return:.3f} length={steps}"
            )
    finally:
        close_env(env)

    report = build_report(
        args, checkpoint_path, device, source_dataset, initial_states_path,
        horizon, terminate_on_success, progress_schema, results,
    )
    atomic_json(output, report)
    print_final_summary(report, output)


if __name__ == "__main__":
    main()
