#!/usr/bin/env python3
"""Evaluate one Stage 3 actor checkpoint in the exact Stage 1 environment."""

import argparse
import copy
import json
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
from common import extract_canonical_observation  # noqa: E402
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


def main():
    args = parse_args()
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
    results = []
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
                success = env_success(env)
                if raw_done or (terminate_on_success and success):
                    break
            truncated = bool(not raw_done and (success or steps >= horizon))
            row = {"initial_seed": seed, "success": int(success), "episode_return": episode_return,
                   "episode_length": steps, "terminated": bool(raw_done), "truncated": truncated}
            results.append(row)
            print(f"[{position:03d}/{len(seeds):03d}] seed={seed} success={int(success)} return={episode_return:.3f} length={steps}")
    finally:
        close_env(env)

    successes = sum(row["success"] for row in results)
    report = {
        "stage": "stage3_actor_initialization_candidate_evaluation", "checkpoint": str(checkpoint_path),
        "device": str(device), "deterministic": args.deterministic, "seed_start": args.seed_start,
        "num_seeds": args.num_seeds, "successes": successes,
        "success_rate": successes / len(results) if results else None,
        "mean_episode_return": float(np.mean([row["episode_return"] for row in results])) if results else None,
        "mean_episode_length": float(np.mean([row["episode_length"] for row in results])) if results else None,
        "horizon": horizon, "terminate_on_success": terminate_on_success,
        "source_dataset": str(source_dataset), "initial_states": str(initial_states_path), "episodes": results,
    }
    output = Path(args.output) if args.output else checkpoint_path.parents[1] / f"evaluation_{checkpoint_path.stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    print("=" * 80)
    print(f"Success: {successes}/{len(results)} = {report['success_rate']:.4f}")
    print(f"Report: {output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
