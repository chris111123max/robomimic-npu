#!/usr/bin/env python3
"""Environment-only startup, step, individual reset/rebuild, and shutdown smoke."""
from __future__ import annotations

import argparse
import json

import numpy as np

from stage3_progressive_vector_env import StaggeredVectorEnv


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--expert-dataset",
        help=(
            "Override config expert_dataset. Required when using the raw "
            "progressive config whose expert_dataset is null."
        ),
    )
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--vector-steps", type=int, default=10)
    return parser.parse_args()


def main():
    args = arguments()
    with open(args.config, encoding="utf-8") as handle:
        cfg = json.load(handle)

    dataset = args.expert_dataset or cfg.get("expert_dataset")
    if not dataset:
        raise ValueError(
            "No expert_dataset resolved. Pass --expert-dataset or use "
            "shared/config_resolved.json."
        )

    parallel = cfg["parallel_env"]
    num_envs = int(
        args.num_envs
        if args.num_envs is not None
        else parallel["num_envs"]
    )
    if num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.vector_steps <= 0:
        raise ValueError("--vector-steps must be positive")

    vector = None
    valid_transitions = 0
    fatal_transitions = 0
    reset_seed = int(cfg["train_seed_base"]) + num_envs

    try:
        vector = StaggeredVectorEnv(
            dataset,
            num_envs,
            cfg["train_seed_base"],
            delay=parallel["env_startup_delay_sec"],
            timeout=parallel["env_startup_timeout_sec"],
            start_method=parallel["multiprocessing_start_method"],
            command_timeout=parallel.get(
                "env_command_timeout_sec",
                parallel["env_startup_timeout_sec"],
            ),
        )
        assert vector.alive_worker_count() == num_envs

        zero_action = np.zeros(
            vector.action_low.shape,
            dtype=np.float32,
        )
        actions = np.stack([zero_action] * num_envs)

        for step in range(args.vector_steps):
            results = vector.step(actions)
            for env_id, message in results:
                if message[0] == "OK":
                    valid_transitions += 1
                elif message[0] == "FATAL":
                    fatal_transitions += 1
                    # Exercise env-local recovery instead of failing all workers.
                    vector.reset(
                        env_id,
                        int(cfg["train_seed_base"]) + num_envs + env_id,
                        rebuild=True,
                    )
                else:
                    raise RuntimeError(
                        f"Unexpected smoke result env={env_id}: {message}"
                    )

            if step == 0:
                # Reset exactly one env. Other workers stay alive.
                vector.reset(0, reset_seed)
                assert vector.alive_worker_count() == num_envs

        alive_before_close = vector.alive_worker_count()
        vector.close()
        alive_after_close = vector.alive_worker_count()
        vector = None

        assert alive_before_close == num_envs
        assert alive_after_close == 0

        print(
            json.dumps(
                {
                    "status": "PASS",
                    "num_envs": num_envs,
                    "vector_steps": args.vector_steps,
                    "valid_transitions": valid_transitions,
                    "fatal_transitions_recovered": fatal_transitions,
                    "individual_reset_env_id": 0,
                    "clean_shutdown": True,
                    "alive_workers_after_close": alive_after_close,
                }
            )
        )
    finally:
        if vector is not None:
            vector.close(force=True)


if __name__ == "__main__":
    main()
