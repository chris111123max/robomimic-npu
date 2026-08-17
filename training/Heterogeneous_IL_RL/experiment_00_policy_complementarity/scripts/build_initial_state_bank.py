#!/usr/bin/env python3
"""Build a resumable bank of persisted, paired EnvRobosuite initial states."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from utils.env_utils import (
    capture_initial_condition, close_environment, create_dataset_environment,
    initialize_observation_utils_from_checkpoint,
)
from utils.result_utils import (
    atomic_json, atomic_npz, observation_hash, read_json, simulator_state_hash,
    state_vector_hash, validate_state_bank_file,
)


def desired_seeds(config, count):
    seed_cfg = config["seed_generation"]
    rng = np.random.RandomState(int(seed_cfg["meta_seed"]))
    low = int(seed_cfg.get("minimum_seed", 0))
    high = int(seed_cfg.get("maximum_seed_exclusive", 2147483647))
    if high - low < count:
        raise ValueError("Configured seed range is smaller than num_initial_conditions")
    values = set()
    while len(values) < count:
        values.update(int(value) for value in rng.randint(low, high, size=max(16, count - len(values))))
    return list(sorted(values))[:count]


def ensure_seed_manifest(run_dir, config, count):
    path = run_dir / "seed_manifest.json"
    if path.exists():
        manifest = read_json(path)
        if int(manifest["meta_seed"]) != int(config["seed_generation"]["meta_seed"]):
            raise RuntimeError("Existing seed manifest uses a different meta_seed")
        if len(manifest["environment_seeds"]) != count:
            raise RuntimeError(
                f"Existing run has {len(manifest['environment_seeds'])} seeds, but {count} were requested; use a new run directory"
            )
        return manifest
    manifest = {
        "meta_seed": int(config["seed_generation"]["meta_seed"]),
        "num_initial_conditions": count,
        "environment_seeds": desired_seeds(config, count),
    }
    atomic_json(path, manifest)
    return manifest


def build(config_path, run_dir, num_seeds=None, force_rebuild=False):
    config = read_json(config_path)
    run_dir = Path(run_dir)
    count = int(num_seeds or config["seed_generation"]["num_initial_conditions"])
    seed_manifest = ensure_seed_manifest(run_dir, config, count)
    state_dir = run_dir / "initial_states"
    state_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "initial_state_manifest.json"
    existing = read_json(manifest_path) if manifest_path.exists() else {"states": []}
    entries = {int(row["initial_state_id"]): row for row in existing.get("states", [])}

    # This stage runs in a fresh subprocess and has not loaded a RolloutPolicy.
    # EnvRobosuite.get_observation still requires the global ObsUtils registry.
    reference_checkpoint = initialize_observation_utils_from_checkpoint(
        config["policies"]["bc"]["checkpoint_path"]
    )
    observation_keys = list(reference_checkpoint["shape_metadata"]["all_shapes"].keys())
    env, env_metadata = create_dataset_environment(config["dataset_path"])
    if env_metadata["env_name"] != config["expected_environment_name"]:
        raise RuntimeError(f"Dataset environment is {env_metadata['env_name']}, expected {config['expected_environment_name']}")
    try:
        for state_id, environment_seed in enumerate(seed_manifest["environment_seeds"]):
            state_path = state_dir / f"state_{state_id:06d}.npz"
            old = entries.get(state_id)
            if not force_rebuild and old and state_path.exists() and validate_state_bank_file(state_path, old):
                print(f"[build {state_id + 1:03d}/{count:03d}] valid, skipping {state_path.name}", flush=True)
                continue
            state, observation = capture_initial_condition(
                env=env,
                environment_seed=environment_seed,
                meta_seed=seed_manifest["meta_seed"],
                initial_state_id=state_id,
                observation_keys=observation_keys,
            )
            arrays = {
                "model": np.asarray(state["model"]),
                "states": np.asarray(state["states"]),
                "ep_meta": np.asarray(state.get("ep_meta", "")),
                "observation_names": np.asarray(sorted(observation)),
            }
            arrays.update({"obs__" + key: np.asarray(observation[key]) for key in sorted(observation)})
            atomic_npz(state_path, compressed=True, **arrays)
            entry = {
                "initial_state_id": state_id,
                "environment_seed": int(environment_seed),
                "state_file": str(state_path.relative_to(run_dir)),
                "state_hash": simulator_state_hash(state),
                "state_vector_hash": state_vector_hash(state["states"]),
                "verified_observation_keys": observation_keys,
                "observation_hash": observation_hash(
                    {key: observation[key] for key in observation_keys}
                ),
            }
            entries[state_id] = entry
            payload = {
                "environment_name": env_metadata["env_name"],
                "num_initial_conditions": count,
                "states": [entries[index] for index in sorted(entries) if index < count],
            }
            atomic_json(manifest_path, payload)
            print(f"[build {state_id + 1:03d}/{count:03d}] saved {state_path.name}", flush=True)
    finally:
        close_environment(env)

    final = read_json(manifest_path)
    if len(final["states"]) != count:
        raise RuntimeError(f"State bank is incomplete: {len(final['states'])}/{count}")
    print(f"Initial-state bank ready: {count} states in {state_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-seeds", type=int)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    build(args.config, args.run_dir, args.num_seeds, args.force_rebuild)


if __name__ == "__main__":
    main()
