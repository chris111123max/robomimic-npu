#!/usr/bin/env python3
"""Fail-fast validation before any research rollouts."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from utils.env_utils import (close_environment, create_environment,
                             exp00_environment_stream_seed, initialize_observation_utils,
                             restore_initial_state, seed_environment_stream)
from utils.exp00_reader import RNN, TRANSFORMER, validate_source
from utils.policy_loader import checkpoint_metadata, load_policy, release_policy, select_device
from utils.result_utils import read_json
from utils.state_utils import (load_exp00_state, observation_hash, simulator_state_hash,
                               state_vector_hash)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def validate_metadata(name, metadata, dataset_meta, config):
    env_meta = metadata["environment_metadata"]
    if env_meta.get("env_name") != config["environment_name"] or env_meta.get("type") != dataset_meta.get("type"):
        raise RuntimeError(f"{name} environment metadata is incompatible")
    for key in ("robots", "env_configuration", "controller_configs", "gripper_types", "control_freq"):
        a, b = env_meta.get("env_kwargs", {}).get(key), dataset_meta.get("env_kwargs", {}).get(key)
        if a is not None and b is not None and canonical(a) != canonical(b):
            raise RuntimeError(f"{name} env_kwargs.{key} differs from dataset")
    if metadata["action_dimension"] != int(config["action_dimension"]):
        raise RuntimeError(f"{name} action dimension mismatch")
    if metadata["checkpoint_horizon"] != int(config["horizon"]):
        raise RuntimeError(f"{name} checkpoint horizon mismatch")
    if name == RNN and metadata["temporal_type"] != "rnn":
        raise RuntimeError("RNN checkpoint did not load as an RNN policy")
    if name == TRANSFORMER and metadata["temporal_type"] != "transformer":
        raise RuntimeError("Transformer checkpoint did not load as a Transformer policy")
    if metadata["uses_action_history"]:
        raise RuntimeError("This implementation requires explicit action-history replay support")


def validate_npu_masks(masks):
    if len(masks) != 4 or len(set(masks)) != 4:
        raise RuntimeError("Exactly four unique NPU masks are required")
    probe = ("import torch, torch_npu; assert torch.npu.is_available(); "
             "x=torch.ones(16, device='npu:0'); y=(x*x).sum(); torch.npu.synchronize(); "
             "print(float(y.cpu()))")
    for mask in masks:
        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(mask)
        result = subprocess.run([sys.executable, "-c", probe], env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        print(f"[validate] NPU mask {mask}: {result.stdout.strip()}")
        if result.returncode:
            raise RuntimeError(f"NPU mask {mask} inference probe failed")


def validate(config_path, run_dir, source_run):
    config = read_json(config_path)
    run_dir = Path(run_dir)
    rows, states, seed_manifest, manifest = validate_source(config, source_run, run_dir / "source_run_manifest.json")
    for policy in (RNN, TRANSFORMER):
        if not Path(config["policies"][policy]["checkpoint_path"]).is_file():
            raise FileNotFoundError(config["policies"][policy]["checkpoint_path"])
    if not Path(config["dataset_path"]).is_file():
        raise FileNotFoundError(config["dataset_path"])
    for entry in states:
        state, observation = load_exp00_state(Path(source_run) / entry["state_file"])
        keys = entry["verified_observation_keys"]
        if (simulator_state_hash(state) != entry["state_hash"] or
                state_vector_hash(state["states"]) != entry["state_vector_hash"] or
                observation_hash({key: observation[key] for key in keys}) != entry["observation_hash"]):
            raise RuntimeError(f"Experiment 00 initial-state artifact is corrupt: {entry['state_file']}")
    with tempfile.NamedTemporaryFile(dir=str(run_dir), prefix=".write_test_", delete=True):
        pass

    initialize_observation_utils(config["policies"][RNN]["checkpoint_path"])
    env, dataset_meta = create_environment(config["dataset_path"])
    device = select_device()
    try:
        initial = env.reset()
        if dataset_meta["env_name"] != config["environment_name"]:
            raise RuntimeError("Dataset environment name mismatch")
        if int(env.action_dimension) != int(config["action_dimension"]):
            raise RuntimeError("Environment action dimension mismatch")
        state, saved_observation = load_exp00_state(Path(source_run) / states[0]["state_file"])
        seed_environment_stream(exp00_environment_stream_seed(seed_manifest["meta_seed"], 0))
        restored = restore_initial_state(env, state, saved_observation, states[0])
        if not np.array_equal(env.get_state()["states"], state["states"]):
            raise RuntimeError("Official reset_to did not restore the first Experiment 00 state")
        for name in (RNN, TRANSFORMER):
            path = config["policies"][name]["checkpoint_path"]
            metadata = checkpoint_metadata(name, path)
            validate_metadata(name, metadata, dataset_meta, config)
            missing = [key for key in metadata["observation_keys"] if key not in restored]
            if missing:
                raise RuntimeError(f"{name} observations unavailable: {missing}")
            policy, _, _ = load_policy(name, path, device)
            try:
                policy.start_episode()
                action = np.asarray(policy(ob=restored))
                if action.shape != (int(config["action_dimension"]),):
                    raise RuntimeError(f"{name} action shape is {action.shape}")
                print(f"[validate] {name}: class={policy.policy_class_name}, device={device}, "
                      f"frame_stack={metadata['frame_stack']}, context={metadata['context_length']}, "
                      f"low_noise_eval={metadata['low_noise_eval']}, action={action.shape}")
            finally:
                release_policy(policy)
    finally:
        close_environment(env)
    validate_npu_masks([str(x) for x in config["npu_masks"]])
    print(f"[validate] fixed Experiment 00 statistics: OK ({len(rows)} states)")
    print("VALIDATION COMPLETE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-run", required=True)
    args = parser.parse_args()
    validate(args.config, args.run_dir, args.source_run)


if __name__ == "__main__":
    main()
    # CANN helper threads can keep native teardown hooks alive after successful
    # validation. This script is always launched as a disposable subprocess.
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
