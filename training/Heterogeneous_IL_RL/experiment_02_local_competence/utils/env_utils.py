"""Dataset-metadata environment creation and strict official reset_to checks."""

import copy
import hashlib
import random

import numpy as np

from utils.state_utils import observation_hash, simulator_state_hash, state_vector_hash


def initialize_observation_utils(checkpoint_path):
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils
    checkpoint = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(checkpoint_path))
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint)
    ObsUtils.initialize_obs_utils_with_config(config)


def create_environment(dataset_path):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    metadata = FileUtils.get_env_metadata_from_dataset(dataset_path=str(dataset_path))
    env = EnvUtils.create_env_from_metadata(
        env_meta=copy.deepcopy(metadata), render=False, render_offscreen=False,
        use_image_obs=False, use_depth_obs=False)
    return env, metadata


def seed_environment_stream(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))


def exp00_environment_stream_seed(meta_seed, initial_state_id):
    """Byte-for-byte copy of Experiment 00's paired reset stream derivation."""
    material = f"{int(meta_seed)}:environment:{int(initial_state_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "little") & 0x7FFFFFFF


def restore_initial_state(env, state, saved_observation, manifest_entry):
    if simulator_state_hash(state) != manifest_entry["state_hash"]:
        raise RuntimeError("Experiment 00 initial state hash mismatch")
    observation = env.reset_to(state)
    restored = env.get_state()
    if not np.array_equal(np.asarray(state["states"]), np.asarray(restored["states"])):
        raise RuntimeError("Initial reset_to state-vector mismatch")
    keys = manifest_entry["verified_observation_keys"]
    selected = {key: observation[key] for key in keys}
    if observation_hash(selected) != manifest_entry["observation_hash"]:
        raise RuntimeError("Initial reset_to observation mismatch")
    return observation


def restore_branch_state(env, branch):
    expected_state = branch["state"]
    expected_hash = branch["metadata"]["state_hash"]
    if simulator_state_hash(expected_state) != expected_hash:
        raise RuntimeError("Persisted branch state hash is corrupt")
    observation = env.reset_to(expected_state)
    restored = env.get_state()
    restored_hash = simulator_state_hash(restored)
    if restored_hash != expected_hash:
        raise RuntimeError(f"Branch restore full-state hash mismatch: {restored_hash} != {expected_hash}")
    if state_vector_hash(restored["states"]) != branch["metadata"]["state_vector_hash"]:
        raise RuntimeError("Branch restore state-vector hash mismatch")
    if observation_hash(observation) != branch["metadata"]["obs_hash"]:
        raise RuntimeError("Branch restore observation hash mismatch")
    return observation, restored_hash


def task_succeeded(env):
    result = env.is_success()
    if "task" not in result:
        raise RuntimeError("Environment success dictionary lacks 'task'")
    return bool(result["task"])


def close_environment(env):
    raw = getattr(env, "env", None)
    close = getattr(raw, "close", None)
    if callable(close):
        close()
