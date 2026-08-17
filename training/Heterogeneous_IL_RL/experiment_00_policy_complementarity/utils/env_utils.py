"""Environment creation, seeding, and strict state-bank restoration."""

import copy
import random

import numpy as np

import robomimic.utils.env_utils as RobomimicEnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from utils.result_utils import observation_hash, simulator_state_hash, state_vector_hash


def load_dataset_env_metadata(dataset_path):
    return FileUtils.get_env_metadata_from_dataset(dataset_path=str(dataset_path))


def initialize_observation_utils_from_checkpoint(checkpoint_path):
    """Initialize robomimic's process-global observation modality registry.

    EnvRobosuite.get_observation requires this registry even when no policy has
    been constructed yet. Official evaluators initialize it as a side effect of
    policy_from_checkpoint; the standalone state-bank builder must do so
    explicitly from the same checkpoint config.
    """
    checkpoint = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(checkpoint_path))
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint)
    ObsUtils.initialize_obs_utils_with_config(config)
    return checkpoint


def create_dataset_environment(dataset_path):
    metadata = load_dataset_env_metadata(dataset_path)
    env = RobomimicEnvUtils.create_env_from_metadata(
        env_meta=copy.deepcopy(metadata),
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        use_depth_obs=False,
    )
    return env, metadata


def seed_environment(env, seed):
    """Seed all initialization RNGs used by the current robosuite stack."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    candidates = [env, getattr(env, "env", None)]
    for candidate in candidates:
        method = getattr(candidate, "seed", None)
        if callable(method):
            try:
                method(seed)
            except TypeError:
                pass


def capture_initial_condition(env, environment_seed):
    seed_environment(env, environment_seed)
    observation = env.reset()
    state = env.get_state()
    # A restore immediately after capture ensures the persisted representation is usable.
    restored_observation = env.reset_to(state)
    restored_state = env.get_state()
    if not np.array_equal(np.asarray(state["states"]), np.asarray(restored_state["states"])):
        raise RuntimeError("EnvRobosuite reset_to did not exactly restore the simulator state vector")
    if observation_hash(observation) != observation_hash(restored_observation):
        raise RuntimeError("EnvRobosuite reset_to did not reproduce the initial low-dimensional observation")
    return state, restored_observation


def restore_and_verify(env, state, saved_observation, manifest_entry):
    expected_state_hash = simulator_state_hash(state)
    if expected_state_hash != manifest_entry["state_hash"]:
        raise RuntimeError("Persisted state hash does not match initial_state_manifest.json")
    if state_vector_hash(state["states"]) != manifest_entry["state_vector_hash"]:
        raise RuntimeError("Persisted simulator state-vector hash does not match manifest")
    if observation_hash(saved_observation) != manifest_entry["observation_hash"]:
        raise RuntimeError("Persisted initial observation hash does not match manifest")

    restored_observation = env.reset_to(state)
    restored_state = env.get_state()
    if not np.array_equal(np.asarray(state["states"]), np.asarray(restored_state["states"])):
        raise RuntimeError("Simulator state vector differs immediately after reset_to")
    restored_observation_hash = observation_hash(restored_observation)
    if restored_observation_hash != manifest_entry["observation_hash"]:
        raise RuntimeError("Environment observation differs immediately after reset_to")
    return restored_observation


def task_succeeded(env):
    return bool(env.is_success()["task"])


def close_environment(env):
    raw = getattr(env, "env", None)
    close = getattr(raw, "close", None)
    if callable(close):
        close()
