"""Environment creation, seeding, and strict state-bank restoration."""

import copy
import hashlib
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


def deterministic_environment_stream_seed(meta_seed, initial_state_id):
    material = f"{int(meta_seed)}:environment:{int(initial_state_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "little") & 0x7FFFFFFF


def select_observation_keys(observation, observation_keys):
    missing = [key for key in observation_keys if key not in observation]
    if missing:
        raise RuntimeError(f"Restored environment is missing policy observations: {missing}")
    return {key: observation[key] for key in observation_keys}


def capture_initial_condition(
    env, environment_seed, meta_seed, initial_state_id, observation_keys,
):
    seed_environment(env, environment_seed)
    env.reset()
    state = env.get_state()
    restore_seed = deterministic_environment_stream_seed(meta_seed, initial_state_id)

    # Compare two independent official restore operations. The observation made
    # by the original random reset can contain transient observable caches that
    # reset_to legitimately reconstructs differently; every evaluated policy
    # starts through reset_to, so restore-to-restore reproducibility is the
    # relevant paired condition.
    restored_observations = []
    for _ in range(2):
        seed_environment(env, restore_seed)
        restored_observation = env.reset_to(state)
        restored_state = env.get_state()
        if not np.array_equal(np.asarray(state["states"]), np.asarray(restored_state["states"])):
            raise RuntimeError("EnvRobosuite reset_to did not exactly restore the simulator state vector")
        restored_observations.append(restored_observation)

    first = select_observation_keys(restored_observations[0], observation_keys)
    second = select_observation_keys(restored_observations[1], observation_keys)
    if observation_hash(first) != observation_hash(second):
        raise RuntimeError(
            "Repeated reset_to calls did not reproduce the checkpoint observation keys"
        )
    return state, restored_observations[1]


def restore_and_verify(env, state, saved_observation, manifest_entry):
    expected_state_hash = simulator_state_hash(state)
    if expected_state_hash != manifest_entry["state_hash"]:
        raise RuntimeError("Persisted state hash does not match initial_state_manifest.json")
    if state_vector_hash(state["states"]) != manifest_entry["state_vector_hash"]:
        raise RuntimeError("Persisted simulator state-vector hash does not match manifest")
    observation_keys = manifest_entry.get("verified_observation_keys", sorted(saved_observation))
    saved_policy_observation = select_observation_keys(saved_observation, observation_keys)
    if observation_hash(saved_policy_observation) != manifest_entry["observation_hash"]:
        raise RuntimeError("Persisted initial observation hash does not match manifest")

    restored_observation = env.reset_to(state)
    restored_state = env.get_state()
    if not np.array_equal(np.asarray(state["states"]), np.asarray(restored_state["states"])):
        raise RuntimeError("Simulator state vector differs immediately after reset_to")
    restored_policy_observation = select_observation_keys(restored_observation, observation_keys)
    restored_observation_hash = observation_hash(restored_policy_observation)
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
