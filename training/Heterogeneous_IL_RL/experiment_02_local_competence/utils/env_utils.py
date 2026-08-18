"""Dataset-metadata environment creation and strict official reset_to checks."""

import copy
import hashlib
import random

import numpy as np

from utils.state_utils import copy_observation, observation_hash, simulator_state_hash, state_vector_hash


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


def _validate_restored_observation(expected, restored, atol, strict=False):
    expected_keys, restored_keys = set(expected), set(restored)
    if expected_keys != restored_keys:
        missing = sorted(expected_keys - restored_keys)
        extra = sorted(restored_keys - expected_keys)
        raise RuntimeError(
            f"Branch restore observation keys mismatch: missing={missing}, extra={extra}"
        )
    worst_key, worst_error = None, 0.0
    for key in sorted(expected_keys):
        expected_value = np.asarray(expected[key])
        restored_value = np.asarray(restored[key])
        if expected_value.shape != restored_value.shape:
            raise RuntimeError(
                f"Branch restore observation shape mismatch for {key}: "
                f"{restored_value.shape} != {expected_value.shape}"
            )
        if not np.all(np.isfinite(restored_value)):
            raise RuntimeError(f"Branch restore observation contains non-finite values for {key}")
        error = float(np.max(np.abs(expected_value - restored_value))) if expected_value.size else 0.0
        if error > worst_error:
            worst_key, worst_error = key, error
        if strict and not np.allclose(
                expected_value, restored_value, rtol=0.0, atol=float(atol)):
            raise RuntimeError(
                f"Branch restore observation mismatch for {key}: "
                f"max_abs_error={error:.9g}, atol={float(atol):.9g}"
            )
    return worst_key, worst_error


def restore_branch_state(env, branch, observation_atol=1e-6,
                         strict_observation=False):
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
    saved_observation = branch["observation"]
    if observation_hash(saved_observation) != branch["metadata"]["obs_hash"]:
        raise RuntimeError("Persisted branch observation hash is corrupt")
    # env.step can return observable-cache values, while reset_to performs
    # sim.forward() and force-updates derived observables. Those values can
    # differ even when the complete simulator state is restored byte-for-byte.
    # The persisted obs_t is the actual branch-point policy input and remains
    # authoritative. Exact action reconstruction is the behavioral gate.
    observation_diagnostic = _validate_restored_observation(
        saved_observation, observation, observation_atol,
        strict=bool(strict_observation),
    )
    # The branch condition explicitly includes the recorded obs_t. Use that exact
    # observation for the first policy call after restoring the physical state.
    return copy_observation(saved_observation), restored_hash, observation_diagnostic


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
