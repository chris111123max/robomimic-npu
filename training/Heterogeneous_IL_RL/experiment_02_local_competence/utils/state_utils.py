"""Full EnvRobosuite state, branch condition, and trajectory persistence."""

import hashlib
import json
from pathlib import Path

import numpy as np

from utils.result_utils import atomic_npz
from utils.rng_utils import decode_rng_state, encode_rng_state


def _hash_piece(hasher, name, value):
    array = np.asarray(value)
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(json.dumps(array.shape).encode("ascii"))
    hasher.update(np.ascontiguousarray(array).tobytes() if array.dtype.kind not in "USO"
                  else str(array.tolist()).encode("utf-8"))


def simulator_state_hash(state):
    hasher = hashlib.sha256()
    for key in ("model", "states", "ep_meta"):
        if state.get(key) is not None:
            _hash_piece(hasher, key, state[key])
    return hasher.hexdigest()


def state_vector_hash(states):
    hasher = hashlib.sha256()
    _hash_piece(hasher, "states", states)
    return hasher.hexdigest()


def observation_hash(observation):
    hasher = hashlib.sha256()
    for key in sorted(observation):
        _hash_piece(hasher, key, observation[key])
    return hasher.hexdigest()


def copy_observation(observation):
    return {key: np.asarray(value).copy() for key, value in observation.items()}


def load_exp00_state(path):
    with np.load(path, allow_pickle=False) as archive:
        state = {"model": str(archive["model"].item()), "states": np.array(archive["states"], copy=True)}
        ep_meta = str(archive["ep_meta"].item()) if "ep_meta" in archive else ""
        if ep_meta:
            state["ep_meta"] = ep_meta
        keys = [str(x) for x in archive["observation_names"].tolist()]
        obs = {key: np.array(archive["obs__" + key], copy=True) for key in keys}
    return state, obs


def trajectory_arrays(observations, actions, rewards, dones, successes, **extra):
    names = sorted(observations[0]) if observations else []
    arrays = {
        "observation_names": np.asarray(names), "actions": np.asarray(actions),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "environment_dones": np.asarray(dones, dtype=np.bool_),
        "task_success": np.asarray(successes, dtype=np.bool_),
        "step_indices": np.arange(len(actions), dtype=np.int64),
    }
    for key in names:
        arrays["obs__" + key] = np.stack([np.asarray(obs[key]) for obs in observations])
    arrays.update(extra)
    return arrays


def save_trajectory(path, observations, actions, rewards, dones, successes, compressed=True, **extra):
    atomic_npz(path, compressed=compressed,
               **trajectory_arrays(observations, actions, rewards, dones, successes, **extra))


def load_trajectory(path):
    with np.load(path, allow_pickle=False) as archive:
        names = [str(x) for x in archive["observation_names"].tolist()]
        length = int(archive["actions"].shape[0])
        observations = [{key: np.array(archive["obs__" + key][i], copy=True) for key in names}
                        for i in range(length)]
        return {
            "observations": observations, "actions": np.array(archive["actions"], copy=True),
            "rewards": np.array(archive["rewards"], copy=True),
            "dones": np.array(archive["environment_dones"], copy=True),
            "successes": np.array(archive["task_success"], copy=True),
        }


def trajectory_length(path):
    """Read only the action length for fast crash-resume validation."""
    with np.load(path, allow_pickle=False) as archive:
        return int(archive["actions"].shape[0])


def save_branch_state(path, state, observation, rng_state, original_action, branch_step, metadata):
    names = sorted(observation)
    arrays = {
        "model": np.asarray(state["model"]), "states": np.asarray(state["states"]),
        "ep_meta": np.asarray(state.get("ep_meta", "")), "observation_names": np.asarray(names),
        "rng_state": encode_rng_state(rng_state), "original_action": np.asarray(original_action),
        "branch_step": np.asarray(int(branch_step), dtype=np.int64),
        "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    }
    arrays.update({"obs__" + key: np.asarray(observation[key]) for key in names})
    atomic_npz(path, compressed=True, **arrays)


def load_branch_state(path):
    with np.load(path, allow_pickle=False) as archive:
        state = {"model": str(archive["model"].item()), "states": np.array(archive["states"], copy=True)}
        ep_meta = str(archive["ep_meta"].item())
        if ep_meta:
            state["ep_meta"] = ep_meta
        names = [str(x) for x in archive["observation_names"].tolist()]
        return {
            "state": state,
            "observation": {key: np.array(archive["obs__" + key], copy=True) for key in names},
            "rng_state": decode_rng_state(archive["rng_state"]),
            "original_action": np.array(archive["original_action"], copy=True),
            "branch_step": int(archive["branch_step"].item()),
            "metadata": json.loads(str(archive["metadata_json"].item())),
        }


def branch_file_valid(path, expected_step=None):
    try:
        item = load_branch_state(path)
        return (expected_step is None or item["branch_step"] == int(expected_step)) and \
               simulator_state_hash(item["state"]) == item["metadata"]["state_hash"] and \
               observation_hash(item["observation"]) == item["metadata"]["obs_hash"]
    except Exception:
        return False
