"""Crash-safe persistence and deterministic fingerprints for experiment outputs."""

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


POLICY_ORDER = ("bc", "bc_gmm", "bc_gmm_rnn", "bc_gmm_transformer")


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_npz(path, compressed=True, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            writer = np.savez_compressed if compressed else np.savez
            writer(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _hash_piece(hasher, name, value):
    array = np.asarray(value)
    hasher.update(name.encode("utf-8"))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(json.dumps(array.shape).encode("ascii"))
    if array.dtype.kind in ("U", "S", "O"):
        hasher.update(str(array.tolist()).encode("utf-8"))
    else:
        hasher.update(np.ascontiguousarray(array).tobytes())


def observation_hash(observation):
    hasher = hashlib.sha256()
    for key in sorted(observation):
        _hash_piece(hasher, key, observation[key])
    return hasher.hexdigest()


def simulator_state_hash(state):
    hasher = hashlib.sha256()
    for key in ("model", "states", "ep_meta"):
        if key in state and state[key] is not None:
            _hash_piece(hasher, key, state[key])
    return hasher.hexdigest()


def state_vector_hash(state_vector):
    hasher = hashlib.sha256()
    _hash_piece(hasher, "states", state_vector)
    return hasher.hexdigest()


def load_state_bank_file(path):
    with np.load(path, allow_pickle=False) as archive:
        state = {
            "model": str(archive["model"].item()),
            "states": np.array(archive["states"], copy=True),
        }
        ep_meta = str(archive["ep_meta"].item())
        if ep_meta:
            state["ep_meta"] = ep_meta
        obs_names = [str(item) for item in archive["observation_names"].tolist()]
        observation = {name: np.array(archive["obs__" + name], copy=True) for name in obs_names}
    return state, observation


def validate_state_bank_file(path, expected):
    try:
        state, observation = load_state_bank_file(path)
    except Exception:
        return False
    return (
        simulator_state_hash(state) == expected["state_hash"]
        and state_vector_hash(state["states"]) == expected["state_vector_hash"]
        and observation_hash(observation) == expected["observation_hash"]
    )


def validate_trajectory(path, expected_length=None):
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = ("actions", "rewards", "environment_dones", "task_success", "step_indices", "observation_names")
            if any(name not in archive for name in required):
                return False
            length = int(archive["actions"].shape[0])
            if expected_length is not None and length != int(expected_length):
                return False
            if any(int(archive[name].shape[0]) != length for name in required[1:-1]):
                return False
            for name in archive["observation_names"].tolist():
                if int(archive["obs__" + str(name)].shape[0]) != length:
                    return False
    except Exception:
        return False
    return True


RESULT_FIELDS = [
    "initial_state_id", "environment_seed", "policy_name", "checkpoint_path", "success",
    "episode_return", "episode_length", "success_step", "termination_reason", "state_hash",
    "observation_hash", "policy_sampling_seed", "trajectory_path", "wall_time_seconds",
]

ERROR_FIELDS = [
    "initial_state_id", "environment_seed", "policy_name", "checkpoint_path", "termination_reason",
    "state_hash", "error_type", "error_message", "traceback", "wall_time_seconds",
]


def collect_episode_records(run_dir):
    episode_root = Path(run_dir) / "raw_results" / "episodes"
    records = []
    if episode_root.exists():
        for path in sorted(episode_root.glob("*/*.json")):
            try:
                record = read_json(path)
                record["_result_file"] = str(path)
                records.append(record)
            except Exception:
                records.append({
                    "status": "error", "policy_name": path.parent.name,
                    "initial_state_id": path.stem, "termination_reason": "error",
                    "error_type": "CorruptResultFile", "error_message": "Could not parse result file",
                    "traceback": "", "_result_file": str(path),
                })
    return records


def rebuild_result_tables(run_dir):
    records = collect_episode_records(run_dir)
    completed = [record for record in records if record.get("status") == "complete"]
    errors = [record for record in records if record.get("status") == "error"]
    key = lambda row: (POLICY_ORDER.index(row.get("policy_name")) if row.get("policy_name") in POLICY_ORDER else 999,
                       int(row.get("initial_state_id", -1)) if str(row.get("initial_state_id", "")).isdigit() else -1)
    completed.sort(key=key)
    errors.sort(key=key)
    root = Path(run_dir) / "raw_results"
    atomic_csv(root / "rollout_results.csv", RESULT_FIELDS, completed)
    atomic_csv(root / "errors.csv", ERROR_FIELDS, errors)
    return completed, errors
