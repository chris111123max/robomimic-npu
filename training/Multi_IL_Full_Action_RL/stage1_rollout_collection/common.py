"""Shared utilities for Stage 1 rollout collection and validation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "multi_il_full_action_rl.stage1.v1"
VALID_POLICY_IDS = ("bc_gmm", "bc_rnn", "bc_transformer")


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def git_commit(repo_root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def sha256_array(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(canonical_json(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def simulator_state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        value = state[key]
        if isinstance(value, np.ndarray):
            digest.update(sha256_array(value).encode("ascii"))
        else:
            digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


def observation_hash(observation):
    digest = hashlib.sha256()
    for key in sorted(observation):
        digest.update(key.encode("utf-8"))
        digest.update(sha256_array(observation[key]).encode("ascii"))
    return digest.hexdigest()


def copy_observation(observation):
    return {key: np.asarray(value).copy() for key, value in observation.items()}


def extract_canonical_observation(policy_observation, keys, expected_shapes=None):
    """Extract raw current-frame values from raw or FrameStackWrapper observations."""
    result = {}
    for key in keys:
        if key not in policy_observation:
            raise KeyError(f"Canonical observation key {key!r} is absent")
        value = np.asarray(policy_observation[key])
        if expected_shapes is None:
            result[key] = value.copy()
            continue
        expected = tuple(expected_shapes[key])
        if value.shape == expected:
            result[key] = value.copy()
        elif value.ndim == len(expected) + 1 and tuple(value.shape[1:]) == expected:
            result[key] = value[-1].copy()
        else:
            raise RuntimeError(
                f"Observation shape mismatch for {key}: policy env returned {value.shape}, "
                f"canonical shape is {expected}"
            )
    return result


def compare_observations(expected, actual, atol):
    if set(expected) != set(actual):
        raise RuntimeError(
            f"Canonical observation keys differ: {sorted(expected)} != {sorted(actual)}"
        )
    worst_key, worst_error = None, 0.0
    for key in sorted(expected):
        left, right = np.asarray(expected[key]), np.asarray(actual[key])
        if left.shape != right.shape:
            raise RuntimeError(f"Canonical observation shape differs for {key}: {left.shape} != {right.shape}")
        error = float(np.max(np.abs(left - right))) if left.size else 0.0
        if error > worst_error:
            worst_key, worst_error = key, error
        if not np.allclose(left, right, rtol=0.0, atol=float(atol)):
            raise RuntimeError(
                f"Same-state canonical observation differs for {key}: "
                f"max_abs_error={error:.9g}, atol={float(atol):.9g}"
            )
    return worst_key, worst_error


def seed_everything(seed):
    """Seed Python, NumPy, Torch, and the visible accelerator when available."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        npu = getattr(torch, "npu", None)
        if npu is not None and hasattr(npu, "manual_seed_all"):
            npu.manual_seed_all(seed)
    except ImportError:
        pass


def try_seed_environment(env, seed):
    """Best-effort API seeding; exact cross-policy identity is enforced with reset_to."""
    seeded = []
    seen = set()
    current = env
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        method = getattr(current, "seed", None)
        if callable(method):
            try:
                method(int(seed))
                seeded.append(type(current).__name__)
            except (AttributeError, NotImplementedError, TypeError):
                pass
        current = getattr(current, "env", None)
    return seeded


def infer_algo_type(config):
    if bool(config.algo.transformer.enabled):
        return "BC-Transformer-GMM" if bool(config.algo.gmm.enabled) else "BC-Transformer"
    if bool(config.algo.rnn.enabled):
        return "BC-RNN-GMM" if bool(config.algo.gmm.enabled) else "BC-RNN"
    return "BC-GMM" if bool(config.algo.gmm.enabled) else "BC"


def configured_low_dim_keys(config):
    keys = list(config.observation.modalities.obs.low_dim)
    if config.observation.modalities.obs.rgb or config.observation.modalities.obs.depth:
        raise RuntimeError("Stage 1 currently requires canonical low-dimensional checkpoints")
    if not keys:
        raise RuntimeError("Checkpoint has no configured low-dimensional observations")
    return keys


def discounted_returns(rewards, gamma):
    rewards = np.asarray(rewards, dtype=np.float64)
    result = np.empty_like(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + float(gamma) * running
        result[index] = running
    return result

