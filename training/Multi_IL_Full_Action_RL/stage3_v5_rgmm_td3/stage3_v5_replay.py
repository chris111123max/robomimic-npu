"""Boundary-aligned Actor windows; Critic replay retains the v3 contract."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import h5py
import json

V3 = Path(__file__).resolve().parents[1] / "stage3_v3_rgmm_td3"
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))
from stage3_v3_replay import (CORE, OfflineDemonstrations as _OfflineDemonstrations,
                              OnlineSequenceReplay as _OnlineSequenceReplay,
                              final_transition)

SOURCE_NAMES = ("rnn", "transformer", "gmm")
SOURCE_IDS = {name: index for index, name in enumerate(SOURCE_NAMES)}
CANONICAL_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                  "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos", "object")


class Stage1OfflineSequenceReplay:
    """Native, read-only loader for Stage1 `/episodes/*` transition HDF5 files."""
    def __init__(self, path, source, seed=0):
        if source not in SOURCE_IDS: raise ValueError(f"Unknown source {source}")
        self.path, self.source, self.source_id = str(Path(path).resolve()), source, SOURCE_IDS[source]
        self.rng = np.random.default_rng(int(seed)); self.episodes = []; self.samples_drawn = 0
        with h5py.File(self.path, "r") as handle:
            if "episodes" not in handle: raise RuntimeError(f"{path}: expected Stage1 /episodes")
            keys = tuple(json.loads(handle.attrs["canonical_observation_keys"]))
            if keys != CANONICAL_KEYS: raise RuntimeError(f"{path}: incompatible canonical observation order")
            for name in sorted(handle["episodes"]): self.episodes.append(self._episode(handle["episodes"][name]))
        if not self.episodes: raise RuntimeError(f"{path}: no episodes")
        self.transition_count = sum(len(x["actions"]) for x in self.episodes)
        self.schema = {"obs_dim": 59, "action_dim": 14, "canonical_keys": CANONICAL_KEYS}

    def _episode(self, group):
        required = ("obs", "next_obs", "actions", "rewards", "dones", "terminated", "truncated")
        missing = [key for key in required if key not in group]
        if missing: raise RuntimeError(f"{group.name}: missing {missing}")
        # h5py cannot always perform a direct HDF5->bool conversion. Read the
        # native array first, then let NumPy cast it explicitly.
        actions = np.asarray(group["actions"][...], np.float32); length = len(actions)
        obs = np.concatenate([np.asarray(group["obs"][key][...], np.float32).reshape(length, -1) for key in CANONICAL_KEYS], axis=1)
        nxt = np.concatenate([np.asarray(group["next_obs"][key][...], np.float32).reshape(length, -1) for key in CANONICAL_KEYS], axis=1)
        terminated, truncated = np.asarray(group["terminated"][...], bool).reshape(-1, 1), np.asarray(group["truncated"][...], bool).reshape(-1, 1)
        dones = np.asarray(group["dones"][...], bool).reshape(-1, 1)
        if obs.shape != (length,59) or nxt.shape != (length,59) or actions.shape != (length,14): raise RuntimeError(f"{group.name}: dimensional contract failed")
        if not np.array_equal(dones, terminated | truncated): raise RuntimeError(f"{group.name}: done contract failed")
        if not all(np.isfinite(x).all() for x in (obs,nxt,actions)): raise RuntimeError(f"{group.name}: non-finite values")
        def scalar(name, default):
            value = group.attrs[name] if name in group.attrs else (group[name][()] if name in group else default)
            return np.asarray(value).reshape(-1)[0].item()
        return {"observations":obs,"next_observations":nxt,"actions":actions,"rewards":np.asarray(group["rewards"][...],np.float32).reshape(-1,1),"dones":dones,"terminated":terminated,"truncated":truncated,"terminals":terminated.astype(np.float32),"episode_steps":np.arange(length,dtype=np.int64),"episode_id":int(scalar("episode_id",-1)),"seed":int(scalar("initial_seed",-1)),"success":bool(scalar("episode_success",False))}

    def sample_sequences(self, count, length, aligned=False, horizon=10):
        eligible = [ep for ep in self.episodes if len(ep["actions"]) >= int(length)]
        if not eligible: raise RuntimeError("No valid Stage1 sequence")
        ids = self.rng.integers(len(eligible), size=int(count)); out={key:[] for key in CORE+("episode_steps","terminated","truncated","dones")}
        for index in ids:
            ep=eligible[int(index)]; starts=(len(ep["actions"])-int(length))//horizon+1 if aligned else len(ep["actions"])-int(length)+1
            start=int(self.rng.integers(starts))*(horizon if aligned else 1)
            for key in out: out[key].append(ep[key][start:start+int(length)])
        self.samples_drawn += int(count)
        result={key:np.stack(value) for key,value in out.items()}; result["source_id"]=np.full(int(count),self.source_id,np.int8); return result

    def state_dict(self): return {"rng_state":self.rng.bit_generator.state,"samples_drawn":self.samples_drawn}
    def load_state_dict(self, value): self.rng.bit_generator.state=value["rng_state"]; self.samples_drawn=int(value["samples_drawn"])


def _sample_sequence_batch(episodes, rng, count, length):
    """Sample the reference distribution with batched RNG/index generation."""
    eligible = [episode for episode in episodes
                if len(episode["actions"]) >= int(length)]
    if not eligible:
        raise RuntimeError("No complete episode is available for recurrent sampling")
    count = int(count)
    episode_ids = rng.integers(len(eligible), size=count)
    start_counts = np.asarray(
        [len(episode["actions"]) - int(length) + 1 for episode in eligible],
        dtype=np.int64,
    )
    starts = rng.integers(start_counts[episode_ids], size=count)
    return {
        key: np.stack([
            eligible[int(episode_id)][key][int(start):int(start) + int(length)]
            for episode_id, start in zip(episode_ids, starts)
        ])
        for key in CORE + ("episode_steps",)
    }


class OfflineDemonstrations(_OfflineDemonstrations):
    """Stage3-v4 replay with batched sequence index generation."""

    def sample_sequences(self, count, length):
        return _sample_sequence_batch(self.episodes, self.rng, count, length)


class BalancedOfflineDemonstrations:
    """Fixed-size 3-way offline mixture with rotating 43/43/42 remainder."""
    def __init__(self, datasets, seed):
        self.sources = [Stage1OfflineSequenceReplay(path, name, int(seed) + index)
                        for index, (name, path) in enumerate(zip(SOURCE_NAMES, datasets))]
        self.rng = np.random.default_rng(int(seed)); self._remainder_cursor = 0
        self.episodes = sum((source.episodes for source in self.sources), [])

    def _all_episodes(self):
        return self.episodes

    def sample_sequences(self, count, length, aligned=False, horizon=10):
        base = int(count) // 3
        # For 128 this produces 43/43/42 and rotates the short source.
        counts = [base, base, base]
        for offset in range(int(count) - 3 * base):
            counts[(self._remainder_cursor + offset) % 3] += 1
        self._remainder_cursor = (self._remainder_cursor + 1) % 3
        pieces = [source.sample_sequences(n, length, aligned, horizon) for source, n in zip(self.sources, counts)]
        keys = CORE + ("episode_steps", "terminated", "truncated", "dones", "source_id")
        result = {key: np.concatenate([piece[key] for piece in pieces], axis=0) for key in keys}
        permutation = self.rng.permutation(int(count))
        return {key: value[permutation] for key, value in result.items()}

    def state_dict(self): return {"rotation_index":self._remainder_cursor,"rng_state":self.rng.bit_generator.state,"sources":[x.state_dict() for x in self.sources]}
    def load_state_dict(self, value):
        self._remainder_cursor=int(value["rotation_index"]); self.rng.bit_generator.state=value["rng_state"]
        for sampler,state in zip(self.sources,value["sources"]): sampler.load_state_dict(state)


class OnlineSequenceReplay(_OnlineSequenceReplay):
    """Stage3-v4 replay with batched sequence index generation."""

    def sample_sequences(self, count, length):
        return _sample_sequence_batch(
            list(self.episodes) + list(self.current.values()),
            self.rng, count, length,
        )

    def finish(self, env_id, success=False):
        episode = self.current.pop(int(env_id), None)
        if episode:
            saved = {key: np.asarray(value) for key, value in episode.items()}
            saved["success"] = bool(success)
            self.add_episode(saved)

    @classmethod
    def load(cls, path):
        payload = np.load(path, allow_pickle=True).item()
        replay = cls(payload["capacity"])
        replay.transitions = int(payload["transitions"])
        from collections import deque
        replay.episodes = deque(payload["episodes"])
        replay.current = payload.get("current", {})
        replay.rng.bit_generator.state = payload["rng_state"]
        return replay


def _sample_aligned(source, count, length, horizon):
    if isinstance(source, (Stage1OfflineSequenceReplay, BalancedOfflineDemonstrations)):
        return source.sample_sequences(count, length, aligned=True, horizon=horizon)
    episodes = source.episodes if isinstance(source, OfflineDemonstrations) else source._all_episodes()
    eligible = [episode for episode in episodes if len(episode["actions"]) >= length]
    if not eligible:
        raise RuntimeError("No complete boundary-aligned Actor window available")
    count = int(count)
    # Draw the same distribution as the reference implementation: episodes
    # are uniform, then complete horizon-aligned starts are uniform within the
    # selected episode. Vectorized draws remove one RNG call per sample.
    episode_ids = source.rng.integers(len(eligible), size=count)
    window_counts = np.asarray(
        [(len(episode["actions"]) - int(length)) // int(horizon) + 1
         for episode in eligible], dtype=np.int64)
    start_ids = source.rng.integers(
        window_counts[episode_ids], size=count) * int(horizon)
    result = {}
    for key in CORE + ("episode_steps",):
        result[key] = np.stack([
            np.asarray(eligible[int(episode_id)][key][int(start):int(start) + int(length)])
            for episode_id, start in zip(episode_ids, start_ids)
        ])
    expected_steps = start_ids[:, None] + np.arange(int(length), dtype=np.int64)[None, :]
    if not np.array_equal(result["episode_steps"], expected_steps):
        raise RuntimeError("Aligned replay sampler produced a non-boundary window")
    return result


def aligned_sequence_batch(offline, online, count, length, horizon=10):
    if int(count) <= 0 or int(count) % 2 or int(length) != int(horizon):
        raise ValueError("Aligned Actor batch requires even count and one RNN horizon")
    half = int(count) // 2
    left = _sample_aligned(offline, half, int(length), int(horizon))
    right = _sample_aligned(online, half, int(length), int(horizon))
    batch = {key: np.concatenate((left[key], right[key]), axis=0)
             for key in CORE + ("episode_steps",)}
    batch["is_offline"] = np.concatenate((np.ones(half, np.float32),
                                           np.zeros(half, np.float32)))
    if not np.all(batch["episode_steps"][:, 0] % horizon == 0):
        raise RuntimeError("Actor batch starts outside a hidden-state reset boundary")
    return batch
