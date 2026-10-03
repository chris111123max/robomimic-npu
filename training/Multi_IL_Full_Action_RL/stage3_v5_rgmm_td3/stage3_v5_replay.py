"""Boundary-aligned Actor windows and full-prefix recurrent Critic replay."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import h5py
import json
from contextlib import nullcontext

V3 = Path(__file__).resolve().parents[1] / "stage3_v3_rgmm_td3"
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))
from stage3_v5_replay_core import (CORE, OfflineDemonstrations as _OfflineDemonstrations,
                              OnlineSequenceReplay as _OnlineSequenceReplay,
                              final_transition, symmetric_sequence_batch)

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
        self.samples_by_purpose = {"critic": 0, "actor": 0, "diagnostic": 0}
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
        # ``dones`` retains the source boundary record, while ``terminals`` is
        # the Bellman mask: neither a true termination nor a time-limit reset
        # has a valid in-episode successor to bootstrap from.
        terminals = (terminated | truncated).astype(np.float32)
        return {"observations":obs,"next_observations":nxt,"actions":actions,"rewards":np.asarray(group["rewards"][...],np.float32).reshape(-1,1),"dones":dones,"terminated":terminated,"truncated":truncated,"terminals":terminals,"episode_steps":np.arange(length,dtype=np.int64),"episode_id":int(scalar("episode_id",-1)),"seed":int(scalar("initial_seed",-1)),"success":bool(scalar("episode_success",False))}

    def sample_sequences(self, count, length, aligned=False, horizon=10, purpose="critic"):
        eligible = [ep for ep in self.episodes if len(ep["actions"]) >= int(length)]
        if not eligible: raise RuntimeError("No valid Stage1 sequence")
        ids = self.rng.integers(len(eligible), size=int(count)); out={key:[] for key in CORE+("episode_steps","terminated","truncated","dones")}
        for index in ids:
            ep=eligible[int(index)]; starts=(len(ep["actions"])-int(length))//horizon+1 if aligned else len(ep["actions"])-int(length)+1
            start=int(self.rng.integers(starts))*(horizon if aligned else 1)
            for key in out: out[key].append(ep[key][start:start+int(length)])
        self.samples_drawn += int(count)
        self.samples_by_purpose[purpose] = self.samples_by_purpose.get(purpose, 0) + int(count)
        result={key:np.stack(value) for key,value in out.items()}; result["source_id"]=np.full(int(count),self.source_id,np.int8); return result

    def sample_critic_prefixes(self, count, length, purpose="critic"):
        result = _sample_prefix_sequence_batch(self.episodes, self.rng, count, length)
        self.samples_drawn += int(count)
        self.samples_by_purpose[purpose] = self.samples_by_purpose.get(purpose, 0) + int(count)
        result["source_id"] = np.full(int(count), self.source_id, np.int8)
        return result

    def sample_actor_prefixes(self, count, length, horizon=10, purpose="actor"):
        result = _sample_aligned_prefix_batch(
            self.episodes, self.rng, count, length, horizon)
        self.samples_drawn += int(count)
        self.samples_by_purpose[purpose] = self.samples_by_purpose.get(purpose, 0) + int(count)
        result["source_id"] = np.full(int(count), self.source_id, np.int8)
        return result

    def state_dict(self): return {"rng_state":self.rng.bit_generator.state,"samples_drawn":self.samples_drawn,"samples_by_purpose":dict(self.samples_by_purpose)}
    def load_state_dict(self, value):
        self.rng.bit_generator.state=value["rng_state"]; self.samples_drawn=int(value["samples_drawn"])
        self.samples_by_purpose.update(value.get("samples_by_purpose", {}))


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


def _sample_prefix_sequence_batch(episodes, rng, count, length):
    """Preserve the old final-transition sampling distribution but return episode prefixes."""
    eligible = [episode for episode in episodes
                if len(episode["actions"]) >= int(length)]
    if not eligible:
        raise RuntimeError("No complete episode is available for recurrent sampling")
    count = int(count)
    episode_ids = rng.integers(len(eligible), size=count)
    start_counts = np.asarray(
        [len(episode["actions"]) - int(length) + 1 for episode in eligible],
        dtype=np.int64)
    starts = rng.integers(start_counts[episode_ids], size=count)
    fields = {key: [] for key in CORE + ("episode_steps",)}
    lengths = []
    for episode_id, start in zip(episode_ids, starts):
        episode = eligible[int(episode_id)]
        stop = int(start) + int(length)
        lengths.append(stop)
        for key in fields:
            fields[key].append(np.asarray(episode[key][:stop]))
    fields["sequence_lengths"] = np.asarray(lengths, dtype=np.int64)
    fields["sample_window_starts"] = np.asarray(starts, dtype=np.int64)
    return fields


def _merge_prefix_batches(left, right):
    merged = {}
    for key in CORE + ("episode_steps",):
        merged[key] = list(left[key]) + list(right[key])
    merged["sequence_lengths"] = np.concatenate(
        (left["sequence_lengths"], right["sequence_lengths"])).astype(np.int64)
    merged["sample_window_starts"] = np.concatenate(
        (left["sample_window_starts"], right["sample_window_starts"])).astype(np.int64)
    return merged


def _pad_prefix_batch(batch):
    lengths = np.asarray(batch["sequence_lengths"], dtype=np.int64)
    if len(lengths) == 0 or np.any(lengths <= 0):
        raise RuntimeError("Invalid full-prefix Critic batch")
    max_length = int(lengths.max())
    result = {
        "observations": np.zeros((len(lengths), max_length, 59), np.float32),
        "actions": np.zeros((len(lengths), max_length, 14), np.float32),
        "rewards": np.zeros((len(lengths), max_length, 1), np.float32),
        "next_observations": np.zeros((len(lengths), max_length, 59), np.float32),
        "terminals": np.zeros((len(lengths), max_length, 1), np.float32),
        "episode_steps": np.zeros((len(lengths), max_length), np.int64),
        "valid_mask": np.zeros((len(lengths), max_length), bool),
        "sequence_lengths": lengths,
        "sample_window_starts": np.asarray(batch["sample_window_starts"], dtype=np.int64),
    }
    for row, length in enumerate(lengths):
        length = int(length)
        for key in CORE + ("episode_steps",):
            result[key][row, :length] = batch[key][row]
        result["valid_mask"][row, :length] = True
        if result["episode_steps"][row, 0] != 0:
            raise RuntimeError("Full-prefix Critic replay must begin at episode step zero")
    return result


def final_transition(sequence_batch):
    """Gather each row's sampled final transition, ignoring right padding."""
    lengths = np.asarray(sequence_batch.get("sequence_lengths"), dtype=np.int64)
    if lengths.ndim != 1 or len(lengths) != len(sequence_batch["actions"]):
        raise ValueError("Full-prefix sequence_lengths missing or malformed")
    rows = np.arange(len(lengths))
    indices = lengths - 1
    return {key: value[rows, indices].copy() for key, value in sequence_batch.items()
            if key in CORE}


class OfflineDemonstrations(_OfflineDemonstrations):
    """Stage3-v5-compatible replay with batched sequence index generation."""

    def sample_sequences(self, count, length):
        return _sample_sequence_batch(self.episodes, self.rng, count, length)

    def sample_actor_prefixes(self, count, length, horizon=10, purpose="actor"):
        del purpose
        return _sample_aligned_prefix_batch(
            self.episodes, self.rng, count, length, horizon)


class BalancedOfflineDemonstrations:
    """Fixed-size 3-way offline mixture with rotating 43/43/42 remainder."""
    def __init__(self, datasets, seed):
        self.sources = [Stage1OfflineSequenceReplay(path, name, int(seed) + index)
                        for index, (name, path) in enumerate(zip(SOURCE_NAMES, datasets))]
        self.rng = np.random.default_rng(int(seed)); self._remainder_cursor = 0
        self.purpose_cursors = {"critic": 0, "actor": 0, "diagnostic": 0}
        self.episodes = sum((source.episodes for source in self.sources), [])

    def _all_episodes(self):
        return self.episodes

    def sample_sequences(self, count, length, aligned=False, horizon=10, purpose="critic"):
        base = int(count) // 3
        # For 128 this produces 43/43/42 and rotates the short source.
        counts = [base, base, base]
        cursor = self.purpose_cursors.get(purpose, 0)
        for offset in range(int(count) - 3 * base):
            counts[(cursor + offset) % 3] += 1
        self.purpose_cursors[purpose] = (cursor + 1) % 3
        self._remainder_cursor = self.purpose_cursors["critic"]
        pieces = [source.sample_sequences(n, length, aligned, horizon, purpose=purpose) for source, n in zip(self.sources, counts)]
        keys = CORE + ("episode_steps", "terminated", "truncated", "dones", "source_id")
        result = {key: np.concatenate([piece[key] for piece in pieces], axis=0) for key in keys}
        permutation = self.rng.permutation(int(count))
        return {key: value[permutation] for key, value in result.items()}

    def sample_critic_prefixes(self, count, length, purpose="critic"):
        base = int(count) // 3
        counts = [base, base, base]
        cursor = self.purpose_cursors.get(purpose, 0)
        for offset in range(int(count) - 3 * base):
            counts[(cursor + offset) % 3] += 1
        self.purpose_cursors[purpose] = (cursor + 1) % 3
        self._remainder_cursor = self.purpose_cursors["critic"]
        pieces = [source.sample_critic_prefixes(n, length, purpose=purpose)
                  for source, n in zip(self.sources, counts)]
        result = {key: sum((list(piece[key]) for piece in pieces), [])
                  for key in CORE + ("episode_steps",)}
        for key in ("sequence_lengths", "sample_window_starts", "source_id"):
            result[key] = np.concatenate([piece[key] for piece in pieces], axis=0)
        permutation = self.rng.permutation(int(count))
        for key in CORE + ("episode_steps",):
            result[key] = [result[key][int(i)] for i in permutation]
        for key in ("sequence_lengths", "sample_window_starts", "source_id"):
            result[key] = result[key][permutation]
        return result

    def sample_actor_prefixes(self, count, length, horizon=10, purpose="actor"):
        base = int(count) // 3
        counts = [base, base, base]
        cursor = self.purpose_cursors.get(purpose, 0)
        for offset in range(int(count) - 3 * base):
            counts[(cursor + offset) % 3] += 1
        self.purpose_cursors[purpose] = (cursor + 1) % 3
        pieces = [source.sample_actor_prefixes(n, length, horizon, purpose)
                  for source, n in zip(self.sources, counts)]
        fixed = CORE + ("episode_steps",)
        result = {key: np.concatenate([piece[key] for piece in pieces], axis=0)
                  for key in fixed}
        for key in tuple(f"critic_{name}" for name in fixed):
            result[key] = sum((list(piece[key]) for piece in pieces), [])
        for key in ("critic_sequence_lengths", "actor_window_starts", "source_id"):
            result[key] = np.concatenate([piece[key] for piece in pieces], axis=0)
        permutation = self.rng.permutation(int(count))
        for key in fixed:
            result[key] = result[key][permutation]
        for key in tuple(f"critic_{name}" for name in fixed):
            result[key] = [result[key][int(i)] for i in permutation]
        for key in ("critic_sequence_lengths", "actor_window_starts", "source_id"):
            result[key] = result[key][permutation]
        return result

    def state_dict(self): return {"rotation_index":self._remainder_cursor,"purpose_cursors":dict(self.purpose_cursors),"rng_state":self.rng.bit_generator.state,"sources":[x.state_dict() for x in self.sources]}
    def load_state_dict(self, value):
        self._remainder_cursor=int(value["rotation_index"]); self.rng.bit_generator.state=value["rng_state"]
        self.purpose_cursors.update(value.get("purpose_cursors", {"critic":self._remainder_cursor}))
        for sampler,state in zip(self.sources,value["sources"]): sampler.load_state_dict(state)


class OnlineSequenceReplay(_OnlineSequenceReplay):
    """Stage3-v5 online replay with batched sequence index generation."""

    def __init__(self, capacity_transitions, seed=0):
        super().__init__(capacity_transitions, seed)
        self.diagnostic_rng = np.random.default_rng(int(seed) + 90173)
        self.diagnostic_reservoir = {True: [], False: []}
        self.diagnostic_seen = {True: 0, False: 0}
        self.fixed_critic_diagnostic_set = None

    def add_episode(self, episode):
        super().add_episode(episode)
        label = bool(episode.get("success", False))
        self.diagnostic_seen[label] += 1
        reservoir = self.diagnostic_reservoir[label]
        slot = (len(reservoir) if len(reservoir) < 64 else
                int(self.diagnostic_rng.integers(self.diagnostic_seen[label])))
        if slot < 64:
            saved = {key: np.asarray(value).copy() for key, value in episode.items()}
            saved["diagnostic_episode_id"] = sum(self.diagnostic_seen.values()) - 1
            if slot == len(reservoir):
                reservoir.append(saved)
            else:
                reservoir[slot] = saved

    def add(self, env_id, observation, action, reward, next_observation, terminal,
            episode_step, terminated=None, truncated=None):
        """Append a transition while retaining Gym termination semantics."""
        terminated_value = bool(terminal if terminated is None else terminated)
        truncated_value = bool(False if truncated is None else truncated)
        terminal_for_td = terminated_value or truncated_value
        super().add(env_id, observation, action, reward, next_observation,
                    terminal_for_td, episode_step)
        episode = self.current[int(env_id)]
        episode.setdefault("terminated", []).append(
            np.asarray([terminated_value], bool))
        episode.setdefault("truncated", []).append(
            np.asarray([bool(truncated_value)], bool))
        episode.setdefault("dones", []).append(
            np.asarray([terminal_for_td], bool))

    def sample_sequences(self, count, length):
        return _sample_sequence_batch(
            list(self.episodes) + list(self.current.values()),
            self.rng, count, length,
        )

    def sample_critic_prefixes(self, count, length):
        return _sample_prefix_sequence_batch(
            list(self.episodes) + list(self.current.values()),
            self.rng, count, length)

    def sample_actor_prefixes(self, count, length, horizon=10):
        return _sample_aligned_prefix_batch(
            list(self.episodes) + list(self.current.values()),
            self.rng, count, length, horizon)

    def finish(self, env_id, success=False):
        episode = self.current.pop(int(env_id), None)
        if episode:
            saved = {key: np.asarray(value) for key, value in episode.items()}
            saved["success"] = bool(success)
            self.add_episode(saved)

    def save(self, path):
        np.save(path, {"capacity": self.capacity, "transitions": self.transitions,
                       "rng_state": self.rng.bit_generator.state,
                       "episodes": list(self.episodes), "current": self.current,
                       "diagnostic_rng_state": self.diagnostic_rng.bit_generator.state,
                       "diagnostic_reservoir": self.diagnostic_reservoir,
                       "diagnostic_seen": self.diagnostic_seen,
                       "fixed_critic_diagnostic_set": self.fixed_critic_diagnostic_set},
                allow_pickle=True)

    @classmethod
    def load(cls, path):
        payload = np.load(path, allow_pickle=True).item()
        replay = cls(payload["capacity"])
        replay.transitions = int(payload["transitions"])
        from collections import deque
        replay.episodes = deque(payload["episodes"])
        replay.current = payload.get("current", {})
        replay.rng.bit_generator.state = payload["rng_state"]
        replay.diagnostic_rng.bit_generator.state = payload.get("diagnostic_rng_state", replay.diagnostic_rng.bit_generator.state)
        replay.diagnostic_reservoir = payload.get("diagnostic_reservoir", {True: [], False: []})
        replay.diagnostic_seen = payload.get("diagnostic_seen", {True: 0, False: 0})
        replay.fixed_critic_diagnostic_set = payload.get("fixed_critic_diagnostic_set")
        if "diagnostic_reservoir" not in payload:
            for episode in replay.episodes:
                label = bool(episode.get("success", False))
                replay.diagnostic_seen[label] += 1
                if len(replay.diagnostic_reservoir[label]) < 64:
                    replay.diagnostic_reservoir[label].append(episode)
        return replay


def _sample_aligned_prefix_batch(episodes, rng, count, length, horizon):
    """Sample Actor windows while retaining full Critic prefixes for those timesteps."""
    eligible = [episode for episode in episodes if len(episode["actions"]) >= int(length)]
    if not eligible:
        raise RuntimeError("No complete boundary-aligned Actor window available")
    count, length, horizon = int(count), int(length), int(horizon)
    episode_ids = rng.integers(len(eligible), size=count)
    window_counts = np.asarray(
        [(len(episode["actions"]) - length) // horizon + 1
         for episode in eligible], dtype=np.int64)
    starts = rng.integers(window_counts[episode_ids], size=count) * horizon
    actor = {key: [] for key in CORE + ("episode_steps",)}
    prefix = {key: [] for key in CORE + ("episode_steps",)}
    lengths = []
    for episode_id, start in zip(episode_ids, starts):
        episode = eligible[int(episode_id)]
        start, stop = int(start), int(start) + length
        for key in actor:
            actor[key].append(np.asarray(episode[key][start:stop]))
            prefix[key].append(np.asarray(episode[key][:stop]))
        lengths.append(stop)
    result = {key: np.stack(value) for key, value in actor.items()}
    result.update({f"critic_{key}": value for key, value in prefix.items()})
    result["critic_sequence_lengths"] = np.asarray(lengths, dtype=np.int64)
    result["actor_window_starts"] = np.asarray(starts, dtype=np.int64)
    return result


def _pad_actor_prefix_fields(batch):
    lengths = np.asarray(batch["critic_sequence_lengths"], dtype=np.int64)
    max_length = int(lengths.max())
    result = {
        "critic_observations": np.zeros((len(lengths), max_length, 59), np.float32),
        "critic_actions": np.zeros((len(lengths), max_length, 14), np.float32),
        "critic_episode_steps": np.zeros((len(lengths), max_length), np.int64),
        "critic_sequence_lengths": lengths,
        "actor_window_starts": np.asarray(batch["actor_window_starts"], dtype=np.int64),
    }
    for row, length in enumerate(lengths):
        length = int(length)
        result["critic_observations"][row, :length] = batch["critic_observations"][row]
        result["critic_actions"][row, :length] = batch["critic_actions"][row]
        result["critic_episode_steps"][row, :length] = batch["critic_episode_steps"][row]
    return result


def _sample_aligned(source, count, length, horizon, purpose="actor"):
    if isinstance(source, (Stage1OfflineSequenceReplay, BalancedOfflineDemonstrations)):
        return source.sample_sequences(count, length, aligned=True, horizon=horizon, purpose=purpose)
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


def aligned_sequence_batch(offline, online, count, length, horizon=10, profiler=None):
    if int(count) <= 0 or int(count) % 2 or int(length) != int(horizon):
        raise ValueError("Aligned Actor batch requires even count and one RNN horizon")
    half = int(count) // 2
    with profiler.measure("actor_replay_sample_ms") if profiler else nullcontext():
        left = offline.sample_actor_prefixes(
            half, int(length), int(horizon), purpose="actor")
        right = online.sample_actor_prefixes(
            half, int(length), int(horizon))
    fixed = CORE + ("episode_steps",)
    batch = {key: np.concatenate((left[key], right[key]), axis=0)
             for key in fixed}
    prefix = {}
    for key in tuple(f"critic_{name}" for name in fixed):
        prefix[key] = list(left[key]) + list(right[key])
    prefix["critic_sequence_lengths"] = np.concatenate(
        (left["critic_sequence_lengths"], right["critic_sequence_lengths"]))
    prefix["actor_window_starts"] = np.concatenate(
        (left["actor_window_starts"], right["actor_window_starts"]))
    batch.update(_pad_actor_prefix_fields(prefix))
    batch["is_offline"] = np.concatenate(
        (np.ones(half, np.float32), np.zeros(half, np.float32)))
    if not np.all(batch["episode_steps"][:, 0] % horizon == 0):
        raise RuntimeError("Actor batch starts outside a hidden-state reset boundary")
    return batch

def symmetric_sequence_batch(offline, online, count, length, profiler=None):
    if int(count) != 256:
        raise ValueError("V5 Critic batch must remain 128 offline + 128 online")
    with profiler.measure("offline_replay_sample_ms") if profiler else nullcontext():
        left = offline.sample_critic_prefixes(128, length, purpose="critic")
    with profiler.measure("online_replay_sample_ms") if profiler else nullcontext():
        right = online.sample_critic_prefixes(128, length)
    with profiler.measure("batch_prepare_ms") if profiler else nullcontext():
        result = _pad_prefix_batch(_merge_prefix_batches(left, right))
        result["is_offline"] = np.r_[np.ones(128, np.float32), np.zeros(128, np.float32)]
    return result


def source_sample_metrics(offline):
    sources = offline.sources if hasattr(offline, "sources") else [offline]
    return {f"{purpose}_offline_{name}_samples": sum(
                source.samples_by_purpose.get(purpose, 0)
                for source in sources if source.source == name)
            for purpose in ("critic", "actor") for name in SOURCE_NAMES}
