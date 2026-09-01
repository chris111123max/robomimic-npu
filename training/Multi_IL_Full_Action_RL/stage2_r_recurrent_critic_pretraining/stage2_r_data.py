"""Read-only Stage1 episode loading through pomdp-baselines SeqReplayBuffer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

POLICIES = ("bc_rnn", "bc_transformer", "bc_gmm")
CANONICAL_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                  "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos", "object"]


def decode(value):
    return value.decode() if isinstance(value, bytes) else value


def initial_seed(group):
    return int(group.attrs["initial_seed"] if "initial_seed" in group.attrs else group["initial_seed"][0])


def flatten(group, keys):
    values = [np.asarray(group[key], dtype=np.float32).reshape(len(group[key]), -1) for key in keys]
    return np.concatenate(values, axis=1)


@dataclass
class Episode:
    source: str
    seed: int
    success: bool
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    done: np.ndarray
    next_actor_action: np.ndarray | None = None

    @property
    def length(self):
        return int(self.action.shape[0])


def load_episodes(source, path, seeds):
    episodes = []
    with h5py.File(Path(path), "r") as handle:
        policy_id = str(decode(handle.attrs.get("policy_id", "")))
        if policy_id != source:
            raise RuntimeError(f"Expected {source}, HDF5 contains {policy_id}: {path}")
        keys = json.loads(decode(handle.attrs["canonical_observation_keys"]))
        if keys != CANONICAL_KEYS:
            raise RuntimeError(f"Canonical observation order mismatch in {path}: {keys}")
        lookup = {initial_seed(group): group for group in handle["episodes"].values()}
        for seed in seeds:
            if seed not in lookup:
                raise RuntimeError(f"{source} missing seed {seed}")
            group = lookup[seed]
            state = flatten(group["obs"], keys)
            next_state = flatten(group["next_obs"], keys)
            action = np.asarray(group["actions"], dtype=np.float32)
            reward = np.asarray(group["rewards"], dtype=np.float32).reshape(-1, 1)
            terminated = np.asarray(group["terminated"], dtype=np.bool_).reshape(-1, 1)
            truncated = np.asarray(group["truncated"], dtype=np.bool_).reshape(-1, 1)
            done = np.logical_or(terminated, truncated).astype(np.float32)
            length = len(action)
            if any(len(value) != length for value in (state, next_state, reward, done)):
                raise RuntimeError(f"Transition length mismatch for {source} seed {seed}")
            success = bool(group.attrs["success"] if "success" in group.attrs else group["episode_success"][0])
            episodes.append(Episode(source, seed, success, state, action, reward, next_state, done))
    return episodes


def attach_actor_actions(episodes, actor, device):
    """Run the frozen actor over each complete episode, preserving its 10-step resets."""
    import torch
    for episode in episodes:
        all_states = np.concatenate((episode.state[:1], episode.next_state), axis=0)
        states = torch.as_tensor(all_states, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            first = actor.deterministic_sequence(states)[0]
            second = actor.deterministic_sequence(states)[0]
        if not torch.equal(first, second) or not torch.isfinite(first).all():
            raise RuntimeError(f"Frozen Stage3-R actor is non-deterministic/non-finite: {episode.source} {episode.seed}")
        episode.next_actor_action = first[1:].detach().cpu().numpy().astype(np.float32)


class NativeSequenceSource:
    """Thin loader around the unmodified vendored efficient sequence buffer."""
    def __init__(self, episodes, sequence_length, sample_weight_baseline, buffer_class):
        self.episodes = episodes
        self.source = episodes[0].source
        self.sequence_length = int(sequence_length)
        capacity = sum(episode.length + 1 for episode in episodes) + 1
        # Auxiliary 14D next-policy action rides beside state in replay storage.
        # It is stripped before every critic call; Critic_RNN always receives 59D.
        self.buffer = buffer_class(capacity, 59 + 14, 14, self.sequence_length,
                                   float(sample_weight_baseline), np.float32)
        for episode in episodes:
            if episode.next_actor_action is None:
                raise RuntimeError("Actor targets must be attached before filling replay")
            current_actor = np.concatenate((np.zeros((1, 14), np.float32), episode.next_actor_action[:-1]), axis=0)
            obs = np.concatenate((episode.state, current_actor), axis=1)
            obs2 = np.concatenate((episode.next_state, episode.next_actor_action), axis=1)
            self.buffer.add_episode(obs, episode.action, episode.reward, episode.done, obs2)
        self.sampled_sequences = 0
        self.effective_timesteps = 0

    def sample(self, count):
        batch = self.buffer.random_episodes(int(count))
        self.sampled_sequences += int(count)
        self.effective_timesteps += int(batch["mask"].sum())
        return batch

    @property
    def transitions(self):
        return sum(episode.length for episode in self.episodes)


class BalancedSampler:
    def __init__(self, sources, seed):
        self.sources = sources
        self.rng = np.random.default_rng(seed)
        self.offset = 0

    def allocation(self, batch_size):
        names = list(self.sources); base, remainder = divmod(int(batch_size), len(names))
        counts = {name: base for name in names}
        for index in range(remainder):
            counts[names[(self.offset + index) % len(names)]] += 1
        self.offset = (self.offset + remainder) % len(names)
        return counts

    def sample(self, batch_size):
        counts = self.allocation(batch_size)
        pieces = [(name, self.sources[name].sample(count)) for name, count in counts.items() if count]
        merged = {key: np.concatenate([piece[key] for _, piece in pieces], axis=1) for key in pieces[0][1]}
        permutation = self.rng.permutation(int(batch_size))
        return {key: value[:, permutation] for key, value in merged.items()}, counts


def validation_sequences(episodes, sequence_length):
    """Deterministic non-overlapping sequences with padding and trajectory identity."""
    records = []
    T = int(sequence_length)
    for episode in episodes:
        current_actor = np.concatenate((np.zeros((1, 14), np.float32), episode.next_actor_action[:-1]), axis=0)
        obs_aug = np.concatenate((episode.state, current_actor), axis=1)
        obs2_aug = np.concatenate((episode.next_state, episode.next_actor_action), axis=1)
        for start in range(0, episode.length, T):
            stop = min(start + T, episode.length); valid = stop - start
            item = {}
            for key, value, width in (
                ("obs", obs_aug, 73), ("obs2", obs2_aug, 73), ("act", episode.action, 14),
                ("rew", episode.reward, 1), ("term", episode.done, 1)):
                array = np.zeros((T, width), dtype=np.float32); array[:valid] = value[start:stop]; item[key] = array
            item["mask"] = np.zeros((T, 1), np.float32); item["mask"][:valid] = 1.0
            item.update(source=episode.source, seed=episode.seed, success=episode.success, start=start, valid=valid)
            records.append(item)
    return records
