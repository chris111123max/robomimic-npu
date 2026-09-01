"""Frozen BC-GMM component inference and exact backbone transfer utilities."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

CANONICAL_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
                  "robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos", "object"]
SHAPES = {"robot0_eef_pos": [3], "robot0_eef_quat": [4], "robot0_gripper_qpos": [2],
          "robot1_eef_pos": [3], "robot1_eef_quat": [4], "robot1_gripper_qpos": [2], "object": [41]}


def decode(value):
    return value.decode() if isinstance(value, bytes) else value


def seed_of(group):
    return int(group.attrs["initial_seed"] if "initial_seed" in group.attrs else group["initial_seed"][0])


def load_policy(checkpoint, device):
    import robomimic.utils.file_utils as FileUtils
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=str(checkpoint), device=device, verbose=False)
    policy.start_episode()
    if policy.obs_normalization_stats is not None:
        raise RuntimeError("Stage3C-v2 backbone transfer requires an unnormalized low-dimensional checkpoint input")
    if policy.action_normalization_stats is not None:
        raise RuntimeError("Stage3C-v2 currently requires checkpoint actions already in environment space")
    gmm = policy.policy.nets["policy"]
    gmm.eval()
    for parameter in gmm.parameters():
        parameter.requires_grad_(False)
    return policy, gmm


def linear_layers(module):
    return [child for child in module.modules() if isinstance(child, torch.nn.Linear)]


def slices(order):
    result, cursor = {}, 0
    for key in order:
        width = int(np.prod(SHAPES[key]))
        result[key] = slice(cursor, cursor + width)
        cursor += width
    return result


def transfer_backbone(gmm, actor):
    teacher_order = list(gmm.nets["encoder"].nets["obs"].obs_shapes)
    teacher = linear_layers(gmm.nets["mlp"])
    if len(teacher) != 2 or len(actor.fcs) != 2:
        raise RuntimeError("Expected exactly two teacher and Student backbone Linear layers")
    teacher_slices, student_slices = slices(teacher_order), slices(CANONICAL_KEYS)
    with torch.no_grad():
        for key in CANONICAL_KEYS:
            actor.fcs[0].weight[:, student_slices[key]].copy_(teacher[0].weight[:, teacher_slices[key]])
        actor.fcs[0].bias.copy_(teacher[0].bias)
        actor.fcs[1].weight.copy_(teacher[1].weight)
        actor.fcs[1].bias.copy_(teacher[1].bias)
    return teacher_order


def backbone_sanity(gmm, actor, canonical_states, teacher_order):
    canonical_slices = slices(CANONICAL_KEYS)
    teacher_input = torch.cat([canonical_states[:, canonical_slices[key]] for key in teacher_order], dim=1)
    teacher_layers = linear_layers(gmm.nets["mlp"])
    with torch.no_grad():
        teacher_h = F.relu(teacher_layers[1](F.relu(teacher_layers[0](teacher_input))))
        student_h = F.relu(actor.fcs[1](F.relu(actor.fcs[0](canonical_states))))
    difference = student_h - teacher_h
    return {"num_states": len(canonical_states), "mse": float(difference.square().mean().cpu()),
            "mae": float(difference.abs().mean().cpu()),
            "max_abs_error": float(difference.abs().max().cpu()),
            "teacher_order": teacher_order, "student_order": CANONICAL_KEYS,
            "fc1_input_columns_permuted_by_observation_field": teacher_order != CANONICAL_KEYS}


def build_cache(dataset_path, cache_path, policy, gmm, device, seeds, batch_size):
    rows = []
    with h5py.File(dataset_path, "r") as source:
        keys = json.loads(decode(source.attrs["canonical_observation_keys"]))
        if keys != CANONICAL_KEYS:
            raise RuntimeError(f"Unexpected canonical keys: {keys}")
        lookup = {seed_of(group): (name, group) for name, group in source["episodes"].items()}
        for seed in seeds:
            name, group = lookup[seed]
            parts = [np.asarray(group["obs"][key], np.float32).reshape(len(group["actions"]), -1) for key in keys]
            state = np.concatenate(parts, axis=1)
            action = np.asarray(group["actions"], np.float32)
            success = bool(group.attrs.get("success", group["episode_success"][0]))
            rows.append((seed, int(group.attrs.get("episode_id", name.split("_")[-1])), success, state, action))
    outputs = {key: [] for key in ("states", "saved_actions", "targets", "selected_components",
                                   "component_probabilities", "nearest_distances", "seeds",
                                   "episode_ids", "timesteps", "success")}
    per_dim_abs = np.zeros(14, np.float64)
    for seed, episode_id, success, states, actions in rows:
        offset = 0
        for start in range(0, len(states), batch_size):
            stop = min(start + batch_size, len(states))
            obs = OrderedDict()
            cursor = 0
            for key in CANONICAL_KEYS:
                width = int(np.prod(SHAPES[key]))
                obs[key] = states[start:stop, cursor:cursor + width].reshape((-1, *SHAPES[key]))
                cursor += width
            prepared = policy._prepare_observation(obs, batched_ob=True)
            with torch.no_grad():
                dist = gmm.forward_train(prepared)
                probs = dist.mixture_distribution.probs
                means = dist.component_distribution.base_dist.loc
                saved = torch.as_tensor(actions[start:stop], device=device)
                if means.shape != (len(saved), 5, 14) or saved.shape != (len(saved), 14):
                    raise RuntimeError(f"Unexpected component/action shapes: means={means.shape}, saved={saved.shape}")
                distances = torch.linalg.vector_norm(means - saved[:, None, :], dim=2)
                selected = distances.argmin(dim=1)
                index = torch.arange(len(selected), device=device)
                target = means[index, selected]
                probability = probs[index, selected]
                minimum = distances[index, selected]
            target_np = target.cpu().numpy().astype(np.float32)
            per_dim_abs += np.abs(target_np - actions[start:stop]).sum(axis=0)
            count = stop - start
            outputs["states"].append(states[start:stop]); outputs["saved_actions"].append(actions[start:stop])
            outputs["targets"].append(target_np); outputs["selected_components"].append(selected.cpu().numpy())
            outputs["component_probabilities"].append(probability.cpu().numpy())
            outputs["nearest_distances"].append(minimum.cpu().numpy())
            outputs["seeds"].append(np.full(count, seed, np.int64)); outputs["episode_ids"].append(np.full(count, episode_id, np.int64))
            outputs["timesteps"].append(np.arange(offset, offset + count, dtype=np.int64)); outputs["success"].append(np.full(count, success, np.bool_))
            offset += count
    outputs = {key: np.concatenate(value) for key, value in outputs.items()}
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(cache_path, "w") as target:
        target.attrs["canonical_observation_keys"] = json.dumps(CANONICAL_KEYS)
        target.attrs["distance_definition"] = "Euclidean L2 in 14D environment action space"
        for key, value in outputs.items():
            target.create_dataset(key, data=value, compression="gzip", compression_opts=1)
    d = outputs["nearest_distances"]
    selected = outputs["selected_components"]
    statistics = {
        "num_transitions": len(d), "mean_min_distance": float(d.mean()), "median_min_distance": float(np.median(d)),
        "p90_min_distance": float(np.quantile(d, .90)), "p95_min_distance": float(np.quantile(d, .95)),
        "p99_min_distance": float(np.quantile(d, .99)), "max_min_distance": float(d.max()),
        "selected_component_histogram": {f"mode{i}": int((selected == i).sum()) for i in range(5)},
        "selected_component_proportions": {f"mode{i}": float((selected == i).mean()) for i in range(5)},
        "per_dimension_mae": (per_dim_abs / len(d)).tolist(),
        "fraction_distance_below_1e-4": float((d < 1e-4).mean()),
        "fraction_distance_below_1e-3": float((d < 1e-3).mean()),
        "fraction_distance_below_1e-2": float((d < 1e-2).mean()),
        "action_space": "post-tanh environment action; checkpoint has no action normalization",
        "action_space_confirmed": True,
        "component_means_shape_verified": ["B", 5, 14],
        "saved_action_shape_verified": ["B", 14],
    }
    return statistics


class CachedTargets:
    def __init__(self, path, seeds):
        with h5py.File(path, "r") as handle:
            all_seeds = handle["seeds"][:]
            mask = np.isin(all_seeds, np.asarray(seeds))
            self.states = handle["states"][:][mask].astype(np.float32, copy=False)
            self.targets = handle["targets"][:][mask].astype(np.float32, copy=False)
            success = handle["success"][:][mask]
            selected_seeds = all_seeds[mask]
        unique = np.unique(selected_seeds)
        self.statistics = {"episodes": len(unique), "success_episodes": int(sum(bool(success[np.flatnonzero(selected_seeds == seed)[0]]) for seed in unique)),
                           "failure_episodes": int(len(unique) - sum(bool(success[np.flatnonzero(selected_seeds == seed)[0]]) for seed in unique)),
                           "transitions": len(self.states)}

    def __len__(self):
        return len(self.states)
