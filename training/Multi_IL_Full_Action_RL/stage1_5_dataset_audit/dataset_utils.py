#!/usr/bin/env python3
"""Read-only discovery and streaming analysis helpers for Stage 1.5."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np


POLICIES = ("bc_rnn", "bc_transformer", "bc_gmm")
CANDIDATE_SUFFIXES = {".hdf5", ".h5", ".json", ".jsonl", ".npz", ".pkl", ".pickle"}
IGNORED_SUFFIXES = {".pth", ".pt", ".ckpt", ".mp4", ".avi", ".mov", ".mkv", ".log", ".out"}
IGNORED_PATH_TERMS = {
    "checkpoint", "checkpoints", "tensorboard", "tb", "video", "videos",
    "wandb", "models", "model", "launcher_logs", "logs",
}

FIELD_ALIASES = {
    "obs": ("obs", "observations", "observation"),
    "next_obs": ("next_obs", "next_observations", "next_observation"),
    "actions": ("actions", "action"),
    "rewards": ("rewards", "reward"),
    "dones": ("dones", "done"),
    "terminated": ("terminated", "terminals", "terminal"),
    "truncated": ("truncated", "timeouts", "timeout"),
    "success": ("episode_success", "success", "task_success"),
    "seed": ("initial_seed", "env_seed", "environment_seed", "seed"),
    "episode_id": ("episode_id", "trajectory_id", "traj_id"),
    "policy_id": ("policy_id", "policy", "policy_name"),
}


def utc_timestamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).astimezone().isoformat()


def require_output_outside_readonly_root(path, readonly_root):
    """Reject output paths inside the Stage 1 artifact tree."""
    output = Path(path).resolve()
    root = Path(readonly_root).resolve()
    try:
        inside = os.path.commonpath((str(root), str(output))) == str(root)
    except ValueError:
        inside = False
    if inside:
        raise RuntimeError(
            f"Refusing to write Stage 1.5 output inside read-only training_runs: {output}"
        )
    return output


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def decode_json_attribute(value):
    value = json_scalar(value)
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return value


def normalize_policy(value):
    if value is None:
        return "unknown"
    text = str(json_scalar(value)).lower().replace("-", "_").replace(" ", "_")
    if "transformer" in text:
        return "bc_transformer"
    if "rnn" in text or "lstm" in text:
        return "bc_rnn"
    if "gmm" in text and "rnn" not in text and "transformer" not in text:
        return "bc_gmm"
    return "unknown"


def path_policy(path):
    return normalize_policy("/".join(part.lower() for part in Path(path).parts))


def probable_run_id(path):
    path = Path(path)
    for parent in (path.parent, *path.parents):
        if (parent / "collection_summary.json").is_file() or (parent / "run_manifest.json").is_file():
            return parent.name
    for part in reversed(path.parts):
        digits = "".join(character for character in part if character.isdigit())
        if len(digits) >= 12:
            return part
    return None


def ignored_candidate(path):
    path = Path(path)
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return True
    lowered_parts = {part.lower() for part in path.parts}
    return bool(lowered_parts & IGNORED_PATH_TERMS)


def dataset_descriptor(name, dataset):
    return {
        "path": name,
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
        "chunks": None if dataset.chunks is None else list(dataset.chunks),
        "compression": dataset.compression,
        "size": int(dataset.size),
    }


def hdf5_schema(handle):
    groups, datasets = [], []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group):
            groups.append({
                "path": name,
                "attributes": {key: json_scalar(value) for key, value in obj.attrs.items()},
            })
        elif isinstance(obj, h5py.Dataset):
            datasets.append(dataset_descriptor(name, obj))

    handle.visititems(visitor)
    return {
        "root_attributes": {key: json_scalar(value) for key, value in handle.attrs.items()},
        "groups": groups,
        "datasets": datasets,
    }


def _first_existing(container, aliases):
    for name in aliases:
        if name in container:
            return name
    return None


def detect_hdf5_layout(handle):
    if "episodes" in handle and isinstance(handle["episodes"], h5py.Group):
        groups = [f"episodes/{name}" for name in sorted(handle["episodes"].keys())]
        return "episodes_group", groups
    if "data" in handle and isinstance(handle["data"], h5py.Group):
        groups = [
            f"data/{name}" for name in sorted(handle["data"].keys())
            if isinstance(handle[f"data/{name}"], h5py.Group)
        ]
        if groups:
            return "robomimic_data_group", groups
    return "root_arrays", [""]


def resolve_episode_fields(group):
    fields = {}
    for concept, aliases in FIELD_ALIASES.items():
        fields[concept] = _first_existing(group, aliases)
    return fields


def transition_evidence(group, fields):
    evidence = {
        "state": fields["obs"] is not None,
        "action": fields["actions"] is not None,
        "reward": fields["rewards"] is not None,
        "next_state": fields["next_obs"] is not None,
        "done": any(fields[name] is not None for name in ("dones", "terminated", "truncated")),
    }
    missing = [name for name, present in evidence.items() if not present]
    return evidence, missing


def _read_constant(dataset):
    if dataset.shape == ():
        return json_scalar(dataset[()])
    if dataset.shape[0] == 0:
        return None
    return json_scalar(dataset[0])


def _episode_seed(group, fields):
    name = fields.get("seed")
    if name is not None and isinstance(group[name], h5py.Dataset):
        value = _read_constant(group[name])
        return None if value is None else int(value)
    for key in FIELD_ALIASES["seed"]:
        if key in group.attrs:
            return int(json_scalar(group.attrs[key]))
    return None


def _episode_length(group, fields):
    name = fields.get("actions")
    if name is not None and isinstance(group[name], h5py.Dataset) and group[name].shape:
        return int(group[name].shape[0])
    if "num_samples" in group.attrs:
        return int(group.attrs["num_samples"])
    return None


def _policy_from_hdf5(handle, episode_groups):
    for key in FIELD_ALIASES["policy_id"]:
        if key in handle.attrs:
            policy = normalize_policy(handle.attrs[key])
            if policy != "unknown":
                return policy, f"root_attribute:{key}"
    for group_path in episode_groups[:1]:
        group = handle if not group_path else handle[group_path]
        fields = resolve_episode_fields(group)
        name = fields.get("policy_id")
        if name is not None and isinstance(group[name], h5py.Dataset):
            policy = normalize_policy(_read_constant(group[name]))
            if policy != "unknown":
                return policy, f"dataset:{group_path}/{name}"
    return "unknown", None


def inspect_hdf5(path, include_schema=True):
    result = {
        "format": "hdf5",
        "content_validated": False,
        "is_rollout_dataset": False,
        "missing_fields": [],
    }
    with h5py.File(path, "r") as handle:
        layout, episode_groups = detect_hdf5_layout(handle)
        result["layout"] = layout
        if include_schema:
            result["schema"] = hdf5_schema(handle)
        policy, source = _policy_from_hdf5(handle, episode_groups)
        result["content_policy"] = policy
        result["policy_evidence"] = source
        episodes, transitions, seeds = 0, 0, []
        duplicate_seed_counter = Counter()
        consistent_evidence = None
        missing_union = set()
        for group_path in episode_groups:
            group = handle if not group_path else handle[group_path]
            fields = resolve_episode_fields(group)
            evidence, missing = transition_evidence(group, fields)
            if consistent_evidence is None:
                consistent_evidence = evidence
            else:
                consistent_evidence = {
                    key: consistent_evidence[key] and evidence[key] for key in evidence
                }
            missing_union.update(missing)
            length = _episode_length(group, fields)
            if length is not None:
                episodes += 1
                transitions += length
            seed = _episode_seed(group, fields)
            if seed is not None:
                seeds.append(seed)
                duplicate_seed_counter[seed] += 1
        result.update({
            "content_validated": True,
            "episodes": episodes if episodes else None,
            "transitions": transitions if episodes else None,
            "seed_count": len(seeds),
            "unique_seed_count": len(set(seeds)),
            "seeds": seeds,
            "duplicate_seeds": sorted(seed for seed, count in duplicate_seed_counter.items() if count > 1),
            "transition_evidence": consistent_evidence or {},
            "missing_fields": sorted(missing_union),
        })
        result["is_rollout_dataset"] = bool(
            episodes and consistent_evidence and all(consistent_evidence.values())
        )
        if layout == "root_arrays" and not result["is_rollout_dataset"]:
            fields = resolve_episode_fields(handle)
            evidence, missing = transition_evidence(handle, fields)
            length = _episode_length(handle, fields)
            result.update({
                "episodes": 1 if length else None,
                "transitions": length,
                "transition_evidence": evidence,
                "missing_fields": missing,
                "is_rollout_dataset": bool(length and all(evidence.values())),
            })
    return result


def _flatten_mapping_keys(value, prefix="", depth=0, limit=5000):
    keys = []
    if depth > 5 or limit <= 0:
        return keys
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            keys.append(name)
            if len(keys) >= limit:
                break
            keys.extend(_flatten_mapping_keys(child, name, depth + 1, limit - len(keys)))
    elif isinstance(value, list) and value:
        keys.extend(_flatten_mapping_keys(value[0], f"{prefix}[0]", depth + 1, limit))
    return keys[:limit]


def inspect_json_file(path):
    result = {
        "format": "jsonl" if path.suffix.lower() == ".jsonl" else "json",
        "content_validated": False,
        "is_rollout_dataset": False,
        "missing_fields": [],
    }
    if path.stat().st_size > 128 * 1024 * 1024:
        result["error"] = "JSON file exceeds 128 MiB safe inspection limit"
        return result
    if path.suffix.lower() == ".jsonl":
        rows = []
        total = 0
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                total += 1
                if len(rows) < 100:
                    rows.append(json.loads(line))
        payload = rows
        result["records"] = total
    else:
        payload = read_json(path)
    keys = _flatten_mapping_keys(payload)
    lowered = {key.rsplit("/", 1)[-1].replace("[0]", "").lower() for key in keys}
    evidence = {
        concept: any(alias in lowered for alias in aliases)
        for concept, aliases in {
            "state": FIELD_ALIASES["obs"],
            "action": FIELD_ALIASES["actions"],
            "reward": FIELD_ALIASES["rewards"],
            "next_state": FIELD_ALIASES["next_obs"],
            "done": FIELD_ALIASES["dones"] + FIELD_ALIASES["terminated"] + FIELD_ALIASES["truncated"],
        }.items()
    }
    result.update({
        "content_validated": True,
        "top_level_type": type(payload).__name__,
        "observed_keys": keys,
        "transition_evidence": evidence,
        "missing_fields": [key for key, present in evidence.items() if not present],
        "is_rollout_dataset": all(evidence.values()),
    })
    return result


def inspect_npz(path):
    result = {
        "format": "npz",
        "content_validated": False,
        "is_rollout_dataset": False,
    }
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: {"shape": list(archive[key].shape), "dtype": str(archive[key].dtype)}
            for key in archive.files
        }
    lowered = {key.lower(): key for key in arrays}
    evidence = {
        "state": any(alias in lowered for alias in FIELD_ALIASES["obs"]),
        "action": any(alias in lowered for alias in FIELD_ALIASES["actions"]),
        "reward": any(alias in lowered for alias in FIELD_ALIASES["rewards"]),
        "next_state": any(alias in lowered for alias in FIELD_ALIASES["next_obs"]),
        "done": any(alias in lowered for alias in (
            FIELD_ALIASES["dones"] + FIELD_ALIASES["terminated"] + FIELD_ALIASES["truncated"]
        )),
    }
    result.update({
        "content_validated": True,
        "arrays": arrays,
        "transition_evidence": evidence,
        "missing_fields": [key for key, present in evidence.items() if not present],
        "is_rollout_dataset": all(evidence.values()),
    })
    return result


def inspect_candidate(path, include_schema=True):
    path = Path(path).resolve()
    stat = path.stat()
    base = {
        "path": str(path),
        "parent_directory": str(path.parent),
        "file_size_bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "suffix": path.suffix.lower(),
        "path_policy": path_policy(path),
        "possible_run_id": probable_run_id(path),
        "ignored": False,
    }
    try:
        if path.suffix.lower() in {".hdf5", ".h5"}:
            detail = inspect_hdf5(path, include_schema=include_schema)
        elif path.suffix.lower() in {".json", ".jsonl"}:
            detail = inspect_json_file(path)
        elif path.suffix.lower() == ".npz":
            detail = inspect_npz(path)
        else:
            detail = {
                "format": "pickle",
                "content_validated": False,
                "is_rollout_dataset": False,
                "error": "Pickle not deserialized during discovery because loading can execute code",
            }
    except Exception as exception:
        detail = {
            "format": path.suffix.lower().lstrip("."),
            "content_validated": False,
            "is_rollout_dataset": False,
            "error": f"{type(exception).__name__}: {exception}",
        }
    result = {**base, **detail}
    content_policy = result.get("content_policy", "unknown")
    if content_policy != "unknown":
        result["possible_policy_identity"] = content_policy
        result["policy_confidence"] = "content"
    elif base["path_policy"] != "unknown":
        result["possible_policy_identity"] = base["path_policy"]
        result["policy_confidence"] = "path_only"
    else:
        result["possible_policy_identity"] = "unknown"
        result["policy_confidence"] = "unknown"
    return result


def scan_candidates(training_runs_root, include_schema=True):
    root = Path(training_runs_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    results = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in CANDIDATE_SUFFIXES:
            continue
        if ignored_candidate(path):
            continue
        results.append(inspect_candidate(path, include_schema=include_schema))
    return results


def dataset_family(path):
    path = Path(path)
    if path.parent.name.lower() in POLICIES:
        return str(path.parent.parent.resolve())
    return str(path.parent.resolve())


def select_formal_datasets(candidates, expected_episodes=100):
    eligible = [
        row for row in candidates
        if row.get("format") == "hdf5"
        and row.get("is_rollout_dataset") is True
        and row.get("possible_policy_identity") in POLICIES
        and row.get("policy_confidence") == "content"
    ]
    families = defaultdict(dict)
    conflicts = defaultdict(lambda: defaultdict(list))
    for row in eligible:
        family = dataset_family(row["path"])
        policy = row["possible_policy_identity"]
        conflicts[family][policy].append(row)
        current = families[family].get(policy)
        if current is None or (
            int(row.get("episodes") or -1), int(row.get("unique_seed_count") or -1)
        ) > (
            int(current.get("episodes") or -1), int(current.get("unique_seed_count") or -1)
        ):
            families[family][policy] = row

    evaluated = []
    for family, policy_rows in families.items():
        if set(policy_rows) != set(POLICIES):
            continue
        seed_sets = {policy: set(row.get("seeds") or []) for policy, row in policy_rows.items()}
        exact_counts = all(int(row.get("episodes") or -1) == expected_episodes for row in policy_rows.values())
        exact_unique = all(int(row.get("unique_seed_count") or -1) == expected_episodes for row in policy_rows.values())
        same_seeds = len({tuple(sorted(items)) for items in seed_sets.values()}) == 1
        duplicate_free = all(not row.get("duplicate_seeds") for row in policy_rows.values())
        ambiguous_within_family = any(len(conflicts[family][policy]) > 1 for policy in POLICIES)
        score = (
            1000 * int(exact_counts) +
            1000 * int(exact_unique) +
            1000 * int(same_seeds) +
            500 * int(duplicate_free) +
            100 * int((Path(family) / "collection_summary.json").is_file()) -
            500 * int("worker_shards" in Path(family).parts) -
            100 * int(ambiguous_within_family)
        )
        evaluated.append({
            "family": family,
            "score": score,
            "exact_episode_count": exact_counts,
            "exact_unique_seed_count": exact_unique,
            "same_seed_sets": same_seeds,
            "duplicate_free": duplicate_free,
            "ambiguous_within_family": ambiguous_within_family,
            "datasets": {policy: row["path"] for policy, row in policy_rows.items()},
            "seed_count": len(next(iter(seed_sets.values()))) if same_seeds else None,
        })
    evaluated.sort(key=lambda row: (row["score"], row["family"]), reverse=True)
    formally_valid = [
        row for row in evaluated
        if row["exact_episode_count"] and row["exact_unique_seed_count"]
        and row["same_seed_sets"] and row["duplicate_free"]
        and not row["ambiguous_within_family"]
    ]
    if not formally_valid:
        status, selected, reason = "incomplete", None, "No unambiguous three-policy 100-seed family"
    elif len(formally_valid) == 1:
        status, selected, reason = "selected", formally_valid[0], "Unique validated 100-seed family"
    elif formally_valid[0]["score"] > formally_valid[1]["score"]:
        status, selected, reason = "selected", formally_valid[0], "Unique highest-scoring validated family"
    else:
        status, selected, reason = "ambiguous", None, "Multiple equally valid 100-seed families"
    return {
        "status": status,
        "reason": reason,
        "expected_episodes_per_policy": expected_episodes,
        "selected_family": None if selected is None else selected["family"],
        "datasets": {} if selected is None else selected["datasets"],
        "candidate_families": evaluated,
    }


def format_schema_text(candidates):
    lines = []
    for row in candidates:
        lines.append("=" * 100)
        lines.append(f"PATH: {row['path']}")
        lines.append(f"FORMAT: {row.get('format')}  ROLLOUT: {row.get('is_rollout_dataset')}")
        lines.append(f"POLICY: {row.get('possible_policy_identity')} ({row.get('policy_confidence')})")
        if row.get("error"):
            lines.append(f"ERROR: {row['error']}")
        schema = row.get("schema")
        if not schema:
            for key in ("observed_keys", "arrays", "transition_evidence", "missing_fields"):
                if key in row:
                    lines.append(f"{key}: {json.dumps(row[key], ensure_ascii=False, sort_keys=True)}")
            continue
        lines.append("ROOT ATTRIBUTES:")
        for key, value in sorted(schema.get("root_attributes", {}).items()):
            lines.append(f"  @{key} = {value!r}")
        lines.append("GROUPS:")
        for group in schema.get("groups", []):
            lines.append(f"  /{group['path']}")
            for key, value in sorted(group.get("attributes", {}).items()):
                lines.append(f"    @{key} = {value!r}")
        lines.append("DATASETS:")
        for dataset in schema.get("datasets", []):
            lines.append(
                f"  /{dataset['path']} shape={tuple(dataset['shape'])} "
                f"dtype={dataset['dtype']} compression={dataset['compression']}"
            )
    return "\n".join(lines) + "\n"


def iter_dataset_chunks(dataset, max_elements=1_000_000):
    if dataset.shape == ():
        yield np.asarray([dataset[()]])
        return
    if dataset.shape[0] == 0:
        return
    per_row = int(np.prod(dataset.shape[1:])) if len(dataset.shape) > 1 else 1
    rows = max(1, int(max_elements // max(per_row, 1)))
    for start in range(0, dataset.shape[0], rows):
        yield np.asarray(dataset[start:min(dataset.shape[0], start + rows)])


def numeric_counts(dataset):
    nan_count = inf_count = 0
    for values in iter_dataset_chunks(dataset):
        if np.issubdtype(values.dtype, np.number):
            nan_count += int(np.isnan(values).sum())
            inf_count += int(np.isinf(values).sum())
    return nan_count, inf_count


class HDF5RolloutReader:
    """Strict read-only adapter for an inspected episode-oriented HDF5 dataset."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.handle = h5py.File(self.path, "r")
        self.layout, self.group_paths = detect_hdf5_layout(self.handle)
        if self.layout not in {"episodes_group", "robomimic_data_group"}:
            self.close()
            raise RuntimeError(f"Unsupported analysis layout {self.layout!r}: {self.path}")
        self.policy, self.policy_evidence = _policy_from_hdf5(self.handle, self.group_paths)
        if self.policy == "unknown":
            self.close()
            raise RuntimeError(f"Policy identity is not recoverable from HDF5 content: {self.path}")

    def close(self):
        if getattr(self, "handle", None) is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def episodes(self):
        for group_path in self.group_paths:
            group = self.handle[group_path]
            fields = resolve_episode_fields(group)
            evidence, missing = transition_evidence(group, fields)
            if missing:
                raise RuntimeError(f"{self.path}:{group_path} missing_field={missing}")
            yield group_path, group, fields

    def progress_mapping(self):
        if "progress_observation_schema" in self.handle.attrs:
            schema = decode_json_attribute(self.handle.attrs["progress_observation_schema"])
            if isinstance(schema, dict) and schema.get("fields"):
                mapping = {}
                for name in ("payload_in_target_bin", "trash_in_trash_bin"):
                    field = schema["fields"].get(name)
                    if field is not None and field.get("canonical_key") and "flat_index" in field:
                        mapping[name] = {
                            "mode": "embedded",
                            "observation_key": str(field["canonical_key"]),
                            "flat_index": int(field["flat_index"]),
                            "evidence": "HDF5 root progress_observation_schema",
                        }
                if len(mapping) == 2:
                    return mapping
        if not self.group_paths:
            return None
        group = self.handle[self.group_paths[0]]
        fields = resolve_episode_fields(group)
        obs_name = fields["obs"]
        next_name = fields["next_obs"]
        if obs_name and next_name and isinstance(group[obs_name], h5py.Group):
            names = ("payload_in_target_bin", "trash_in_trash_bin")
            if all(name in group[obs_name] and name in group[next_name] for name in names):
                return {
                    name: {
                        "mode": "named",
                        "observation_key": name,
                        "evidence": "named observation datasets",
                    }
                    for name in names
                }
        return None


def observation_schema(group, fields):
    result = {"obs": {}, "next_obs": {}}
    for concept in ("obs", "next_obs"):
        name = fields[concept]
        obj = group[name]
        if isinstance(obj, h5py.Dataset):
            result[concept]["__array__"] = {
                "shape": list(obj.shape[1:]), "dtype": str(obj.dtype)
            }
        elif isinstance(obj, h5py.Group):
            for key, value in obj.items():
                if isinstance(value, h5py.Dataset):
                    result[concept][key] = {
                        "shape": list(value.shape[1:]), "dtype": str(value.dtype)
                    }
    return result


def observation_datasets(group, field_name):
    obj = group[field_name]
    if isinstance(obj, h5py.Dataset):
        return {"__array__": obj}
    return {key: value for key, value in obj.items() if isinstance(value, h5py.Dataset)}


def scalar_from_episode(group, fields, concept):
    name = fields.get(concept)
    if name is not None and isinstance(group[name], h5py.Dataset):
        value = _read_constant(group[name])
        return json_scalar(value)
    for alias in FIELD_ALIASES[concept]:
        if alias in group.attrs:
            return json_scalar(group.attrs[alias])
    return None


def final_progress(group, fields, mapping):
    if mapping is None:
        return None
    next_name = fields["next_obs"]
    next_obj = group[next_name]
    result = {}
    for field_name, descriptor in mapping.items():
        if descriptor["mode"] == "named":
            dataset = next_obj[descriptor["observation_key"]]
            value = np.asarray(dataset[-1]).reshape(-1)[0]
        else:
            if not isinstance(next_obj, h5py.Group):
                return None
            dataset = next_obj[descriptor["observation_key"]]
            value = np.asarray(dataset[-1]).reshape(-1)[descriptor["flat_index"]]
        if float(value) not in (0.0, 1.0):
            raise RuntimeError(f"Non-boolean progress value {field_name}={value}")
        result[field_name] = bool(value)
    return result
