#!/usr/bin/env python3
"""Run Stage 1 seed shards on separate NPUs and merge them into one dataset."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
COLLECTOR = Path(__file__).resolve().parent / "collect_multi_il_rollouts.py"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SCHEMA_VERSION, VALID_POLICY_IDS, atomic_json, git_commit, read_json
from collect_multi_il_rollouts import same_seed_summary
from validate_dataset import validate


def now():
    return datetime.now(timezone.utc).astimezone().isoformat()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(EXPERIMENT_ROOT / "configs/stage1_three_policies.json"))
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--npu-ids", default="0,1,2,3")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root")
    parser.add_argument("--run-root")
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--mc-gamma", type=float)
    return parser.parse_args()


def parse_npu_ids(value):
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids:
        raise ValueError("--npu-ids is empty")
    if len(ids) != len(set(ids)):
        raise ValueError("--npu-ids contains duplicates")
    return ids


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def copy_attributes(source, destination):
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def require_equal_attribute(handles, name, context):
    values = [json_value(handle.attrs[name]) for handle in handles]
    if any(value != values[0] for value in values[1:]):
        raise RuntimeError(f"{context}: shard attribute {name!r} differs")
    return values[0]


def worker_paths(worker_root, worker_index):
    worker_id = f"worker_{worker_index:02d}"
    return {
        "worker_id": worker_id,
        "data": worker_root / "datasets" / worker_id,
        "run": worker_root / "runs" / worker_id,
        "seed_file": worker_root / "seeds" / f"{worker_id}.json",
        "log": worker_root / "logs" / f"{worker_id}.log",
    }


def launch_workers(args, assignments, npu_ids, worker_root):
    processes = []
    for worker_index, (npu_id, seeds) in enumerate(zip(npu_ids, assignments)):
        paths = worker_paths(worker_root, worker_index)
        atomic_json(paths["seed_file"], {"seeds": seeds})
        command = [
            sys.executable, "-u", str(COLLECTOR),
            "--config", str(Path(args.config).resolve()),
            "--seed-list", str(paths["seed_file"]),
            "--run-id", paths["worker_id"],
            "--output-root", str(worker_root / "datasets"),
            "--run-root", str(worker_root / "runs"),
            "--device", "npu:0",
        ]
        if args.horizon is not None:
            command.extend(("--horizon", str(args.horizon)))
        if args.mc_gamma is not None:
            command.extend(("--mc-gamma", str(args.mc_gamma)))
        environment = os.environ.copy()
        environment["ASCEND_RT_VISIBLE_DEVICES"] = str(npu_id)
        environment["NPU_ID"] = str(npu_id)
        environment.setdefault("OMP_NUM_THREADS", "1")
        environment.setdefault("MKL_NUM_THREADS", "1")
        environment.setdefault("OPENBLAS_NUM_THREADS", "1")
        log_handle = paths["log"].open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append({
            "process": process,
            "log_handle": log_handle,
            "command": command,
            "npu_id": npu_id,
            "seeds": seeds,
            **paths,
        })
        print(
            f"[parallel] launched {paths['worker_id']} pid={process.pid} "
            f"physical_npu={npu_id} logical_device=npu:0 seeds={len(seeds)} "
            f"log={paths['log']}",
            flush=True,
        )
    return processes


def stop_workers(workers):
    for worker in workers:
        process = worker["process"]
        if process.poll() is None:
            process.terminate()
    for worker in workers:
        process = worker["process"]
        if process.poll() is None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def wait_workers(workers):
    try:
        pending = set(range(len(workers)))
        while pending:
            for index in tuple(pending):
                worker = workers[index]
                return_code = worker["process"].poll()
                if return_code is None:
                    continue
                pending.remove(index)
                if return_code:
                    stop_workers(workers)
                    raise RuntimeError(
                        "Parallel collection worker failure: "
                        f"worker={worker['worker_id']} return_code={return_code} log={worker['log']}"
                    )
                print(
                    f"[parallel] completed {worker['worker_id']} physical_npu={worker['npu_id']}",
                    flush=True,
                )
            if pending:
                time.sleep(1.0)
    finally:
        for worker in workers:
            worker["log_handle"].close()


def locate_seed(assignments):
    result = {}
    for worker_index, seeds in enumerate(assignments):
        for local_index, seed in enumerate(seeds):
            result[int(seed)] = (worker_index, local_index)
    return result


def merge_initial_states(final_data, final_run, worker_root, seeds, assignments):
    locations = locate_seed(assignments)
    sources = [
        h5py.File(worker_paths(worker_root, index)["data"] / "initial_states.hdf5", "r")
        for index in range(len(assignments))
    ]
    summaries = []
    try:
        for name in ("schema_version", "canonical_observation_keys", "canonical_observation_shapes"):
            require_equal_attribute(sources, name, "initial_states")
        with h5py.File(final_data / "initial_states.hdf5", "w") as destination:
            copy_attributes(sources[0], destination)
            destination.attrs["parallel_num_workers"] = len(assignments)
            seed_group = destination.create_group("seeds")
            for global_index, seed in enumerate(seeds):
                worker_index, local_index = locations[seed]
                source_group = sources[worker_index][f"seeds/episode_{local_index:06d}"]
                name = f"episode_{global_index:06d}"
                sources[worker_index].copy(source_group, seed_group, name=name)
                copied = seed_group[name]
                copied.attrs["episode_index"] = global_index
                summaries.append({
                    "episode_index": global_index,
                    "initial_seed": seed,
                    "state_hash": json_value(copied.attrs["state_hash"]),
                    "state_vector_hash": json_value(copied.attrs["state_vector_hash"]),
                    "observation_hash": json_value(copied.attrs["observation_hash"]),
                    "environment_seed_apis": json.loads(
                        json_value(copied.attrs["environment_seed_apis"])
                    ),
                    "source_worker": worker_index,
                })
            keys = json.loads(destination.attrs["canonical_observation_keys"])
            shapes = json.loads(destination.attrs["canonical_observation_shapes"])
    finally:
        for source in sources:
            source.close()
    manifest = {
        "canonical_observation_keys": keys,
        "canonical_observation_shapes": shapes,
        "states": summaries,
        "guarantee": "Each seed is reset identically across all policies inside one NPU worker.",
        "parallel_num_workers": len(assignments),
    }
    atomic_json(final_run / "initial_state_manifest.json", manifest)
    return keys, shapes


def aggregate_policy(policy_id, final_data, final_run, worker_root, seeds, assignments, npu_ids):
    locations = locate_seed(assignments)
    shard_dirs = [worker_paths(worker_root, index)["data"] / policy_id for index in range(len(assignments))]
    sources = [h5py.File(path / "transitions.hdf5", "r") for path in shard_dirs]
    shard_rows = [read_json(path / "episodes.json") for path in shard_dirs]
    shard_metadata = [read_json(path / "metadata.json") for path in shard_dirs]
    rows_by_worker = [
        {int(row["initial_seed"]): row for row in rows}
        for rows in shard_rows
    ]
    merged_rows, total_transitions = [], 0
    schema_attributes = (
        "schema_version", "policy_id", "checkpoint", "environment_metadata",
        "canonical_observation_keys", "canonical_observation_shapes", "action_shape",
        "observation_representation", "progress_observation_schema",
        "policy_input_representation", "success_and_failure_saved", "mc_return_gamma",
    )
    try:
        for name in schema_attributes:
            require_equal_attribute(sources, name, policy_id)
        output_path = final_data / policy_id / "transitions.hdf5"
        output_path.parent.mkdir(parents=True, exist_ok=False)
        with h5py.File(output_path, "w") as destination:
            copy_attributes(sources[0], destination)
            destination.attrs["parallel_num_workers"] = len(assignments)
            destination.attrs["parallel_physical_npu_ids"] = json.dumps(npu_ids)
            episodes_group = destination.create_group("episodes")
            for global_index, seed in enumerate(seeds):
                worker_index, local_index = locations[seed]
                source_name = f"episode_{local_index:06d}"
                destination_name = f"episode_{global_index:06d}"
                sources[worker_index].copy(
                    sources[worker_index][f"episodes/{source_name}"],
                    episodes_group,
                    name=destination_name,
                )
                copied = episodes_group[destination_name]
                length = int(copied["actions"].shape[0])
                del copied["episode_id"]
                copied.create_dataset("episode_id", data=np.full(length, global_index, dtype=np.int64))
                copied.attrs["episode_id"] = global_index
                row = dict(rows_by_worker[worker_index][seed])
                row["episode_id"] = global_index
                row["source_worker"] = worker_index
                row["physical_npu_id"] = npu_ids[worker_index]
                merged_rows.append(row)
                total_transitions += length
            destination.attrs["num_episodes"] = len(seeds)
            destination.attrs["total_transitions"] = total_transitions
    finally:
        for source in sources:
            source.close()

    success_count = sum(int(row["success"]) for row in merged_rows)
    metadata = dict(shard_metadata[0])
    metadata.update({
        "device": "parallel_npu_workers",
        "physical_npu_ids": npu_ids,
        "parallel_num_workers": len(assignments),
        "worker_seed_counts": [len(items) for items in assignments],
        "start_time": min(row["start_time"] for row in shard_metadata),
        "end_time": max(row["end_time"] for row in shard_metadata),
        "num_episodes": len(merged_rows),
        "success_episodes": success_count,
        "failure_episodes": len(merged_rows) - success_count,
        "success_rate": success_count / len(merged_rows),
        "total_transitions": total_transitions,
        "mean_episode_length": float(np.mean([row["episode_length"] for row in merged_rows])),
        "mean_episode_return": float(np.mean([row["episode_return"] for row in merged_rows])),
        "success_filter_applied": False,
    })
    atomic_json(final_data / policy_id / "episodes.json", merged_rows)
    atomic_json(final_data / policy_id / "metadata.json", metadata)
    atomic_json(final_run / f"{policy_id}_metadata.json", metadata)
    return merged_rows, metadata


def aggregate(args, config, run_id, final_data, final_run, worker_root, seeds, assignments, npu_ids, started):
    final_data.mkdir(parents=True, exist_ok=False)
    atomic_json(final_data / "seed_list.json", {"seeds": seeds})
    atomic_json(final_run / "seed_list.json", {"seeds": seeds})
    canonical_keys, canonical_shapes = merge_initial_states(
        final_data, final_run, worker_root, seeds, assignments
    )
    policy_episodes, policy_metadata = {}, {}
    for policy_id in VALID_POLICY_IDS:
        rows, metadata = aggregate_policy(
            policy_id, final_data, final_run, worker_root, seeds, assignments, npu_ids
        )
        policy_episodes[policy_id] = rows
        policy_metadata[policy_id] = metadata
    outcome_rows, patterns = same_seed_summary(policy_episodes, seeds)
    atomic_json(final_data / "same_seed_outcomes.json", outcome_rows)
    first_worker_summary = read_json(worker_paths(worker_root, 0)["data"] / "collection_summary.json")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "start_time": started,
        "end_time": now(),
        "num_seeds": len(seeds),
        "num_policies": len(VALID_POLICY_IDS),
        "total_episodes": len(seeds) * len(VALID_POLICY_IDS),
        "total_transitions": sum(row["total_transitions"] for row in policy_metadata.values()),
        "environment_metadata": first_worker_summary["environment_metadata"],
        "canonical_observation_keys": canonical_keys,
        "canonical_observation_shapes": canonical_shapes,
        "action_shape": first_worker_summary["action_shape"],
        "same_seed_success_patterns": patterns,
        "policies": policy_metadata,
        "runtime_errors": 0,
        "parallel_collection": {
            "num_workers": len(assignments),
            "physical_npu_ids": npu_ids,
            "worker_seed_counts": [len(items) for items in assignments],
            "partition": "round_robin_by_seed",
            "worker_shards": str(worker_root),
        },
    }
    atomic_json(final_data / "collection_summary.json", summary)
    atomic_json(final_run / "collection_summary.json", summary)
    return summary


def cleanup_worker_shards(worker_root):
    """Remove intermediate HDF5/JSON shards after the merged dataset is complete.

    Worker logs and seed assignments remain under the run directory for auditability;
    only duplicated intermediate datasets and metadata are removed.
    """
    for name in ("datasets", "runs"):
        path = worker_root / name
        if path.exists():
            shutil.rmtree(path)


def main():
    args = parse_args()
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    npu_ids = parse_npu_ids(args.npu_ids)
    if args.num_episodes < len(npu_ids):
        raise ValueError("Number of episodes must be at least the number of NPU workers")
    config = read_json(args.config)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root or config["paths"]["dataset_root"])
    run_root = Path(args.run_root or config["paths"]["run_root"])
    final_data, final_run = output_root / run_id, run_root / run_id
    if final_data.exists() or final_run.exists():
        raise FileExistsError(f"Run already exists: data={final_data}, metadata={final_run}")
    final_run.mkdir(parents=True)
    worker_root = final_run / "worker_shards"
    for name in ("datasets", "runs", "seeds", "logs"):
        (worker_root / name).mkdir(parents=True)
    seeds = list(range(args.seed_start, args.seed_start + args.num_episodes))
    assignments = [seeds[index::len(npu_ids)] for index in range(len(npu_ids))]
    started = now()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "stage": "stage1_parallel_rollout_collection",
        "run_id": run_id,
        "start_time": started,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "command": shlex.join(sys.argv),
        "git_commit": git_commit(REPO_ROOT),
        "dataset_directory": str(final_data),
        "run_directory": str(final_run),
        "physical_npu_ids": npu_ids,
        "worker_assignments": assignments,
    }
    atomic_json(final_run / "run_manifest.json", manifest)
    workers = []

    def handle_signal(signum, _frame):
        print(f"[parallel] received signal {signum}; stopping workers", flush=True)
        stop_workers(workers)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    try:
        print(
            f"[parallel] run={run_id} workers={len(npu_ids)} physical_npus={npu_ids} "
            f"episodes_per_policy={len(seeds)} assignments={[len(x) for x in assignments]}",
            flush=True,
        )
        workers = launch_workers(args, assignments, npu_ids, worker_root)
        wait_workers(workers)
        print("[parallel] all workers completed; aggregating final dataset", flush=True)
        summary = aggregate(
            args, config, run_id, final_data, final_run, worker_root,
            seeds, assignments, npu_ids, started,
        )
        validation_report = validate(final_data)
        atomic_json(final_run / "validation_report.json", validation_report)
        cleanup_worker_shards(worker_root)
        manifest.update({"status": "complete", "end_time": summary["end_time"]})
        atomic_json(final_run / "run_manifest.json", manifest)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        print(f"PARALLEL COLLECTION COMPLETE: {final_data}", flush=True)
    except BaseException as exception:
        stop_workers(workers)
        manifest.update({
            "status": "failed",
            "end_time": now(),
            "error": f"{type(exception).__name__}: {exception}",
        })
        atomic_json(final_run / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
