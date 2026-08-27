#!/usr/bin/env python3
"""Collect Stage 3B teacher-labelled corrective transitions on mixed-policy rollouts."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from stage3b_common import (
    atomic_json,
    close_env,
    env_success,
    extract_progress,
    flatten_canonical,
    prepare_rollout,
    read_json,
    seed_everything,
    student_action,
    teacher_action,
    validate_action,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("stage3b_config.json")))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--mode", choices=("teacher-sanity", "collect"), required=True)
    parser.add_argument("--round-id", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=None)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--worker-id", type=int, default=None)
    return parser.parse_args()


def episode_group_lookup(handle):
    result = {}
    for group in handle["episodes"].values():
        result[int(group.attrs["initial_seed"])] = group
    return result


def progress_score(success, ever_trash, ever_payload):
    if success:
        return 3
    if ever_trash and ever_payload:
        return 2
    if ever_trash or ever_payload:
        return 1
    return 0


def rollout_episode(resources, config, seed, beta, round_id, episode_id):
    teacher = resources["teacher"]
    env = resources["env"]
    student = resources["student"]
    device = resources["device"]
    keys, shapes = resources["keys"], resources["shapes"]
    progress_schema = resources["progress_schema"]
    mixture_rng = np.random.default_rng(int(config["random_seed"]) + 100000 * round_id + seed)

    seed_everything(seed)
    teacher.start_episode()
    observation = env.reset_to(copy.deepcopy(resources["initial_states"][seed]))
    restored = env.get_state()
    expected_state = np.asarray(resources["initial_states"][seed]["states"])
    if not np.array_equal(expected_state, np.asarray(restored["states"])):
        raise RuntimeError(f"reset_to state-vector mismatch for seed {seed}")
    seed_everything(seed)
    arrays = {
        "state_59d": [], "teacher_action_14d": [], "student_action_14d": [],
        "executed_action_14d": [], "executed_by_teacher": [], "episode_seed": [],
        "episode_id": [], "timestep": [], "reward": [], "terminated": [],
        "truncated": [], "trash_in_bin": [], "payload_in_bin": [],
        "beta": [], "round_id": [],
    }
    episode_return = 0.0
    success = False
    ever_trash = ever_payload = False
    final_trash = final_payload = False
    teacher_executions = 0

    for timestep in range(int(config["horizon"])):
        _, state = flatten_canonical(observation, keys, shapes)
        action_teacher = teacher_action(teacher, observation)
        action_student = student_action(student, state, device)
        execute_teacher = bool(mixture_rng.random() < float(beta))
        action_exec = action_teacher if execute_teacher else action_student
        action_exec = validate_action(action_exec, "executed")
        next_observation, reward, env_done, _ = env.step(action_exec)
        next_canonical, _ = flatten_canonical(next_observation, keys, shapes)
        progress = extract_progress(next_canonical, progress_schema)
        final_trash = progress["trash_in_trash_bin"]
        final_payload = progress["payload_in_target_bin"]
        ever_trash = ever_trash or final_trash
        ever_payload = ever_payload or final_payload
        success = success or env_success(env)
        stop_for_success = bool(config["terminate_on_success"] and success)
        horizon_reached = timestep + 1 >= int(config["horizon"])
        terminated = bool(env_done)
        truncated = bool((stop_for_success or horizon_reached) and not terminated)

        arrays["state_59d"].append(state)
        arrays["teacher_action_14d"].append(action_teacher)
        arrays["student_action_14d"].append(action_student)
        arrays["executed_action_14d"].append(action_exec)
        arrays["executed_by_teacher"].append(execute_teacher)
        arrays["episode_seed"].append(seed)
        arrays["episode_id"].append(episode_id)
        arrays["timestep"].append(timestep)
        arrays["reward"].append(float(reward))
        arrays["terminated"].append(terminated)
        arrays["truncated"].append(truncated)
        arrays["trash_in_bin"].append(final_trash)
        arrays["payload_in_bin"].append(final_payload)
        arrays["beta"].append(float(beta))
        arrays["round_id"].append(round_id)
        teacher_executions += int(execute_teacher)
        episode_return += float(reward)
        observation = next_observation
        if terminated or stop_for_success or horizon_reached:
            break

    length = len(arrays["reward"])
    arrays["success"] = [success] * length
    converted = {}
    float_vectors = {"state_59d", "teacher_action_14d", "student_action_14d", "executed_action_14d"}
    bool_fields = {"executed_by_teacher", "success", "terminated", "truncated", "trash_in_bin", "payload_in_bin"}
    int_fields = {"episode_seed", "episode_id", "timestep", "round_id"}
    for name, values in arrays.items():
        if name in float_vectors:
            converted[name] = np.asarray(values, dtype=np.float32)
        elif name in bool_fields:
            converted[name] = np.asarray(values, dtype=np.bool_)
        elif name in int_fields:
            converted[name] = np.asarray(values, dtype=np.int64)
        else:
            converted[name] = np.asarray(values, dtype=np.float32)
    if converted["state_59d"].shape != (length, 59):
        raise RuntimeError(f"Corrective state shape mismatch: {converted['state_59d'].shape}")
    for name in ("teacher_action_14d", "student_action_14d", "executed_action_14d"):
        if converted[name].shape != (length, 14) or not np.isfinite(converted[name]).all():
            raise RuntimeError(f"Corrective action invalid: {name} {converted[name].shape}")
    summary = {
        "round_id": round_id, "beta": float(beta), "seed": seed,
        "success": bool(success), "episode_return": episode_return, "episode_length": length,
        "teacher_action_count": teacher_executions,
        "student_action_count": length - teacher_executions,
        "teacher_action_fraction": teacher_executions / length,
        "student_action_fraction": (length - teacher_executions) / length,
        "trash_ever": bool(ever_trash), "payload_ever": bool(ever_payload),
        "trash_final": bool(final_trash), "payload_final": bool(final_payload),
        "progress_score": progress_score(success, ever_trash, ever_payload),
    }
    return converted, summary


def write_corrective_dataset(path, resources, config, round_id, beta, episodes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "multi_il_full_action_rl.stage3b.dagger.v1"
        handle.attrs["round_id"] = round_id
        handle.attrs["beta"] = beta
        handle.attrs["target_definition"] = "frozen BC-RNN action on actual mixed-policy history"
        handle.attrs["canonical_observation_keys"] = json.dumps(resources["keys"])
        handle.attrs["canonical_observation_shapes"] = json.dumps(resources["shapes"], sort_keys=True)
        root = handle.create_group("episodes")
        for episode_id, (arrays, summary) in enumerate(episodes):
            group = root.create_group(f"episode_{episode_id:06d}")
            for name, value in arrays.items():
                group.create_dataset(name, data=value, compression="gzip", compression_opts=1)
            for name, value in summary.items():
                group.attrs[name] = value


def run_teacher_sanity(args, config, run_dir):
    seeds = config["teacher_sanity_seeds"]
    resources = prepare_rollout(config, args.student_checkpoint, seeds, args.device)
    rows = []
    try:
        with h5py.File(config["stage1_rnn_dataset"], "r") as source:
            lookup = episode_group_lookup(source)
            for episode_id, seed in enumerate(seeds):
                arrays, summary = rollout_episode(resources, config, seed, 1.0, 0, episode_id)
                behavior = np.asarray(lookup[seed]["actions"], dtype=np.float32)
                teacher = arrays["teacher_action_14d"]
                common = min(len(behavior), len(teacher))
                difference = np.abs(teacher[:common] - behavior[:common])
                max_abs = float(difference.max()) if difference.size else float("inf")
                exact = bool(teacher.shape == behavior.shape and np.array_equal(teacher, behavior))
                source_success = bool(lookup[seed].attrs["success"])
                passed = bool(
                    teacher.shape == behavior.shape
                    and max_abs <= float(config["teacher_sanity_action_atol"])
                    and summary["success"] == source_success
                )
                rows.append({
                    "seed": seed, "passed": passed, "exact_action_match": exact,
                    "max_abs_action_error": max_abs,
                    "teacher_length": len(teacher), "stage1_length": len(behavior),
                    "teacher_success": summary["success"], "stage1_success": source_success,
                })
                print(
                    f"[teacher-sanity] seed={seed} pass={int(passed)} exact={int(exact)} "
                    f"max_abs={max_abs:.10g} length={len(teacher)}/{len(behavior)} "
                    f"success={int(summary['success'])}/{int(source_success)}"
                )
    finally:
        close_env(resources["env"])
    report = {
        "status": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "beta": 1.0,
        "teacher_frozen": True,
        "teacher_parameter_count": resources["teacher_parameter_count"],
        "action_atol": config["teacher_sanity_action_atol"],
        "seeds": rows,
    }
    output = Path(args.output) if args.output else run_dir / "teacher_sanity.json"
    atomic_json(output, report)
    print(f"Teacher beta=1.0 sanity: {report['status']} | {output}")
    if report["status"] != "PASS":
        raise RuntimeError("Teacher beta=1.0 sanity failed; DAgger collection is blocked")


def run_collection(args, config, run_dir):
    sanity_path = run_dir / "teacher_sanity.json"
    if not sanity_path.is_file() or read_json(sanity_path).get("status") != "PASS":
        raise RuntimeError(f"Passing teacher sanity is required before collection: {sanity_path}")
    train_start, train_end = config["train_seeds"]
    seed_start = train_start if args.seed_start is None else args.seed_start
    num_seeds = train_end - train_start + 1 if args.num_seeds is None else args.num_seeds
    seeds = list(range(seed_start, seed_start + num_seeds))
    if any(seed < train_start or seed > train_end for seed in seeds):
        raise RuntimeError("DAgger collection may only use train seeds 10000..10079")
    round_specs = {int(item["round_id"]): float(item["beta"]) for item in config["rounds"]}
    if args.round_id not in round_specs:
        raise RuntimeError(f"Unknown configured DAgger round: {args.round_id}")
    if not np.isclose(args.beta, round_specs[args.round_id], rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Round {args.round_id} beta mismatch: command={args.beta}, "
            f"config={round_specs[args.round_id]}"
        )
    resources = prepare_rollout(config, args.student_checkpoint, seeds, args.device)
    if args.round_id == 1:
        if int(resources["student_payload"].get("round_id", -1)) != 0:
            raise RuntimeError("Round 1 collection must start from the Stage 3B Round-0 Actor")
    elif int(resources["student_payload"].get("round_id", -1)) != args.round_id - 1:
        raise RuntimeError(
            f"Round {args.round_id} collection must use Round {args.round_id - 1} Actor"
        )
    episodes = []
    try:
        for episode_id, seed in enumerate(seeds):
            arrays, summary = rollout_episode(
                resources, config, seed, args.beta, args.round_id, episode_id
            )
            episodes.append((arrays, summary))
            print(
                f"[round={args.round_id} beta={args.beta:.3f} {episode_id + 1:03d}/{len(seeds):03d}] "
                f"seed={seed} success={int(summary['success'])} return={summary['episode_return']:.3f} "
                f"length={summary['episode_length']} teacher_fraction={summary['teacher_action_fraction']:.3f} "
                f"student_fraction={summary['student_action_fraction']:.3f} "
                f"trash_ever={int(summary['trash_ever'])} payload_ever={int(summary['payload_ever'])} "
                f"progress={summary['progress_score']}"
            )
    finally:
        close_env(resources["env"])
    output = Path(args.output) if args.output else run_dir / "datasets" / f"round{args.round_id}_corrective.hdf5"
    write_corrective_dataset(output, resources, config, args.round_id, args.beta, episodes)
    summaries = [summary for _, summary in episodes]
    transition_count = sum(row["episode_length"] for row in summaries)
    report = {
        "round_id": args.round_id, "beta": args.beta, "episodes": len(episodes),
        "seeds": seeds, "success_count": sum(int(row["success"]) for row in summaries),
        "success_rate": float(np.mean([row["success"] for row in summaries])),
        "trash_ever_rate": float(np.mean([row["trash_ever"] for row in summaries])),
        "payload_ever_rate": float(np.mean([row["payload_ever"] for row in summaries])),
        "mean_progress_score": float(np.mean([row["progress_score"] for row in summaries])),
        "corrective_transitions": transition_count,
        "target_is_teacher_action": True, "heldout_seeds_used": False,
        "dataset_shapes": {
            "state_59d": [transition_count, 59],
            "teacher_action_14d": [transition_count, 14],
            "student_action_14d": [transition_count, 14],
            "executed_action_14d": [transition_count, 14],
        },
        "dataset": str(output), "episode_results": summaries,
    }
    summary_output = (
        Path(args.summary_output) if args.summary_output
        else run_dir / f"round_{args.round_id}" / "collection_summary.json"
    )
    atomic_json(summary_output, report)
    with h5py.File(output, "r") as handle:
        reloaded = sum(int(group["state_59d"].shape[0]) for group in handle["episodes"].values())
    if reloaded != report["corrective_transitions"]:
        raise RuntimeError("Corrective dataset reload transition count mismatch")
    print(
        f"Round {args.round_id} collection complete | episodes={len(episodes)} "
        f"transitions={reloaded} dataset={output}"
    )


def merge_worker_datasets(shards, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing corrective dataset: {output}")
    episode_index = 0
    with h5py.File(output, "w") as destination:
        destination_root = destination.create_group("episodes")
        for shard_index, shard in enumerate(shards):
            with h5py.File(shard, "r") as source:
                if shard_index == 0:
                    for name, value in source.attrs.items():
                        destination.attrs[name] = value
                    destination.attrs["parallel_worker_count"] = len(shards)
                for source_group in source["episodes"].values():
                    source.copy(
                        source_group,
                        destination_root,
                        name=f"episode_{episode_index:06d}",
                    )
                    episode_index += 1
    return episode_index


def run_parallel_collection(args, config, run_dir, num_workers):
    train_start, train_end = config["train_seeds"]
    seed_start = train_start if args.seed_start is None else args.seed_start
    num_seeds = train_end - train_start + 1 if args.num_seeds is None else args.num_seeds
    seeds = list(range(seed_start, seed_start + num_seeds))
    if any(seed < train_start or seed > train_end for seed in seeds):
        raise RuntimeError("Parallel DAgger collection may only use train seeds 10000..10079")
    if num_workers > len(seeds):
        num_workers = len(seeds)
    partitions = [part.tolist() for part in np.array_split(np.asarray(seeds), num_workers) if len(part)]
    worker_root = run_dir / "datasets" / "workers" / f"round_{args.round_id}"
    summary_root = run_dir / f"round_{args.round_id}" / "workers"
    log_root = run_dir / "logs"
    for path in (worker_root, summary_root, log_root):
        path.mkdir(parents=True, exist_ok=True)

    processes = []
    for worker_id, worker_seeds in enumerate(partitions):
        shard = worker_root / f"worker_{worker_id:02d}.hdf5"
        summary = summary_root / f"worker_{worker_id:02d}_summary.json"
        log_path = log_root / f"round{args.round_id}_collection_worker_{worker_id:02d}.log"
        command = [
            sys.executable, "-u", Path(__file__).resolve(),
            "--config", args.config, "--run-dir", run_dir,
            "--student-checkpoint", args.student_checkpoint,
            "--mode", "collect", "--round-id", args.round_id,
            "--beta", args.beta, "--seed-start", worker_seeds[0],
            "--num-seeds", len(worker_seeds), "--device", args.device,
            "--output", shard, "--summary-output", summary,
            "--num-workers", 1, "--worker-id", worker_id,
        ]
        stream = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            [str(item) for item in command], stdout=stream,
            stderr=subprocess.STDOUT, text=True,
        )
        processes.append((worker_id, worker_seeds, shard, summary, log_path, stream, process))
        print(
            f"[parallel collection] worker={worker_id:02d} pid={process.pid} "
            f"seeds={worker_seeds[0]}..{worker_seeds[-1]} log={log_path}"
        )

    failures = []
    for worker_id, worker_seeds, shard, summary, log_path, stream, process in processes:
        return_code = process.wait()
        stream.close()
        if return_code:
            failures.append({
                "worker_id": worker_id, "return_code": return_code,
                "seeds": worker_seeds, "log": str(log_path),
            })
        else:
            print(f"[parallel collection] worker={worker_id:02d} complete")
    if failures:
        raise RuntimeError(f"Parallel DAgger collection worker failures: {failures}")

    output = (
        Path(args.output) if args.output
        else run_dir / "datasets" / f"round{args.round_id}_corrective.hdf5"
    )
    shard_paths = [item[2] for item in processes]
    episode_count = merge_worker_datasets(shard_paths, output)
    worker_reports = [read_json(item[3]) for item in processes]
    episode_results = sorted(
        [row for report in worker_reports for row in report["episode_results"]],
        key=lambda row: row["seed"],
    )
    observed_seeds = [int(row["seed"]) for row in episode_results]
    if observed_seeds != seeds or episode_count != len(seeds):
        raise RuntimeError(
            f"Parallel aggregate seed mismatch: expected={seeds}, observed={observed_seeds}"
        )
    transition_count = sum(int(row["episode_length"]) for row in episode_results)
    report = {
        "round_id": args.round_id, "beta": args.beta,
        "parallel_workers": len(partitions), "episodes": episode_count,
        "seeds": seeds,
        "success_count": sum(int(row["success"]) for row in episode_results),
        "success_rate": float(np.mean([row["success"] for row in episode_results])),
        "trash_ever_rate": float(np.mean([row["trash_ever"] for row in episode_results])),
        "payload_ever_rate": float(np.mean([row["payload_ever"] for row in episode_results])),
        "mean_progress_score": float(np.mean([row["progress_score"] for row in episode_results])),
        "corrective_transitions": transition_count,
        "target_is_teacher_action": True, "heldout_seeds_used": False,
        "dataset_shapes": {
            "state_59d": [transition_count, 59],
            "teacher_action_14d": [transition_count, 14],
            "student_action_14d": [transition_count, 14],
            "executed_action_14d": [transition_count, 14],
        },
        "dataset": str(output), "worker_shards": [str(path) for path in shard_paths],
        "episode_results": episode_results,
    }
    atomic_json(run_dir / f"round_{args.round_id}" / "collection_summary.json", report)
    with h5py.File(output, "r") as handle:
        reloaded = sum(int(group["state_59d"].shape[0]) for group in handle["episodes"].values())
    if reloaded != transition_count:
        raise RuntimeError("Parallel corrective dataset reload transition count mismatch")
    print(
        f"Parallel Round {args.round_id} collection complete | workers={len(partitions)} "
        f"episodes={episode_count} transitions={transition_count} dataset={output}"
    )


def main():
    args = parse_args()
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("--beta must be in [0, 1]")
    config = read_json(args.config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "config.json", config)
    for name in ("datasets", "checkpoints", "logs", "evaluations"):
        (run_dir / name).mkdir(exist_ok=True)
    if args.mode == "teacher-sanity":
        run_teacher_sanity(args, config, run_dir)
    else:
        num_workers = int(
            args.num_workers if args.num_workers is not None else config.get("collection_workers", 1)
        )
        if num_workers <= 0:
            raise ValueError("--num-workers must be positive")
        if num_workers == 1:
            run_collection(args, config, run_dir)
        else:
            run_parallel_collection(args, config, run_dir, num_workers)


if __name__ == "__main__":
    main()
