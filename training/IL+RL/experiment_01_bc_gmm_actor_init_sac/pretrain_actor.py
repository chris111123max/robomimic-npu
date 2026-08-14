#!/usr/bin/env python3
"""Stage 1: distill a frozen BC-GMM into a SAC-compatible Actor."""

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from rlkit.torch.sac.policies import TanhGaussianPolicy
import rlkit.torch.pytorch_util as ptu

import robomimic.utils.file_utils as FileUtils

from train import (
    ParallelEnvPool,
    PURE,
    configure_device,
    initialize_observation_modalities,
    make_stage2_config,
    policy_actions,
    set_random_seeds,
)


def decode_demo_keys(values):
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def load_observation_split(dataset_path, filter_key, observation_keys, shapes):
    """Read observation states only; expert action/reward/done datasets are untouched."""
    arrays = {key: [] for key in observation_keys}
    with h5py.File(dataset_path, "r") as dataset:
        mask_path = f"mask/{filter_key}"
        if mask_path not in dataset:
            raise KeyError(f"Dataset split mask not found: {mask_path}")
        demo_keys = decode_demo_keys(dataset[mask_path][()])
        if not demo_keys:
            raise RuntimeError(f"Dataset split {filter_key!r} contains no demonstrations")
        for demo_key in demo_keys:
            obs_group_path = f"data/{demo_key}/obs"
            if obs_group_path not in dataset:
                raise KeyError(f"Observation group missing: {obs_group_path}")
            obs_group = dataset[obs_group_path]
            lengths = []
            for key in observation_keys:
                if key not in obs_group:
                    raise KeyError(f"Observation key {key!r} missing in demo {demo_key}")
                value = np.asarray(obs_group[key][()], dtype=np.float32)
                if tuple(value.shape[1:]) != tuple(shapes[key]):
                    raise AssertionError(
                        f"{demo_key}/{key} expected trailing shape {shapes[key]}, got {value.shape}"
                    )
                arrays[key].append(value)
                lengths.append(value.shape[0])
            if len(set(lengths)) != 1:
                raise AssertionError(f"Observation lengths disagree in demo {demo_key}: {lengths}")
    arrays = OrderedDict(
        (key, np.concatenate(arrays[key], axis=0)) for key in observation_keys
    )
    state_count = next(iter(arrays.values())).shape[0]
    if any(value.shape[0] != state_count for value in arrays.values()):
        raise AssertionError("Observation split arrays have inconsistent state counts")
    return arrays, demo_keys


def flatten_states(states, observation_keys):
    flattened = np.concatenate(
        [states[key].reshape(states[key].shape[0], -1) for key in observation_keys],
        axis=1,
    ).astype(np.float32, copy=False)
    if flattened.shape[1] != 59:
        raise AssertionError(f"Student observation expected [N,59], got {flattened.shape}")
    return flattened


def flat_to_observation_dict(flat_batch, observation_keys, shapes):
    flat_batch = np.asarray(flat_batch, dtype=np.float32)
    result = OrderedDict()
    offset = 0
    for key in observation_keys:
        width = int(np.prod(shapes[key]))
        result[key] = flat_batch[:, offset : offset + width].reshape(
            (flat_batch.shape[0],) + tuple(shapes[key])
        )
        offset += width
    if offset != flat_batch.shape[1]:
        raise AssertionError(f"Observation reconstruction consumed {offset}/{flat_batch.shape[1]}")
    return result


def module_fingerprint(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_torch_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu"):
        torch.npu.manual_seed_all(seed)


def generate_teacher_actions(teacher, states, batch_size):
    count = next(iter(states.values())).shape[0]
    outputs = []
    teacher.start_episode()
    with torch.no_grad():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            batch = OrderedDict((key, value[start:end]) for key, value in states.items())
            action = np.asarray(teacher(batch, batched_ob=True), dtype=np.float32)
            if action.shape != (end - start, 14):
                raise AssertionError(f"Teacher action expected {(end-start,14)}, got {action.shape}")
            if not np.isfinite(action).all():
                raise FloatingPointError("Teacher action contains non-finite values")
            if np.max(np.abs(action)) > 1.001:
                raise AssertionError(f"Teacher action outside [-1,1]: max={np.max(np.abs(action))}")
            outputs.append(action)
    return np.concatenate(outputs, axis=0)


def load_or_create_teacher_cache(
    teacher, train_states, valid_states, cache_dir, teacher_checkpoint, teacher_config
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "teacher_targets.npz"
    metadata_path = cache_dir / "metadata.json"
    expected_metadata = {
        "source": "derived from BC-GMM teacher outputs; never copied from HDF5 actions",
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_semantics": "RolloutPolicy -> BC_GMM.get_action -> GMMActorNetwork.forward -> MixtureSameFamily.sample",
        "low_noise_eval": bool(teacher_config["low_noise_eval"]),
        "target_generation_seed": int(teacher_config["target_generation_seed"]),
        "train_state_count": int(next(iter(train_states.values())).shape[0]),
        "valid_state_count": int(next(iter(valid_states.values())).shape[0]),
    }
    if cache_path.is_file() and metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata != expected_metadata:
            raise RuntimeError("Existing Teacher target cache metadata does not match this run")
        with np.load(cache_path) as cache:
            train_actions = np.asarray(cache["train_actions"], dtype=np.float32)
            valid_actions = np.asarray(cache["valid_actions"], dtype=np.float32)
        print("Loaded stable Teacher target cache:", cache_path)
    else:
        set_torch_seed(teacher_config["target_generation_seed"])
        train_actions = generate_teacher_actions(
            teacher, train_states, teacher_config["cache_batch_size"]
        )
        valid_actions = generate_teacher_actions(
            teacher, valid_states, teacher_config["cache_batch_size"]
        )
        np.savez_compressed(
            cache_path, train_actions=train_actions, valid_actions=valid_actions
        )
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(expected_metadata, stream, indent=4, ensure_ascii=False)
        print("Created stable Teacher target cache:", cache_path)
    if train_actions.shape != (expected_metadata["train_state_count"], 14):
        raise AssertionError(f"Cached train Teacher actions have shape {train_actions.shape}")
    if valid_actions.shape != (expected_metadata["valid_state_count"], 14):
        raise AssertionError(f"Cached valid Teacher actions have shape {valid_actions.shape}")
    with np.load(cache_path) as disk_cache:
        disk_train_actions = np.asarray(disk_cache["train_actions"], dtype=np.float32)
        disk_valid_actions = np.asarray(disk_cache["valid_actions"], dtype=np.float32)
    if not np.array_equal(train_actions, disk_train_actions):
        raise AssertionError("Train Teacher targets changed after cache reload")
    if not np.array_equal(valid_actions, disk_valid_actions):
        raise AssertionError("Valid Teacher targets changed after cache reload")
    return train_actions, valid_actions, cache_path


def evaluate_parallel_policy(
    env_pool, action_function, seeds, max_episode_steps, terminate_on_success, label
):
    worker_ids = list(range(len(seeds)))
    observations = env_pool.reset(
        worker_ids, seeds, max_episode_steps, terminate_on_success
    )
    returns = {worker_id: 0.0 for worker_id in worker_ids}
    lengths = {worker_id: 0 for worker_id in worker_ids}
    successes = {worker_id: False for worker_id in worker_ids}
    active = set(worker_ids)
    start_time = time.perf_counter()
    while active:
        active_ids = sorted(active)
        obs_batch = np.stack([observations[worker_id] for worker_id in active_ids])
        actions = action_function(obs_batch)
        transitions = env_pool.step(active_ids, actions)
        for worker_id in active_ids:
            next_obs, reward, done, info = transitions[worker_id]
            observations[worker_id] = next_obs
            returns[worker_id] += reward
            lengths[worker_id] += 1
            successes[worker_id] = bool(info["success"])
            if done:
                active.remove(worker_id)
    metrics = {
        "Success_Rate": float(np.mean(list(successes.values()))),
        "Return_Mean": float(np.mean(list(returns.values()))),
        "Return_Std": float(np.std(list(returns.values()))),
        "Episode_Length_Mean": float(np.mean(list(lengths.values()))),
        "Eval_Time": time.perf_counter() - start_time,
    }
    print(label, json.dumps(metrics, ensure_ascii=False))
    return metrics


def diagnostic_metrics(student_action, teacher_action):
    difference = student_action - teacher_action
    return {
        "mse": float(np.mean(np.square(difference))),
        "mae": float(np.mean(np.abs(difference))),
        "per_dim_mse": np.mean(np.square(difference), axis=0).tolist(),
        "per_dim_mae": np.mean(np.abs(difference), axis=0).tolist(),
    }


def save_actor_checkpoint(
    path, actor, optimizer, epoch, train_metrics, validation_metrics, eval_success,
    teacher_success, retention, config, teacher_checkpoint, observation_keys
):
    checkpoint = {
        "actor": PURE.cpu_tree(actor.state_dict()),
        "optimizer": PURE.cpu_tree(optimizer.state_dict()),
        "epoch": int(epoch),
        "train_mse": train_metrics["mse"],
        "validation_mse": validation_metrics["mse"],
        "eval_success_rate": eval_success,
        "teacher_success_rate": teacher_success,
        "retention_ratio": retention,
        "obs_keys": list(observation_keys),
        "obs_dim": 59,
        "action_dim": 14,
        "teacher_checkpoint_path": str(teacher_checkpoint),
        "config": config,
        "seed": config["stage1"]["training"]["seed"],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)
    print("Saved Stage1 Actor:", path)


def run_stage1(raw_config, config_path, run_dir, smoke_test=False):
    stage1 = json.loads(json.dumps(raw_config["stage1"]))
    environment = json.loads(json.dumps(raw_config["environment"]))
    if smoke_test:
        stage1["training"].update(num_epochs=1, train_steps_per_epoch=1)
        stage1["validation"].update(fixed_num_states=100, steps_per_epoch=1)
        stage1["evaluation"].update(every_n_epochs=1, max_episode_steps=5)
        environment["max_episode_steps"] = 5

    teacher_checkpoint = Path(raw_config["paths"]["teacher_checkpoint"]).expanduser().resolve()
    dataset_path = Path(raw_config["paths"]["dataset"]).expanduser().resolve()
    if not teacher_checkpoint.is_file():
        raise FileNotFoundError(f"Epoch 1850 BC-GMM Teacher not found: {teacher_checkpoint}")
    if teacher_checkpoint.name != raw_config["teacher"]["checkpoint_filename"]:
        raise AssertionError(f"Unexpected Teacher filename: {teacher_checkpoint.name}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    stage1_dir = run_dir / "stage1_actor_pretraining"
    logs_dir = stage1_dir / "logs"
    models_dir = stage1_dir / "models"
    evaluation_dir = stage1_dir / "evaluation"
    teacher_targets_dir = stage1_dir / "teacher_targets"
    for path in (logs_dir, models_dir, evaluation_dir, teacher_targets_dir):
        path.mkdir(parents=True, exist_ok=False)

    original_stdout, original_stderr = sys.stdout, sys.stderr
    log_handle = (logs_dir / "log.txt").open("a", encoding="utf-8", buffering=1)
    if raw_config["logging"]["terminal_output_to_txt"]:
        sys.stdout = PURE.TeeStream(original_stdout, log_handle)
        sys.stderr = PURE.TeeStream(original_stderr, log_handle)
    writer = None
    if raw_config["logging"]["tensorboard"]:
        try:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(str(logs_dir / "tb"))
        except ImportError:
            print("WARNING: tensorboardX unavailable; Stage1 TensorBoard disabled")

    env_pool = None
    try:
        device_config = {
            "device": raw_config["device"],
        }
        configure_device(device_config)
        set_random_seeds(stage1["training"]["seed"])
        initialize_observation_modalities(make_stage2_config(raw_config, smoke_test=False))
        print("Teacher Checkpoint:", teacher_checkpoint)
        print("Teacher device:", ptu.device)
        teacher, teacher_ckpt = FileUtils.policy_from_checkpoint(
            device=ptu.device, ckpt_path=str(teacher_checkpoint), verbose=True
        )
        checkpoint_epoch = teacher_ckpt.get("epoch")
        if checkpoint_epoch is not None and int(checkpoint_epoch) != int(
            raw_config["teacher"]["expected_epoch"]
        ):
            raise AssertionError(f"Teacher checkpoint epoch mismatch: {checkpoint_epoch}")
        teacher_low_noise_eval = bool(teacher.policy.algo_config.gmm.low_noise_eval)
        if teacher_low_noise_eval is not bool(raw_config["teacher"]["low_noise_eval"]):
            raise AssertionError(
                "Teacher checkpoint gmm.low_noise_eval does not match experiment config"
            )
        print("Teacher policy class:", teacher.policy.__class__.__name__)
        print("Teacher GMM low_noise_eval:", teacher_low_noise_eval)
        teacher_module = teacher.policy.nets
        teacher.policy.set_eval()
        for parameter in teacher_module.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in teacher_module.parameters()):
            raise AssertionError("Teacher freezing failed")
        teacher_fingerprint_before = module_fingerprint(teacher_module)

        observation_keys = raw_config["observation"]["keys"]
        observation_shapes = raw_config["observation"]["shapes"]
        train_states, train_demos = load_observation_split(
            dataset_path, stage1["validation"]["train_filter_key"], observation_keys, observation_shapes
        )
        valid_states, valid_demos = load_observation_split(
            dataset_path, stage1["validation"]["filter_key"], observation_keys, observation_shapes
        )
        if smoke_test:
            # A smoke run validates the complete data chain without evaluating
            # the Teacher over the full dataset. Formal runs keep every state.
            train_states = OrderedDict(
                (key, value[:100]) for key, value in train_states.items()
            )
            valid_states = OrderedDict(
                (key, value[:100]) for key, value in valid_states.items()
            )
        train_flat = flatten_states(train_states, observation_keys)
        valid_flat = flatten_states(valid_states, observation_keys)
        print("Train split demos/states:", len(train_demos), train_flat.shape)
        print("Valid split demos/states:", len(valid_demos), valid_flat.shape)

        train_teacher, valid_teacher, cache_path = load_or_create_teacher_cache(
            teacher, train_states, valid_states, teacher_targets_dir,
            teacher_checkpoint, raw_config["teacher"]
        )
        print("Teacher action shapes:", train_teacher.shape, valid_teacher.shape)
        print("Teacher target cache stable:", np.array_equal(train_teacher[0], train_teacher[0].copy()))

        env_config = make_stage2_config(raw_config, smoke_test=False)
        env_config["environment"].update(environment)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=str(dataset_path))
        env_pool = ParallelEnvPool(
            env_config, env_meta, environment["parallel_envs"], stage1["training"]["seed"]
        )
        evaluation_seeds = stage1["evaluation"]["seeds"]
        print("Stage1 Evaluation Seeds:", evaluation_seeds)
        set_torch_seed(raw_config["teacher"]["target_generation_seed"] + 1)
        teacher.policy.set_eval()
        teacher_metrics = evaluate_parallel_policy(
            env_pool,
            lambda flat: np.asarray(
                teacher(
                    flat_to_observation_dict(flat, observation_keys, observation_shapes),
                    batched_ob=True,
                ),
                dtype=np.float32,
            ),
            evaluation_seeds,
            stage1["evaluation"]["max_episode_steps"],
            stage1["evaluation"]["terminate_on_success"],
            "Teacher Benchmark",
        )
        with (evaluation_dir / "teacher_benchmark.json").open("w", encoding="utf-8") as stream:
            json.dump(teacher_metrics, stream, indent=4, ensure_ascii=False)

        set_random_seeds(stage1["training"]["seed"])
        actor = TanhGaussianPolicy(
            obs_dim=59,
            action_dim=14,
            hidden_sizes=stage1["network"]["actor_hidden_sizes"],
        ).to(ptu.device)
        optimizer = torch.optim.Adam(actor.parameters(), lr=stage1["optimizer"]["actor_lr"])
        initial_actor_fingerprint = module_fingerprint(actor)
        initial_log_std = {
            name: value.detach().cpu().clone()
            for name, value in actor.state_dict().items()
            if "last_fc_log_std" in name
        }
        rng = np.random.RandomState(stage1["training"]["seed"])
        fixed_valid_indices = rng.choice(
            valid_flat.shape[0],
            size=stage1["validation"]["fixed_num_states"],
            replace=valid_flat.shape[0] < stage1["validation"]["fixed_num_states"],
        )
        metrics_path = logs_dir / "metrics.jsonl"
        best_key = None
        latest_eval_success = -1.0
        latest_retention = None
        teacher_success = teacher_metrics["Success_Rate"]

        for epoch in range(1, stage1["training"]["num_epochs"] + 1):
            epoch_start = time.perf_counter()
            actor.train()
            train_student_batches, train_teacher_batches = [], []
            for _ in range(stage1["training"]["train_steps_per_epoch"]):
                indices = rng.randint(0, train_flat.shape[0], size=stage1["training"]["batch_size"])
                obs_tensor = torch.from_numpy(train_flat[indices]).float().to(ptu.device)
                target_tensor = torch.from_numpy(train_teacher[indices]).float().to(ptu.device)
                optimizer.zero_grad(set_to_none=True)
                student_action, _, log_std, _, _, _, _, _ = actor(
                    obs_tensor, deterministic=True
                )
                if student_action.shape != target_tensor.shape or student_action.shape[1] != 14:
                    raise AssertionError(
                        f"Stage1 action shape mismatch: {student_action.shape} vs {target_tensor.shape}"
                    )
                loss = F.mse_loss(student_action, target_tensor)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite Stage1 MSE: {loss}")
                loss.backward()
                optimizer.step()
                train_student_batches.append(student_action.detach().cpu().numpy())
                train_teacher_batches.append(target_tensor.detach().cpu().numpy())

            actor.eval()
            valid_student_batches, valid_teacher_batches, log_std_batches = [], [], []
            with torch.no_grad():
                batch_size = stage1["training"]["batch_size"]
                validation_limit = min(
                    len(fixed_valid_indices),
                    stage1["validation"]["steps_per_epoch"] * batch_size,
                )
                for start in range(0, validation_limit, batch_size):
                    indices = fixed_valid_indices[start : start + batch_size]
                    obs_tensor = torch.from_numpy(valid_flat[indices]).float().to(ptu.device)
                    student_action, _, log_std, _, _, _, _, _ = actor(
                        obs_tensor, deterministic=True
                    )
                    valid_student_batches.append(student_action.cpu().numpy())
                    valid_teacher_batches.append(valid_teacher[indices])
                    log_std_batches.append(log_std.cpu().numpy())

            train_metrics = diagnostic_metrics(
                np.concatenate(train_student_batches), np.concatenate(train_teacher_batches)
            )
            validation_metrics = diagnostic_metrics(
                np.concatenate(valid_student_batches), np.concatenate(valid_teacher_batches)
            )
            log_std_values = np.concatenate(log_std_batches)
            epoch_metrics = {
                "Stage1_Epoch": epoch,
                "Train_MSE": train_metrics["mse"],
                "Validation_MSE": validation_metrics["mse"],
                "Validation_MAE": validation_metrics["mae"],
                "Per_Action_Dim_MSE": validation_metrics["per_dim_mse"],
                "Per_Action_Dim_MAE": validation_metrics["per_dim_mae"],
                "Student_Log_Std_Mean": float(np.mean(log_std_values)),
                "Student_Log_Std_Std": float(np.std(log_std_values)),
                "Time_Epoch": time.perf_counter() - epoch_start,
            }

            should_eval = epoch % stage1["evaluation"]["every_n_epochs"] == 0
            if should_eval:
                student_metrics = evaluate_parallel_policy(
                    env_pool,
                    lambda flat: policy_actions(actor, flat, deterministic=True),
                    evaluation_seeds,
                    stage1["evaluation"]["max_episode_steps"],
                    stage1["evaluation"]["terminate_on_success"],
                    f"Stage1 Student Epoch {epoch}",
                )
                latest_eval_success = student_metrics["Success_Rate"]
                if teacher_success > 0:
                    latest_retention = latest_eval_success / teacher_success
                    target_reached = latest_retention >= stage1["selection"]["target_retention"]
                else:
                    latest_retention = None
                    target_reached = False
                    print("WARNING: Teacher benchmark success is zero; retention metric is undefined.")
                epoch_metrics.update(
                    Stage1_Eval_Success_Rate=latest_eval_success,
                    Stage1_Eval_Return_Mean=student_metrics["Return_Mean"],
                    Stage1_Eval_Return_Std=student_metrics["Return_Std"],
                    Stage1_Eval_Episode_Length_Mean=student_metrics["Episode_Length_Mean"],
                    Teacher_Success_Rate=teacher_success,
                    Success_Retention_Ratio=latest_retention,
                    Target_Retention=stage1["selection"]["target_retention"],
                    Target_Reached=target_reached,
                )
                with (evaluation_dir / f"student_epoch_{epoch:03d}.json").open(
                    "w", encoding="utf-8"
                ) as stream:
                    json.dump(epoch_metrics, stream, indent=4, ensure_ascii=False)

                selection_key = (-latest_eval_success, validation_metrics["mse"], epoch)
                save_actor_checkpoint(
                    models_dir / f"actor_epoch_{epoch:03d}.pth", actor, optimizer,
                    epoch, train_metrics, validation_metrics, latest_eval_success,
                    teacher_success, latest_retention, raw_config, teacher_checkpoint,
                    observation_keys,
                )
                if best_key is None or selection_key < best_key:
                    best_key = selection_key
                    save_actor_checkpoint(
                        models_dir / "best_actor.pth", actor, optimizer, epoch,
                        train_metrics, validation_metrics, latest_eval_success,
                        teacher_success, latest_retention, raw_config,
                        teacher_checkpoint, observation_keys,
                    )

            save_actor_checkpoint(
                models_dir / "last_actor.pth", actor, optimizer, epoch,
                train_metrics, validation_metrics, latest_eval_success,
                teacher_success, latest_retention, raw_config, teacher_checkpoint,
                observation_keys,
            )
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")
            print("\nStage1 Epoch Summary")
            print(json.dumps(epoch_metrics, indent=4, ensure_ascii=False))
            if writer is not None:
                for key, value in epoch_metrics.items():
                    if isinstance(value, (int, float)) and value is not None:
                        writer.add_scalar(key, value, epoch)
                writer.flush()

        teacher_fingerprint_after = module_fingerprint(teacher_module)
        if teacher_fingerprint_after != teacher_fingerprint_before:
            raise AssertionError("Frozen BC-GMM Teacher parameters changed")
        if module_fingerprint(actor) == initial_actor_fingerprint:
            raise AssertionError("Student Actor parameters did not change")
        final_state = actor.state_dict()
        for name, initial_value in initial_log_std.items():
            if not torch.equal(final_state[name].detach().cpu(), initial_value):
                raise AssertionError(f"Stage1 directly changed unsupervised log_std parameter {name}")
        print("Stage1 completed successfully.")
        print("Teacher parameters unchanged: True")
        print("Student parameters updated: True")
        print("Student log_std head directly updated by MSE: False")
        print("Teacher target cache:", cache_path)
        print("Best Actor:", models_dir / "best_actor.pth")
        return models_dir / "best_actor.pth"
    finally:
        if writer is not None:
            writer.close()
        if env_pool is not None:
            env_pool.close()
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log_handle.close()


def main():
    parser = argparse.ArgumentParser(description="Stage 1 BC-GMM Actor distillation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    run_stage1(
        config,
        config_path,
        args.run_dir.expanduser().resolve(),
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
