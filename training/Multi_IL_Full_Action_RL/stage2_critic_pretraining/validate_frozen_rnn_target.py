#!/usr/bin/env python3
"""Stage 2.0: validate frozen BC-RNN inference on Stage 1 histories."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import deque
from pathlib import Path

import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_ROOT.parents[1]
STAGE1_DIR = PROJECT_ROOT / "stage1_rollout_collection"
sys.path.insert(0, str(STAGE1_DIR))

from common import seed_everything  # noqa: E402


POLICIES = ("bc_rnn", "bc_transformer", "bc_gmm")
DISPLAY_NAMES = {
    "bc_rnn": "RNN",
    "bc_transformer": "Transformer",
    "bc_gmm": "GMM",
}
DEFAULT_SEEDS = (10000, 10001, 10002, 10003, 10004)
DEFAULT_TRAINING_RUNS = Path("/data/home/3220251075/lerobot_workspace/training_runs")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "analysis/stage1_5_selected_datasets.json"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "analysis/stage2/frozen_rnn_target_validation.json"),
    )
    parser.add_argument(
        "--training-runs-root",
        default=str(DEFAULT_TRAINING_RUNS),
        help="Read-only boundary containing all selected Stage 1 datasets and checkpoint.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    return parser.parse_args()


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


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def validate_paths(args):
    root = Path(args.training_runs_root).resolve()
    output = Path(args.output).resolve()
    if inside(output, root):
        raise RuntimeError(f"Refusing to write validation output inside training_runs: {output}")
    manifest = read_json(args.manifest)
    if manifest.get("status") != "selected":
        raise RuntimeError(
            f"Stage 1.5 manifest status is {manifest.get('status')!r}; refusing to guess datasets"
        )
    selected = manifest.get("datasets", {})
    for policy in POLICIES:
        if policy not in selected:
            raise RuntimeError(f"Selected manifest has no {policy} dataset")
        path = Path(selected[policy]).resolve()
        if not inside(path, root):
            raise RuntimeError(f"Selected dataset is outside training_runs: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        selected[policy] = str(path)
    return manifest, selected, root, output


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def checkpoint_from_rnn_dataset(path, training_runs_root):
    with h5py.File(path, "r") as handle:
        if str(decode(handle.attrs.get("policy_id", ""))) != "bc_rnn":
            raise RuntimeError(f"RNN dataset content policy_id is invalid: {path}")
        checkpoint = decode(handle.attrs.get("checkpoint"))
    if not checkpoint:
        raise RuntimeError("bc_rnn HDF5 root attribute 'checkpoint' is missing")
    checkpoint = Path(str(checkpoint)).resolve()
    if not inside(checkpoint, training_runs_root):
        raise RuntimeError(f"Recorded RNN checkpoint is outside training_runs: {checkpoint}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def select_device(name):
    import torch
    wants_npu = name == "auto" or str(name).split(":", 1)[0] == "npu"
    if wants_npu:
        try:
            import torch_npu  # noqa: F401 - registers torch.npu / the "npu" device type
        except ImportError as exception:
            if name != "auto":
                raise RuntimeError(
                    f"Requested device {name!r}, but torch_npu is not importable"
                ) from exception
    if name != "auto":
        device = torch.device(name)
        if device.type == "npu":
            npu = getattr(torch, "npu", None)
            if npu is None or not npu.is_available():
                raise RuntimeError(f"Requested device {name!r}, but Ascend NPU is unavailable")
            npu.set_device(device)
        return device
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        device = torch.device("npu:0")
        npu.set_device(device)
        return device
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_frozen_policy(checkpoint_path, device_name):
    import robomimic.utils.file_utils as FileUtils

    device = select_device(device_name)
    checkpoint = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(checkpoint_path))
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint)
    policy, loaded_checkpoint = FileUtils.policy_from_checkpoint(
        ckpt_dict=checkpoint, device=device, verbose=False
    )
    if not bool(config.algo.rnn.enabled):
        raise RuntimeError("Checkpoint recorded by bc_rnn dataset is not an RNN policy")
    model = getattr(policy, "policy", None)
    if model is None:
        raise RuntimeError("Official RolloutPolicy does not expose its underlying policy")
    model.set_eval()
    networks = getattr(model, "nets", None)
    if networks is None:
        raise RuntimeError("Loaded BC-RNN algorithm does not expose its network ModuleDict")
    networks.eval()
    parameter_count = 0
    for parameter in networks.parameters():
        parameter.requires_grad_(False)
        parameter_count += parameter.numel()
    if parameter_count == 0:
        raise RuntimeError("Loaded BC-RNN contains no parameters")
    if any(parameter.requires_grad for parameter in networks.parameters()):
        raise RuntimeError("Failed to freeze every BC-RNN parameter")
    frame_stack = int(config.train.frame_stack)
    stochastic = bool(config.algo.gmm.enabled)
    return policy, loaded_checkpoint, config, device, parameter_count, frame_stack, stochastic


class ObservationHistory:
    """Recreate the current-frame or FrameStackWrapper policy input."""

    def __init__(self, frame_stack):
        self.frame_stack = int(frame_stack)
        self.history = deque(maxlen=self.frame_stack)

    def append(self, observation, timestep, previous_action, action_dimension):
        current = {key: np.asarray(value).copy() for key, value in observation.items()}
        if self.frame_stack > 1:
            current["timesteps"] = np.asarray([timestep])
            current["actions"] = (
                np.zeros(action_dimension, dtype=np.float64)
                if timestep == 0 else np.asarray(previous_action).copy()
            )
        if not self.history:
            for _ in range(self.frame_stack):
                self.history.append({key: value.copy() for key, value in current.items()})
        else:
            self.history.append(current)
        if self.frame_stack == 1:
            return current
        return {
            key: np.stack([item[key] for item in self.history], axis=0)
            for key in current
        }


def episode_lookup(handle):
    if "episodes" not in handle:
        raise RuntimeError(f"Dataset has no /episodes group: {handle.filename}")
    lookup = {}
    for name, group in handle["episodes"].items():
        if "initial_seed" in group.attrs:
            seed = int(group.attrs["initial_seed"])
        elif "initial_seed" in group:
            seed = int(np.asarray(group["initial_seed"][0]))
        else:
            raise RuntimeError(f"missing initial_seed: {handle.filename}:/episodes/{name}")
        if seed in lookup:
            raise RuntimeError(f"duplicate initial_seed={seed}: {handle.filename}")
        lookup[seed] = group
    return lookup


class RunningActions:
    def __init__(self):
        self.count = self.nan = self.inf = self.saturation = 0
        self.minimum = np.inf
        self.maximum = -np.inf
        self.abs_sum = self.l2_sum = 0.0
        self.l2_max = 0.0
        self.distance_sum = self.distance_sq_sum = 0.0
        self.distance_max = 0.0

    def update(self, target, behavior):
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        behavior = np.asarray(behavior, dtype=np.float64).reshape(-1)
        if target.shape != behavior.shape:
            raise RuntimeError(f"Target/behavior action shapes differ: {target.shape} != {behavior.shape}")
        self.nan += int(np.isnan(target).sum())
        self.inf += int(np.isinf(target).sum())
        finite = target[np.isfinite(target)]
        if finite.size:
            self.minimum = min(self.minimum, float(finite.min()))
            self.maximum = max(self.maximum, float(finite.max()))
            self.abs_sum += float(np.abs(finite).sum())
            self.saturation += int(np.count_nonzero(np.abs(finite) > 0.99))
        norm = float(np.linalg.norm(target))
        if np.isfinite(norm):
            self.l2_sum += norm
            self.l2_max = max(self.l2_max, norm)
        distance = float(np.linalg.norm(behavior - target))
        if np.isfinite(distance):
            self.distance_sum += distance
            self.distance_sq_sum += distance * distance
            self.distance_max = max(self.distance_max, distance)
        self.count += 1

    def summary(self, action_dimension):
        denominator = self.count * action_dimension
        return {
            "num_transitions": self.count,
            "nan_count": self.nan,
            "inf_count": self.inf,
            "action_min": None if self.minimum == np.inf else self.minimum,
            "action_max": None if self.maximum == -np.inf else self.maximum,
            "action_abs_mean": None if not denominator else self.abs_sum / denominator,
            "action_l2_mean": None if not self.count else self.l2_sum / self.count,
            "action_l2_max": None if not self.count else self.l2_max,
            "saturation_fraction": None if not denominator else self.saturation / denominator,
            "behavior_target_l2_mean": None if not self.count else self.distance_sum / self.count,
            "behavior_target_l2_rms": None if not self.count else np.sqrt(self.distance_sq_sum / self.count),
            "behavior_target_l2_max": None if not self.count else self.distance_max,
        }


def replay_source(policy, dataset_path, source_policy, seeds, frame_stack, self_replay):
    import torch

    aggregate = RunningActions()
    error_sum = error_abs_sum = 0.0
    error_max = 0.0
    error_elements = 0
    exact_equal = True
    with h5py.File(dataset_path, "r") as handle:
        content_policy = str(decode(handle.attrs.get("policy_id", "")))
        if content_policy != source_policy:
            raise RuntimeError(
                f"Manifest/content policy mismatch: expected={source_policy}, actual={content_policy}"
            )
        lookup = episode_lookup(handle)
        missing = [seed for seed in seeds if seed not in lookup]
        if missing:
            raise RuntimeError(f"Dataset {dataset_path} is missing requested seeds: {missing}")
        action_dimension = None
        for seed in seeds:
            group = lookup[seed]
            if "obs" not in group or "actions" not in group:
                raise RuntimeError(f"missing obs/actions for seed={seed}: {dataset_path}")
            observations = group["obs"]
            actions = group["actions"]
            keys = sorted(observations.keys())
            if not keys:
                raise RuntimeError(f"empty observation dictionary for seed={seed}")
            length = int(actions.shape[0])
            if any(int(observations[key].shape[0]) != length for key in keys):
                raise RuntimeError(f"observation/action length mismatch for seed={seed}")
            current_action_dimension = int(np.prod(actions.shape[1:]))
            if action_dimension is None:
                action_dimension = current_action_dimension
            elif current_action_dimension != action_dimension:
                raise RuntimeError("Action dimension changes between episodes")
            policy.start_episode()
            seed_everything(seed)
            history = ObservationHistory(frame_stack)
            for timestep in range(length):
                observation = {
                    key: np.asarray(observations[key][timestep]).copy() for key in keys
                }
                previous_action = None if timestep == 0 else np.asarray(actions[timestep - 1])
                policy_observation = history.append(
                    observation, timestep, previous_action, action_dimension
                )
                with torch.no_grad():
                    target = np.asarray(policy(ob=policy_observation)).copy()
                behavior = np.asarray(actions[timestep]).copy()
                aggregate.update(target, behavior)
                if self_replay:
                    difference = np.asarray(target, dtype=np.float64) - np.asarray(behavior, dtype=np.float64)
                    finite = difference[np.isfinite(difference)]
                    if finite.size:
                        error_sum += float(np.square(finite).sum())
                        error_abs_sum += float(np.abs(finite).sum())
                        error_max = max(error_max, float(np.abs(finite).max()))
                        error_elements += int(finite.size)
                    exact_equal = exact_equal and np.array_equal(target, behavior)
    result = aggregate.summary(action_dimension)
    if self_replay:
        result.update({
            "action_mse": None if not error_elements else error_sum / error_elements,
            "action_mae": None if not error_elements else error_abs_sum / error_elements,
            "max_abs_error": error_max,
            "exact_action_reproduction": exact_equal,
        })
    return result


def severity(status):
    return {"PASS": 0, "WARN": 1, "FAIL": 2}[status]


def main():
    args = parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("--seeds contains duplicates")
    manifest, datasets, training_runs_root, output = validate_paths(args)
    checkpoint_path = checkpoint_from_rnn_dataset(datasets["bc_rnn"], training_runs_root)
    policy, _, config, device, parameter_count, frame_stack, stochastic = load_frozen_policy(
        checkpoint_path, args.device
    )

    self_result = replay_source(
        policy, datasets["bc_rnn"], "bc_rnn", args.seeds, frame_stack, self_replay=True
    )
    self_result["seeds"] = args.seeds
    self_result["episodes"] = len(args.seeds)
    self_result["stochastic_policy"] = stochastic
    if self_result["nan_count"] or self_result["inf_count"] or not self_result["num_transitions"]:
        self_status = "FAIL"
        self_reason = "non-finite action or empty replay"
    elif self_result["exact_action_reproduction"]:
        self_status = "PASS"
        self_reason = "stored actions reproduced exactly"
    else:
        self_status = "WARN"
        self_reason = (
            "finite replay differs from stored actions; GMM sampling is stochastic"
            if stochastic else
            "finite replay differs from stored actions; inspect recurrent input reconstruction"
        )
    self_result.update({"status": self_status, "status_reason": self_reason})

    off_policy = {}
    for source in ("bc_transformer", "bc_gmm"):
        result = replay_source(
            policy, datasets[source], source, args.seeds, frame_stack, self_replay=False
        )
        result["seeds"] = args.seeds
        result["episodes"] = len(args.seeds)
        result["status"] = (
            "FAIL" if result["nan_count"] or result["inf_count"] or not result["num_transitions"]
            else "PASS"
        )
        off_policy[source] = result
    off_status = max((row["status"] for row in off_policy.values()), key=severity)

    behavior_distance = {
        "bc_rnn": {
            key: self_result[key] for key in (
                "behavior_target_l2_mean", "behavior_target_l2_rms", "behavior_target_l2_max"
            )
        }
    }
    for source, row in off_policy.items():
        behavior_distance[source] = {
            key: row[key] for key in (
                "behavior_target_l2_mean", "behavior_target_l2_rms", "behavior_target_l2_max"
            )
        }
    final_status = max((self_status, off_status), key=severity)
    report = {
        "stage": "2.0_frozen_rnn_target_validation",
        "stage1_data_read_only": True,
        "manifest": str(Path(args.manifest).resolve()),
        "selected_datasets": datasets,
        "rnn_checkpoint": str(checkpoint_path),
        "device": str(device),
        "frozen_parameter_count": parameter_count,
        "all_parameters_frozen": True,
        "model_eval": True,
        "frame_stack": frame_stack,
        "rnn_self_replay": self_result,
        "off_policy_history": {**off_policy, "status": off_status},
        "behavior_target_distance": behavior_distance,
        "final_status": final_status,
        "critic_training_started": False,
        "sac_started": False,
    }
    atomic_json(output, report)

    print("=" * 50)
    print("Stage 2.0 Frozen RNN Target Sanity Check")
    print("=" * 50)
    print("\nRNN self replay")
    print("episodes:", len(args.seeds))
    print("MSE:", self_result["action_mse"])
    print("MAE:", self_result["action_mae"])
    print("Max error:", self_result["max_abs_error"])
    print("NaN / Inf:", self_result["nan_count"], "/", self_result["inf_count"])
    print("Status:", self_status)
    print("\nOff-policy history")
    print(f"{'Source':12s} {'NaN':>5s} {'Inf':>5s} {'ActionNormMean':>16s} {'ActionNormMax':>15s}")
    for source in ("bc_transformer", "bc_gmm"):
        row = off_policy[source]
        print(
            f"{DISPLAY_NAMES[source]:12s} {row['nan_count']:5d} {row['inf_count']:5d} "
            f"{row['action_l2_mean']:16.6f} {row['action_l2_max']:15.6f}"
        )
    print("\nOff-policy status:", off_status)
    print("\nFINAL STATUS:", final_status)
    print("=" * 50)


if __name__ == "__main__":
    main()
