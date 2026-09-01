#!/usr/bin/env python3
"""Read-only BC-GMM -> SAC Actor structural and action-semantics audit."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
STAGE3_DIR = PROJECT_DIR / "stage3_actor_initialization"
for path in (STAGE3_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from actor_network import build_actor  # noqa: E402
import robomimic.models.policy_nets as PolicyNets  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402


DEFAULT_CHECKPOINT = (
    "/data/home/3220251075/lerobot_workspace/training_runs/Pure IL/"
    "two_arm_transport_bc_gmm_official_ph_low_dim/models/"
    "model_epoch_1850_low_dim_v15_success_0.3.pth"
)
DEFAULT_DATASET = (
    "/data/home/3220251075/lerobot_workspace/training_runs/"
    "Multi_IL_Full_Action_RL/stage1_rollout_collection/datasets/"
    "20260820_160603/bc_gmm/transitions.hdf5"
)
CANONICAL_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot1_eef_pos",
    "robot1_eef_quat",
    "robot1_gripper_qpos",
    "object",
]
CANONICAL_SHAPES = {
    "robot0_eef_pos": [3],
    "robot0_eef_quat": [4],
    "robot0_gripper_qpos": [2],
    "robot1_eef_pos": [3],
    "robot1_eef_quat": [4],
    "robot1_gripper_qpos": [2],
    "object": [41],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--output",
        default=str(THIS_DIR / "bc_gmm_sac_compatibility_audit.json"),
    )
    return parser.parse_args()


def select_device(name):
    if name.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("NPU requested but torch_npu cannot be imported") from exc
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but unavailable")
        torch.npu.set_device(torch.device(name))
    return torch.device(name)


def seed_everything(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(int(seed))


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode(value.item())
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def linear_layers(module):
    rows = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            rows.append({
                "name": name,
                "in_features": int(child.in_features),
                "out_features": int(child.out_features),
                "weight_shape": list(child.weight.shape),
                "bias_shape": list(child.bias.shape),
            })
    return rows


def module_types(module, cls):
    return [name for name, child in module.named_modules() if isinstance(child, cls)]


def encoder_audit(gmm):
    group_encoder = gmm.nets["encoder"]
    observation_encoder = group_encoder.nets["obs"]
    rows = []
    for key in observation_encoder.obs_shapes:
        network = observation_encoder.obs_nets[key]
        randomizers = observation_encoder.obs_randomizers[key]
        rows.append({
            "key": key,
            "shape": list(observation_encoder.obs_shapes[key]),
            "encoder": None if network is None else network.__class__.__name__,
            "encoder_parameter_count": 0 if network is None else sum(
                int(parameter.numel()) for parameter in network.parameters()
            ),
            "randomizers": [
                None if randomizer is None else randomizer.__class__.__name__
                for randomizer in randomizers
            ],
        })
    return {
        "group_encoder_class": group_encoder.__class__.__name__,
        "observation_encoder_class": observation_encoder.__class__.__name__,
        "concat_order": [row["key"] for row in rows],
        "keys": rows,
        "output_dim": int(observation_encoder.output_shape()[0]),
        "has_learned_encoder": any(row["encoder_parameter_count"] for row in rows),
    }


def load_transition(dataset_path, observation_order):
    with h5py.File(dataset_path, "r") as handle:
        policy_id = decode(handle.attrs.get("policy_id", ""))
        if policy_id and policy_id != "bc_gmm":
            raise RuntimeError(f"Expected Stage1 bc_gmm dataset, got {policy_id!r}")
        dataset_keys = json.loads(decode(handle.attrs["canonical_observation_keys"]))
        dataset_shapes = json.loads(decode(handle.attrs["canonical_observation_shapes"]))
        episode_name = sorted(handle["episodes"].keys())[0]
        episode = handle["episodes"][episode_name]
        if "initial_seed" in episode.attrs:
            seed = int(episode.attrs["initial_seed"])
        else:
            seed = int(episode["initial_seed"][0])
        timestep = 0
        observation = OrderedDict()
        for key in observation_order:
            if key not in episode["obs"]:
                raise RuntimeError(f"Stage1 episode lacks checkpoint observation key {key!r}")
            observation[key] = np.asarray(episode["obs"][key][timestep]).copy()
        saved_action = np.asarray(episode["actions"][timestep], dtype=np.float32)
    return {
        "episode": episode_name,
        "seed": seed,
        "timestep": timestep,
        "observation": observation,
        "saved_action": saved_action,
        "dataset_keys": dataset_keys,
        "dataset_shapes": dataset_shapes,
    }


def forward_sanity(rollout_policy, gmm, transition):
    seed = transition["seed"]
    rollout_policy.start_episode()
    # Stage1 reseeds immediately before the first policy call. Timestep zero is
    # deliberately used so no earlier categorical / Gaussian samples are needed.
    seed_everything(seed)
    predicted = np.asarray(rollout_policy(ob=transition["observation"]), dtype=np.float32)
    saved = transition["saved_action"]
    difference = predicted - saved

    prepared = rollout_policy._prepare_observation(transition["observation"])
    with torch.no_grad():
        distribution = gmm.forward_train(prepared)
    raw_distribution = getattr(distribution, "base_dist", distribution)
    probs = raw_distribution.mixture_distribution.probs[0].detach().cpu().numpy()
    component = raw_distribution.component_distribution.base_dist
    means = component.loc[0].detach().cpu().numpy()
    scales = component.scale[0].detach().cpu().numpy()
    nearest_mode = int(np.argmin(np.square(means - predicted[None]).mean(axis=1)))
    return {
        "episode": transition["episode"],
        "initial_seed": seed,
        "timestep": transition["timestep"],
        "saved_action": saved.tolist(),
        "restored_policy_action": predicted.tolist(),
        "mse": float(np.square(difference).mean()),
        "mae": float(np.abs(difference).mean()),
        "max_abs_error": float(np.abs(difference).max()),
        "exact_match": bool(np.array_equal(predicted, saved)),
        "saved_action_range": [float(saved.min()), float(saved.max())],
        "predicted_action_range": [float(predicted.min()), float(predicted.max())],
        "mixture_probabilities": probs.tolist(),
        "highest_probability_mode": int(np.argmax(probs)),
        "nearest_component_mean_to_sample": nearest_mode,
        "component_means_shape": list(means.shape),
        "component_scales_shape": list(scales.shape),
        "component_scale_range_eval": [float(scales.min()), float(scales.max())],
        "pass": bool(np.allclose(predicted, saved, rtol=0.0, atol=1e-6)),
    }


def compare_backbones(gmm_layers, sac_layers, encoder):
    rows = []
    layer_count = max(len(gmm_layers), len(sac_layers))
    for index in range(layer_count):
        left = gmm_layers[index] if index < len(gmm_layers) else None
        right = sac_layers[index] if index < len(sac_layers) else None
        compatible = bool(
            left is not None and right is not None
            and left["weight_shape"] == right["weight_shape"]
            and left["bias_shape"] == right["bias_shape"]
        )
        rows.append({
            "index": index,
            "bc_gmm": left,
            "sac": right,
            "directly_compatible": compatible,
        })
    compatible_count = sum(int(row["directly_compatible"]) for row in rows)
    exact_architecture = bool(
        not encoder["has_learned_encoder"]
        and len(gmm_layers) == len(sac_layers)
        and compatible_count == len(rows)
    )
    if exact_architecture:
        verdict = "FULL BACKBONE COMPATIBLE"
    elif compatible_count:
        verdict = "PARTIALLY COMPATIBLE"
    else:
        verdict = "NOT PRACTICALLY COMPATIBLE"
    return rows, verdict


def print_report(report):
    print("=" * 80)
    print("BC-GMM -> SAC Actor Structural Compatibility Audit")
    print("=" * 80)
    bc = report["bc_gmm"]
    sac = report["sac_actor"]
    print(f"BC-GMM class: {bc['class']}")
    print(f"BC-GMM observation order: {bc['encoder']['concat_order']}")
    print(f"BC-GMM MLP dims: {bc['mlp_dims']}")
    print(f"BC-GMM decoder: {bc['decoder']}")
    print(f"SAC class: {sac['class']}")
    print(f"SAC MLP dims: {sac['mlp_dims']}")
    print("-" * 80)
    print(f"{'layer':<8} {'BC-GMM weight':<22} {'SAC weight':<22} compatible")
    for row in report["backbone_comparison"]:
        left = None if row["bc_gmm"] is None else row["bc_gmm"]["weight_shape"]
        right = None if row["sac"] is None else row["sac"]["weight_shape"]
        print(f"{row['index']:<8} {str(left):<22} {str(right):<22} {row['directly_compatible']}")
    sanity = report["forward_sanity"]
    print("-" * 80)
    print(
        "Stage1 action sanity: "
        f"MSE={sanity['mse']:.12g} MAE={sanity['mae']:.12g} "
        f"max_abs={sanity['max_abs_error']:.12g} pass={sanity['pass']}"
    )
    print(f"Compatibility verdict: {report['compatibility_verdict']}")
    print(f"Recommendation: {report['recommendation']}")
    print(f"Report: {report['output']}")
    print("No Actor training was started. Stage 4 was not modified.")
    print("=" * 80)


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.dataset)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"BC-GMM checkpoint not found: {checkpoint_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Stage1 BC-GMM dataset not found: {dataset_path}")
    device = select_device(args.device)

    checkpoint = FileUtils.maybe_dict_from_checkpoint(ckpt_path=str(checkpoint_path))
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint)
    rollout_policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_dict=checkpoint, device=device, verbose=False
    )
    algo = rollout_policy.policy
    algo.set_eval()
    gmm = algo.nets["policy"]
    if not isinstance(gmm, PolicyNets.GMMActorNetwork):
        raise RuntimeError(f"Restored policy is not GMMActorNetwork: {type(gmm).__name__}")
    encoder = encoder_audit(gmm)
    gmm_backbone = linear_layers(gmm.nets["mlp"])
    gmm_decoder = {
        name: linear_layers(module)[0]
        for name, module in gmm.nets["decoder"].nets.items()
    }

    sac = build_actor(
        state_dim=59, action_dim=14, hidden_dims=(256, 256),
        initial_log_std=-3.0, freeze_log_std=True, device=device,
    )
    sac_backbone = [
        {
            "name": f"fc{index}",
            "in_features": int(layer.in_features),
            "out_features": int(layer.out_features),
            "weight_shape": list(layer.weight.shape),
            "bias_shape": list(layer.bias.shape),
        }
        for index, layer in enumerate(sac.fcs)
    ]
    comparison, verdict = compare_backbones(gmm_backbone, sac_backbone, encoder)

    transition = load_transition(dataset_path, encoder["concat_order"])
    dataset_order_matches_checkpoint = transition["dataset_keys"] == encoder["concat_order"]
    checkpoint_order_matches_canonical = encoder["concat_order"] == CANONICAL_KEYS
    dataset_shapes_match = all(
        list(transition["dataset_shapes"][key]) == CANONICAL_SHAPES[key]
        for key in CANONICAL_KEYS
    )
    sanity = forward_sanity(rollout_policy, gmm, transition)

    recommendation = (
        "A. backbone transfer + head distillation"
        if verdict == "FULL BACKBONE COMPATIBLE"
        else "B. direct BC-GMM -> current SAC Actor action distillation"
    )
    report = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "device": str(device),
        "bc_gmm": {
            "algo_class": algo.__class__.__name__,
            "class": gmm.__class__.__name__,
            "module_repr": repr(gmm),
            "observation_input_dim": encoder["output_dim"],
            "encoder": encoder,
            "mlp_dims": [
                gmm_backbone[0]["in_features"],
                *[row["out_features"] for row in gmm_backbone],
            ],
            "mlp_linear_layers": gmm_backbone,
            "activation": gmm.nets["mlp"]._act.__name__,
            "layer_norm_modules": module_types(gmm.nets["mlp"], nn.LayerNorm),
            "dropout_modules": module_types(gmm.nets["mlp"], nn.Dropout),
            "decoder": gmm_decoder,
            "action_dim": int(gmm.ac_dim),
            "num_modes": int(gmm.num_modes),
            "min_std": float(gmm.min_std),
            "std_activation": gmm.std_activation,
            "low_noise_eval": bool(gmm.low_noise_eval),
            "use_tanh_distribution": bool(gmm.use_tanh),
            "deterministic_action_convention": (
                "In eval with low_noise_eval=True, forward still calls "
                "MixtureSameFamily.sample(): categorical component sampled from logits, "
                "then a Normal sample with scale 1e-4. It is not argmax-mode mean or "
                "weighted mixture mean; reproducibility requires the Stage1 RNG seed/order."
            ),
            "action_range_semantics": (
                "Component means are tanh-bounded because use_tanh=False; the subsequent "
                "1e-4 Normal sample is not clipped and can numerically exceed [-1, 1] slightly."
            ),
            "checkpoint_actor_layer_dims": list(config.algo.actor_layer_dims),
            "checkpoint_gmm_num_modes": int(config.algo.gmm.num_modes),
            "observation_normalization_applied": (
                rollout_policy.obs_normalization_stats is not None
            ),
            "action_unnormalization_applied_by_rollout_policy": (
                rollout_policy.action_normalization_stats is not None
            ),
        },
        "sac_actor": {
            "class": sac.__class__.__name__,
            "builder_file": str(STAGE3_DIR / "actor_network.py"),
            "mlp_dims": [59, 256, 256],
            "backbone_linear_layers": sac_backbone,
            "activation": "torch.nn.functional.relu",
            "shared_backbone": True,
            "layer_norm": bool(sac.layer_norm),
            "dropout": False,
            "residual": False,
            "mu_head": {
                "name": "last_fc",
                "weight_shape": list(sac.last_fc.weight.shape),
                "bias_shape": list(sac.last_fc.bias.shape),
            },
            "log_std_head": {
                "name": "last_fc_log_std",
                "type": "network output from nn.Linear; its weights/bias are parameters",
                "weight_shape": list(sac.last_fc_log_std.weight.shape),
                "bias_shape": list(sac.last_fc_log_std.bias.shape),
                "initialized_bias": -3.0,
                "frozen_in_current_distillation": not any(
                    parameter.requires_grad for parameter in sac.last_fc_log_std.parameters()
                ),
                "clamp": [-20, 2],
            },
            "deterministic_action": "tanh(mu)",
            "stochastic_action": "tanh(Normal(mu, exp(log_std)))",
            "action_dim": 14,
        },
        "state_ordering": {
            "checkpoint_encoder_order": encoder["concat_order"],
            "stage1_dataset_canonical_order": transition["dataset_keys"],
            "required_canonical_order": CANONICAL_KEYS,
            "dataset_order_matches_checkpoint": dataset_order_matches_checkpoint,
            "checkpoint_order_matches_required_canonical": checkpoint_order_matches_canonical,
            "canonical_shapes_match": dataset_shapes_match,
            "equivalent_59d_input": bool(
                encoder["output_dim"] == 59
                and not encoder["has_learned_encoder"]
                and dataset_order_matches_checkpoint
                and checkpoint_order_matches_canonical
                and dataset_shapes_match
            ),
        },
        "backbone_comparison": comparison,
        "compatibility_verdict": verdict,
        "direct_copy_layers": [
            {
                "bc_gmm": row["bc_gmm"]["name"],
                "sac": row["sac"]["name"],
            }
            for row in comparison if row["directly_compatible"]
        ],
        "forward_sanity": sanity,
        "recommendation": recommendation,
        "training_started": False,
        "stage4_modified": False,
        "output": str(Path(args.output)),
    }
    if not report["state_ordering"]["equivalent_59d_input"]:
        report["recommendation"] = (
            "BLOCKED: BC-GMM and SAC do not currently receive the same canonical 59D state"
        )
    if not sanity["pass"]:
        report["recommendation"] = (
            "BLOCKED pending RNG/action-preprocessing diagnosis: restored BC-GMM did not "
            "reproduce the saved Stage1 timestep-0 action"
        )
    atomic_json(args.output, report)
    print_report(report)


if __name__ == "__main__":
    main()
