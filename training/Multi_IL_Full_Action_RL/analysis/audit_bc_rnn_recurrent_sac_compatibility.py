#!/usr/bin/env python3
"""Read-only BC-RNN -> recurrent SAC compatibility audit.

This script restores the real robomimic policy, sequentially replays complete
Stage 1 episodes, and strictly loads the three Stage 2.1 critic checkpoints.
It never trains or mutates a checkpoint / dataset.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
STAGE2_DIR = PROJECT_DIR / "stage2_critic_pretraining"
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))

from critic_network import TwinCritic  # noqa: E402
import robomimic.models.base_nets as BaseNets  # noqa: E402
import robomimic.models.policy_nets as PolicyNets  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402


DEFAULT_BC_RNN = (
    "/data/home/3220251075/lerobot_workspace/training_runs/Pure IL/"
    "two_arm_transport_bc_rnn_official_ph_low_dim/"
    "two_arm_transport_bc_rnn_official_ph_low_dim/20260811151701/models/"
    "model_epoch_1000_low_dim_v15_success_0.9.pth"
)
DEFAULT_DATASET = (
    "/data/home/3220251075/lerobot_workspace/training_runs/"
    "Multi_IL_Full_Action_RL/stage1_rollout_collection/datasets/"
    "20260820_160603/bc_rnn/transitions.hdf5"
)
STAGE2_ROOT = (
    "/data/home/3220251075/lerobot_workspace/training_runs/"
    "Multi_IL_Full_Action_RL/stage2_1_critic_pretraining/20260826_163642"
)
DEFAULT_CRITICS = {
    "random": f"{STAGE2_ROOT}/random_critic/model_init.pth",
    "rnn_only": f"{STAGE2_ROOT}/rnn_only_critic/checkpoints/best.pth",
    "multi_il": f"{STAGE2_ROOT}/multi_il_critic/checkpoints/best.pth",
}
CANONICAL_SHAPES = OrderedDict([
    ("robot0_eef_pos", [3]),
    ("robot0_eef_quat", [4]),
    ("robot0_gripper_qpos", [2]),
    ("robot1_eef_pos", [3]),
    ("robot1_eef_quat", [4]),
    ("robot1_gripper_qpos", [2]),
    ("object", [41]),
])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_BC_RNN)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--random-critic", default=DEFAULT_CRITICS["random"])
    parser.add_argument("--rnn-only-critic", default=DEFAULT_CRITICS["rnn_only"])
    parser.add_argument("--multi-il-critic", default=DEFAULT_CRITICS["multi_il"])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--output", default=str(THIS_DIR / "bc_rnn_recurrent_sac_compatibility_audit.json")
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
        torch.npu.set_device(name)
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


def encoder_audit(actor):
    group = actor.nets["encoder"]
    obs_encoder = group.nets["obs"]
    rows = []
    for key in obs_encoder.obs_shapes:
        network = obs_encoder.obs_nets[key]
        rows.append({
            "key": key,
            "shape": list(obs_encoder.obs_shapes[key]),
            "encoder": None if network is None else network.__class__.__name__,
            "parameter_count": 0 if network is None else sum(
                int(parameter.numel()) for parameter in network.parameters()
            ),
        })
    return {
        "class": obs_encoder.__class__.__name__,
        "concat_order": [row["key"] for row in rows],
        "fields": rows,
        "output_dim": int(obs_encoder.output_shape()[0]),
        "direct_concat": not any(row["parameter_count"] for row in rows),
    }


def named_parameter_shapes(module):
    return [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "requires_grad": bool(parameter.requires_grad),
        }
        for name, parameter in module.named_parameters()
    ]


def module_tree(module):
    return [
        {"name": name or "<root>", "class": child.__class__.__name__}
        for name, child in module.named_modules()
    ]


def sequential_replay(rollout_policy, dataset_path, observation_keys, episode_count):
    rows = []
    all_differences = []
    with h5py.File(dataset_path, "r") as handle:
        policy_id = decode(handle.attrs.get("policy_id", ""))
        if policy_id and policy_id != "bc_rnn":
            raise RuntimeError(f"Expected bc_rnn Stage1 dataset, got {policy_id!r}")
        dataset_keys = json.loads(decode(handle.attrs["canonical_observation_keys"]))
        episode_names = sorted(handle["episodes"].keys())[:episode_count]
        if len(episode_names) < 2:
            raise RuntimeError("At least two complete Stage1 episodes are required")
        for episode_name in episode_names:
            episode = handle[f"episodes/{episode_name}"]
            length = int(episode.attrs.get("episode_length", len(episode["actions"])))
            if length != len(episode["actions"]):
                raise RuntimeError(f"Incomplete episode {episode_name}: action length mismatch")
            seed = int(episode.attrs.get("initial_seed", episode["initial_seed"][0]))
            rollout_policy.start_episode()
            seed_everything(seed)
            predicted = []
            for timestep in range(length):
                observation = OrderedDict(
                    (key, np.asarray(episode[f"obs/{key}"][timestep]).copy())
                    for key in observation_keys
                )
                predicted.append(np.asarray(rollout_policy(ob=observation), dtype=np.float32))
            predicted = np.stack(predicted)
            saved = np.asarray(episode["actions"], dtype=np.float32)
            difference = predicted - saved
            all_differences.append(difference)
            rows.append({
                "episode": episode_name,
                "initial_seed": seed,
                "length": length,
                "mse": float(np.square(difference).mean()),
                "mae": float(np.abs(difference).mean()),
                "max_abs_error": float(np.abs(difference).max()),
                "allclose_atol_1e-6": bool(np.allclose(predicted, saved, rtol=0.0, atol=1e-6)),
            })
    difference = np.concatenate(all_differences, axis=0)
    return {
        "dataset_canonical_keys": dataset_keys,
        "episodes": rows,
        "total_transitions": int(difference.shape[0]),
        "aggregate_mse": float(np.square(difference).mean()),
        "aggregate_mae": float(np.abs(difference).mean()),
        "aggregate_max_abs_error": float(np.abs(difference).max()),
        "all_episodes_allclose_atol_1e-6": all(row["allclose_atol_1e-6"] for row in rows),
        "method": (
            "For each complete episode: RolloutPolicy.start_episode(), reseed with initial_seed, "
            "then call the restored policy once per stored observation in temporal order."
        ),
    }


def critic_audit(path):
    payload = torch.load(path, map_location="cpu")
    state = payload["critic_state_dict"]
    architecture = payload.get("architecture") or {
        "state_dim": 59,
        "action_dim": 14,
        "hidden_dims": [256, 256],
        "twin_q": True,
    }
    state_dim = int(architecture.get("state_dim", 59))
    action_dim = int(architecture.get("action_dim", 14))
    hidden_dims = [int(value) for value in architecture.get("hidden_dims", [256, 256])]
    critic = TwinCritic(state_dim, action_dim, hidden_dims)
    critic.load_state_dict(state, strict=True)
    critic.eval()
    with torch.no_grad():
        q1, q2 = critic(torch.zeros(2, state_dim), torch.zeros(2, action_dim))
    recurrent_keys = [key for key in state if any(token in key.lower() for token in ("rnn", "lstm", "gru"))]
    return {
        "path": str(path),
        "strict_load": True,
        "class": critic.__class__.__name__,
        "architecture": {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "input_dim": state_dim + action_dim,
            "hidden_dims": hidden_dims,
            "twin_q": True,
            "q_output_shapes": [list(q1.shape), list(q2.shape)],
        },
        "current_state_action_only": not recurrent_keys,
        "recurrent_parameter_keys": recurrent_keys,
        "parameter_shapes": {key: list(value.shape) for key, value in state.items()},
        "reusable_with_recurrent_actor": bool(
            state_dim == 59 and action_dim == 14 and not recurrent_keys
        ),
    }


def print_report(report):
    actor = report["bc_rnn"]
    replay = report["sequential_replay"]
    rnn = actor["recurrent_core"]
    print("=" * 80)
    print("BC-RNN -> Recurrent SAC Compatibility Audit")
    print("=" * 80)
    print(f"Algorithm / policy: {actor['algorithm_class']} / {actor['policy_class']}")
    print(f"Observation: {actor['observation_encoder']['concat_order']} -> 59D")
    print(
        f"RNN: {rnn['type']} input={rnn['input_size']} hidden={rnn['hidden_size']} "
        f"layers={rnn['num_layers']} bidirectional={rnn['bidirectional']} dropout={rnn['dropout']}"
    )
    print(f"Action head: {actor['action_head']['classification']}")
    print(
        "Sequential replay: "
        f"MSE={replay['aggregate_mse']:.12g} MAE={replay['aggregate_mae']:.12g} "
        f"max_abs={replay['aggregate_max_abs_error']:.12g}"
    )
    print("Stage2 critics reusable:", all(
        item["reusable_with_recurrent_actor"] for item in report["stage2_critics"].values()
    ))
    print("Verdict:", report["verdict"])
    print("Report:", report["output"])
    print("No training was started. Stage 2 and Stage 4 were not modified.")
    print("=" * 80)


def main():
    args = parse_args()
    required = [
        args.checkpoint, args.dataset, args.random_critic,
        args.rnn_only_critic, args.multi_il_critic,
    ]
    missing = [path for path in required if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required audit inputs:\n" + "\n".join(missing))
    if args.episodes < 2:
        raise ValueError("--episodes must be at least 2")
    device = select_device(args.device)

    checkpoint = FileUtils.maybe_dict_from_checkpoint(ckpt_path=args.checkpoint)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint)
    rollout_policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_dict=checkpoint, device=device, verbose=False
    )
    algo = rollout_policy.policy
    actor = algo.nets["policy"]
    if not isinstance(actor, PolicyNets.RNNGMMActorNetwork):
        raise RuntimeError(f"Expected RNNGMMActorNetwork, got {type(actor).__name__}")
    recurrent_wrapper = actor.nets["rnn"]
    recurrent = recurrent_wrapper.nets
    if not isinstance(recurrent, (nn.LSTM, nn.GRU, nn.RNN)):
        raise RuntimeError(f"Unsupported recurrent module: {type(recurrent).__name__}")
    encoder = encoder_audit(actor)
    decoder = actor.nets["decoder"].nets
    modes = int(actor.num_modes)
    action_dim = int(actor.ac_dim)

    # RNN_MIMO_MLP always feeds its observation encoder directly into RNN_Base.
    # actor_layer_dims controls only the optional per-timestep post-RNN MLP.
    pre_mlp = None
    post_mlp = actor.nets["mlp"] if "mlp" in actor.nets else None
    replay = sequential_replay(
        rollout_policy, args.dataset, encoder["concat_order"], args.episodes
    )
    critic_paths = {
        "random": args.random_critic,
        "rnn_only": args.rnn_only_critic,
        "multi_il": args.multi_il_critic,
    }
    critics = {name: critic_audit(path) for name, path in critic_paths.items()}
    canonical_order = list(CANONICAL_SHAPES)
    structure_compatible = bool(
        encoder["direct_concat"]
        and encoder["output_dim"] == 59
        and recurrent.input_size == 59
        and recurrent.hidden_size == 400
        and recurrent.num_layers == 2
        and not recurrent.bidirectional
        and pre_mlp is None
        and post_mlp is None
    )
    all_critics_reusable = all(item["reusable_with_recurrent_actor"] for item in critics.values())
    verdict = (
        "RECURRENT CORE + NEW HEAD"
        if structure_compatible and all_critics_reusable
        else "MAJOR INCOMPATIBILITY"
    )

    report = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "device": str(device),
        "bc_rnn": {
            "algorithm_class": algo.__class__.__name__,
            "policy_class": actor.__class__.__name__,
            "actor_network_class": actor.__class__.__name__,
            "module_repr": repr(actor),
            "module_tree": module_tree(actor),
            "named_parameters": named_parameter_shapes(actor),
            "observation_encoder": encoder,
            "pre_rnn_mlp": None if pre_mlp is None else repr(pre_mlp),
            "post_rnn_mlp": None if post_mlp is None else repr(post_mlp),
            "recurrent_core": {
                "wrapper_class": recurrent_wrapper.__class__.__name__,
                "type": recurrent.__class__.__name__,
                "input_size": int(recurrent.input_size),
                "hidden_size": int(recurrent.hidden_size),
                "num_layers": int(recurrent.num_layers),
                "bidirectional": bool(recurrent.bidirectional),
                "dropout": float(recurrent.dropout),
                "output_shape": ["batch", "time", int(recurrent.hidden_size)],
                "hidden_state_convention": (
                    f"(h_t, c_t), each [{recurrent.num_layers}, batch, {recurrent.hidden_size}]"
                    if isinstance(recurrent, nn.LSTM)
                    else f"h_t [{recurrent.num_layers}, batch, {recurrent.hidden_size}]"
                ),
            },
            "action_head": {
                "classification": "B. GMM",
                "action_dim": action_dim,
                "num_modes": modes,
                "mean": {"module": repr(decoder["mean"]), "output_shape": ["batch", "time", modes, action_dim]},
                "scale": {"module": repr(decoder["scale"]), "output_shape": ["batch", "time", modes, action_dim]},
                "logits": {"module": repr(decoder["logits"]), "output_shape": ["batch", "time", modes]},
                "min_std": float(actor.min_std),
                "std_activation": actor.std_activation,
                "low_noise_eval": bool(actor.low_noise_eval),
                "deterministic_evaluation_convention": (
                    "Not mathematically deterministic: eval forces every component scale to 1e-4, "
                    "then samples a categorical component from logits and a Normal action. Exact "
                    "reproduction requires the same episode seed and sampling order."
                ),
                "action_semantics": (
                    "use_tanh=False; component means are tanh-bounded, but the final low-noise "
                    "Normal sample is not clipped and can slightly exceed [-1, 1]."
                ),
            },
            "checkpoint_config": {
                "actor_layer_dims": list(config.algo.actor_layer_dims),
                "rnn_horizon_reset_interval": int(config.algo.rnn.horizon),
                "open_loop": bool(config.algo.rnn.open_loop),
            },
        },
        "transfer_to_recurrent_sac": {
            "directly_copy": [
                "observation encoder (parameter-free canonical concatenation)",
                "pre-RNN path (identity; no parameters)",
                "all LSTM weights and biases",
                "post-RNN path (identity; no parameters)",
            ],
            "new_parameters": [
                "mu head: Linear(400, 14)",
                "log_std head: Linear(400, 14), followed by SAC log-std clamp",
            ],
            "head_connection": (
                "At every timestep, feed the LSTM output [B,T,400] to separate 400->14 "
                "mu and log_std heads; sample with reparameterization and tanh squash."
            ),
            "gmm_heads_directly_reusable_as_standard_sac_head": False,
        },
        "sequential_replay": replay,
        "stage2_critics": critics,
        "stage2_critic_interpretation": (
            "Each checkpoint strictly loads into TwinCritic: two independent "
            "73->256->256->1 networks. It consumes only the current canonical 59D state "
            "and current 14D action; recurrent actor hidden state is not a critic input."
        ),
        "verdict": verdict,
        "training_started": False,
        "stage2_modified": False,
        "stage4_started_or_modified": False,
        "output": str(Path(args.output)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print_report(report)


if __name__ == "__main__":
    main()
