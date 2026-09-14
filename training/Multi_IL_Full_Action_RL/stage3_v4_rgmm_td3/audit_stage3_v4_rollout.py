#!/usr/bin/env python3
"""Read-only checkpoint-backed audit of the actual online GMM executor."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

V3 = Path(__file__).resolve().parents[1] / "stage3_v3_rgmm_td3"
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))
from stage3_v3_actor import (BatchedGMMExecutor, CANONICAL_KEYS,
                             distribution_tensors, flat_to_obs,
                             load_exact_actor, obs_to_flat)


def rng_snapshot(device):
    cpu = torch.get_rng_state()
    npu = torch.npu.get_rng_state() if device.type == "npu" else None
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu, npu, cuda


def rng_restore(device, snapshot):
    torch.set_rng_state(snapshot[0])
    if snapshot[1] is not None:
        torch.npu.set_rng_state(snapshot[1])
    if snapshot[2] is not None:
        torch.cuda.set_rng_state(snapshot[2], device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-rnn-checkpoint", required=True)
    parser.add_argument("--expert-dataset", help="defaults to the dataset embedded in BC checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output", help="optional JSON file; existing files are never overwritten")
    args = parser.parse_args()
    if not 1 <= args.samples <= 64:
        parser.error("--samples must be between 1 and 64")
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
    device = torch.device(args.device)
    actor, rollout, metadata = load_exact_actor(args.bc_rnn_checkpoint, device)
    cfg = json.loads(rollout.policy.global_config.dump())
    configured_data = cfg["train"]["data"]
    embedded = configured_data[0]["path"] if isinstance(configured_data, list) else configured_data
    dataset = Path(args.expert_dataset or embedded)
    with h5py.File(dataset, "r") as handle:
        first = handle["data"]["demo_0"]["obs"]
        observation = {key: np.asarray(first[key][0], np.float32) for key in CANONICAL_KEYS}
    flat = torch.as_tensor(np.stack([obs_to_flat(observation)] * args.samples),
                           dtype=torch.float32, device=device)
    scale = torch.as_tensor(rollout.action_normalization_stats["actions"]["scale"],
                            dtype=torch.float32, device=device).reshape(1, 1, 1, 14)
    offset = torch.as_tensor(rollout.action_normalization_stats["actions"]["offset"],
                             dtype=torch.float32, device=device).reshape(1, 1, 1, 14)
    with torch.no_grad():
        actor.train()
        learned_dist, _ = actor.forward_train_step(flat_to_obs(flat), rnn_state=None)
        learned = distribution_tensors(learned_dist)
        actor.eval()
        rollout_dist, _ = actor.forward_train_step(flat_to_obs(flat), rnn_state=None)
        actual = distribution_tensors(rollout_dist)
        executor = BatchedGMMExecutor(actor, scale, offset, args.samples, 10)
        snapshot = rng_snapshot(device)
        executed = np.stack(executor.actions_for(
            list(range(args.samples)), [observation] * args.samples,
            external_noise_std=0.0, action_low=None, action_high=None))
        rng_restore(device, snapshot)
        # Mirror torch.distributions.MixtureSameFamily.sample: categorical
        # first, then all component normals, then gather the selected mode.
        mode = rollout_dist.mixture_distribution.sample()
        all_samples = rollout_dist.component_distribution.sample()
        chosen = all_samples[torch.arange(args.samples, device=device), mode]
        reconstructed = (chosen * scale.reshape(1, 14) + offset.reshape(1, 14))
        selected_mean = (actual["means_normalized"][
            torch.arange(args.samples, device=device), mode]
            * scale.reshape(1, 14) + offset.reshape(1, 14))
        reconstructed_np = reconstructed.cpu().numpy()
        selected_mean_np = selected_mean.cpu().numpy()
        noise = executed - selected_mean_np
        sigma = actual["scales"]
        learned_sigma = learned["scales"]
        mean_diff = float((learned["means_normalized"] - actual["means_normalized"]).abs().max())
        probs_diff = float((learned["probs"] - actual["probs"]).abs().max())
        fixed_std = bool(torch.allclose(sigma, torch.full_like(sigma, 1e-4)))
        reconstruction_ok = bool(np.allclose(executed, reconstructed_np, atol=1e-5))
        result = {
            "status": "PASS" if (metadata["low_noise_eval"] and fixed_std and
                                 mean_diff <= 1e-5 and probs_diff <= 1e-5 and
                                 reconstruction_ok) else "FAIL",
            "decision": ("Case A: categorical mode + fixed 1e-4 Gaussian"
                         if fixed_std and reconstruction_ok else "UNRESOLVED; inspect mismatch"),
            "online_rollout_actor_mode": "eval (temporarily inside BatchedGMMExecutor)",
            "observation_source": f"{dataset}:data/demo_0/obs[0]",
            "recurrent_hidden": "zero/None at episode start",
            "learned_sigma_mean": float(learned_sigma.mean()),
            "learned_sigma_min": float(learned_sigma.min()),
            "learned_sigma_max": float(learned_sigma.max()),
            "rollout_sigma_mean": float(sigma.mean()),
            "rollout_sigma_min": float(sigma.min()),
            "rollout_sigma_max": float(sigma.max()),
            "mean_max_abs_diff_train_vs_eval": mean_diff,
            "probs_max_abs_diff_train_vs_eval": probs_diff,
            "executor_reconstruction_max_abs_diff": float(np.max(np.abs(executed - reconstructed_np))),
            "selected_mode": mode.cpu().tolist(),
            "executed_action": executed.tolist(),
            "selected_component_mean": selected_mean_np.tolist(),
            "executed_minus_selected_mean": noise.tolist(),
            "mean_abs_noise": float(np.mean(np.abs(noise))),
            "max_abs_noise": float(np.max(np.abs(noise))),
            "noise_units": "environment action (after checkpoint normalization scale and offset)",
        }
    if args.output:
        with open(args.output, "x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
