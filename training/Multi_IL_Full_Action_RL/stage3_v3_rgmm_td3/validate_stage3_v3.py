#!/usr/bin/env python3
"""CPU/NPU synthetic validation of Stage3-v3 invariants (no environment rollout)."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

from stage3_v3_actor import (BatchedGMMExecutor, _zero_rows, distribution_tensors,
                             flat_to_obs, load_exact_actor, module_hash,
                             recurrent_distributions)
from stage3_v3_agent import RecurrentGMMTD3, bc_lambda

ROOT = Path(__file__).resolve().parents[3]
STAGE2 = ROOT / "training" / "Multi_IL_Full_Action_RL" / "stage2_new_critic_pretraining"
if str(STAGE2) not in sys.path:
    sys.path.insert(0, str(STAGE2))
from critic_network import build_critic  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-rnn-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default=str(Path(__file__).with_name("stage3_v3_config.json")))
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
    device = torch.device(args.device)
    config = json.load(open(args.config, encoding="utf-8"))
    torch.manual_seed(123)
    actor, rollout, metadata = load_exact_actor(args.bc_rnn_checkpoint, device)
    source_hash = module_hash(actor)
    clone = copy.deepcopy(actor)
    result = clone.load_state_dict(actor.state_dict(), strict=True)
    strict = not result.missing_keys and not result.unexpected_keys and module_hash(clone) == source_hash
    observations = torch.randn(4, 20, 59, device=device)
    steps = torch.arange(20, device=device).repeat(4, 1)
    left, hidden_left = recurrent_distributions(actor.eval(), observations, steps, 10)
    right, hidden_right = recurrent_distributions(clone.eval(), observations, steps, 10)
    keys = ("means_normalized", "scales", "logits", "probs")
    maximum = max(float((distribution_tensors(a)[key] - distribution_tensors(b)[key]).abs().max())
                  for a, b in zip(left, right) for key in keys)
    maximum = max(maximum, *(float((a - b).abs().max())
                               for a, b in zip(hidden_left, hidden_right)))
    reset_mask = torch.tensor([False, True, False, True], device=device)
    hidden_probe = (torch.randn(2, 4, 7, device=device),
                    torch.randn(2, 4, 7, device=device))
    hidden_expected = tuple(value.clone() for value in hidden_probe)
    for value in hidden_expected:
        value[:, reset_mask, :] = 0
    hidden_actual = _zero_rows(hidden_probe, reset_mask)
    hidden_reset_exact = all(torch.equal(actual, expected)
                             for actual, expected in zip(hidden_actual, hidden_expected))

    critic = build_critic(59, 14, [256, 256], "relu", True, device)
    scale = torch.as_tensor(rollout.action_normalization_stats["actions"]["scale"],
                            dtype=torch.float32, device=device).reshape(1, 1, 1, 14)
    offset = torch.as_tensor(rollout.action_normalization_stats["actions"]["offset"],
                            dtype=torch.float32, device=device).reshape(1, 1, 1, 14)
    # Exercise the same batched rollout path used by the 16-env trainer. This
    # catches hidden-state packing and action-shape regressions without
    # starting MuJoCo environments.
    batched_obs = flat_to_obs(observations[:, 0])
    observation_rows = [
        {key: value[index].detach().cpu().numpy() for key, value in batched_obs.items()}
        for index in range(4)
    ]
    executor = BatchedGMMExecutor(clone.eval(), scale, offset, 4, horizon=10)
    batched_actions = executor.actions_for(range(4), observation_rows)
    executor.reset_indices([1])
    reset_actions = executor.actions_for(range(4), observation_rows)
    batched_executor_ok = (
        len(batched_actions) == 4
        and all(np.asarray(action).shape == (14,) for action in batched_actions)
        and all(np.isfinite(action).all() for action in batched_actions)
        and len(reset_actions) == 4
        and all(np.asarray(action).shape == (14,) for action in reset_actions)
    )
    agent = RecurrentGMMTD3(clone.train(), critic, config, device, scale, offset)
    frozen_hash = module_hash(agent.actor)
    gate_9999 = agent.maybe_open_gate(9999, True, True)
    frozen = module_hash(agent.actor) == frozen_hash and not gate_9999
    gate_10000 = agent.maybe_open_gate(10000, True, True)
    latched = gate_10000 and agent.maybe_open_gate(10001, False, False) is False and agent.actor_gate_open
    batch = {
        "observations": observations.detach().cpu().numpy(),
        "next_observations": (observations + 0.01).detach().cpu().numpy(),
        "actions": np.zeros((4, 20, 14), np.float32),
        "rewards": np.zeros((4, 20, 1), np.float32),
        "terminals": np.zeros((4, 20, 1), np.float32),
        "episode_steps": np.tile(np.arange(20, dtype=np.int64), (4, 1)),
        "is_offline": np.asarray([1, 1, 0, 0], np.float32),
    }
    # Use valid actions from the transferred GMM for the synthetic NLL update;
    # arbitrary all-zero actions can be many standard deviations off-support
    # and obscure the finite-gradient invariant with an artificial 1e8 loss.
    with torch.no_grad():
        synthetic_dists, _ = recurrent_distributions(
            agent.actor, observations, steps, 10)
        for time_index, distribution in enumerate(synthetic_dists):
            normalized = distribution_tensors(distribution)["means_normalized"][:, 0]
            batch["actions"][:, time_index] = (
                normalized * scale.reshape(1, 14) + offset.reshape(1, 14)
            ).detach().cpu().numpy()
    before = module_hash(agent.actor)
    metrics = agent.actor_update(batch, 10000)
    actor_changed = module_hash(agent.actor) != before
    checks = {
        "exact_native_class": metadata["network_class"] == "RNNGMMActorNetwork",
        "parameter_count": metadata["parameter_count"] == 2078945,
        "strict_transfer": strict, "equivalence": maximum <= 1e-5,
        "frozen_before_10k": frozen, "gate_latched_at_10k": latched,
        "actor_update_changes_actor": actor_changed,
        "actor_loss_finite": all(np.isfinite(value) for value in metrics.values()),
        "bc_schedule": bool(np.allclose(
            [bc_lambda(config["bc_lambda_schedule"], step)
             for step in (10000, 100000, 300000, 500000)],
            [1.0, 1.0, 0.2, 0.0], atol=1e-6, rtol=0.0)),
        "no_online_cql": config["online_cql"]["enabled"] is False,
        "no_sac_alpha": "alpha_lr" not in config and "target_entropy" not in config,
        "exact_50_50": config["offline_fraction"] == config["online_fraction"] == 0.5,
        "utd_one": config["utd"] == 1,
        "policy_delay_four": config["policy_delay"] == 4,
        "batched_executor": batched_executor_ok,
        "hidden_reset_exact": hidden_reset_exact,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps({"status": status, "checks": checks,
                      "equivalence_max_abs_diff": maximum,
                      "actor_metrics": metrics}, indent=2))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
