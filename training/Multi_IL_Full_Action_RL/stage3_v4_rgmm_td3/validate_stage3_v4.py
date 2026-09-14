#!/usr/bin/env python3
"""Checkpoint-backed, no-environment Stage3-v4 contract smoke test."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

V3 = Path(__file__).resolve().parents[1] / "stage3_v3_rgmm_td3"
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))
from stage3_v3_actor import (distribution_tensors, load_exact_actor, module_hash,
                             recurrent_distributions, target_final_distribution)
from stage3_v4_agent import RecurrentGMMTD3
from stage3_v4_gmm_math import single_expected_q
from stage3_v4_boundary import aligned_start
from stage3_v4_replay import final_transition

ROOT = Path(__file__).resolve().parents[3]
STAGE2 = ROOT / "training" / "Multi_IL_Full_Action_RL" / "stage2_new_critic_pretraining"
if str(STAGE2) not in sys.path:
    sys.path.insert(0, str(STAGE2))
from critic_network import build_critic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc-rnn-checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default=str(Path(__file__).with_name("stage3_v4_config.json")))
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
    device = torch.device(args.device)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    torch.manual_seed(123)
    np.random.seed(123)
    actor, rollout, metadata = load_exact_actor(args.bc_rnn_checkpoint, device)
    clone = copy.deepcopy(actor).to(device)
    strict_result = clone.load_state_dict(actor.state_dict(), strict=True)
    observations = torch.randn(4, 11, 59, device=device)
    steps = torch.arange(11, device=device).repeat(4, 1)
    left, _ = recurrent_distributions(actor.eval(), observations, steps, 10)
    right, _ = recurrent_distributions(clone.eval(), observations, steps, 10)
    keys = ("means_normalized", "scales", "logits", "probs")
    transfer_diff = max(float((distribution_tensors(a)[key] - distribution_tensors(b)[key]).abs().max())
                        for a, b in zip(left, right) for key in keys)
    with torch.no_grad():
        optimized, _ = target_final_distribution(actor, observations, steps, 10)
    target_diff = max(float((distribution_tensors(left[-1])[key]
                             - distribution_tensors(optimized)[key]).abs().max())
                      for key in keys)
    scale = torch.as_tensor(rollout.action_normalization_stats["actions"]["scale"],
                            dtype=torch.float32, device=device).reshape(1, 1, 1, 14)
    offset = torch.as_tensor(rollout.action_normalization_stats["actions"]["offset"],
                             dtype=torch.float32, device=device).reshape(1, 1, 1, 14)
    critic = build_critic(59, 14, [256, 256], "relu", True, device)
    agent = RecurrentGMMTD3(clone.train(), critic, config, device, scale, offset)

    vector_expected, vector_q, _, _ = agent._expected_q_sequence(
        observations[:, :10], left[:10])
    per_step_expected = torch.stack([
        agent._expected_q(agent.critic, observations[:, index], left[index],
                          twin_min=False)[0]
        for index in range(10)], dim=1)

    # Learned std remains in the network but cannot enter the RL objective.
    train_distributions, _ = recurrent_distributions(
        actor.train(), observations, steps, 10)
    distribution = train_distributions[0]
    params = distribution_tensors(distribution)
    eps = torch.ones(4, 5, 1, 14, device=device)
    mean_only, q1, q2, _, actions = agent._expected_q(
        agent.critic, observations[:, 0], distribution, twin_min=False)
    manual_q = agent.critic.q1(
        observations[:, 0, None].expand(-1, 5, -1).reshape(-1, 59),
        actions.reshape(-1, 14)).reshape(4, 5)
    manual = (params["probs"] * manual_q).sum(-1)
    with torch.no_grad():
        target_distribution, _ = target_final_distribution(
            agent.target_actor, observations, steps, 10)
        target_mean, _, _, _, _ = agent._expected_q(
            agent.target_critic, observations[:, -1], target_distribution)
        target_stds = distribution_tensors(target_distribution)["scales"]
        sampled_diagnostic, _, _, _, _ = single_expected_q(
            agent.critic, observations[:, 0], distribution,
            agent.action_scale, agent.action_offset,
            twin_min=False, epsilon=eps)
    frozen_hash = module_hash(agent.actor)
    critic_hash = module_hash(agent.critic)
    critic_sequence = {
        "observations": observations.detach().cpu().numpy(),
        "next_observations": observations.detach().cpu().numpy(),
        "actions": np.zeros((4, 11, 14), np.float32),
        "rewards": np.zeros((4, 11, 1), np.float32),
        "terminals": np.zeros((4, 11, 1), np.float32),
        "episode_steps": np.tile(np.arange(11, dtype=np.int64), (4, 1)),
    }
    agent.critic_update(final_transition(critic_sequence), critic_sequence,
                        collect_metrics=False)
    critic_warmup_changed = module_hash(agent.critic) != critic_hash
    gate_before = agent.maybe_open_gate(9999, True, True)
    gate_at = agent.maybe_open_gate(10000, True, True)

    # Actor sequence starts on a real zero-hidden boundary and is one horizon.
    fake_episode = {"episode_steps": np.arange(30), "actions": np.zeros((30, 14))}
    aligned_starts = [aligned_start(fake_episode, 10, 10, np.random.default_rng(index))
                      for index in range(8)]
    batch = {
        "observations": observations[:, :10].detach().cpu().numpy(),
        "next_observations": observations[:, :10].detach().cpu().numpy(),
        "actions": np.zeros((4, 10, 14), np.float32),
        "rewards": np.zeros((4, 10, 1), np.float32),
        "terminals": np.zeros((4, 10, 1), np.float32),
        "episode_steps": np.tile(np.arange(10, dtype=np.int64), (4, 1)),
        "is_offline": np.asarray([1, 1, 0, 0], np.float32),
    }
    before = module_hash(agent.actor)
    metrics = agent.actor_update(batch, 10000)
    gradient_keys = ("actor_grad_norm_rnn", "actor_grad_norm_gmm_mean",
                     "actor_grad_norm_gmm_std", "actor_grad_norm_gmm_logits")
    gradients = {key: metrics[key] for key in gradient_keys}
    checks = {
        "exact_native_class": metadata["network_class"] == "RNNGMMActorNetwork",
        "parameter_count": metadata["parameter_count"] == 2078945,
        "strict_transfer": not strict_result.missing_keys and not strict_result.unexpected_keys,
        "step0_output_equivalence": transfer_diff <= 1e-5,
        "target_recurrent_equivalence": target_diff <= 1e-5,
        "warmup_actor_frozen": not gate_before and frozen_hash == module_hash(agent.actor),
        "warmup_critic_changed": critic_warmup_changed,
        "target_low_noise_contract": agent.target_actor.low_noise_eval is True
                                     and not agent.target_actor.training
                                     and torch.allclose(target_stds,
                                                        torch.full_like(target_stds, 1e-4)),
        "gate_opens_at_10k": gate_at and agent.actor_gate_open,
        "actor_updated_after_gate": module_hash(agent.actor) != before,
        "mean_logit_rnn_gradients": all(np.isfinite(value) for value in gradients.values())
                                    and gradients["actor_grad_norm_gmm_mean"] > 0
                                    and gradients["actor_grad_norm_gmm_logits"] > 0
                                    and gradients["actor_grad_norm_rnn"] > 0,
        "std_head_no_rl_gradient": gradients["actor_grad_norm_gmm_std"] == 0,
        "component_mean_q_batched_equivalence": q2 is None and torch.allclose(q1, manual_q)
                                                 and torch.allclose(mean_only, manual),
        "time_mode_q_vectorized_equivalence": vector_q.shape == (4, 10, 5)
                                              and torch.allclose(vector_expected,
                                                                 per_step_expected, atol=1e-5),
        "sampled_diagnostic_differs_but_no_grad": (not sampled_diagnostic.requires_grad
                                                    and not torch.allclose(sampled_diagnostic, mean_only)),
        "target_mean_value_finite": bool(torch.isfinite(target_mean).all()),
        "actor_loss_raw_component_mean_q": np.isclose(metrics["actor_rl_loss"],
                                               metrics["actor_total_loss"]),
        "no_bc": config["bc_weight"] == metrics["lambda_bc"] == 0
                 and not config["adaptive_bc_enabled"],
        "no_q_normalization": config["actor_q_scale_normalization"] is False,
        "policy_delay_four": config["policy_delay"] == 4,
        "actor_batch_64": config["recurrent_replay"]["actor_sequence_batch_size"] == 64,
        "diagnostic_one_sample_per_mode": config["diagnostic_learned_std_samples_per_mode"] == 1,
        "case_a_objective": config["rl_policy_expectation"] == "categorical_component_mean",
        "aligned_same_episode": all(start in (0, 10, 20) for start in aligned_starts),
        "exact_50_50": config["offline_fraction"] == config["online_fraction"] == 0.5,
        "utd_one": config["utd"] == 1,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    status = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps({"status": status, "checks": checks,
                      "transfer_max_abs_diff": transfer_diff,
                      "target_max_abs_diff": target_diff,
                      "gradient_norms": gradients,
                      "gradient_nonzero_by_group": {key: value > 0 for key, value in gradients.items()},
                      "sampled_diagnostic_minus_mean_q": float(
                          (sampled_diagnostic - mean_only).mean())}, indent=2))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
