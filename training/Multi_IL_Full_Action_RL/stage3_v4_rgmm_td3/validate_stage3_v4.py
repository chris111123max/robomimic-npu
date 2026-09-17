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
                             recurrent_distributions, target_final_distribution, flat_to_obs)
from stage3_v4_agent import RecurrentGMMTD3, target_final_distribution_vectorized
from stage3_v4_gmm_math import single_expected_q, full_sequence_component_mean_q
from stage3_v4_boundary import aligned_start
from stage3_v4_replay import aligned_sequence_batch, final_transition

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
    parser.add_argument("--compile-backend", choices=("none", "torchair"), default="none")
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
        legacy_target, _ = target_final_distribution(actor, observations, steps, 10)
        optimized, _ = target_final_distribution_vectorized(actor, observations, steps, 10)
    legacy_target_diff = max(float((distribution_tensors(legacy_target)[key]
                                   - distribution_tensors(optimized)[key]).abs().max())
                             for key in keys)
    target_diff = max(float((distribution_tensors(left[-1])[key]
                             - distribution_tensors(optimized)[key]).abs().max())
                      for key in keys)
    scale = torch.as_tensor(rollout.action_normalization_stats["actions"]["scale"],
                            dtype=torch.float32, device=device).reshape(1, 1, 1, 14)
    offset = torch.as_tensor(rollout.action_normalization_stats["actions"]["offset"],
                             dtype=torch.float32, device=device).reshape(1, 1, 1, 14)
    critic = build_critic(59, 14, [256, 256], "relu", True, device)
    agent = RecurrentGMMTD3(clone.train(), critic, config, device, scale, offset)
    if args.compile_backend == "torchair":
        from stage3_v4_execution import enable_npu_compile
        enable_npu_compile(agent)

    vector_expected, vector_q, _, _ = agent._expected_q_sequence(
        observations[:, :10], left[:10])
    per_step_expected = torch.stack([
        agent._expected_q(agent.critic, observations[:, index], left[index],
                          twin_min=False)[0]
        for index in range(10)], dim=1)

    # Learned std remains in the network but cannot enter the RL objective.
    train_distributions, _ = recurrent_distributions(
        actor.train(), observations, steps, 10)
    native_distribution = actor.forward_train(
        flat_to_obs(observations[:, :10]),
        rnn_init_state=None, return_state=False)
    native_expected, _, _, _ = full_sequence_component_mean_q(
        agent.critic, observations[:, :10], native_distribution, scale, offset)
    legacy_expected, _, _, _ = agent._expected_q_sequence(
        observations[:, :10], train_distributions[:10])
    parameters = tuple(actor.parameters())
    native_grads = torch.autograd.grad(-native_expected.mean(), parameters,
                                        allow_unused=True, retain_graph=True)
    legacy_grads = torch.autograd.grad(-legacy_expected.mean(), parameters,
                                        allow_unused=True, retain_graph=True)
    gradient_equivalence = all(
        (a is None and b is None) or (a is not None and b is not None
                                     and torch.allclose(a, b, rtol=1e-3, atol=1e-5))
        for a, b in zip(native_grads, legacy_grads))
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
    # Check the complete frozen interval before opening the 10k gate and before
    # any Actor optimizer step. Comparing this hash after actor_update() would
    # incorrectly fail whenever the post-gate Actor update works as intended.
    warmup_actor_unchanged = frozen_hash == module_hash(agent.actor)
    gate_at = agent.maybe_open_gate(10000, True, True)

    # Actor sequence starts on a real zero-hidden boundary and is one horizon.
    fake_episode = {"episode_steps": np.arange(30), "actions": np.zeros((30, 14))}
    aligned_starts = [aligned_start(fake_episode, 10, 10, np.random.default_rng(index))
                      for index in range(8)]
    class SyntheticReplay:
        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)
            self.episodes = []
            for offset in range(3):
                length = 30 + offset
                steps = np.arange(length, dtype=np.int64)
                self.episodes.append({
                    "observations": np.zeros((length, 59), np.float32),
                    "actions": np.zeros((length, 14), np.float32),
                    "rewards": np.full((length, 1), seed, np.float32),
                    "next_observations": np.zeros((length, 59), np.float32),
                    "terminals": np.zeros((length, 1), np.float32),
                    "episode_steps": steps,
                })
        def _all_episodes(self):
            return self.episodes
        def sample_sequences(self, count, length):
            from stage3_v4_replay import _sample_sequence_batch
            return _sample_sequence_batch(self.episodes, self.rng, count, length)
    replay_batch = aligned_sequence_batch(SyntheticReplay(7), SyntheticReplay(8), 64, 10, 10)
    from stage3_v4_execution import prepare_round_batches
    prepared_critics, prepared_actors = prepare_round_batches(
        SyntheticReplay(7), SyntheticReplay(8), 2, config, device, True)
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
        "target_sequence_vectorized_equivalence": legacy_target_diff <= 1e-5,
        "warmup_actor_frozen": not gate_before and warmup_actor_unchanged,
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
        "policy_delay_one": config["policy_delay"] == 1,
        "native_actor_output_equivalence": torch.allclose(native_expected, legacy_expected, rtol=1e-4, atol=1e-5),
        "native_actor_gradient_equivalence": gradient_equivalence,
        "round_prefetch_batch_contract": (
            len(prepared_critics) == len(prepared_actors) == 2
            and prepared_critics[0][0]["observations"].shape == (256, 59)
            and prepared_actors[0]["observations"].shape == (64, 10, 59)
            and torch.all(prepared_critics[0][0]["rewards"][:128] == 7)
            and torch.all(prepared_critics[0][0]["rewards"][128:] == 8)
            and torch.all(prepared_actors[0]["episode_steps"][:, 0] % 10 == 0)),
        "actor_batch_64": config["recurrent_replay"]["actor_sequence_batch_size"] == 64,
        "diagnostic_one_sample_per_mode": config["diagnostic_learned_std_samples_per_mode"] == 1,
        "case_a_objective": config["rl_policy_expectation"] == "categorical_component_mean",
        "formal_evaluation_seeds_ten": config["evaluation_seeds"] == list(range(20000, 20010)),
        "competence_seed_contract": config["competence_evaluation_seeds"] == list(range(20000, 20020)),
        "smoke_evaluation_seeds_two": config["smoke_evaluation_seeds"] == [20000, 20001],
        "aligned_same_episode": all(start in (0, 10, 20) for start in aligned_starts),
        "replay_batch_shape_boundary_ratio": (
            replay_batch["observations"].shape == (64, 10, 59)
            and replay_batch["actions"].shape == (64, 10, 14)
            and int(replay_batch["is_offline"].sum()) == 32
            and np.array_equal(
                replay_batch["episode_steps"],
                replay_batch["episode_steps"][:, :1] + np.arange(10)[None, :])
            and np.all(replay_batch["episode_steps"][:, 0] % 10 == 0)),
        "exact_50_50": config["offline_fraction"] == config["online_fraction"] == 0.5,
        "utd_one": config["utd"] == 1,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    status = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps({"status": status, "checks": checks,
                      "transfer_max_abs_diff": transfer_diff,
                      "target_max_abs_diff": target_diff,
                      "target_legacy_vs_vectorized_max_abs_diff": legacy_target_diff,
                      "gradient_norms": gradients,
                      "gradient_nonzero_by_group": {key: value > 0 for key, value in gradients.items()},
                      "sampled_diagnostic_minus_mean_q": float(
                          (sampled_diagnostic - mean_only).mean())}, indent=2))
    raise SystemExit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
