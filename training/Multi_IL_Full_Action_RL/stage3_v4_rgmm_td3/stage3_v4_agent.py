"""Stochastic GMM-expectation TD3-style updates for an exact recurrent Actor."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch

V3 = Path(__file__).resolve().parents[1] / "stage3_v3_rgmm_td3"
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))
from stage3_v3_actor import (distribution_tensors, module_hash,
                             recurrent_distributions, target_final_distribution)
from stage3_v4_gmm_math import single_expected_q, sequence_expected_q

ROOT = Path(__file__).resolve().parents[3]
STAGE2 = ROOT / "training" / "Multi_IL_Full_Action_RL" / "stage2_new_critic_pretraining"
if str(STAGE2) not in sys.path:
    sys.path.insert(0, str(STAGE2))
from critic_network import load_stage2_critic_checkpoint  # noqa: E402


def strict_stage2_load(path, device):
    critic, payload = load_stage2_critic_checkpoint(path, device)
    expected = {"obs_dim": 59, "action_dim": 14, "hidden_dims": [256, 256],
                "activation": "relu", "layer_norm": True}
    for key, value in expected.items():
        if payload["model_config"].get(key) != value:
            raise RuntimeError(f"Stage2 Critic {key} contract changed")
    if float(payload.get("gamma", -1)) != 0.99:
        raise RuntimeError("Stage2 Critic gamma is not 0.99")
    return critic, payload


def grad_norm(parameters):
    values = [parameter.grad.detach().norm(2) for parameter in parameters
              if parameter.grad is not None]
    return float(torch.stack(values).norm(2).item()) if values else 0.0


class RecurrentGMMTD3:
    def __init__(self, actor, critic, config, device, action_scale, action_offset):
        self.actor = actor.train()
        self.target_actor = copy.deepcopy(actor).to(device)
        self.target_actor.eval()
        # In the native robomimic network, eval()+low_noise_eval replaces all
        # learned stds with 1e-4. Keep the independent Polyak target and eval
        # mode, but retain its learned GMM variance for the stochastic target.
        self.target_actor.low_noise_eval = False
        self.target_actor.requires_grad_(False)
        self.critic = critic
        self.target_critic = copy.deepcopy(critic).to(device)
        self.target_critic.requires_grad_(False)
        self.config, self.device = config, device
        self.action_scale = action_scale.to(device)
        self.action_offset = action_offset.to(device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(config["actor_lr"]), weight_decay=0.0)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=float(config["critic_lr"]),
            weight_decay=float(config["critic_weight_decay"]))
        self.critic_updates = 0
        self.actor_updates = 0
        self.actor_gate_open = False
        self.gate_open_step = None
        self.initial_actor = {key: value.detach().cpu().clone()
                              for key, value in actor.state_dict().items()}
        self.reference_actor = copy.deepcopy(actor).to(device).eval()
        self.reference_actor.requires_grad_(False)

    def maybe_open_gate(self, env_steps, equivalence_pass, competence_pass):
        if (not self.actor_gate_open and equivalence_pass and competence_pass
                and int(env_steps) >= int(self.config["actor_gate"]["warmup_env_steps"])):
            self.actor_gate_open = True
            self.gate_open_step = int(env_steps)
            return True
        return False

    def _tensor_batch(self, batch):
        return {key: torch.as_tensor(
                    value, dtype=(torch.long if key == "episode_steps" else torch.float32),
                    device=self.device)
                for key, value in batch.items()}

    def _expected_q(self, critic, states, distribution, twin_min=True,
                    epsilon=None):
        """Enumerate modes; reparameterize Gaussian samples; one batched Q call."""
        return single_expected_q(
            critic, states, distribution, self.action_scale, self.action_offset,
            samples=int(self.config["gmm_rl_samples_per_mode"]),
            twin_min=twin_min, epsilon=epsilon)

    def _expected_q_sequence(self, states, distributions, epsilon=None):
        """One Q1 forward over [batch,time,mode,sample], with full gradients."""
        return sequence_expected_q(
            self.critic, states, distributions, self.action_scale, self.action_offset,
            samples=int(self.config["gmm_rl_samples_per_mode"]), epsilon=epsilon)

    def critic_update(self, batch, target_sequence, collect_metrics=True):
        b = self._tensor_batch(batch)
        target = self._tensor_batch({key: target_sequence[key]
                                     for key in ("next_observations", "episode_steps")})
        horizon = int(self.config["actor_source_contract"]["rnn_horizon"])
        with torch.no_grad():
            distribution, _ = target_final_distribution(
                self.target_actor, target["next_observations"],
                target["episode_steps"] + 1, horizon=horizon)
            expected_next, target_q1, target_q2, target_tensors, _ = self._expected_q(
                self.target_critic, target["next_observations"][:, -1], distribution)
            td_target = b["rewards"] + float(self.config["gamma"]) * (
                1.0 - b["terminals"]) * expected_next.reshape(-1, 1)
        q1, q2 = self.critic(b["observations"], b["actions"])
        loss_q1 = torch.nn.functional.mse_loss(q1, td_target)
        loss_q2 = torch.nn.functional.mse_loss(q2, td_target)
        loss = loss_q1 + loss_q2
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Stage3-v4 Critic loss")
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        critic_grad = grad_norm(self.critic.parameters()) if collect_metrics else None
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.config["critic_max_grad_norm"]))
        self.critic_optimizer.step()
        self.critic_updates += 1
        if not collect_metrics:
            # Avoid synchronizing several NPU scalar tensors back to the host
            # on every transition. The trainer only needs detailed Critic
            # diagnostics at train_metrics_interval_updates boundaries.
            return {}
        with torch.no_grad():
            zero_epsilon = torch.zeros_like(target_tensors["means_normalized"]).unsqueeze(-2).expand(
                -1, -1, int(self.config["gmm_rl_samples_per_mode"]), -1)
            mean_approx, _, _, _, _ = self._expected_q(
                self.target_critic, target["next_observations"][:, -1], distribution,
                epsilon=zero_epsilon)
        return {
            "critic_loss_q1": float(loss_q1.item()),
            "critic_loss_q2": float(loss_q2.item()),
            "td_target_mean": float(td_target.mean().item()),
            "q1_mean": float(q1.mean().item()), "q2_mean": float(q2.mean().item()),
            "qmin_mean": float(torch.minimum(q1, q2).mean().item()),
            "q1_abs_mean": float(q1.abs().mean().item()),
            "q2_abs_mean": float(q2.abs().mean().item()),
            "q1_std": float(q1.std(unbiased=False).item()),
            "q2_std": float(q2.std(unbiased=False).item()),
            "q1_min": float(q1.min().item()), "q1_max": float(q1.max().item()),
            "q2_min": float(q2.min().item()), "q2_max": float(q2.max().item()),
            "critic_grad_norm": critic_grad,
            "target_expected_sampled_q": float(expected_next.mean().item()),
            "target_q_component_mean_estimate": float(mean_approx.mean().item()),
            "target_q_gmm_sample_estimate": float(expected_next.mean().item()),
            "target_q_estimate_gap": float((expected_next - mean_approx).mean().item()),
        }

    def actor_update(self, sequences, env_steps, collect_metrics=True):
        if not self.actor_gate_open:
            raise RuntimeError("Actor update attempted while competence gate is closed")
        self.actor.train()
        b = self._tensor_batch({key: sequences[key]
                                for key in ("observations", "episode_steps")})
        burn = int(self.config["recurrent_replay"]["burn_in"])
        horizon = int(self.config["actor_source_contract"]["rnn_horizon"])
        distributions, _ = recurrent_distributions(
            self.actor, b["observations"], b["episode_steps"], horizon=horizon,
            no_grad_prefix=burn)
        learn_distributions = distributions[burn:]
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        try:
            expected, q1, tensors, _ = self._expected_q_sequence(
                b["observations"][:, burn:], learn_distributions)
            actor_rl = -expected.mean()
            if not torch.isfinite(actor_rl):
                raise FloatingPointError("Non-finite Stage3-v4 Actor loss")
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_rl.backward()
            actor_grad = grad_norm(self.actor.parameters()) if collect_metrics else None
            head_grads = {}
            if collect_metrics:
                named = dict(self.actor.named_parameters())
                for label, token in (("gmm_mean", "per_step_net.nets.mean"),
                                     ("gmm_std", "per_step_net.nets.scale"),
                                     ("gmm_logits", "per_step_net.nets.logits"),
                                     ("rnn", "nets.rnn.nets")):
                    head_grads[f"actor_grad_norm_{label}"] = grad_norm(
                        value for name, value in named.items() if token in name)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.config["actor_max_grad_norm"]))
            self.actor_optimizer.step()
        finally:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)
        self.actor_updates += 1
        if not collect_metrics:
            return {}
        probabilities = tensors["probs"]
        entropy = (-(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)).mean()
        sampled = q1.detach().reshape(-1)
        std_values = tensors["scales"].detach().reshape(-1)
        with torch.no_grad():
            zero_epsilon = torch.zeros_like(tensors["means_normalized"]).unsqueeze(-2).expand(
                -1, -1, -1, int(self.config["gmm_rl_samples_per_mode"]), -1)
            mean_estimate, _, _, _ = self._expected_q_sequence(
                b["observations"][:, burn:], learn_distributions,
                epsilon=zero_epsilon)
            mean_estimate = mean_estimate.mean()
            means = tensors["means_normalized"]
            distances = torch.cdist(means, means)
            modes = means.shape[-2]
            mask = ~torch.eye(modes, dtype=torch.bool, device=means.device)
            pairwise = distances[..., mask].mean()
        return {
            "actor_rl_loss": float(actor_rl.item()),
            "actor_total_loss": float(actor_rl.item()),
            "lambda_bc": 0.0,
            "actor_grad_norm_total": actor_grad,
            **head_grads,
            "gmm_entropy": float(entropy.item()),
            "gmm_std_mean": float(std_values.mean().item()),
            "gmm_std_min": float(std_values.min().item()),
            "gmm_std_max": float(std_values.max().item()),
            "actor_expected_sampled_q": float((-actor_rl).item()),
            "actor_sampled_q_mean": float(sampled.mean().item()),
            "actor_sampled_q_std": float(sampled.std(unbiased=False).item()),
            "actor_sampled_q_min": float(sampled.min().item()),
            "actor_sampled_q_max": float(sampled.max().item()),
            "actor_q_component_mean_estimate": float(mean_estimate.item()),
            "actor_q_gmm_sample_estimate": float((-actor_rl).item()),
            "actor_q_estimate_gap": float((-actor_rl - mean_estimate).item()),
            "policy_delay": int(self.config["policy_delay"]),
            "actor_sequence_batch": int(self.config["recurrent_replay"]["actor_sequence_batch_size"]),
            "gmm_rl_samples_per_mode": int(self.config["gmm_rl_samples_per_mode"]),
            "mixture_prob_mean": float(1.0 / self.config["actor_source_contract"]["num_modes"]),
            "mixture_prob_max": float(probabilities.max(-1).values.mean().item()),
            "component_q_mean": float(sampled.mean().item()),
            "component_mean_pairwise_distance": float(pairwise.item()),
            "effectively_active_modes": float(math.exp(entropy.item())),
            "actor_parameter_drift_l2": self.parameter_drift(),
        }

    def parameter_drift(self):
        total = 0.0
        with torch.no_grad():
            for key, value in self.actor.state_dict().items():
                difference = value.detach().cpu() - self.initial_actor[key]
                total += float(difference.square().sum())
        return math.sqrt(total)

    @torch.no_grad()
    def gmm_diagnostics(self, sequences):
        b = self._tensor_batch({key: sequences[key]
                                for key in ("observations", "episode_steps")})
        burn = int(self.config["recurrent_replay"]["burn_in"])
        horizon = int(self.config["actor_source_contract"]["rnn_horizon"])
        actor_mode, reference_mode = self.actor.training, self.reference_actor.training
        self.actor.train(); self.reference_actor.train()
        current, _ = recurrent_distributions(
            self.actor, b["observations"], b["episode_steps"], horizon, no_grad_prefix=burn)
        initial, _ = recurrent_distributions(
            self.reference_actor, b["observations"], b["episode_steps"], horizon,
            no_grad_prefix=burn)
        current_tensors = [distribution_tensors(value) for value in current[burn:]]
        initial_tensors = [distribution_tensors(value) for value in initial[burn:]]
        probs = torch.cat([value["probs"].reshape(-1, value["probs"].shape[-1])
                           for value in current_tensors])
        modes = probs.shape[-1]
        usage = torch.bincount(probs.argmax(-1), minlength=modes).float()
        usage = usage / usage.sum().clamp_min(1)
        mean_drift = torch.cat([
            (a["means_normalized"] - b0["means_normalized"]).reshape(-1, 14)
            for a, b0 in zip(current_tensors, initial_tensors)])
        logit_drift = torch.cat([
            (a["logits"] - b0["logits"]).reshape(-1, modes)
            for a, b0 in zip(current_tensors, initial_tensors)])
        diagnostic_action_drift = torch.cat([
            ((a["probs"].unsqueeze(-1) * a["means_normalized"]).sum(-2)
             - (b0["probs"].unsqueeze(-1) * b0["means_normalized"]).sum(-2)).reshape(-1, 14)
            for a, b0 in zip(current_tensors, initial_tensors)])
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1)
        std_values = torch.cat([value["scales"].reshape(-1) for value in current_tensors])
        std_per_mode = torch.cat([value["scales"].reshape(-1, modes, 14)
                                  for value in current_tensors], dim=0).mean((0, 2))
        self.actor.train(actor_mode); self.reference_actor.train(reference_mode)
        return {
            "gmm_entropy": float(entropy.mean().item()),
            "gmm_std_mean": float(std_values.mean().item()),
            "gmm_std_median": float(std_values.median().item()),
            "gmm_std_min": float(std_values.min().item()),
            "gmm_std_max": float(std_values.max().item()),
            "gmm_std_mean_per_mode": std_per_mode.cpu().tolist(),
            "mixture_probability_mean_per_mode": probs.mean(0).cpu().tolist(),
            "mixture_probability_max": float(probs.max(-1).values.mean().item()),
            "component_usage_frequency": usage.cpu().tolist(),
            "effectively_active_modes": float(entropy.exp().mean().item()),
            "component_mean_drift_mse": float(mean_drift.square().mean().item()),
            "component_mean_drift_max_abs": float(mean_drift.abs().max().item()),
            "logit_drift_mse": float(logit_drift.square().mean().item()),
            "logit_drift_max_abs": float(logit_drift.abs().max().item()),
            "diagnostic_weighted_mean_action_drift_mse": float(
                diagnostic_action_drift.square().mean().item()),
            "diagnostic_action_note": "weighted mixture mean is diagnostic only, never executed",
            "parameter_drift_l2": self.parameter_drift(),
        }

    @torch.no_grad()
    def polyak_update(self):
        tau = float(self.config["tau"])
        for source, target in zip(self.critic.parameters(), self.target_critic.parameters()):
            target.mul_(1.0 - tau).add_(source, alpha=tau)
        if self.actor_gate_open:
            for source, target in zip(self.actor.parameters(), self.target_actor.parameters()):
                target.mul_(1.0 - tau).add_(source, alpha=tau)

    def initial_hashes(self):
        return {"actor": module_hash(self.actor), "target_actor": module_hash(self.target_actor),
                "critic": module_hash(self.critic), "target_critic": module_hash(self.target_critic)}
