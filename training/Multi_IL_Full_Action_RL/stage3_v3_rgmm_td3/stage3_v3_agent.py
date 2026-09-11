"""TD3-style updates for an exact robomimic recurrent GMM Actor."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch

from stage3_v3_actor import (distribution_tensors, environment_means,
                             module_hash, normalize_actions,
                             recurrent_distributions)

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


def bc_lambda(schedule, env_steps):
    points = sorted((int(row["step"]), float(row["value"])) for row in schedule)
    step = int(env_steps)
    if step <= points[0][0]:
        return points[0][1]
    for (left_step, left), (right_step, right) in zip(points, points[1:]):
        if step <= right_step:
            ratio = (step - left_step) / max(1, right_step - left_step)
            return left + ratio * (right - left)
    return points[-1][1]


def grad_norm(parameters):
    values = [parameter.grad.detach().norm(2) for parameter in parameters
              if parameter.grad is not None]
    return float(torch.stack(values).norm(2).item()) if values else 0.0


class RecurrentGMMTD3:
    def __init__(self, actor, critic, config, device, action_scale, action_offset):
        self.actor = actor.train()
        self.target_actor = copy.deepcopy(actor).to(device)
        self.target_actor.eval()
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

    def _expected_q(self, critic, states, distribution, twin_min=True):
        tensors = distribution_tensors(distribution)
        means = environment_means(distribution, self.action_scale, self.action_offset)
        batch_shape = means.shape[:-2]
        modes = means.shape[-2]
        flat_states = states.unsqueeze(-2).expand(*batch_shape, modes, 59).reshape(-1, 59)
        q1, q2 = critic(flat_states, means.reshape(-1, 14))
        q1 = q1.reshape(*batch_shape, modes)
        q2 = q2.reshape(*batch_shape, modes)
        selected = torch.minimum(q1, q2) if twin_min else q1
        expected = (tensors["probs"] * selected).sum(-1)
        return expected, q1, q2, tensors, means

    def critic_update(self, batch, target_sequence, collect_metrics=True):
        b = self._tensor_batch(batch)
        target = self._tensor_batch(target_sequence)
        burn = int(self.config["recurrent_replay"]["burn_in"])
        horizon = int(self.config["actor_source_contract"]["rnn_horizon"])
        with torch.no_grad():
            distributions, _ = recurrent_distributions(
                self.target_actor, target["next_observations"],
                target["episode_steps"] + 1, horizon=horizon,
                no_grad_prefix=burn)
            distribution = distributions[-1]
            expected_next, _, _, _, _ = self._expected_q(
                self.target_critic, target["next_observations"][:, -1], distribution)
            td_target = b["rewards"] + float(self.config["gamma"]) * (
                1.0 - b["terminals"]) * expected_next.reshape(-1, 1)
        q1, q2 = self.critic(b["observations"], b["actions"])
        loss_q1 = torch.nn.functional.mse_loss(q1, td_target)
        loss_q2 = torch.nn.functional.mse_loss(q2, td_target)
        loss = loss_q1 + loss_q2
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite Stage3-v3 Critic loss")
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
        return {
            "critic_loss_q1": float(loss_q1.item()),
            "critic_loss_q2": float(loss_q2.item()),
            "td_target_mean": float(td_target.mean().item()),
            "q1_mean": float(q1.mean().item()), "q2_mean": float(q2.mean().item()),
            "qmin_mean": float(torch.minimum(q1, q2).mean().item()),
            "critic_grad_norm": critic_grad,
        }

    def actor_update(self, sequences, env_steps):
        if not self.actor_gate_open:
            raise RuntimeError("Actor update attempted while competence gate is closed")
        self.actor.train()
        b = self._tensor_batch(sequences)
        burn = int(self.config["recurrent_replay"]["burn_in"])
        horizon = int(self.config["actor_source_contract"]["rnn_horizon"])
        distributions, _ = recurrent_distributions(
            self.actor, b["observations"], b["episode_steps"], horizon=horizon,
            no_grad_prefix=burn)
        learn_distributions = distributions[burn:]
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        rl_terms, nll_terms, entropies, max_probs, pairwise = [], [], [], [], []
        component_q_values = []
        for local, distribution in enumerate(learn_distributions):
            index = burn + local
            expected, q1, _, tensors, means = self._expected_q(
                self.critic, b["observations"][:, index], distribution,
                twin_min=False)
            rl_terms.append(-expected.mean())
            normalized_actions = normalize_actions(
                b["actions"][:, index:index + 1], self.action_scale,
                self.action_offset)[:, 0]
            offline = b["is_offline"].bool()
            if not bool(offline.any()):
                raise RuntimeError("Actor BC-GMM NLL requires offline demonstrations")
            nll_terms.append(-distribution.log_prob(normalized_actions)[offline].mean())
            probabilities = tensors["probs"]
            entropies.append((-(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)).mean())
            max_probs.append(probabilities.max(-1).values.mean())
            component_q_values.append(q1.mean())
            distances = torch.cdist(means, means)
            modes = means.shape[-2]
            mask = ~torch.eye(modes, dtype=torch.bool, device=means.device)
            pairwise.append(distances[..., mask].mean())
        actor_rl = torch.stack(rl_terms).mean()
        actor_bc = torch.stack(nll_terms).mean()
        weight = bc_lambda(self.config["bc_lambda_schedule"], env_steps)
        total = actor_rl + weight * actor_bc
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite Stage3-v3 Actor loss")
        self.actor_optimizer.zero_grad(set_to_none=True)
        total.backward()
        actor_grad = grad_norm(self.actor.parameters())
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float(self.config["actor_max_grad_norm"]))
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)
        self.actor_updates += 1
        return {
            "actor_rl_loss": float(actor_rl.item()),
            "actor_bc_loss_raw": float(actor_bc.item()),
            "actor_bc_loss_weighted": float((weight * actor_bc).item()),
            "actor_total_loss": float(total.item()),
            "lambda_bc": weight, "actor_grad_norm": actor_grad,
            "gmm_entropy": float(torch.stack(entropies).mean().item()),
            "mixture_prob_mean": float(1.0 / self.config["actor_source_contract"]["num_modes"]),
            "mixture_prob_max": float(torch.stack(max_probs).mean().item()),
            "component_q_mean": float(torch.stack(component_q_values).mean().item()),
            "expected_component_q": float((-actor_rl).item()),
            "component_mean_pairwise_distance": float(torch.stack(pairwise).mean().item()),
            "effectively_active_modes": float(math.exp(torch.stack(entropies).mean().item())),
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
        b = self._tensor_batch(sequences)
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
        self.actor.train(actor_mode); self.reference_actor.train(reference_mode)
        return {
            "gmm_entropy": float(entropy.mean().item()),
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
