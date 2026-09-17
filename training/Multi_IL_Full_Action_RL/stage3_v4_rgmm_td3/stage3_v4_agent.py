"""Low-noise rollout-aligned GMM component-mean TD3-style updates."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch

V3 = Path(__file__).resolve().parents[1] / "stage3_v3_rgmm_td3"
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))
from stage3_v3_actor import (distribution_tensors, flat_to_obs, module_hash,
                             recurrent_distributions)
from stage3_v4_gmm_math import (single_component_mean_q, sequence_component_mean_q,
                                 single_expected_q, sequence_expected_q,
                                 full_sequence_component_mean_q)

ROOT = Path(__file__).resolve().parents[3]
STAGE2 = ROOT / "training" / "Multi_IL_Full_Action_RL" / "stage2_new_critic_pretraining"
if str(STAGE2) not in sys.path:
    sys.path.insert(0, str(STAGE2))
from critic_network import load_stage2_critic_checkpoint  # noqa: E402


def _last_reset_starts_from_numpy(episode_steps, horizon):
    """Return the same last-reset positions without synchronizing the NPU.

    Replay batches are NumPy arrays before ``_tensor_batch`` moves observations
    to the accelerator.  Computing this metadata here avoids the old
    device->host ``episode_steps.cpu()`` round trip inside every Critic update.
    """
    values = np.asarray(episode_steps, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("episode_steps must have shape [B,T]")
    starts = []
    for row in values:
        positions = np.flatnonzero((row + 1) % int(horizon) == 0)
        starts.append(int(positions[-1]) if len(positions) else 0)
    return starts


@torch.no_grad()
def target_final_distribution_vectorized(actor, observations, episode_steps=None,
                                         horizon=10, starts=None):
    """Compute the final target distribution with full-sequence RNN kernels.

    The legacy helper launches one encoder/RNN call per timestep and applies
    hidden resets in Python.  A context can contain at most a few reset
    boundaries; grouping rows by their last boundary lets the native RNN run
    each contiguous suffix in one call while preserving zero-state semantics.
    """
    if observations.ndim != 3 or observations.shape[-1] != 59:
        raise ValueError("Recurrent observations must have shape [B,T,59]")
    if observations.shape[1] < 1:
        raise ValueError("Invalid target recurrent context")
    batch, time_steps = observations.shape[:2]
    if starts is None:
        if episode_steps is None or episode_steps.shape != observations.shape[:2]:
            raise ValueError("Invalid target recurrent context")
        # Compatibility path for validators and callers that already have the
        # metadata on the host. Training passes ``starts`` explicitly below.
        if torch.is_tensor(episode_steps):
            reset_cpu = episode_steps.remainder(int(horizon)).eq(0).detach().cpu().numpy()
        else:
            reset_cpu = (np.asarray(episode_steps, dtype=np.int64)
                         % int(horizon) == 0)
        starts = []
        for row in reset_cpu:
            positions = np.flatnonzero(row)
            starts.append(int(positions[-1]) if len(positions) else 0)
    if len(starts) != batch or any(int(start) < 0 or int(start) >= time_steps
                                   for start in starts):
        raise ValueError("Invalid target reset-start metadata")
    # Keep raw distribution parameters instead of slicing MixtureSameFamily
    # objects.  torch.distributions.Distribution does not guarantee tensor-
    # style indexing (MixtureSameFamily is not subscriptable on torch-npu).
    result_loc = [None] * batch
    result_scale = [None] * batch
    result_logits = [None] * batch
    for start in sorted(set(starts)):
        rows = [index for index, value in enumerate(starts) if value == start]
        row_index = torch.as_tensor(rows, dtype=torch.long, device=observations.device)
        suffix = observations.index_select(0, row_index)[:, start:]
        distribution = actor.forward_train(
            flat_to_obs(suffix), rnn_init_state=None, return_state=False)
        last = distribution.component_distribution.base_dist.loc.shape[1] - 1
        base = distribution.component_distribution.base_dist
        final_loc = base.loc[:, last]
        final_scale = base.scale[:, last]
        final_logits = distribution.mixture_distribution.logits[:, last]
        for position, row in enumerate(rows):
            result_loc[row] = final_loc[position:position + 1]
            result_scale[row] = final_scale[position:position + 1]
            result_logits[row] = final_logits[position:position + 1]
    # Concatenate distribution tensors explicitly to avoid relying on private
    # Distribution slicing behavior for heterogeneous row groups.
    base_loc = torch.cat(result_loc, dim=0)
    base_scale = torch.cat(result_scale, dim=0)
    logits = torch.cat(result_logits, dim=0)
    component = torch.distributions.Independent(
        torch.distributions.Normal(base_loc, base_scale), 1)
    return torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=logits), component), None


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
    return torch.stack(values).norm(2) if values else 0.0


def read_scalar_metrics(metrics):
    """Copy all diagnostic scalars in one device-to-host transfer."""
    keys = [key for key, value in metrics.items() if torch.is_tensor(value)]
    if keys:
        values = torch.stack([metrics[key].detach().reshape(()) for key in keys]).cpu().tolist()
        metrics.update(zip(keys, values))
    return metrics


class RecurrentGMMTD3:
    def __init__(self, actor, critic, config, device, action_scale, action_offset):
        self.actor = actor.train()
        self.target_actor = copy.deepcopy(actor).to(device)
        self.target_actor.eval()
        # Match the online executor's eval()+low_noise_eval=True contract.
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
        return single_component_mean_q(
            critic, states, distribution, self.action_scale, self.action_offset,
            twin_min=twin_min)

    def _expected_q_sequence(self, states, distributions):
        return sequence_component_mean_q(
            self.critic, states, distributions, self.action_scale, self.action_offset)

    def critic_update(self, batch, target_sequence, collect_metrics=True):
        b = self._tensor_batch(batch)
        # ``episode_steps`` is replay metadata. Keep it on the host and derive
        # reset boundaries before moving tensors to the NPU; this removes one
        # synchronizing device->host copy per Critic update.
        horizon = int(self.config["actor_source_contract"]["rnn_horizon"])
        target_starts = _last_reset_starts_from_numpy(
            target_sequence["episode_steps"], horizon)
        target = self._tensor_batch({"next_observations": target_sequence["next_observations"]})
        with torch.no_grad():
            distribution, _ = target_final_distribution_vectorized(
                self.target_actor, target["next_observations"],
                horizon=horizon, starts=target_starts)
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
            # Diagnostic counterfactual only. Never enters td_target or backward.
            self.target_actor.low_noise_eval = False
            try:
                learned_distribution, _ = target_final_distribution_vectorized(
                    self.target_actor, target["next_observations"],
                    horizon=horizon, starts=target_starts)
                sampled_diagnostic, _, _, _, _ = single_expected_q(
                    self.target_critic, target["next_observations"][:, -1],
                    learned_distribution, self.action_scale, self.action_offset,
                    samples=int(self.config["diagnostic_learned_std_samples_per_mode"]))
            finally:
                self.target_actor.low_noise_eval = True
        return read_scalar_metrics({
            "critic_loss_q1": loss_q1,
            "critic_loss_q2": loss_q2,
            "td_target_mean": td_target.mean(),
            "q1_mean": q1.mean(), "q2_mean": q2.mean(),
            "qmin_mean": torch.minimum(q1, q2).mean(),
            "q1_abs_mean": q1.abs().mean(),
            "q2_abs_mean": q2.abs().mean(),
            "q1_std": q1.std(unbiased=False),
            "q2_std": q2.std(unbiased=False),
            "q1_min": q1.min(), "q1_max": q1.max(),
            "q2_min": q2.min(), "q2_max": q2.max(),
            "critic_grad_norm": critic_grad,
            "target_expected_component_mean_q": expected_next.mean(),
            "target_q_component_mean_estimate": expected_next.mean(),
            "target_q_sampled_learned_std_diagnostic": sampled_diagnostic.mean(),
            "target_q_sampled_minus_mean_gap": (sampled_diagnostic - expected_next).mean(),
        })

    def actor_update(self, sequences, env_steps, collect_metrics=True):
        if not self.actor_gate_open:
            raise RuntimeError("Actor update attempted while competence gate is closed")
        self.actor.train()
        b = self._tensor_batch({key: sequences[key]
                                for key in ("observations", "episode_steps")})
        burn = int(self.config["recurrent_replay"]["burn_in"])
        horizon = int(self.config["actor_source_contract"]["rnn_horizon"])
        if burn != 0 or b["observations"].shape[1] != horizon:
            raise ValueError("Full-sequence Actor requires one aligned horizon and zero burn-in")
        # Replay guarantees starts at step 0,10,...; no internal reset exists
        # within this window. A single native RNN call retains BPTT gradients.
        distribution = self.actor.forward_train(
            flat_to_obs(b["observations"]), rnn_init_state=None, return_state=False)
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        try:
            expected, q1, tensors, _ = full_sequence_component_mean_q(
                self.critic, b["observations"], distribution,
                self.action_scale, self.action_offset)
            actor_rl = -expected.mean()
            if not torch.isfinite(actor_rl):
                raise FloatingPointError("Non-finite Stage3-v4 Actor loss")
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_rl.backward()
            actor_grad = grad_norm(self.actor.parameters()) if collect_metrics else None
            head_grads = {}
            if collect_metrics:
                named = dict(self.actor.named_parameters())
                # RNNGMMActorNetwork registers the shared ObservationDecoder under
                # ``nets.decoder`` before it is referenced again by the RNN
                # per-step network. ``named_parameters()`` therefore exposes the
                # canonical decoder names below (duplicate references are removed).
                for label, token in (("gmm_mean", "nets.decoder.nets.mean"),
                                     ("gmm_std", "nets.decoder.nets.scale"),
                                     ("gmm_logits", "nets.decoder.nets.logits"),
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
        component_q = q1.detach().reshape(-1)
        std_values = tensors["scales"].detach().reshape(-1)
        with torch.no_grad():
            # Counterfactual learned-std Q is logged only and has no RL gradient.
            learn_distributions, _ = recurrent_distributions(
                self.actor, b["observations"], b["episode_steps"], horizon=horizon)
            sampled_diagnostic, _, _, _ = sequence_expected_q(
                self.critic, b["observations"][:, burn:], learn_distributions,
                self.action_scale, self.action_offset,
                samples=int(self.config["diagnostic_learned_std_samples_per_mode"]))
            means = tensors["means_normalized"]
            distances = torch.cdist(means, means)
            modes = means.shape[-2]
            mask = ~torch.eye(modes, dtype=torch.bool, device=means.device)
            pairwise = distances[..., mask].mean()
        return read_scalar_metrics({
            "actor_rl_loss": actor_rl,
            "actor_total_loss": actor_rl,
            "lambda_bc": 0.0,
            "actor_grad_norm_total": actor_grad,
            **head_grads,
            "gmm_entropy": entropy,
            "gmm_std_mean": std_values.mean(),
            "gmm_std_min": std_values.min(),
            "gmm_std_max": std_values.max(),
            "actor_expected_component_mean_q": -actor_rl,
            "actor_component_q_mean": component_q.mean(),
            "actor_component_q_std": component_q.std(unbiased=False),
            "actor_component_q_min": component_q.min(),
            "actor_component_q_max": component_q.max(),
            "actor_q_component_mean_estimate": -actor_rl,
            "actor_q_sampled_learned_std_diagnostic": sampled_diagnostic.mean(),
            "actor_q_sampled_minus_mean_gap": sampled_diagnostic.mean() + actor_rl,
            "policy_delay": int(self.config["policy_delay"]),
            "actor_sequence_batch": int(self.config["recurrent_replay"]["actor_sequence_batch_size"]),
            "diagnostic_learned_std_samples_per_mode": int(
                self.config["diagnostic_learned_std_samples_per_mode"]),
            "mixture_prob_mean": float(1.0 / self.config["actor_source_contract"]["num_modes"]),
            "mixture_prob_max": probabilities.max(-1).values.mean(),
            "component_q_mean": component_q.mean(),
            "component_mean_pairwise_distance": pairwise,
            "effectively_active_modes": entropy.exp(),
            "actor_parameter_drift_l2": self.parameter_drift(),
        })

    def parameter_drift(self):
        with torch.no_grad():
            reference = self.reference_actor.state_dict()
            total = torch.stack([(value.detach() - reference[key]).square().sum()
                                 for key, value in self.actor.state_dict().items()]).sum()
        return float(total.sqrt().item())

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
