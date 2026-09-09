"""Stage3-v2 SAC agent: standard SAC target plus time-scheduled BC regularization."""
from __future__ import annotations

import copy
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import torch

from stage3_v2_behavior import bc_schedule, progressive_critic_schedule

ROOT = Path(__file__).resolve().parents[3]
STAGE2 = ROOT / "training" / "Multi_IL_Full_Action_RL" / "stage2_new_critic_pretraining"
RLKIT = ROOT / "rlkit"
for module_path in (STAGE2, RLKIT):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from critic_network import build_critic  # noqa: E402
from rlkit.torch.sac.policies import TanhGaussianPolicy  # noqa: E402


def build_actor(config, device=None):
    actor = TanhGaussianPolicy(
        hidden_sizes=list(config["hidden_dims"]), obs_dim=59, action_dim=14, std=None
    )
    return actor if device is None else actor.to(device)


def state_hash(module):
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def strict_stage2_load(path, device, config):
    payload = torch.load(path, map_location=device)
    model_config = payload.get("model_config", {})
    expected = {
        "obs_dim": 59,
        "action_dim": 14,
        "hidden_dims": list(config["hidden_dims"]),
        "activation": "relu",
        "layer_norm": True,
    }
    for key, value in expected.items():
        if model_config.get(key) != value:
            raise RuntimeError(
                f"Stage2 checkpoint {key}={model_config.get(key)!r}, expected {value!r}"
            )
    if float(payload.get("gamma", -1)) != 0.99:
        raise RuntimeError("Stage2 checkpoint gamma must be 0.99")
    critic = build_critic(59, 14, config["hidden_dims"], "relu", True, device)
    critic.load_state_dict(payload["critic_state_dict"], strict=True)
    return critic, payload


class Stage3V2SAC:
    def __init__(self, actor, critic, config, device, action_low, action_high):
        self.actor = actor
        self.critic = critic
        self.target = copy.deepcopy(critic).to(device)
        self.target.eval()
        self.target.requires_grad_(False)
        self.config = config
        self.device = device
        self.actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=float(config["actor_lr"]), weight_decay=0.0
        )
        self.critic_optimizer = torch.optim.AdamW(
            critic.parameters(),
            lr=float(config["critic_lr"]),
            weight_decay=float(config["critic_weight_decay"]),
        )
        alpha_init = float(config["alpha_init"])
        if not math.isfinite(alpha_init) or alpha_init <= 0:
            raise ValueError("alpha_init must be finite and positive")
        self.log_alpha = torch.tensor(
            math.log(alpha_init), dtype=torch.float32, device=device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=float(config["alpha_lr"]), weight_decay=0.0
        )
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=device)
        if self.action_low.shape != (14,) or self.action_high.shape != (14,):
            raise ValueError("Stage3-v2 requires 14-dimensional action bounds")
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def standard_target_components(self, batch):
        """The only Stage3-v2 TD target: standard entropy-regularized SAC."""
        with torch.no_grad():
            next_action, _, _, next_log_pi, *_ = self.actor(
                batch["next_observations"], reparameterize=True, return_log_prob=True
            )
            target_q1, target_q2 = self.target(batch["next_observations"], next_action)
            target_qmin = torch.minimum(target_q1, target_q2)
            entropy_term = -self.alpha.detach() * next_log_pi
            td_target = batch["rewards"] + float(self.config["gamma"]) * (
                1.0 - batch["terminals"]
            ) * (target_qmin + entropy_term)
        return {
            "next_action": next_action,
            "next_log_pi": next_log_pi,
            "target_qmin": target_qmin,
            "entropy_term": entropy_term,
            "td_target": td_target,
        }

    def cql_components(self, states, q1_data, q2_data):
        cfg = self.config["cql"]
        batch_size = len(states)
        random_count = int(cfg["num_random_actions"])
        policy_count = int(cfg["num_policy_actions"])
        if random_count != 10 or policy_count != 1:
            raise RuntimeError("Stage3-v2 CQL contract requires K_random=10, K_policy=1")
        with torch.no_grad():
            policy_action = self.actor(
                states, reparameterize=True, return_log_prob=False
            )[0].detach()
        random_action = torch.rand(
            (batch_size, random_count, 14), dtype=states.dtype, device=states.device
        )
        random_action = self.action_low + (self.action_high - self.action_low) * random_action
        candidates = torch.cat((policy_action[:, None, :], random_action), dim=1)
        expanded = states[:, None, :].expand(-1, 1 + random_count, -1).reshape(-1, 59)
        ood_q1, ood_q2 = self.critic(expanded, candidates.reshape(-1, 14))
        ood_q1 = ood_q1.reshape(batch_size, -1)
        ood_q2 = ood_q2.reshape(batch_size, -1)
        loss_q1 = torch.logsumexp(ood_q1, dim=1).mean() - q1_data.mean()
        loss_q2 = torch.logsumexp(ood_q2, dim=1).mean() - q2_data.mean()
        q_data = torch.minimum(q1_data, q2_data).reshape(-1)
        q_policy = torch.minimum(ood_q1[:, 0], ood_q2[:, 0])
        q_random = torch.minimum(ood_q1[:, 1:], ood_q2[:, 1:])
        return {
            "loss": loss_q1 + loss_q2,
            "loss_q1": loss_q1,
            "loss_q2": loss_q2,
            "q_data": q_data,
            "q_policy": q_policy,
            "q_random_max": q_random.max(dim=1).values,
        }

    @staticmethod
    def bc_components(actor, batch):
        offline = batch["is_online"].reshape(-1) < 0.5
        online_rnn = (batch["is_online"].reshape(-1) > 0.5) & (
            batch["behavior_source"].reshape(-1) < 0.5
        )
        mask = offline | online_rnn
        count = int(mask.sum().item())
        if not count:
            zero = batch["observations"].sum() * 0.0
            return zero, count, mask
        deterministic = actor(batch["observations"][mask], deterministic=True)[0]
        loss = torch.nn.functional.mse_loss(deterministic, batch["action_rnn"][mask])
        return loss, count, mask

    def update(self, batch, env_steps):
        b = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in batch.items()
        }
        critic_schedule = progressive_critic_schedule(self.config, env_steps)
        actor_schedule = bc_schedule(self.config, env_steps)
        bc_loss, bc_count, _ = self.bc_components(self.actor, b)
        lambda_bc = float(actor_schedule["lambda_bc"])
        bc_weighted = lambda_bc * bc_loss

        if actor_schedule["actor_objective"] == "bc_only":
            self.actor_optimizer.zero_grad(set_to_none=True)
            bc_weighted.backward()
            self.actor_optimizer.step()
            self.updates += 1
            return {
                **critic_schedule,
                **actor_schedule,
                "critic_loss": None,
                "critic_td_loss": None,
                "q1_loss": None,
                "q2_loss": None,
                "q1_mean": None,
                "q2_mean": None,
                "qmin_mean": None,
                "target_qmin_mean": None,
                "td_target_mean": None,
                "td_error": None,
                "cql_loss_raw": 0.0,
                "cql_loss_weighted": 0.0,
                "q_data_mean": None,
                "q_policy_mean": None,
                "q_random_max_mean": None,
                "actor_sac_loss": 0.0,
                "bc_loss_raw": float(bc_loss.item()),
                "bc_loss_weighted": float(bc_weighted.item()),
                "actor_total_loss": float(bc_weighted.item()),
                "bc_mask_count": bc_count,
                "bc_mask_fraction": bc_count / len(b["observations"]),
                "alpha": float(self.alpha.item()),
                "alpha_loss": 0.0,
                "policy_entropy": None,
            }

        for group in self.critic_optimizer.param_groups:
            group["lr"] = critic_schedule["critic_lr_effective"]
        target = self.standard_target_components(b)
        q1, q2 = self.critic(b["observations"], b["actions"])
        q1_loss = torch.nn.functional.mse_loss(q1, target["td_target"])
        q2_loss = torch.nn.functional.mse_loss(q2, target["td_target"])
        critic_td_loss = q1_loss + q2_loss
        cql = self.cql_components(b["observations"], q1, q2)
        cql_weighted = float(self.config["cql"]["lambda"]) * cql["loss"]
        critic_loss = critic_td_loss + cql_weighted
        if critic_schedule["critic_update_enabled"]:
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_optimizer.step()

        action, _, _, log_pi, *_ = self.actor(
            b["observations"], reparameterize=True, return_log_prob=True
        )
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        actor_q1, actor_q2 = self.critic(b["observations"], action)
        actor_sac_loss = (
            self.alpha.detach() * log_pi - torch.minimum(actor_q1, actor_q2)
        ).mean()
        actor_total = actor_sac_loss + bc_weighted
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_total.backward()
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        alpha_loss = -(
            self.log_alpha * (log_pi.detach() + float(self.config["target_entropy"]))
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        tau = float(critic_schedule["target_tau_effective"])
        if tau > 0:
            with torch.no_grad():
                for source, destination in zip(
                    self.critic.parameters(), self.target.parameters()
                ):
                    destination.mul_(1.0 - tau).add_(source, alpha=tau)
        self.updates += 1
        td_error = torch.cat((q1 - target["td_target"], q2 - target["td_target"]))
        return {
            **critic_schedule,
            **actor_schedule,
            "critic_loss": float(critic_loss.item()),
            "critic_td_loss": float(critic_td_loss.item()),
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
            "qmin_mean": float(torch.minimum(q1, q2).mean().item()),
            "target_qmin_mean": float(target["target_qmin"].mean().item()),
            "td_target_mean": float(target["td_target"].mean().item()),
            "td_error": float(td_error.abs().mean().item()),
            "cql_loss_raw": float(cql["loss"].item()),
            "cql_loss_weighted": float(cql_weighted.item()),
            "q_data_mean": float(cql["q_data"].mean().item()),
            "q_policy_mean": float(cql["q_policy"].mean().item()),
            "q_random_max_mean": float(cql["q_random_max"].mean().item()),
            "actor_sac_loss": float(actor_sac_loss.item()),
            "bc_loss_raw": float(bc_loss.item()),
            "bc_loss_weighted": float(bc_weighted.item()),
            "actor_total_loss": float(actor_total.item()),
            "bc_mask_count": bc_count,
            "bc_mask_fraction": bc_count / len(b["observations"]),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
            "policy_entropy": float((-log_pi).mean().item()),
        }

    def actor_vs_rnn_diagnostics(self, observations, rnn_actions):
        states = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        teachers = torch.as_tensor(rnn_actions, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            actions = self.actor(states, deterministic=True)[0]
            difference = actions - teachers
        return {
            "actor_vs_rnn_action_mse": float(difference.square().mean().item()),
            "actor_vs_rnn_action_l2": float(difference.norm(dim=1).mean().item()),
        }

    def source_diagnostics(self, batch):
        b = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in batch.items() if key in {
                "observations", "actions", "rewards", "next_observations", "terminals",
                "action_rnn", "is_online", "behavior_source"
            }
        }
        with torch.no_grad():
            q1, q2 = self.critic(b["observations"], b["actions"])
            cql = self.cql_components(b["observations"], q1, q2)
            target = self.standard_target_components(b)
        return {
            "q_data_mean": float(cql["q_data"].mean().item()),
            "q_policy_mean": float(cql["q_policy"].mean().item()),
            "q_random_max_mean": float(cql["q_random_max"].mean().item()),
            "target_qmin_mean": float(target["target_qmin"].mean().item()),
            "td_target_mean": float(target["td_target"].mean().item()),
        }
