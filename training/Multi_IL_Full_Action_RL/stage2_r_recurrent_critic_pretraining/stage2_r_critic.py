"""Native pomdp-baselines recurrent twin critic and Stage2-R TD operations."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
VENDOR = HERE.parent / "third_party" / "pomdp_baselines"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from policies.models.recurrent_critic import Critic_RNN  # noqa: E402
from policies.rl.sac import SAC  # noqa: E402
import torchkit.pytorch_utils as ptu  # noqa: E402


def set_device(device):
    ptu.set_device(device)


def architecture(config):
    return {
        "implementation": "pomdp-baselines policies.models.recurrent_critic.Critic_RNN",
        "encoder": config["encoder"], "state_dim": 59, "action_dim": 14,
        "history_input": "embedded(previous_action[14], reward[1], canonical_observation[59])",
        "action_embedding_size": config["action_embedding_size"],
        "observation_embedding_size": config["observation_embedding_size"],
        "reward_embedding_size": config["reward_embedding_size"],
        "rnn_hidden_size": config["rnn_hidden_size"], "rnn_num_layers": config["rnn_num_layers"],
        "shortcut": "current canonical observation + current action",
        "q_heads": 2, "q_head_hidden_layers": config["dqn_layers"], "q_output_dim": 1,
        "sequence_length": config["sequence_length"]
    }


def make_pair(config, device):
    # SAC supplies the official continuous twin-Q head builder used by Critic_RNN.
    algo = SAC(automatic_entropy_tuning=False, entropy_alpha=0.0, action_dim=14)
    critic = Critic_RNN(59, 14, config["encoder"], algo,
                        config["action_embedding_size"], config["observation_embedding_size"],
                        config["reward_embedding_size"], config["rnn_hidden_size"],
                        config["dqn_layers"], config["rnn_num_layers"]).to(device)
    target = copy.deepcopy(critic).to(device).requires_grad_(False)
    return critic, target


def tensor_batch(batch, device):
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in batch.items() if key in ("obs", "obs2", "act", "rew", "term", "mask")}


def aligned(batch):
    """Create native Critic_RNN T+1 history and current/next action alignment."""
    obs, obs2 = batch["obs"][..., :59], batch["obs2"][..., :59]
    next_actor_action = batch["obs2"][..., 59:]
    T, B, _ = obs.shape
    zeros_a = torch.zeros((1, B, 14), dtype=obs.dtype, device=obs.device)
    zeros_r = torch.zeros((1, B, 1), dtype=obs.dtype, device=obs.device)
    observs = torch.cat((obs[[0]], obs2), dim=0)
    previous_actions = torch.cat((zeros_a, batch["act"]), dim=0)
    previous_rewards = torch.cat((zeros_r, batch["rew"]), dim=0)
    target_actions = torch.cat((zeros_a, next_actor_action), dim=0)
    return observs, previous_actions, previous_rewards, target_actions


def predictions(critic, target, batch, gamma):
    observs, previous_actions, previous_rewards, target_actions = aligned(batch)
    q1, q2 = critic(previous_actions, previous_rewards, observs, batch["act"])
    with torch.no_grad():
        tq1, tq2 = target(previous_actions, previous_rewards, observs, target_actions)
        bootstrap = 1.0 - batch["term"]
        bellman = batch["rew"] + float(gamma) * bootstrap * torch.minimum(tq1[1:], tq2[1:])
    return q1, q2, bellman


def update(critic, target, optimizer, batch, config, update_index):
    q1, q2, bellman = predictions(critic, target, batch, config["gamma"])
    mask = batch["mask"]; valid = torch.clamp(mask.sum(), min=1.0)
    loss1 = (((q1 - bellman) ** 2) * mask).sum() / valid
    loss2 = (((q2 - bellman) ** 2) * mask).sum() / valid
    loss = loss1 + loss2
    optimizer.zero_grad(); loss.backward()
    grad_sq = torch.zeros((), device=loss.device)
    for parameter in critic.parameters():
        if parameter.grad is not None: grad_sq += parameter.grad.detach().pow(2).sum()
    grad_norm = grad_sq.sqrt()
    optimizer.step()
    if update_index % int(config["target_update_interval"]) == 0:
        ptu.soft_update_from_to(critic, target, float(config["tau"]))
    hidden = critic.get_hidden_states(*aligned(batch)[1:3], aligned(batch)[0])
    finite = all(torch.isfinite(value).all() for value in (q1, q2, bellman, loss, grad_norm, hidden))
    if not finite:raise RuntimeError(f"NaN/Inf in recurrent critic update {update_index}")
    valid_hidden = hidden[:-1][mask.bool().expand_as(hidden[:-1])].reshape(-1, hidden.shape[-1])
    return {
        "critic_loss": float(loss.item()), "q1_loss": float(loss1.item()), "q2_loss": float(loss2.item()),
        "q_mean": float(torch.minimum(q1, q2)[mask.bool()].mean().item()),
        "target_q_mean": float(bellman[mask.bool()].mean().item()),
        "target_q_std": float(bellman[mask.bool()].std(unbiased=False).item()),
        "hidden_norm_mean": float(valid_hidden.norm(dim=-1).mean().item()),
        "hidden_norm_std": float(valid_hidden.norm(dim=-1).std(unbiased=False).item()),
        "gradient_norm": float(grad_norm.item()), "nan_inf_count": 0,
        "effective_timesteps": int(mask.sum().item())
    }
