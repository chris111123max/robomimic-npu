"""Stage 3 SAC-compatible squashed Gaussian actor."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
RLKIT_ROOT = REPO_ROOT / "rlkit"
if str(RLKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(RLKIT_ROOT))

from rlkit.torch.sac.policies import TanhGaussianPolicy  # noqa: E402


def build_actor(state_dim=59, action_dim=14, hidden_dims=(256, 256),
                initial_log_std=-3.0, freeze_log_std=False, device=None):
    actor = TanhGaussianPolicy(
        hidden_sizes=list(hidden_dims),
        obs_dim=int(state_dim),
        action_dim=int(action_dim),
        std=None,
    )
    with torch.no_grad():
        actor.last_fc_log_std.weight.zero_()
        actor.last_fc_log_std.bias.fill_(float(initial_log_std))
    actor.last_fc_log_std.requires_grad_(not freeze_log_std)
    if device is not None:
        actor = actor.to(device)
    return actor


def deterministic_action(actor, state):
    return actor(state, deterministic=True)[0]


def stochastic_action_and_log_prob(actor, state):
    output = actor(state, deterministic=False, reparameterize=True, return_log_prob=True)
    return output[0], output[3]


def load_actor_checkpoint(checkpoint_path, device, freeze_log_std=False):
    payload = torch.load(checkpoint_path, map_location=device)
    architecture = payload["architecture"]
    distribution = payload["action_distribution"]
    actor = build_actor(
        state_dim=architecture["state_dim"],
        action_dim=architecture["action_dim"],
        hidden_dims=architecture["hidden_dims"],
        initial_log_std=distribution["initial_log_std"],
        freeze_log_std=freeze_log_std,
        device=device,
    )
    actor.load_state_dict(payload["actor_state_dict"])
    return actor, payload

