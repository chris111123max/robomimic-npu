"""Reusable feed-forward twin Q networks for Stage 2 and later SAC stages."""

from __future__ import annotations

import copy

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim=59, action_dim=14, hidden_dims=(256, 256)):
        super().__init__()
        dimensions = [state_dim + action_dim, *hidden_dims, 1]
        layers = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(input_dim, output_dim), nn.ReLU()))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dims = tuple(int(value) for value in hidden_dims)

    def forward(self, state, action):
        return self.network(torch.cat((state, action), dim=-1))


class TwinCritic(nn.Module):
    def __init__(self, state_dim=59, action_dim=14, hidden_dims=(256, 256)):
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)

    def q_min(self, state, action):
        q1, q2 = self(state, action)
        return torch.minimum(q1, q2)


def make_critic_pair(state_dim, action_dim, hidden_dims, device):
    critic = TwinCritic(state_dim, action_dim, hidden_dims).to(device)
    target = copy.deepcopy(critic).to(device)
    target.requires_grad_(False)
    target.eval()
    return critic, target


@torch.no_grad()
def soft_update(source, target, tau):
    for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
        target_parameter.mul_(1.0 - tau)
        target_parameter.add_(source_parameter, alpha=tau)

