"""Vectorized GMM component-mean Q expectations and diagnostic sampling."""
from __future__ import annotations

import torch


def tensors(distribution):
    base = distribution.component_distribution.base_dist
    return {"means_normalized": base.loc, "scales": base.scale,
            "logits": distribution.mixture_distribution.logits,
            "probs": distribution.mixture_distribution.probs}


def single_component_mean_q(critic, states, distribution, action_scale,
                            action_offset, twin_min=True):
    """Evaluate all K component means in one Q call; learned std is unused."""
    params = tensors(distribution)
    means = params["means_normalized"]
    batch, modes, action_dim = means.shape
    actions = (means * action_scale.reshape(1, 1, action_dim)
               + action_offset.reshape(1, 1, action_dim))
    flat_states = states[:, None, :].expand(-1, modes, -1).reshape(-1, states.shape[-1])
    flat_actions = actions.reshape(-1, action_dim)
    if twin_min:
        q1, q2 = critic(flat_states, flat_actions)
        q1, q2 = q1.reshape(batch, modes), q2.reshape(batch, modes)
        selected = torch.minimum(q1, q2)
    else:
        q1 = critic.q1(flat_states, flat_actions).reshape(batch, modes)
        q2, selected = None, q1
    return (params["probs"] * selected).sum(-1), q1, q2, params, actions


def sequence_component_mean_q(critic, states, distributions, action_scale,
                              action_offset):
    """One Q1 forward across sequence time and all mixture components."""
    pieces = [tensors(item) for item in distributions]
    params = {key: torch.stack([item[key] for item in pieces], dim=1)
              for key in ("means_normalized", "scales", "logits", "probs")}
    means = params["means_normalized"]
    batch, time_steps, modes, action_dim = means.shape
    actions = (means * action_scale.reshape(1, 1, 1, action_dim)
               + action_offset.reshape(1, 1, 1, action_dim))
    flat_states = states[:, :, None, :].expand(-1, -1, modes, -1).reshape(-1, states.shape[-1])
    q1 = critic.q1(flat_states, actions.reshape(-1, action_dim)).reshape(batch, time_steps, modes)
    return (params["probs"] * q1).sum(-1), q1, params, actions


def full_sequence_component_mean_q(critic, states, distribution, action_scale, action_offset):
    """Native full-sequence Actor output with identical time/mode Q layout."""
    params = tensors(distribution)
    means = params["means_normalized"]
    batch, time_steps, modes, action_dim = means.shape
    actions = means * action_scale.reshape(1, 1, 1, action_dim) + action_offset.reshape(1, 1, 1, action_dim)
    flat_states = states[:, :, None, :].expand(-1, -1, modes, -1).reshape(-1, states.shape[-1])
    q1 = critic.q1(flat_states, actions.reshape(-1, action_dim)).reshape(batch, time_steps, modes)
    return (params["probs"] * q1).sum(-1), q1, params, actions


def single_expected_q(critic, states, distribution, action_scale, action_offset,
                      samples=1, twin_min=True, epsilon=None):
    params = tensors(distribution)
    means = params["means_normalized"]
    batch, modes, action_dim = means.shape
    shape = (batch, modes, int(samples), action_dim)
    if epsilon is None:
        epsilon = torch.randn(shape, dtype=means.dtype, device=means.device)
    elif tuple(epsilon.shape) != shape:
        raise ValueError("GMM epsilon shape differs from [batch,mode,sample,action]")
    normalized = means.unsqueeze(-2) + params["scales"].unsqueeze(-2) * epsilon
    actions = (normalized * action_scale.reshape(1, 1, 1, action_dim)
               + action_offset.reshape(1, 1, 1, action_dim))
    flat_states = states[:, None, None, :].expand(-1, modes, int(samples), -1).reshape(-1, states.shape[-1])
    flat_actions = actions.reshape(-1, action_dim)
    if twin_min:
        q1, q2 = critic(flat_states, flat_actions)
        q1 = q1.reshape(batch, modes, int(samples))
        q2 = q2.reshape(batch, modes, int(samples))
        selected = torch.minimum(q1, q2)
    else:
        q1 = critic.q1(flat_states, flat_actions).reshape(batch, modes, int(samples))
        q2 = None
        selected = q1
    expected = (params["probs"] * selected.mean(-1)).sum(-1)
    return expected, q1, q2, params, actions


def sequence_expected_q(critic, states, distributions, action_scale, action_offset,
                        samples=1, epsilon=None):
    """One Q1 forward over all sequence timesteps, modes, and samples."""
    pieces = [tensors(item) for item in distributions]
    params = {key: torch.stack([item[key] for item in pieces], dim=1)
              for key in ("means_normalized", "scales", "logits", "probs")}
    means = params["means_normalized"]
    batch, time_steps, modes, action_dim = means.shape
    shape = (batch, time_steps, modes, int(samples), action_dim)
    if epsilon is None:
        epsilon = torch.randn(shape, dtype=means.dtype, device=means.device)
    elif tuple(epsilon.shape) != shape:
        raise ValueError("GMM sequence epsilon shape differs from [B,T,K,M,A]")
    normalized = means.unsqueeze(-2) + params["scales"].unsqueeze(-2) * epsilon
    actions = (normalized * action_scale.reshape(1, 1, 1, 1, action_dim)
               + action_offset.reshape(1, 1, 1, 1, action_dim))
    flat_states = states[:, :, None, None, :].expand(
        -1, -1, modes, int(samples), -1).reshape(-1, states.shape[-1])
    q1 = critic.q1(flat_states, actions.reshape(-1, action_dim)).reshape(
        batch, time_steps, modes, int(samples))
    expected = (params["probs"] * q1.mean(-1)).sum(-1)
    return expected, q1, params, actions
