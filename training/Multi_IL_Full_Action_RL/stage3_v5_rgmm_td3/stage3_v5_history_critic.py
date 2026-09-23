"""Stage2.2 history-aware Critic adapter for the Stage3-v5 trainer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ROOT = Path(__file__).resolve().parents[3]
STAGE2_2 = ROOT / "training" / "Multi_IL_Full_Action_RL" / "stage2_2_history_aware_critic"
if str(STAGE2_2) not in sys.path:
    sys.path.insert(0, str(STAGE2_2))
from history_critic import load_checkpoint  # noqa: E402

CONFIG_PATH = STAGE2_2 / "stage2_2_config.json"


def load_stage2_2_critic_checkpoint(path, device):
    """Strictly load the audited Stage2.2 recurrent Twin-Q checkpoint."""
    config = json.loads(CONFIG_PATH.read_text())
    critic, payload = load_checkpoint(
        path, config, device=device, critic_type="history_aware_twin_q")
    return critic, payload


def previous_actions(actions, episode_steps):
    """Build exact a_(t-1) for episode-prefix replay."""
    if actions.ndim != 3 or episode_steps.shape != actions.shape[:2]:
        raise ValueError("History Critic expects actions [B,T,A] and steps [B,T]")
    result = torch.zeros_like(actions)
    result[:, 1:] = actions[:, :-1]
    result = result.masked_fill(episode_steps.eq(0).unsqueeze(-1), 0.0)
    return result


def _packed_context(network, observations, previous, progress, sequence_lengths):
    """Run one recurrent branch over right-padded episode prefixes."""
    encoded = network._tokens(observations, previous, progress)
    packed = pack_padded_sequence(
        encoded, sequence_lengths.detach().cpu(), batch_first=True,
        enforce_sorted=False)
    packed_context, _ = network.lstm(packed)
    context, _ = pad_packed_sequence(
        packed_context, batch_first=True, total_length=observations.shape[1])
    return context


def encode_replay_contexts(critic, observations, actions, episode_steps,
                           horizon, next_observations=None, sequence_lengths=None):
    """Encode current or successor histories using Stage2.2 full-prefix semantics."""
    if observations.ndim != 3 or actions.ndim != 3:
        raise ValueError("History Critic replay inputs must be rank-three sequences")
    if sequence_lengths is None:
        sequence_lengths = torch.full(
            (observations.shape[0],), observations.shape[1],
            dtype=torch.long, device=observations.device)
    else:
        sequence_lengths = torch.as_tensor(
            sequence_lengths, dtype=torch.long, device=observations.device)
    if torch.any(sequence_lengths <= 0) or torch.any(sequence_lengths > observations.shape[1]):
        raise ValueError("Invalid full-prefix sequence lengths")
    if next_observations is None:
        tokens = observations
        prior = previous_actions(actions, episode_steps)
        steps = episode_steps
    else:
        if next_observations.shape != observations.shape:
            raise ValueError("next_observations must match observations")
        tokens = next_observations
        # At successor state t+1, the previous executed action is exactly a_t.
        prior = actions
        steps = episode_steps + 1
    progress = steps.to(dtype=observations.dtype).unsqueeze(-1) / float(horizon)
    return (
        _packed_context(critic.q1, tokens, prior, progress, sequence_lengths),
        _packed_context(critic.q2, tokens, prior, progress, sequence_lengths),
    )


def final_contexts(contexts, sequence_lengths):
    """Gather each row's final valid recurrent state from a padded prefix batch."""
    lengths = torch.as_tensor(sequence_lengths, dtype=torch.long,
                              device=contexts[0].device)
    rows = torch.arange(len(lengths), device=contexts[0].device)
    indices = lengths - 1
    return contexts[0][rows, indices], contexts[1][rows, indices]


def _gather_tail(value, sequence_lengths, width):
    lengths = torch.as_tensor(sequence_lengths, dtype=torch.long, device=value.device)
    if torch.any(lengths < int(width)):
        raise ValueError("Critic sampling prefix is shorter than the gradient window")
    offsets = torch.arange(int(width), device=value.device).unsqueeze(0)
    indices = lengths.unsqueeze(1) - int(width) + offsets
    expand = indices
    for _ in range(value.ndim - 2):
        expand = expand.unsqueeze(-1)
    expand = expand.expand(value.shape[0], int(width), *value.shape[2:])
    return torch.gather(value, 1, expand)


def _detached_prefix_state(network, observations, previous, progress,
                           prefix_lengths):
    """Reconstruct exact prefix state without retaining a backward graph."""
    lengths = torch.as_tensor(prefix_lengths, dtype=torch.long,
                              device=observations.device)
    batch = observations.shape[0]
    layers = network.lstm.num_layers
    hidden = network.lstm.hidden_size
    h = torch.zeros((layers, batch, hidden), dtype=observations.dtype,
                    device=observations.device)
    c = torch.zeros_like(h)
    active = lengths > 0
    if not bool(active.any()):
        return h, c
    max_prefix = int(lengths.max().item())
    with torch.no_grad():
        encoded = network._tokens(
            observations[active, :max_prefix],
            previous[active, :max_prefix],
            progress[active, :max_prefix])
        packed = pack_padded_sequence(
            encoded, lengths[active].detach().cpu(), batch_first=True,
            enforce_sorted=False)
        _, state = network.lstm(packed)
    h[:, active] = state[0].detach()
    c[:, active] = state[1].detach()
    return h, c


def final_contexts_with_detached_prefix(critic, observations, actions,
                                        episode_steps, horizon,
                                        sequence_lengths, gradient_window):
    """Full-history final states with gradients only through the tail window.

    Prefix tokens reconstruct the Stage2.2 recurrent state under no_grad.
    Only the final gradient_window tokens retain a backward graph.
    """
    lengths = torch.as_tensor(sequence_lengths, dtype=torch.long,
                              device=observations.device)
    width = int(gradient_window)
    previous = previous_actions(actions, episode_steps)
    progress = episode_steps.to(dtype=observations.dtype).unsqueeze(-1) / float(horizon)
    prefix_lengths = lengths - width
    tail_obs = _gather_tail(observations, lengths, width)
    tail_prev = _gather_tail(previous, lengths, width)
    tail_progress = _gather_tail(progress, lengths, width)
    finals = []
    for network in (critic.q1, critic.q2):
        state = _detached_prefix_state(
            network, observations, previous, progress, prefix_lengths)
        tail_context, _ = network.encode_history(
            tail_obs, tail_prev, tail_progress, state)
        finals.append(tail_context[:, -1])
    return tuple(finals)


def component_mean_q(critic, contexts, distribution, action_scale, action_offset,
                     twin_min=True):
    """Evaluate every categorical GMM component from pre-encoded histories."""
    base = distribution.component_distribution.base_dist
    means = base.loc
    probabilities = distribution.mixture_distribution.probs
    action_dim = means.shape[-1]
    broadcast = (1,) * (means.ndim - 1) + (action_dim,)
    actions = means * action_scale.reshape(broadcast) + action_offset.reshape(broadcast)
    if twin_min:
        q1, q2 = critic.q_from_context(contexts, actions)
        q1, q2 = q1.squeeze(-1), q2.squeeze(-1)
        selected = torch.minimum(q1, q2)
    else:
        q1 = critic.q1.q_from_context(contexts[0], actions).squeeze(-1)
        q2, selected = None, q1
    params = {"means_normalized": means, "scales": base.scale,
              "logits": distribution.mixture_distribution.logits,
              "probs": probabilities}
    return (probabilities * selected).sum(-1), q1, q2, params, actions


def sampled_q(critic, contexts, distribution, action_scale, action_offset,
              samples=1, twin_min=True, epsilon=None):
    """Diagnostic learned-std expectation from pre-encoded histories."""
    base = distribution.component_distribution.base_dist
    means, scales = base.loc, base.scale
    probabilities = distribution.mixture_distribution.probs
    action_dim, modes = means.shape[-1], means.shape[-2]
    shape = tuple(means.shape[:-1]) + (int(samples), action_dim)
    if epsilon is None:
        epsilon = torch.randn(shape, dtype=means.dtype, device=means.device)
    elif tuple(epsilon.shape) != shape:
        raise ValueError("History Critic GMM epsilon shape mismatch")
    normalized = means.unsqueeze(-2) + scales.unsqueeze(-2) * epsilon
    broadcast = (1,) * (normalized.ndim - 1) + (action_dim,)
    actions = (normalized * action_scale.reshape(broadcast)
               + action_offset.reshape(broadcast))
    leading = tuple(contexts[0].shape[:-1])
    flat_actions = actions.reshape(*leading, modes * int(samples), action_dim)
    if twin_min:
        q1, q2 = critic.q_from_context(contexts, flat_actions)
        q1 = q1.squeeze(-1).reshape(*leading, modes, int(samples))
        q2 = q2.squeeze(-1).reshape(*leading, modes, int(samples))
        selected = torch.minimum(q1, q2)
    else:
        q1 = critic.q1.q_from_context(contexts[0], flat_actions)
        q1 = q1.squeeze(-1).reshape(*leading, modes, int(samples))
        q2, selected = None, q1
    params = {"means_normalized": means, "scales": scales,
              "logits": distribution.mixture_distribution.logits,
              "probs": probabilities}
    return (probabilities * selected.mean(-1)).sum(-1), q1, q2, params, actions
