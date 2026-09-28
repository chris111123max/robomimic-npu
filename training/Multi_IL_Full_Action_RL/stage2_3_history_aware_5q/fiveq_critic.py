"""Five independent history-aware Q networks for Stage2.3."""
from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
STAGE2_2 = HERE.parent / "stage2_2_history_aware_critic"
if str(STAGE2_2) not in sys.path:
    sys.path.insert(0, str(STAGE2_2))

from history_critic import HistoryQNetwork  # noqa: E402


class HistoryAwareFiveQ(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.qs = nn.ModuleList([HistoryQNetwork(**kwargs) for _ in range(5)])

    def encode_history(self, observations, previous_actions, progress, states=None):
        if states is None:
            states = (None,) * len(self.qs)
        if len(states) != len(self.qs):
            raise ValueError("states must match five-Q ensemble size")
        contexts, next_states = [], []
        for q, state in zip(self.qs, states):
            z, s = q.encode_history(observations, previous_actions, progress, state)
            contexts.append(z)
            next_states.append(s)
        return tuple(contexts), tuple(next_states)

    def q_from_context(self, contexts, actions):
        if len(contexts) != len(self.qs):
            raise ValueError("contexts must match five-Q ensemble size")
        return tuple(q.q_from_context(z, actions) for q, z in zip(self.qs, contexts))

    def forward_sequence(self, observations, previous_actions, progress, actions, burn_in=0):
        values = []
        for q in self.qs:
            value, _ = q.forward_sequence(
                observations, previous_actions, progress, actions, burn_in=burn_in
            )
            values.append(value)
        return tuple(values)

    def diagnostic_forward(self, observations, previous_actions, progress, actions):
        values, diagnostics = [], []
        for q in self.qs:
            value, diagnostic = q.diagnostic_forward(
                observations, previous_actions, progress, actions
            )
            values.append(value)
            diagnostics.append(diagnostic)
        return tuple(values), tuple(diagnostics)


def architecture_config(config):
    return {
        key: config[key]
        for key in (
            "obs_dim",
            "action_dim",
            "token_dim",
            "lstm_hidden_dim",
            "lstm_layers",
            "head_hidden_dim",
            "legacy_replay_burn_in_length",
            "learning_sequence_length",
            "history_semantics",
            "num_qs",
        )
    }


def build_critic(config, device=None):
    if int(config["num_qs"]) != 5:
        raise RuntimeError("Stage2.3 requires exactly five independent Q networks")
    model = HistoryAwareFiveQ(
        obs_dim=config["obs_dim"],
        action_dim=config["action_dim"],
        token_dim=config["token_dim"],
        hidden_dim=config["lstm_hidden_dim"],
        layers=config["lstm_layers"],
        head_hidden_dim=config["head_hidden_dim"],
    )
    return model if device is None else model.to(device)


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def per_q_hashes(model):
    return [state_hash(q.state_dict()) for q in model.qs]


def checkpoint_payload(model, optimizer, config, step, validation=None):
    return {
        "stage_version": "2.3",
        "critic_type": "history_aware_5q",
        "num_qs": 5,
        "architecture": architecture_config(config),
        "critic_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "checkpoint_step": int(step),
        "validation": copy.deepcopy(validation),
        "normalization": copy.deepcopy(config["normalization"]),
        "token_schema": list(config["token_schema"]),
        "horizon": int(config["horizon"]),
        "training_target": config["training_target"],
        "loss": config["loss"],
        "gamma": float(config["gamma"]),
        "history_semantics": config["history_semantics"],
        "dataset_sources": {
            name: f'{config["dataset_root"]}/{name}/transitions.hdf5'
            for name in ("bc_rnn", "bc_transformer", "bc_gmm")
        },
        "sampling_ratios": {
            "rnn_q": [1, 0, 0],
            "multi_q": [1 / 3, 1 / 3, 1 / 3],
        },
        "random_seed": int(config["training_seed"]),
        "terminated_truncated_semantics": config[
            "terminated_truncated_semantics"
        ],
        "best_checkpoint_rule": config["best_checkpoint_rule"],
    }


def load_checkpoint(path, config, device="cpu"):
    payload = torch.load(path, map_location=device)
    expected = {
        "stage_version": "2.3",
        "critic_type": "history_aware_5q",
        "num_qs": 5,
        "architecture": architecture_config(config),
        "normalization": config["normalization"],
        "token_schema": config["token_schema"],
        "horizon": int(config["horizon"]),
        "training_target": config["training_target"],
        "loss": config["loss"],
        "gamma": float(config["gamma"]),
        "terminated_truncated_semantics": config[
            "terminated_truncated_semantics"
        ],
        "history_semantics": config["history_semantics"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"Stage2.3 checkpoint contract mismatch for {key}: "
                f"{payload.get(key)!r} != {value!r}"
            )
    model = build_critic(config, device)
    model.load_state_dict(payload["critic_state_dict"], strict=True)
    return model, payload
