"""Stage2-new's public, SAC-compatible Twin Critic API.

The implementation is deliberately shared with the earlier feed-forward Stage2
module.  Stage2-new enables its optional hidden LayerNorm blocks; legacy code
continues using the default ``layer_norm=False`` behavior unchanged.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

_SHARED_PATH = Path(__file__).resolve().parents[1] / "stage2_critic_pretraining" / "critic_network.py"
_SPEC = importlib.util.spec_from_file_location("_stage2_new_shared_critic", _SHARED_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load shared Critic from {_SHARED_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
TwinCritic = _MODULE.TwinCritic


def build_critic(obs_dim=59, action_dim=14, hidden_dims=(256, 256),
                 activation="relu", layer_norm=True, device=None):
    if activation.lower() != "relu":
        raise ValueError("The shared SAC critic uses ReLU; activation must be 'relu'")
    if not layer_norm:
        raise ValueError("Stage2-new requires LayerNorm in every hidden block")
    critic = TwinCritic(int(obs_dim), int(action_dim), tuple(hidden_dims), layer_norm=True)
    return critic if device is None else critic.to(device)


def model_config_from(config):
    return {
        "obs_dim": int(config["obs_dim"]), "action_dim": int(config["action_dim"]),
        "hidden_dims": [int(item) for item in config["hidden_dims"]],
        "activation": str(config["activation"]), "layer_norm": bool(config["layer_norm"]),
        "implementation_source": "stage2_critic_pretraining/critic_network.py",
        "initialization": "PyTorch nn.Linear default initialization (shared critic)",
    }


def load_stage2_critic_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device)
    required = {"critic_state_dict", "model_config"}
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"Stage2-new checkpoint is missing keys: {sorted(missing)}")
    config = payload["model_config"]
    model = build_critic(obs_dim=config["obs_dim"], action_dim=config["action_dim"],
                         hidden_dims=config["hidden_dims"], activation=config["activation"],
                         layer_norm=config["layer_norm"], device=device)
    model.load_state_dict(payload["critic_state_dict"])
    return model, payload
