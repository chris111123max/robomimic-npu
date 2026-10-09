"""Independent QSE-GMM likelihood; controller actions -> normalized density."""
import math
import torch
from stage3_v5_actor import flat_to_obs

def validate_windows(batch):
    steps = batch["episode_steps"]
    if torch.is_tensor(steps):
        steps = steps.detach().cpu().numpy()
    import numpy as np
    steps = np.asarray(steps)
    if steps.ndim != 2 or steps.shape[1] != 10:
        raise ValueError("Expected full H10 windows")
    if not np.all(steps[:, 0] % 10 == 0) or not np.all(np.diff(steps, axis=1) == 1):
        raise ValueError("Non-aligned or cross-episode window")
    if tuple(batch["observations"].shape) != (*steps.shape, 59):
        raise ValueError("Observation contract")
    if tuple(batch["actions"].shape) != (*steps.shape, 14):
        raise ValueError("Action contract")

def gmm_log_prob(means, logits, std, actions):
    if means.shape[-2:] != (5, 14) or logits.shape != means.shape[:-1]:
        raise ValueError("Expected five components and 14 action dimensions")
    if actions.shape != means.shape[:-2] + (14,):
        raise ValueError("Action shape")
    std = torch.as_tensor(std, dtype=means.dtype, device=means.device)
    if not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Supervision std must be finite and positive")
    if not all(torch.isfinite(x).all() for x in (means, logits, actions)):
        raise FloatingPointError("Nonfinite GMM input")
    log_component = (-0.5 * ((actions.unsqueeze(-2) - means) / std).square()
                     - std.log() - 0.5 * math.log(2 * math.pi)).sum(-1)
    return torch.logsumexp(torch.log_softmax(logits, -1) + log_component, -1)

def good_loss(actor, batch, scale, offset, loss_config, return_distribution=False):
    validate_windows(batch)
    dev = next(actor.parameters()).device
    obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=dev)
    actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=dev)
    scale = scale.reshape(14); offset = offset.reshape(14)
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Invalid action normalization scale")
    normalized = (actions - offset) / scale
    dist = actor.forward_train(flat_to_obs(obs), rnn_init_state=None, return_state=False)
    base = dist.component_distribution.base_dist
    mode = loss_config["std_mode"]
    if mode == "fixed":
        std = torch.as_tensor(loss_config["supervision_std"], dtype=base.loc.dtype, device=dev)
        if std.numel() != 14: raise ValueError("Need calibrated 14-vector")
        std = std.reshape(1, 1, 1, 14)
    elif mode == "learned":
        std = base.scale
    else:
        raise ValueError("Unknown supervision std mode")
    logp = gmm_log_prob(base.loc, dist.mixture_distribution.logits, std, normalized)
    # Normalized-action density: environment density subtracts sum(log(scale)),
    # a constant wrt actor, deliberately excluded from the supervised objective.
    mask = torch.as_tensor(batch.get("mask", torch.ones_like(logp)), dtype=logp.dtype, device=dev)
    weight = torch.as_tensor(batch.get("weights", torch.ones_like(logp)), dtype=logp.dtype, device=dev)
    if mask.shape != logp.shape or weight.shape != logp.shape:
        raise ValueError("Mask/weight shape")
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("Invalid supervision weights")
    if not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Mask must be binary")
    valid_weight = mask * weight
    denom = valid_weight.sum()
    if not torch.isfinite(denom) or denom <= 0:
        raise ValueError("No valid weighted positions")
    loss = -(logp * valid_weight).sum() / denom
    if not torch.isfinite(loss): raise FloatingPointError("Nonfinite good loss")
    return (loss, dist) if return_distribution else loss
