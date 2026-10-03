"""Testing-only Actor optimizer-geometry intervention.

Branches
--------
PRODUCTION_ADAM:
    Exact inherited production Actor update and production Adam step.

SGD_STEP_MATCHED_CONTROL:
    Uses the exact same production Actor loss / backward / grad clipping, but
    replaces Adam's parameter-space direction with the global negative-gradient
    direction. A shadow Adam state, driven by the *same clipped gradients*,
    computes the global L2 step magnitude Adam would have taken on that branch
    at that update. The actual Actor step is then globally rescaled SGD:

        delta_theta = -g / ||g|| * ||delta_theta_shadow_adam||

    Thus global parameter-step magnitude is matched while Adam's per-parameter
    preconditioning geometry is removed.

Production source files and formal checkpoints are never modified.
"""
from __future__ import annotations

import copy
import inspect
import json
import math
import os
import sys
import textwrap
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RL_ROOT = HERE.parents[2]
V5_DIR = RL_ROOT / "stage3_v5_rgmm_td3"
V6_DIR = RL_ROOT / "stage3_v6_dual_2q"
for folder in (V5_DIR, V6_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import stage3_v5_agent as V5  # noqa: E402
import stage3_v6_agent as V6  # noqa: E402
from stage3_v5_actor import module_hash  # noqa: E402


BRANCHES = ("PRODUCTION_ADAM", "SGD_STEP_MATCHED_CONTROL")
Original = V6.RecurrentGMMTD3


def _scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _group(name):
    if name.startswith("nets.rnn.nets."):
        return "rnn"
    if name.startswith("nets.decoder.nets.mean."):
        return "gmm_mean"
    if name.startswith("nets.decoder.nets.logits."):
        return "gmm_logits"
    if name.startswith("nets.decoder.nets.scale."):
        return "gmm_std"
    if "encoder" in name:
        return "encoder"
    return "other"


def _append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _sum_squares(tensors):
    values = [
        tensor.detach().float().square().sum()
        for tensor in tensors
        if tensor is not None
    ]
    if not values:
        return torch.tensor(0.0)
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total


def _group_l2(named_tensors):
    accum = {}
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        key = _group(name)
        value = tensor.detach().float().square().sum()
        accum[key] = accum.get(key, 0.0) + value
    if accum:
        values = list(accum.values())
        total = values[0]
        for value in values[1:]:
            total = total + value
        accum["total"] = total
    return {key: _scalar(value.sqrt()) for key, value in accum.items()}


class GeometryAgent(Original):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry_branch = os.environ["OPT_GEOMETRY_BRANCH"]
        if self.geometry_branch not in BRANCHES:
            raise RuntimeError(
                f"OPT_GEOMETRY_BRANCH must be one of {BRANCHES}, "
                f"got {self.geometry_branch!r}"
            )
        self.geometry_group_out = Path(os.environ["OPT_GEOMETRY_GROUP_OUT"]).resolve()
        self.geometry_group_out.mkdir(parents=True, exist_ok=True)

        # At construction the Actor is the immutable Stage1 source. The random2q
        # critic_ready checkpoint has actor_updates=0, so strict equality after
        # restore is required before the intervention is enabled.
        self.geometry_reference = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.geometry_reference_hash = module_hash(self.geometry_reference)
        self.geometry_configured = False

        # Shadow Adam state exists only for the step-matched control. It never
        # owns or updates Actor parameters.
        self._shadow_adam = {}
        self._shadow_active_parameter_steps = {}
        self._last_geometry_record = None

    def configure_after_resume(self, payload):
        if payload.get("critic_target_mode") != "random2q":
            raise RuntimeError("Optimizer geometry test is random2q-only")
        if payload.get("group") != "multi_q":
            raise RuntimeError("Optimizer geometry test is multi_q-only")
        if int(payload.get("actor_updates", -1)) != 0:
            raise RuntimeError("Source checkpoint must have zero Actor updates")
        if payload["actor_optimizer"].get("state"):
            raise RuntimeError("Source Actor Adam state is not fresh / empty")
        if module_hash(self.actor) != self.geometry_reference_hash:
            raise RuntimeError(
                "critic_ready Actor differs from immutable Stage1 Actor; "
                "step-matched intervention contract invalid"
            )

        group = self.actor_optimizer.param_groups[0]
        if float(group.get("weight_decay", 0.0)) != 0.0:
            raise RuntimeError("Control assumes production Actor Adam weight_decay=0")
        if bool(group.get("amsgrad", False)):
            raise RuntimeError("Control assumes production Actor Adam amsgrad=False")

        self.geometry_configured = True
        _append_jsonl(
            self.geometry_group_out / "geometry_contract.jsonl",
            {
                "testing_only": True,
                "branch": self.geometry_branch,
                "source_env_steps": int(payload["env_steps"]),
                "source_critic_updates": int(payload["updates"]),
                "source_actor_updates": int(payload["actor_updates"]),
                "source_actor_optimizer_state_entries": len(
                    payload["actor_optimizer"].get("state", {})
                ),
                "actor_reference_hash": self.geometry_reference_hash,
                "betas": [float(x) for x in group["betas"]],
                "eps": float(group["eps"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "amsgrad": bool(group.get("amsgrad", False)),
                "semantics": (
                    "PRODUCTION_ADAM exact production optimizer"
                    if self.geometry_branch == "PRODUCTION_ADAM"
                    else "negative-gradient direction with global L2 matched to "
                         "shadow production Adam driven by the same clipped gradients"
                ),
            },
        )

    def _should_record_geometry(self, update_index):
        return (
            int(update_index) <= 10
            or int(update_index) % 25 == 0
        )

    @torch.no_grad()
    def _shadow_adam_target(self):
        """Advance shadow Adam moments and return its exact global step L2.

        PyTorch Adam defaults used by production:
            betas=(0.9,0.999), eps=1e-8, weight_decay=0, amsgrad=False.

        With weight_decay=0 the Adam step depends on gradients and moments but
        not on the shadow parameter values, so no shadow parameter copy is
        necessary.
        """
        group = self.actor_optimizer.param_groups[0]
        beta1, beta2 = (float(x) for x in group["betas"])
        eps = float(group["eps"])
        lr = float(group["lr"])

        update_units = {}
        active_grads = {}
        for name, parameter in self.actor.named_parameters():
            grad = parameter.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("Sparse Actor gradient unsupported")
            g = grad.detach()
            state = self._shadow_adam.get(name)
            if state is None:
                state = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                }
                self._shadow_adam[name] = state
            state["step"] += 1
            state["exp_avg"].mul_(beta1).add_(g, alpha=1.0 - beta1)
            state["exp_avg_sq"].mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
            step = int(state["step"])
            bias1 = 1.0 - beta1 ** step
            bias2 = 1.0 - beta2 ** step
            denom = state["exp_avg_sq"].sqrt().div_(math.sqrt(bias2)).add_(eps)
            unit = state["exp_avg"].div(bias1).div(denom)
            update_units[name] = unit
            active_grads[name] = g
            self._shadow_active_parameter_steps[name] = step

        grad_sq = _sum_squares(active_grads.values())
        adam_unit_sq = _sum_squares(update_units.values())
        grad_l2 = grad_sq.sqrt()
        target_l2 = adam_unit_sq.sqrt() * lr
        return active_grads, update_units, grad_l2, target_l2

    @torch.no_grad()
    def _geometry_optimizer_step(self, env_steps):
        if not self.geometry_configured:
            raise RuntimeError("GeometryAgent used before configure_after_resume")

        next_update = int(self.actor_updates) + 1
        should_record = self._should_record_geometry(next_update)
        lr = float(self.actor_optimizer.param_groups[0]["lr"])

        if self.geometry_branch == "PRODUCTION_ADAM":
            before = None
            if should_record:
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in self.actor.named_parameters()
                }
                grad_named = [
                    (name, None if parameter.grad is None else parameter.grad.detach())
                    for name, parameter in self.actor.named_parameters()
                ]
                grad_l2 = math.sqrt(
                    sum(
                        float(g.float().square().sum().detach().cpu())
                        for _, g in grad_named
                        if g is not None
                    )
                )
            self.actor_optimizer.step()
            if should_record:
                delta_named = [
                    (name, parameter.detach() - before[name])
                    for name, parameter in self.actor.named_parameters()
                ]
                delta_l2 = math.sqrt(
                    sum(
                        float(delta.float().square().sum().detach().cpu())
                        for _, delta in delta_named
                    )
                )
                row = {
                    "testing_only": True,
                    "branch": self.geometry_branch,
                    "actor_update": next_update,
                    "env_steps": int(env_steps),
                    "actor_lr": lr,
                    "clipped_gradient_l2": grad_l2,
                    "actual_parameter_step_l2": delta_l2,
                    "step_match_target_l2": None,
                    "step_match_ratio": None,
                    "gradient_groups_l2": _group_l2(grad_named),
                    "parameter_step_groups_l2": _group_l2(delta_named),
                }
                self._last_geometry_record = row
                _append_jsonl(
                    self.geometry_group_out / "actor_geometry_updates.jsonl",
                    row,
                )
            return

        # STEP_MATCHED_CONTROL: advance shadow Adam moments from the same
        # already-clipped gradients, then move only along global -g.
        active_grads, update_units, grad_l2_t, target_l2_t = self._shadow_adam_target()
        if not bool(torch.isfinite(grad_l2_t)) or not bool(torch.isfinite(target_l2_t)):
            raise FloatingPointError("Non-finite geometry-control step")
        grad_l2_value = _scalar(grad_l2_t)
        target_l2_value = _scalar(target_l2_t)
        scale = (
            target_l2_t / grad_l2_t.clamp_min(1e-30)
            if grad_l2_value > 0.0
            else torch.zeros_like(target_l2_t)
        )
        scale_value = _scalar(scale)

        for name, parameter in self.actor.named_parameters():
            grad = active_grads.get(name)
            if grad is not None:
                parameter.add_(grad, alpha=-scale_value)

        if should_record:
            actual_l2 = grad_l2_value * abs(scale_value)
            row = {
                "testing_only": True,
                "branch": self.geometry_branch,
                "actor_update": next_update,
                "env_steps": int(env_steps),
                "actor_lr": lr,
                "clipped_gradient_l2": grad_l2_value,
                "shadow_adam_parameter_step_l2": target_l2_value,
                "actual_parameter_step_l2": actual_l2,
                "step_match_target_l2": target_l2_value,
                "step_match_ratio": (
                    actual_l2 / target_l2_value if target_l2_value > 0 else 1.0
                ),
                "matched_gradient_scale": scale_value,
                "shadow_adam_active_parameter_count": len(active_grads),
                "shadow_adam_step_min": (
                    min(self._shadow_active_parameter_steps.values())
                    if self._shadow_active_parameter_steps else 0
                ),
                "shadow_adam_step_max": (
                    max(self._shadow_active_parameter_steps.values())
                    if self._shadow_active_parameter_steps else 0
                ),
                "gradient_groups_l2": _group_l2(active_grads.items()),
                "shadow_adam_unit_groups_l2": _group_l2(update_units.items()),
            }
            self._last_geometry_record = row
            _append_jsonl(
                self.geometry_group_out / "actor_geometry_updates.jsonl",
                row,
            )


# Derive the Actor update from the exact production V5 implementation. V6
# inherits this Actor objective unchanged; only its Critic target differs.
source = textwrap.dedent(inspect.getsource(V5.RecurrentGMMTD3.actor_update))
if source.count("self.actor_optimizer.step()") != 1:
    raise RuntimeError("Production Actor optimizer-step source changed")
source = source.replace(
    "self.actor_optimizer.step()",
    "self._geometry_optimizer_step(env_steps)",
    1,
)
namespace = dict(vars(V5))
exec(compile(source, str(HERE / "derived_geometry_actor_update"), "exec"), namespace)
GeometryAgent.actor_update = namespace["actor_update"]
