"""Testing-only online Actor module-ablation agent.

Branches
--------
FULL_PRODUCTION:
    Exact production Stage3-v6 Actor update / Adam step.

MEAN_HEAD_ONLY:
    Production Actor loss, backward, global clipping, Adam hyperparameters and
    LR schedule are unchanged. Immediately before optimizer.step(), gradients
    outside the GMM component-mean head are set to None.

RNN_ONLY:
    Same intervention, retaining only recurrent-core gradients.

This tests whether production Adam updates restricted to one Actor parameter
subspace are sufficient to cause real online closed-loop collapse. Critic,
target networks, replay, collector, environment interaction and random2q target
selection remain production behavior.
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


BRANCH_GROUPS = {
    "FULL_PRODUCTION": None,
    "MEAN_HEAD_ONLY": frozenset({"gmm_mean"}),
    "RNN_ONLY": frozenset({"rnn"}),
}
BRANCHES = tuple(BRANCH_GROUPS)
Original = V6.RecurrentGMMTD3


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


def _l2(named_tensors):
    total = 0.0
    by_group = {}
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        value = float(tensor.detach().float().square().sum().cpu())
        total += value
        key = _group(name)
        by_group[key] = by_group.get(key, 0.0) + value
    return (
        math.sqrt(total),
        {key: math.sqrt(value) for key, value in by_group.items()},
    )


class ModuleOnlineAgent(Original):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module_branch = os.environ["ACTOR_MODULE_BRANCH"]
        if self.module_branch not in BRANCHES:
            raise RuntimeError(
                f"ACTOR_MODULE_BRANCH must be one of {BRANCHES}, "
                f"got {self.module_branch!r}"
            )
        self.module_group_out = Path(
            os.environ["ACTOR_MODULE_GROUP_OUT"]
        ).resolve()
        self.module_group_out.mkdir(parents=True, exist_ok=True)

        self.module_reference = (
            copy.deepcopy(self.actor).eval().requires_grad_(False)
        )
        self.module_reference_hash = module_hash(self.module_reference)
        self.module_configured = False

    @property
    def allowed_actor_groups(self):
        return BRANCH_GROUPS[self.module_branch]

    def configure_after_resume(self, payload):
        if payload.get("critic_target_mode") != "random2q":
            raise RuntimeError("Online module test is random2q-only")
        if payload.get("group") != "multi_q":
            raise RuntimeError("Online module test is multi_q-only")
        if int(payload.get("actor_updates", -1)) != 0:
            raise RuntimeError("Source checkpoint must have zero Actor updates")
        if payload["actor_optimizer"].get("state"):
            raise RuntimeError("Source Actor Adam state must be empty")
        if module_hash(self.actor) != self.module_reference_hash:
            raise RuntimeError(
                "critic_ready Actor differs from immutable Stage1 Actor"
            )

        group = self.actor_optimizer.param_groups[0]
        if float(group.get("weight_decay", 0.0)) != 0.0:
            raise RuntimeError("Expected production Actor Adam weight_decay=0")
        if bool(group.get("amsgrad", False)):
            raise RuntimeError("Expected production Actor Adam amsgrad=False")

        self.module_configured = True
        _append_jsonl(
            self.module_group_out / "module_contract.jsonl",
            {
                "testing_only": True,
                "branch": self.module_branch,
                "allowed_actor_groups": (
                    "ALL"
                    if self.allowed_actor_groups is None
                    else sorted(self.allowed_actor_groups)
                ),
                "source_env_steps": int(payload["env_steps"]),
                "source_critic_updates": int(payload["updates"]),
                "source_actor_updates": int(payload["actor_updates"]),
                "source_actor_optimizer_state_entries": len(
                    payload["actor_optimizer"].get("state", {})
                ),
                "reference_actor_hash": self.module_reference_hash,
                "optimizer": "production torch.optim.Adam",
                "betas": [float(x) for x in group["betas"]],
                "eps": float(group["eps"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "semantics": (
                    "exact production Actor optimizer step"
                    if self.allowed_actor_groups is None
                    else "production Actor loss/backward/clip/Adam with "
                         "grad=None outside allowed parameter subspace"
                ),
            },
        )

    def _should_record(self, next_update):
        return int(next_update) <= 10 or int(next_update) % 25 == 0

    @torch.no_grad()
    def _module_optimizer_step(self, env_steps):
        if not self.module_configured:
            raise RuntimeError("ModuleOnlineAgent used before configure_after_resume")

        next_update = int(self.actor_updates) + 1
        record = self._should_record(next_update)
        allowed = self.allowed_actor_groups

        before = None
        if record:
            before = {
                name: parameter.detach().clone()
                for name, parameter in self.actor.named_parameters()
            }

        pre_mask = [
            (
                name,
                None if parameter.grad is None else parameter.grad.detach().clone(),
            )
            for name, parameter in self.actor.named_parameters()
        ]
        pre_mask_l2, pre_mask_groups = _l2(pre_mask)

        masked_tensor_count = 0
        if allowed is not None:
            for name, parameter in self.actor.named_parameters():
                if parameter.grad is None:
                    continue
                if _group(name) not in allowed:
                    parameter.grad = None
                    masked_tensor_count += 1

        post_mask = [
            (
                name,
                None if parameter.grad is None else parameter.grad.detach(),
            )
            for name, parameter in self.actor.named_parameters()
        ]
        post_mask_l2, post_mask_groups = _l2(post_mask)
        if allowed is not None and post_mask_l2 <= 0.0:
            raise RuntimeError(
                f"{self.module_branch} has zero retained gradient at update {next_update}"
            )

        self.actor_optimizer.step()

        if record:
            delta = [
                (name, parameter.detach() - before[name])
                for name, parameter in self.actor.named_parameters()
            ]
            delta_l2, delta_groups = _l2(delta)
            forbidden_sq = 0.0
            if allowed is not None:
                for name, value in delta:
                    if _group(name) not in allowed:
                        forbidden_sq += float(
                            value.detach().float().square().sum().cpu()
                        )
            row = {
                "testing_only": True,
                "branch": self.module_branch,
                "actor_update": int(next_update),
                "env_steps": int(env_steps),
                "actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
                "allowed_actor_groups": (
                    "ALL" if allowed is None else sorted(allowed)
                ),
                "pre_mask_gradient_l2": pre_mask_l2,
                "post_mask_gradient_l2": post_mask_l2,
                "retained_gradient_fraction_l2": (
                    post_mask_l2 / pre_mask_l2 if pre_mask_l2 > 0 else None
                ),
                "masked_parameter_tensor_count": int(masked_tensor_count),
                "pre_mask_gradient_groups_l2": pre_mask_groups,
                "post_mask_gradient_groups_l2": post_mask_groups,
                "actual_parameter_step_l2": delta_l2,
                "parameter_step_groups_l2": delta_groups,
                "forbidden_parameter_step_l2": math.sqrt(forbidden_sq),
            }
            _append_jsonl(
                self.module_group_out / "actor_module_updates.jsonl",
                row,
            )


# V6 inherits the V5 Actor objective unchanged. Derive the exact production
# method and replace only the single optimizer.step() call.
source = textwrap.dedent(inspect.getsource(V5.RecurrentGMMTD3.actor_update))
if source.count("self.actor_optimizer.step()") != 1:
    raise RuntimeError("Production Actor optimizer-step source changed")
source = source.replace(
    "self.actor_optimizer.step()",
    "self._module_optimizer_step(env_steps)",
    1,
)
namespace = dict(vars(V5))
exec(
    compile(source, str(HERE / "derived_module_online_actor_update"), "exec"),
    namespace,
)
ModuleOnlineAgent.actor_update = namespace["actor_update"]
