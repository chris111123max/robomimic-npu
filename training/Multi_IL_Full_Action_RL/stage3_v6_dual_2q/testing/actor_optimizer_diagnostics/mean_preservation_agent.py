"""Testing-only online component-mean preservation agent.

Branches
--------
FULL_PRODUCTION:
    Exact inherited production Stage3-v6 Actor update.

FULL_MEAN_PRESERVATION:
    Keeps the production Actor objective, Critic, replay, global grad clipping,
    Adam optimizer and LR schedule, but adds an adaptive output-space penalty on
    GMM component means relative to the untouched random2q critic_ready Actor.

Only component means are preserved. No KL / logits / std / hidden-state anchor
is present.

Controller (pre-registered in code)
-----------------------------------
mean_rms <= 0.004 : no preservation gradient
0.004 .. 0.006    : linearly increase anchor/RL gradient-norm target
mean_rms >= 0.006 : target anchor gradient norm = 4 * RL gradient norm

The scalar coefficient is detached: it is a controller, not a differentiable
function of the loss. This experiment tests necessity of component-mean drift,
not a proposed final training algorithm.
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


BRANCHES = ("FULL_PRODUCTION", "FULL_MEAN_PRESERVATION")
Original = V6.RecurrentGMMTD3

MEAN_RMS_SAFE = 0.004
MEAN_RMS_HARD = 0.006
MAX_ANCHOR_RL_GRAD_RATIO = 4.0
MAX_LAMBDA = 1.0e6


def _scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _norm(grads):
    values = [
        grad.detach().float().square().sum()
        for grad in grads
        if grad is not None
    ]
    if not values:
        return torch.tensor(0.0)
    return torch.stack(values).sum().sqrt()


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


def _group_grad_norm(named, grads):
    accum = {}
    for (name, _), grad in zip(named, grads):
        if grad is None:
            continue
        key = _group(name)
        value = grad.detach().float().square().sum()
        accum[key] = accum.get(key, 0.0) + value
    result = {}
    for key, value in accum.items():
        result[key] = _scalar(value.sqrt())
    return result


def _append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


class MeanPreservationAgent(Original):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_preservation_branch = os.environ["MEAN_PRESERVATION_BRANCH"]
        if self.mean_preservation_branch not in BRANCHES:
            raise RuntimeError(self.mean_preservation_branch)
        self.mean_preservation_out = Path(
            os.environ["MEAN_PRESERVATION_GROUP_OUT"]
        ).resolve()
        self.mean_preservation_out.mkdir(parents=True, exist_ok=True)

        self.mean_reference = (
            copy.deepcopy(self.actor).eval().requires_grad_(False)
        )
        self.mean_reference_hash = module_hash(self.mean_reference)
        # Compatibility with the existing testing-only milestone diagnostics.
        self.module_reference = self.mean_reference
        self.module_reference_hash = self.mean_reference_hash
        self.module_branch = self.mean_preservation_branch
        self.mean_preservation_configured = False
        self._mean_record = None

    @property
    def allowed_actor_groups(self):
        # This experiment never masks parameter groups; the intervention is
        # exclusively an output-space component-mean penalty.
        return None

    def configure_after_resume(self, payload):
        if payload.get("critic_target_mode") != "random2q":
            raise RuntimeError("Mean-preservation test is random2q-only")
        if payload.get("group") != "multi_q":
            raise RuntimeError("Mean-preservation test is multi_q-only")
        if int(payload.get("actor_updates", -1)) != 0:
            raise RuntimeError("Source checkpoint must have zero Actor updates")
        if payload["actor_optimizer"].get("state"):
            raise RuntimeError("Source Actor Adam state must be empty")
        if module_hash(self.actor) != self.mean_reference_hash:
            raise RuntimeError(
                "critic_ready Actor differs from immutable Stage1 Actor"
            )
        if any(parameter.requires_grad for parameter in self.mean_reference.parameters()):
            raise RuntimeError("Reference Actor must stay frozen")

        self.mean_preservation_configured = True
        _append_jsonl(
            self.mean_preservation_out / "mean_preservation_contract.jsonl",
            {
                "testing_only": True,
                "branch": self.mean_preservation_branch,
                "source_env_steps": int(payload["env_steps"]),
                "source_critic_updates": int(payload["updates"]),
                "source_actor_updates": int(payload["actor_updates"]),
                "source_actor_optimizer_state_entries": len(
                    payload["actor_optimizer"].get("state", {})
                ),
                "reference_actor_hash": self.mean_reference_hash,
                "mean_rms_safe": MEAN_RMS_SAFE,
                "mean_rms_hard": MEAN_RMS_HARD,
                "max_anchor_RL_grad_ratio": MAX_ANCHOR_RL_GRAD_RATIO,
                "max_lambda": MAX_LAMBDA,
                "anchor": (
                    "mean squared difference of all 5x14 GMM component means "
                    "over all 10 recurrent tokens; no logits/KL/std/hidden term"
                ),
                "controller": (
                    "severity=clamp((mean_rms-safe)/(hard-safe),0,1); "
                    "target_ratio=4*severity; "
                    "lambda=target_ratio*||g_RL||/||g_mean||, detached"
                ),
            },
        )

    def actor_update(self, sequences, env_steps, collect_metrics=True):
        if self.mean_preservation_branch == "FULL_PRODUCTION":
            return super().actor_update(
                sequences, env_steps, collect_metrics=collect_metrics
            )
        return self._preserved_actor_update(
            sequences, env_steps, collect_metrics=collect_metrics
        )

    def _begin_mean_preservation(
        self,
        sequence_distribution,
        batch,
        actor_rl,
        env_steps,
    ):
        if not self.mean_preservation_configured:
            raise RuntimeError("Agent used before configure_after_resume")

        with torch.no_grad():
            reference_distribution = self.mean_reference.forward_train(
                V5.flat_to_obs(batch["observations"]),
                rnn_init_state=None,
                return_state=False,
            )
            reference_means = (
                reference_distribution.component_distribution.base_dist.loc.detach()
            )

        current_means = sequence_distribution.component_distribution.base_dist.loc
        mean_loss = (current_means - reference_means).square().mean()
        mean_rms = mean_loss.clamp_min(0.0).sqrt()

        named = list(self.actor.named_parameters())
        parameters = [parameter for _, parameter in named]
        rl_grads = torch.autograd.grad(
            actor_rl,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        mean_grads = torch.autograd.grad(
            mean_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        rl_norm = _norm(rl_grads)
        mean_norm = _norm(mean_grads)

        severity = (
            (mean_rms.detach() - MEAN_RMS_SAFE)
            / (MEAN_RMS_HARD - MEAN_RMS_SAFE)
        ).clamp(0.0, 1.0)
        target_ratio = severity * MAX_ANCHOR_RL_GRAD_RATIO
        coefficient = (
            target_ratio
            * rl_norm.detach()
            / mean_norm.detach().clamp_min(1e-30)
        ).clamp(0.0, MAX_LAMBDA)

        total_grads = []
        for rl_grad, mean_grad in zip(rl_grads, mean_grads):
            if rl_grad is None and mean_grad is None:
                total_grads.append(None)
            elif mean_grad is None:
                total_grads.append(rl_grad)
            elif rl_grad is None:
                total_grads.append(coefficient * mean_grad)
            else:
                total_grads.append(rl_grad + coefficient * mean_grad)
        total_norm = _norm(total_grads)

        self._mean_record = {
            "testing_only": True,
            "branch": self.mean_preservation_branch,
            "env_steps": int(env_steps),
            "actor_update": int(self.actor_updates) + 1,
            "actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
            "mean_rms_before": _scalar(mean_rms),
            "mean_loss_before": _scalar(mean_loss),
            "controller_severity": _scalar(severity),
            "target_anchor_RL_grad_ratio": _scalar(target_ratio),
            "lambda": _scalar(coefficient),
            "RL_loss_before": _scalar(actor_rl),
            "Q1_before": -_scalar(actor_rl),
            "RL_grad_norm": _scalar(rl_norm),
            "mean_anchor_unit_grad_norm": _scalar(mean_norm),
            "mean_anchor_grad_norm": _scalar(coefficient * mean_norm),
            "actual_anchor_RL_grad_ratio": (
                _scalar(coefficient * mean_norm / rl_norm.clamp_min(1e-30))
            ),
            "total_grad_norm_preclip": _scalar(total_norm),
            "RL_grad_groups": _group_grad_norm(named, rl_grads),
            "mean_anchor_unit_grad_groups": _group_grad_norm(named, mean_grads),
        }
        return mean_loss, coefficient, reference_distribution

    def _finish_mean_preservation(
        self,
        batch,
        actor_rl,
        reference_distribution,
        final_contexts,
    ):
        should_diagnose = (
            int(self.actor_updates) <= 10
            or int(self.actor_updates) % 25 == 0
        )
        row = self._mean_record
        if row is None:
            raise RuntimeError("Missing mean-preservation update record")

        if should_diagnose:
            mode = self.actor.training
            try:
                with torch.no_grad():
                    post_distribution = self.actor.forward_train(
                        V5.flat_to_obs(batch["observations"]),
                        rnn_init_state=None,
                        return_state=False,
                    )
                    current_means = (
                        post_distribution.component_distribution.base_dist.loc
                    )
                    reference_means = (
                        reference_distribution.component_distribution.base_dist.loc
                    )
                    mean_rms_after = (
                        (current_means - reference_means)
                        .square()
                        .mean()
                        .sqrt()
                    )

                    final_distribution = torch.distributions.MixtureSameFamily(
                        torch.distributions.Categorical(
                            logits=post_distribution.mixture_distribution.logits[:, -1]
                        ),
                        torch.distributions.Independent(
                            torch.distributions.Normal(
                                post_distribution.component_distribution.base_dist.loc[:, -1],
                                post_distribution.component_distribution.base_dist.scale[:, -1],
                            ),
                            1,
                        ),
                    )
                    expected_after, _, _, _, _ = V5.component_mean_q(
                        self.critic,
                        final_contexts,
                        final_distribution,
                        self.action_scale,
                        self.action_offset,
                        twin_min=False,
                    )
                    q_after = expected_after.mean()
                row.update(
                    {
                        "mean_rms_after": _scalar(mean_rms_after),
                        "Q1_after_same_critic": _scalar(q_after),
                        "same_critic_optimizer_step_Q1_gain": _scalar(
                            q_after + actor_rl.detach()
                        ),
                        "reference_hash_unchanged": (
                            module_hash(self.mean_reference)
                            == self.mean_reference_hash
                        ),
                    }
                )
            finally:
                self.actor.train(mode)
                self.mean_reference.eval()

        _append_jsonl(
            self.mean_preservation_out / "mean_preservation_updates.jsonl",
            row,
        )
        self._mean_record = None


# Derive the current production Actor method and add only the component-mean
# output penalty. All gates, Q1 objective, backward, clipping, Adam, counters,
# and production diagnostics remain inherited from current source semantics.
source = textwrap.dedent(inspect.getsource(V5.RecurrentGMMTD3.actor_update))
if source.count("actor_rl = -expected.mean()") != 1:
    raise RuntimeError("Production Actor objective source changed")
if source.count("actor_rl.backward()") != 1:
    raise RuntimeError("Production Actor backward source changed")
if source.count("self.actor_updates += 1") != 1:
    raise RuntimeError("Production Actor counter source changed")

source = source.replace(
    "actor_rl = -expected.mean()",
    "actor_rl = -expected.mean()\n"
    "            mean_preservation_loss, mean_preservation_lambda, "
    "mean_preservation_reference = self._begin_mean_preservation("
    "sequence_distribution, b, actor_rl, env_steps)",
    1,
)
source = source.replace(
    "actor_rl.backward()",
    "(actor_rl + mean_preservation_lambda * mean_preservation_loss).backward()",
    1,
)
source = source.replace(
    "self.actor_updates += 1",
    "self.actor_updates += 1\n"
    "        self._finish_mean_preservation("
    "b, actor_rl, mean_preservation_reference, final_contexts)",
    1,
)
source = source.replace(
    '"actor_total_loss": actor_rl,',
    '"actor_total_loss": actor_rl + mean_preservation_lambda * mean_preservation_loss,',
    1,
)

namespace = dict(vars(V5))
exec(
    compile(
        source,
        str(HERE / "derived_mean_preservation_actor_update"),
        "exec",
    ),
    namespace,
)
MeanPreservationAgent._preserved_actor_update = namespace["actor_update"]
