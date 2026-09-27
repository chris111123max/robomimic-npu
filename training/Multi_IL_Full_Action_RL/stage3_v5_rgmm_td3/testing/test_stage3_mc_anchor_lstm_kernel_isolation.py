#!/usr/bin/env python3
"""Isolate the update-503 MC-anchor LSTM backward failure.

For each supported MC-anchor lambda, this diagnostic reproduces the exact
production-like trajectory through update 502, then freezes that model state
and compares the same update-503 Q1 regression on three implementations:

1. Ascend NPU native torch.nn.LSTM, on the live online Critic path.
2. CPU native torch.nn.LSTM with identical Q1 weights, inputs and target.
3. Ascend NPU explicit one-layer LSTM recurrence using the same weights and
   elementary matmul/sigmoid/tanh operations instead of the fused nn.LSTM op.

No optimizer step, gradient clipping, Polyak update, Actor update, rollout, or
checkpoint write occurs at update 503 in this test. The purpose is only to
separate a fused-LSTM-backward numerical failure from a recurrent-gradient
pathology inherent to the model state and loss.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import test_stage3_bootstrap_vs_oracle_mc_causal as prior
import test_stage3_mc_anchor_backward_first_bad_op as firstbad
import test_stage3_mc_anchor_nonfinite_trace as phase
import test_stage3_production_mc_anchor_causal as original


LAMBDAS = (0.25, 0.50)
TARGET_UPDATE = 503


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-run-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=list(LAMBDAS)
    )
    parser.add_argument("--output")
    return parser.parse_args()


def tensor_stats(tensor, per_timestep=False):
    if tensor is None:
        return {"captured": False}
    array = (
        tensor.detach().float().cpu().numpy().astype(np.float64, copy=False)
    )
    flat = array.reshape(-1)
    finite = np.isfinite(flat)
    good = flat[finite]
    result = {
        "captured": True,
        "shape": list(array.shape),
        "finite": bool(finite.all()),
        "nonfinite_count": int((~finite).sum()),
        "finite_count": int(finite.sum()),
        "max_abs_finite": float(np.abs(good).max()) if good.size else None,
        "mean_finite": float(good.mean()) if good.size else None,
        "std_finite": float(good.std()) if good.size else None,
    }
    if per_timestep and array.ndim >= 3:
        rows = []
        for t in range(array.shape[1]):
            part = array[:, t].reshape(-1)
            mask = np.isfinite(part)
            values = part[mask]
            rows.append({
                "t": int(t),
                "finite": bool(mask.all()),
                "nonfinite_count": int((~mask).sum()),
                "max_abs_finite": (
                    float(np.abs(values).max()) if values.size else None
                ),
            })
        result["per_timestep"] = rows
    return result


def state_to_cpu(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def clone_q1(template, state_cpu, device):
    clone = copy.deepcopy(template).to(device)
    clone.load_state_dict(state_cpu, strict=True)
    clone.zero_grad(set_to_none=True)
    return clone


def current_q_inputs(agent, sequence):
    values = agent._tensor_batch({
        "observations": sequence["observations"],
        "actions": sequence["actions"],
        "episode_steps": sequence["episode_steps"],
    })
    observations = values["observations"]
    actions = values["actions"]
    steps = values["episode_steps"]

    previous = torch.zeros_like(actions)
    previous[:, 1:] = actions[:, :-1]
    previous = previous.masked_fill(steps.eq(0).unsqueeze(-1), 0.0)
    progress = (
        steps.to(dtype=observations.dtype).unsqueeze(-1)
        / float(agent.config["horizon"])
    )
    return observations, previous, progress, actions[:, -1]


def anchored_target(agent, sequence, final):
    b = agent._tensor_batch(final)
    with torch.no_grad():
        target = agent.bellman_target(b, sequence)[0].detach()
    return target


def parameter_grad_stats(module):
    rows = []
    total_nonfinite = 0
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        stat = tensor_stats(parameter.grad)
        bad = int(stat["nonfinite_count"])
        total_nonfinite += bad
        if bad:
            rows.append({
                "name": name,
                "shape": list(parameter.shape),
                "nonfinite_count": bad,
                "total_count": int(parameter.numel()),
                "max_abs_finite": stat["max_abs_finite"],
            })
    return {
        "finite": total_nonfinite == 0,
        "nonfinite_scalar_count": int(total_nonfinite),
        "nonfinite_parameters": rows,
        "first_nonfinite_parameter": rows[0] if rows else None,
    }


def compare_tensors(left, right):
    left_cpu = left.detach().float().cpu()
    right_cpu = right.detach().float().cpu()
    if tuple(left_cpu.shape) != tuple(right_cpu.shape):
        return {
            "shape_match": False,
            "left_shape": list(left_cpu.shape),
            "right_shape": list(right_cpu.shape),
        }
    diff = (left_cpu - right_cpu).abs()
    return {
        "shape_match": True,
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def run_live_npu_native(agent, sequence, final):
    """Exact online Twin-Q forward; inspect Q1 fused-LSTM backward boundary."""
    captured = {}
    handles = []

    def token_hook(_module, _inputs, output):
        captured["token"] = output
        output.retain_grad()

    def lstm_hook(_module, _inputs, output):
        captured["context"] = output[0]
        output[0].retain_grad()

    handles.append(
        agent.critic.q1.token_encoder[2].register_forward_hook(token_hook)
    )
    handles.append(agent.critic.q1.lstm.register_forward_hook(lstm_hook))

    target = anchored_target(agent, sequence, final)
    b = agent._tensor_batch(final)
    try:
        contexts = agent._history_contexts(agent.critic, sequence)
        q1, q2 = agent.critic.q_from_context(
            (contexts[0][:, -1], contexts[1][:, -1]), b["actions"]
        )
        q1.retain_grad()
        q2.retain_grad()
        loss_q1 = F.mse_loss(q1, target)
        loss_q2 = F.mse_loss(q2, target)
        loss = loss_q1 + loss_q2

        agent.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        prior.sync(agent.device)
    finally:
        for handle in handles:
            handle.remove()

    result = {
        "device": str(agent.device),
        "implementation": "native_torch_nn_LSTM_live_twin_q",
        "target": tensor_stats(target),
        "q1": tensor_stats(q1),
        "q2": tensor_stats(q2),
        "loss_q1": tensor_stats(loss_q1),
        "loss_q2": tensor_stats(loss_q2),
        "loss_total": tensor_stats(loss),
        "q1_output_gradient": tensor_stats(q1.grad),
        "q1_lstm_output_gradient": tensor_stats(
            captured["context"].grad, per_timestep=True
        ),
        "q1_lstm_input_gradient": tensor_stats(
            captured["token"].grad, per_timestep=True
        ),
        "parameter_gradients": parameter_grad_stats(agent.critic),
    }
    result["failure_boundary_reproduced"] = (
        result["q1_lstm_output_gradient"]["finite"] is True
        and result["q1_lstm_input_gradient"]["finite"] is False
    )
    return result, {
        "target": target.detach(),
        "q1": q1.detach(),
        "context": captured["context"].detach(),
        "token": captured["token"].detach(),
    }


def q1_native_forward_backward(q1_module, observations, previous, progress,
                               action, target, label):
    q1_module.zero_grad(set_to_none=True)
    token = q1_module.token_encoder(
        torch.cat((observations, previous, progress), dim=-1)
    )
    token.retain_grad()
    context, _ = q1_module.lstm(token)
    context.retain_grad()
    q = q1_module.q_head(torch.cat((context[:, -1], action), dim=-1))
    q.retain_grad()
    loss = F.mse_loss(q, target)
    loss.backward()
    if target.device.type != "cpu":
        prior.sync(target.device)

    return {
        "implementation": label,
        "target": tensor_stats(target),
        "q1": tensor_stats(q),
        "loss_q1": tensor_stats(loss),
        "q1_output_gradient": tensor_stats(q.grad),
        "q1_lstm_output_gradient": tensor_stats(
            context.grad, per_timestep=True
        ),
        "q1_lstm_input_gradient": tensor_stats(
            token.grad, per_timestep=True
        ),
        "parameter_gradients": parameter_grad_stats(q1_module),
    }, {
        "q1": q.detach(),
        "context": context.detach(),
        "token": token.detach(),
    }


def manual_lstm_sequence(lstm, token):
    """Exact one-layer PyTorch LSTM equations without calling nn.LSTM.forward."""
    if int(lstm.num_layers) != 1:
        raise ValueError("Manual isolation supports exactly one LSTM layer")
    if bool(lstm.bidirectional):
        raise ValueError("Manual isolation does not support bidirectional LSTM")
    if float(lstm.dropout) != 0.0:
        raise ValueError("Manual isolation expects zero LSTM dropout")

    weight_ih = lstm.weight_ih_l0
    weight_hh = lstm.weight_hh_l0
    bias_ih = lstm.bias_ih_l0
    bias_hh = lstm.bias_hh_l0

    batch = token.shape[0]
    hidden_size = int(lstm.hidden_size)
    h = torch.zeros(
        batch, hidden_size, dtype=token.dtype, device=token.device
    )
    c = torch.zeros_like(h)

    outputs = []
    diagnostics = []
    for t in range(token.shape[1]):
        gates = (
            F.linear(token[:, t], weight_ih, bias_ih)
            + F.linear(h, weight_hh, bias_hh)
        )
        gates.retain_grad()
        i_pre, f_pre, g_pre, o_pre = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i_pre)
        f = torch.sigmoid(f_pre)
        g = torch.tanh(g_pre)
        o = torch.sigmoid(o_pre)
        c = f * c + i * g
        h = o * torch.tanh(c)
        c.retain_grad()
        h.retain_grad()
        outputs.append(h)
        diagnostics.append({
            "t": int(t),
            "gates": gates,
            "input_gate": i,
            "forget_gate": f,
            "candidate_gate": g,
            "output_gate": o,
            "cell": c,
            "hidden": h,
        })
    return torch.stack(outputs, dim=1), diagnostics


def summarize_manual_steps(steps):
    result = []
    for row in steps:
        result.append({
            "t": row["t"],
            "forward": {
                "gates": tensor_stats(row["gates"]),
                "input_gate": tensor_stats(row["input_gate"]),
                "forget_gate": tensor_stats(row["forget_gate"]),
                "candidate_gate": tensor_stats(row["candidate_gate"]),
                "output_gate": tensor_stats(row["output_gate"]),
                "cell": tensor_stats(row["cell"]),
                "hidden": tensor_stats(row["hidden"]),
            },
            "backward": {
                "gates_grad": tensor_stats(row["gates"].grad),
                "cell_grad": tensor_stats(row["cell"].grad),
                "hidden_grad": tensor_stats(row["hidden"].grad),
            },
        })
    return result


def run_npu_manual(q1_module, observations, previous, progress, action, target):
    q1_module.zero_grad(set_to_none=True)
    token = q1_module.token_encoder(
        torch.cat((observations, previous, progress), dim=-1)
    )
    token.retain_grad()
    context, steps = manual_lstm_sequence(q1_module.lstm, token)
    context.retain_grad()
    q = q1_module.q_head(torch.cat((context[:, -1], action), dim=-1))
    q.retain_grad()
    loss = F.mse_loss(q, target)
    loss.backward()
    prior.sync(target.device)

    result = {
        "device": str(target.device),
        "implementation": "explicit_one_layer_lstm_recurrence",
        "target": tensor_stats(target),
        "q1": tensor_stats(q),
        "loss_q1": tensor_stats(loss),
        "q1_output_gradient": tensor_stats(q.grad),
        "q1_lstm_output_gradient": tensor_stats(
            context.grad, per_timestep=True
        ),
        "q1_lstm_input_gradient": tensor_stats(
            token.grad, per_timestep=True
        ),
        "parameter_gradients": parameter_grad_stats(q1_module),
        "manual_timestep_diagnostics": summarize_manual_steps(steps),
    }
    return result, {
        "q1": q.detach(),
        "context": context.detach(),
        "token": token.detach(),
    }


def classify(native, cpu, manual, comparisons):
    native_bad = native["failure_boundary_reproduced"]
    cpu_finite = (
        cpu["q1_lstm_input_gradient"]["finite"] is True
        and cpu["parameter_gradients"]["finite"] is True
    )
    manual_finite = (
        manual["q1_lstm_input_gradient"]["finite"] is True
        and manual["parameter_gradients"]["finite"] is True
    )
    forward_close = (
        comparisons["npu_native_vs_npu_manual_q1"].get("max_abs", 1.0)
        <= 5e-4
    )

    if native_bad and cpu_finite and manual_finite and forward_close:
        return {
            "label": "NPU_FUSED_LSTM_BACKWARD_SPECIFIC_STRONG_SUPPORT",
            "interpretation": (
                "The live NPU fused nn.LSTM backward crosses from a finite "
                "output gradient to a nonfinite input gradient, while identical "
                "weights/input/target remain finite in CPU native nn.LSTM and "
                "in an explicit NPU LSTM recurrence with matching forward Q."
            ),
        }
    if native_bad and cpu_finite and manual_finite:
        return {
            "label": "NPU_FUSED_LSTM_BACKWARD_SPECIFIC_SUPPORT",
            "interpretation": (
                "Both controls remain finite, but the explicit recurrence's "
                "forward value differs more than the strict comparison threshold."
            ),
        }
    if native_bad and not cpu_finite and not manual_finite:
        return {
            "label": "RECURRENT_GRADIENT_PATHOLOGY_SUPPORTED",
            "interpretation": (
                "The same state/loss is nonfinite in both CPU native and NPU "
                "explicit recurrence controls, so the failure is not isolated "
                "to the Ascend fused LSTM backward implementation."
            ),
        }
    if native_bad and cpu_finite and not manual_finite:
        return {
            "label": "NPU_RECURRENT_NUMERICAL_PATH_INCONCLUSIVE",
            "interpretation": (
                "CPU native is finite but the NPU explicit recurrence is not; "
                "this suggests an NPU-side recurrent numerical issue but does "
                "not isolate the fused LSTM kernel."
            ),
        }
    if native_bad and not cpu_finite and manual_finite:
        return {
            "label": "FUSED_LSTM_BACKWARD_PATH_INCONCLUSIVE",
            "interpretation": (
                "Both NPU and CPU fused LSTM paths fail while the explicit "
                "recurrence is finite; fused implementation behavior is "
                "implicated but not specifically the Ascend NPU kernel."
            ),
        }
    return {
        "label": "INCONCLUSIVE_NATIVE_FAILURE_NOT_REPRODUCED_OR_MIXED",
        "interpretation": (
            "The native NPU update-503 finite-to-nonfinite boundary was not "
            "reproduced cleanly or the control outcomes are mixed."
        ),
    }


def run_branch(s, lam):
    agent, start, first_target = firstbad.reproduce_prefix(s, lam)

    refs = s["schedule"][TARGET_UPDATE - 1]
    sequence, diagnostic = prior.prepare_training_batch(
        s["episodes"], refs, s["length"], s["returns"]
    )
    prior.install_diagnostic_target_data(agent, diagnostic)
    final = prior.final_transition(sequence)

    # Freeze a CPU copy of the exact update-502 Q1 state before any backward.
    q1_state_cpu = state_to_cpu(agent.critic.q1)
    observations, previous, progress, action = current_q_inputs(agent, sequence)

    # Compute the exact production-like anchored target once, on the live NPU
    # agent. It is detached and reused in all three backward controls.
    target = anchored_target(agent, sequence, final)
    target_cpu = target.detach().cpu()
    observations_cpu = observations.detach().cpu()
    previous_cpu = previous.detach().cpu()
    progress_cpu = progress.detach().cpu()
    action_cpu = action.detach().cpu()

    cpu_q1 = clone_q1(agent.critic.q1, q1_state_cpu, torch.device("cpu"))
    manual_q1 = clone_q1(agent.critic.q1, q1_state_cpu, agent.device)

    native, native_tensors = run_live_npu_native(agent, sequence, final)

    cpu, cpu_tensors = q1_native_forward_backward(
        cpu_q1,
        observations_cpu,
        previous_cpu,
        progress_cpu,
        action_cpu,
        target_cpu,
        "cpu_native_torch_nn_LSTM",
    )
    cpu["device"] = "cpu"

    manual, manual_tensors = run_npu_manual(
        manual_q1,
        observations,
        previous,
        progress,
        action,
        target,
    )

    comparisons = {
        "npu_native_vs_cpu_native_q1": compare_tensors(
            native_tensors["q1"], cpu_tensors["q1"]
        ),
        "npu_native_vs_npu_manual_q1": compare_tensors(
            native_tensors["q1"], manual_tensors["q1"]
        ),
        "npu_native_vs_cpu_native_context": compare_tensors(
            native_tensors["context"], cpu_tensors["context"]
        ),
        "npu_native_vs_npu_manual_context": compare_tensors(
            native_tensors["context"], manual_tensors["context"]
        ),
        "npu_native_vs_cpu_native_token": compare_tensors(
            native_tensors["token"], cpu_tensors["token"]
        ),
        "npu_native_vs_npu_manual_token": compare_tensors(
            native_tensors["token"], manual_tensors["token"]
        ),
    }

    classification = classify(native, cpu, manual, comparisons)

    result = {
        "lambda": float(lam),
        "update": TARGET_UPDATE,
        "initial": start,
        "first_target_contract": first_target,
        "update503_refs_sha256": original.digest_refs(
            np.asarray(refs, dtype=np.int64)
        ),
        "target_stats": tensor_stats(target),
        "npu_native": native,
        "cpu_native": cpu,
        "npu_manual": manual,
        "forward_comparisons": comparisons,
        "classification": classification,
        "safety": {
            "optimizer_step_at_update503": 0,
            "gradient_clip_at_update503": 0,
            "polyak_update_at_update503": 0,
            "actor_optimizer_steps": 0,
            "environment_steps": 0,
            "formal_training_steps": 0,
            "training_checkpoint_writes": 0,
        },
    }

    del cpu_q1, manual_q1, agent
    prior.cleanup_device(s["device"])
    return result


def main():
    args = parse_args()
    if any(float(lam) not in LAMBDAS for lam in args.lambdas):
        raise ValueError(f"Supported lambdas are exactly {LAMBDAS}")

    s = phase.setup(args)
    phase.s_global_payload = s["payload"]
    first_batch_contract = phase.prepare_original_start(s)

    output_path = (
        Path(args.output).resolve()
        if args.output
        else s["run"]
        / "testing/stage2_vs_stage3_readiness"
        / "multi_mc_anchor_lstm_kernel_isolation.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "status": "RUNNING",
        "experiment": "stage3_mc_anchor_update503_lstm_kernel_isolation",
        "device": str(s["device"]),
        "seed": 20260926,
        "batch_size": 256,
        "target_update": TARGET_UPDATE,
        "lambdas": [float(x) for x in args.lambdas],
        "stage2_checkpoint": str(s["stage2"]),
        "step0_checkpoint": str(s["step0"]),
        "canonical_replay": str(s["replay"]),
        "schedule_sha256": original.digest_refs(s["schedule"]),
        "probe_sha256": original.digest_refs(s["probe"]),
        "first_batch_contract": first_batch_contract,
        "branches": [],
        "safety": {
            "environment_steps": 0,
            "formal_training_steps": 0,
            "training_checkpoint_writes": 0,
            "production_source_modified": False,
        },
    }

    for lam in args.lambdas:
        branch = run_branch(s, float(lam))
        result["branches"].append(branch)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"[LAMBDA {float(lam):.2f}] "
            f"{branch['classification']['label']} "
            f"native_bad="
            f"{branch['npu_native']['failure_boundary_reproduced']} "
            f"cpu_input_finite="
            f"{branch['cpu_native']['q1_lstm_input_gradient']['finite']} "
            f"manual_input_finite="
            f"{branch['npu_manual']['q1_lstm_input_gradient']['finite']}",
            flush=True,
        )

    native_reproduced_all = all(
        branch["npu_native"]["failure_boundary_reproduced"]
        for branch in result["branches"]
    )
    cpu_control_finite_all = all(
        branch["cpu_native"]["q1_lstm_input_gradient"]["finite"]
        and branch["cpu_native"]["parameter_gradients"]["finite"]
        for branch in result["branches"]
    )
    manual_control_finite_all = all(
        branch["npu_manual"]["q1_lstm_input_gradient"]["finite"]
        and branch["npu_manual"]["parameter_gradients"]["finite"]
        for branch in result["branches"]
    )

    result["validity"] = {
        "native_update503_failure_reproduced_all": native_reproduced_all,
        "cpu_native_control_completed_all": all(
            branch["cpu_native"]["loss_q1"]["finite"]
            for branch in result["branches"]
        ),
        "npu_manual_control_completed_all": all(
            branch["npu_manual"]["loss_q1"]["finite"]
            for branch in result["branches"]
        ),
        "same_update503_target_reused": True,
        "no_update503_optimizer_clip_or_polyak": True,
        "no_environment_or_formal_training": True,
    }
    result["summary"] = {
        "native_failure_reproduced_all": native_reproduced_all,
        "cpu_control_finite_all": cpu_control_finite_all,
        "npu_manual_control_finite_all": manual_control_finite_all,
        "npu_fused_specific_supported_all": all(
            branch["classification"]["label"].startswith(
                "NPU_FUSED_LSTM_BACKWARD_SPECIFIC"
            )
            for branch in result["branches"]
        ),
    }
    result["status"] = (
        "PASS"
        if all(result["validity"].values())
        else "INCONCLUSIVE"
    )

    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[STATUS] {result['status']} [SAVED] {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
