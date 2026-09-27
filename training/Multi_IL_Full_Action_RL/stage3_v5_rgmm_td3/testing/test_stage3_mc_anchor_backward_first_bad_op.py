#!/usr/bin/env python3
"""Localize the first bad backward boundary at MC-anchor update 503.

Updates 1..502 follow the exact production-like path used by the previous
nonfinite diagnostic. Only at update 503 do we attach lightweight tensor
hooks to online Critic intermediates. Hook callbacks do no reductions, copies,
device-to-host transfers, or synchronization; they only retain gradient tensor
references for inspection after backward has finished.

The main discriminator is whether dL/d(LSTM output) remains finite while
dL/d(LSTM input) becomes nonfinite. That localizes the first observed
finite-to-nonfinite boundary to the LSTM backward operator instead of the
downstream Q head or token encoder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import test_stage3_bootstrap_vs_oracle_mc_causal as prior
import test_stage3_mc_anchor_nonfinite_trace as phase
import test_stage3_production_mc_anchor_causal as original


DEFAULT_LAMBDAS = (0.25, 0.50)
TARGET_UPDATE = 503
MILESTONES = (100, 250, 500)

# Logical backward order from loss/Q output toward raw history tokens.
CHAIN_SUFFIXES = (
    "q_head.3",
    "q_head.2",
    "q_head.1",
    "q_head.0",
    "lstm",
    "token_encoder.2",
    "token_encoder.1",
    "token_encoder.0",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-run-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS)
    )
    parser.add_argument("--output")
    return parser.parse_args()


def module_map(critic):
    result = {}
    for twin_name in ("q1", "q2"):
        twin = getattr(critic, twin_name)
        result.update({
            f"{twin_name}.token_encoder.0": twin.token_encoder[0],
            f"{twin_name}.token_encoder.1": twin.token_encoder[1],
            f"{twin_name}.token_encoder.2": twin.token_encoder[2],
            f"{twin_name}.lstm": twin.lstm,
            f"{twin_name}.q_head.0": twin.q_head[0],
            f"{twin_name}.q_head.1": twin.q_head[1],
            f"{twin_name}.q_head.2": twin.q_head[2],
            f"{twin_name}.q_head.3": twin.q_head[3],
        })
    return result


def select_tensor(module, output):
    if isinstance(module, torch.nn.LSTM):
        # nn.LSTM returns (sequence_output, (h_n, c_n)). Only sequence_output
        # feeds the final Q prediction.
        return output[0]
    if torch.is_tensor(output):
        return output
    return None


class TensorGradientTap:
    """Register minimal output-gradient taps for one online-Critic forward."""

    def __init__(self, critic):
        self.critic = critic
        self.handles = []
        self.forward_tensors = {}
        self.backward_tensors = {}
        self.backward_event_order = []

    def _forward_hook(self, name):
        def hook(module, _inputs, output):
            tensor = select_tensor(module, output)
            if tensor is None:
                return
            # Detach creates no numerical transform and no device-host sync.
            self.forward_tensors[name] = tensor.detach()
            if not tensor.requires_grad:
                return

            def grad_hook(grad):
                # Deliberately do no isfinite/norm/clone/cpu work here.
                self.backward_event_order.append(name)
                self.backward_tensors[name] = grad.detach()
                return grad

            tensor.register_hook(grad_hook)

        return hook

    def __enter__(self):
        for name, module in module_map(self.critic).items():
            self.handles.append(
                module.register_forward_hook(self._forward_hook(name))
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def tensor_detail(tensor):
    """Host-side statistics, called only after update-503 backward returns."""
    if tensor is None:
        return {"captured": False}
    array = (
        tensor.detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
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
    if array.ndim == 3:
        per_timestep = []
        for t in range(array.shape[1]):
            part = array[:, t].reshape(-1)
            mask = np.isfinite(part)
            values = part[mask]
            per_timestep.append({
                "t": int(t),
                "finite": bool(mask.all()),
                "nonfinite_count": int((~mask).sum()),
                "max_abs_finite": (
                    float(np.abs(values).max()) if values.size else None
                ),
            })
        result["per_timestep"] = per_timestep
    return result


def parameter_grad_summary(critic):
    bad_rows = []
    total_bad = 0
    for name, parameter in critic.named_parameters():
        if parameter.grad is None:
            continue
        stat = tensor_detail(parameter.grad)
        bad = int(stat.get("nonfinite_count", 0))
        total_bad += bad
        if bad:
            bad_rows.append({
                "name": name,
                "shape": list(parameter.shape),
                "nonfinite_count": bad,
                "total_count": int(parameter.numel()),
                "max_abs_finite": stat.get("max_abs_finite"),
            })
    return {
        "finite": total_bad == 0,
        "nonfinite_scalar_count": total_bad,
        "nonfinite_parameters": bad_rows,
        "first_nonfinite_parameter": bad_rows[0] if bad_rows else None,
    }


def parameter_value_summary(critic):
    total_bad = 0
    first = None
    for name, parameter in critic.named_parameters():
        values = parameter.detach().float().cpu().numpy().reshape(-1)
        bad = int((~np.isfinite(values)).sum())
        if bad and first is None:
            first = {
                "name": name,
                "shape": list(parameter.shape),
                "nonfinite_count": bad,
            }
        total_bad += bad
    return {
        "finite": total_bad == 0,
        "nonfinite_scalar_count": total_bad,
        "first_nonfinite_parameter": first,
    }


def classify_boundary(downstream, upstream):
    if downstream.endswith(".lstm") and upstream.endswith(".token_encoder.2"):
        return "LSTM_BACKWARD_OUTPUT_GRAD_FINITE_TO_INPUT_GRAD_NONFINITE"
    if downstream.endswith(".q_head.0") and upstream.endswith(".lstm"):
        return "Q_HEAD_TO_LSTM_OUTPUT"
    if (
        downstream.endswith(".token_encoder.2")
        and upstream.endswith(".token_encoder.1")
    ):
        return "TOKEN_RELU_BACKWARD"
    if (
        downstream.endswith(".token_encoder.1")
        and upstream.endswith(".token_encoder.0")
    ):
        return "TOKEN_LAYERNORM_BACKWARD"
    return "FINITE_TO_NONFINITE_BETWEEN_ADJACENT_TAPS"


def chain_analysis(gradient_stats, twin):
    chain = [f"{twin}.{suffix}" for suffix in CHAIN_SUFFIXES]
    rows = []
    previous = None
    boundary = None
    for name in chain:
        stat = gradient_stats.get(name, {"captured": False})
        row = {
            "name": name,
            "captured": bool(stat.get("captured", False)),
            "finite": stat.get("finite"),
            "nonfinite_count": stat.get("nonfinite_count"),
            "max_abs_finite": stat.get("max_abs_finite"),
        }
        rows.append(row)
        if (
            boundary is None
            and previous is not None
            and previous["captured"]
            and previous["finite"] is True
            and row["captured"]
            and row["finite"] is False
        ):
            boundary = {
                "downstream_finite_tensor": previous["name"],
                "upstream_nonfinite_tensor": row["name"],
                "classification": classify_boundary(
                    previous["name"], row["name"]
                ),
            }
        previous = row
    first_bad = next(
        (
            row
            for row in rows
            if row["captured"] and row["finite"] is False
        ),
        None,
    )
    return {
        "chain": rows,
        "first_bad_in_logical_chain": first_bad,
        "finite_to_nonfinite_boundary": boundary,
    }


def affected_batch_rows(grad_tensor, refs, episode_steps, limit=32):
    """Map bad recurrent-input gradients back to canonical replay rows."""
    if grad_tensor is None:
        return {"bad_row_count": 0, "rows": []}
    array = grad_tensor.detach().float().cpu().numpy()
    if array.ndim < 2:
        return {"bad_row_count": 0, "rows": []}
    axes = tuple(range(1, array.ndim))
    bad_rows = np.flatnonzero((~np.isfinite(array)).any(axis=axes))
    refs = np.asarray(refs, dtype=np.int64)
    steps = np.asarray(episode_steps, dtype=np.int64)
    rows = []
    for row in bad_rows[:limit]:
        rows.append({
            "batch_row": int(row),
            "episode_index": int(refs[row, 0]),
            "window_start": int(refs[row, 1]),
            "episode_steps": steps[row].tolist(),
            "final_episode_step": int(steps[row, -1]),
        })
    return {
        "bad_row_count": int(len(bad_rows)),
        "rows": rows,
        "rows_truncated": bool(len(bad_rows) > limit),
    }


def validate_lambda(lam):
    if lam not in DEFAULT_LAMBDAS:
        raise ValueError(
            f"This localization test is fixed to {DEFAULT_LAMBDAS}; got {lam}"
        )


def reproduce_prefix(s, lam):
    """Match the previous reproducing precursor path through update 502."""
    agent = original.make_agent(
        s["stage2"], s["payload"], s["device"], lam
    )
    start = phase.initial_fingerprint(agent)
    first_target = original.initial_lambda_contract(
        agent,
        s["episodes"],
        s["schedule"][0],
        s["length"],
        s["returns"],
        lam,
    )

    # The earlier precursor run performed the same update-0 probe before
    # entering the optimization loop; preserve it because the failure proved
    # sensitive to broader instrumentation.
    original.record(
        agent,
        s["episodes"],
        s["probe"],
        s["probe_returns"],
        s["labels"],
        s["ids"],
        s["geometry_refs"],
        s["returns"],
        s["device"],
        s["config"],
        s["length"],
        1024,
        lam,
        0,
    )

    for index, refs in enumerate(s["schedule"][: TARGET_UPDATE - 1]):
        update = index + 1
        sequence, diagnostic = prior.prepare_training_batch(
            s["episodes"], refs, s["length"], s["returns"]
        )
        prior.install_diagnostic_target_data(agent, diagnostic)
        final = prior.final_transition(sequence)
        agent.critic_update(final, sequence, collect_metrics=False)
        agent.polyak_update()

        if update in MILESTONES:
            original.record(
                agent,
                s["episodes"],
                s["probe"],
                s["probe_returns"],
                s["labels"],
                s["ids"],
                s["geometry_refs"],
                s["returns"],
                s["device"],
                s["config"],
                s["length"],
                1024,
                lam,
                update,
            )

    return agent, start, first_target


def run_update_503(agent, s, lam):
    refs = s["schedule"][TARGET_UPDATE - 1]
    sequence, diagnostic = prior.prepare_training_batch(
        s["episodes"], refs, s["length"], s["returns"]
    )
    prior.install_diagnostic_target_data(agent, diagnostic)
    final = prior.final_transition(sequence)

    exception = None
    with TensorGradientTap(agent.critic) as tap:
        try:
            # Exact production-like update. The hooks only retain tensor
            # references; all reductions happen after this call returns.
            agent.critic_update(final, sequence, collect_metrics=False)
        except Exception as exc:
            exception = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

    prior.sync(s["device"])

    forward_stats = {
        name: tensor_detail(tensor)
        for name, tensor in tap.forward_tensors.items()
    }
    gradient_stats = {
        name: tensor_detail(tensor)
        for name, tensor in tap.backward_tensors.items()
    }

    event_rows = []
    for index, name in enumerate(tap.backward_event_order):
        stat = gradient_stats.get(name, {})
        event_rows.append({
            "event_index": int(index),
            "name": name,
            "finite": stat.get("finite"),
            "nonfinite_count": stat.get("nonfinite_count"),
            "max_abs_finite": stat.get("max_abs_finite"),
        })
    first_bad_event = next(
        (row for row in event_rows if row["finite"] is False), None
    )

    q1_chain = chain_analysis(gradient_stats, "q1")
    q2_chain = chain_analysis(gradient_stats, "q2")

    q1_lstm_input_grad = tap.backward_tensors.get("q1.token_encoder.2")
    q2_lstm_input_grad = tap.backward_tensors.get("q2.token_encoder.2")

    return {
        "lambda": float(lam),
        "update": TARGET_UPDATE,
        "exception": exception,
        "refs_sha256": original.digest_refs(
            np.asarray(refs, dtype=np.int64)
        ),
        "forward_tensors": forward_stats,
        "backward_tensors": gradient_stats,
        "backward_event_order": event_rows,
        "first_nonfinite_backward_event": first_bad_event,
        "q1_logical_backward_chain": q1_chain,
        "q2_logical_backward_chain": q2_chain,
        "q1_lstm_input_bad_rows": affected_batch_rows(
            q1_lstm_input_grad, refs, sequence["episode_steps"]
        ),
        "q2_lstm_input_bad_rows": affected_batch_rows(
            q2_lstm_input_grad, refs, sequence["episode_steps"]
        ),
        "post_critic_update": {
            "online_parameters": parameter_value_summary(agent.critic),
            "optimizer": phase.optimizer_summary(agent),
            "parameter_gradients_after_clip": parameter_grad_summary(
                agent.critic
            ),
        },
        "observed_q1_lstm_backward_boundary": (
            q1_chain["finite_to_nonfinite_boundary"] is not None
            and q1_chain["finite_to_nonfinite_boundary"]["classification"]
            == "LSTM_BACKWARD_OUTPUT_GRAD_FINITE_TO_INPUT_GRAD_NONFINITE"
        ),
        "environment_steps": 0,
        "formal_training_steps": 0,
        "training_checkpoint_writes": 0,
        "actor_optimizer_steps": 0,
    }


def main():
    args = parse_args()
    for lam in args.lambdas:
        validate_lambda(float(lam))

    s = phase.setup(args)
    phase.s_global_payload = s["payload"]

    # Match the previous diagnostic's pre-run contract work exactly once.
    first_batch_contract = phase.prepare_original_start(s)

    output_path = (
        Path(args.output).resolve()
        if args.output
        else s["run"]
        / "testing/stage2_vs_stage3_readiness"
        / "multi_mc_anchor_backward_first_bad_op.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "status": "RUNNING",
        "experiment": (
            "stage3_mc_anchor_update503_first_bad_backward_boundary"
        ),
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
        lam = float(lam)
        agent, start, first_target = reproduce_prefix(s, lam)
        branch = run_update_503(agent, s, lam)
        branch["initial"] = start
        branch["first_target"] = first_target
        result["branches"].append(branch)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"[LAMBDA {lam:.2f}] "
            f"first_bad={branch['first_nonfinite_backward_event']} "
            f"q1_boundary="
            f"{branch['q1_logical_backward_chain']['finite_to_nonfinite_boundary']}",
            flush=True,
        )
        del agent
        prior.cleanup_device(s["device"])

    captured_all = all(
        branch["q1_logical_backward_chain"][
            "first_bad_in_logical_chain"
        ]
        is not None
        for branch in result["branches"]
    )
    nonfinite_backward_reproduced_all = all(
        branch["first_nonfinite_backward_event"] is not None
        for branch in result["branches"]
    )

    result["validity"] = {
        "first_batch_contract_present": bool(first_batch_contract),
        "fixed_target_update_503": TARGET_UPDATE == 503,
        "only_supported_lambdas": all(
            float(x) in DEFAULT_LAMBDAS for x in args.lambdas
        ),
        "bad_q1_backward_captured_all": captured_all,
        "nonfinite_backward_reproduced_all": (
            nonfinite_backward_reproduced_all
        ),
        "no_environment_or_formal_training": True,
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
