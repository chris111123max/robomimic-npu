#!/usr/bin/env python3
"""Three-stage isolation of the Stage3 MC-anchor update-503 LSTM backward fault.

The previous single-process kernel-isolation experiment was inconclusive because
extra CPU/NPU clones and allocations before the native backward changed the
execution trajectory enough that the update-503 failure disappeared.

This script separates the experiment into independent processes:

REFERENCE
    Reproduce the known failing trajectory through update 502, then execute only
    the lightweight update-503 backward localization already proven to reproduce
    the finite-LSTM-output-grad -> nonfinite-LSTM-input-grad boundary.

SNAPSHOT
    Reproduce the same trajectory through update 502, prepare the exact
    update-503 Q1 regression inputs and anchored target, save a testing-only
    snapshot, then exit WITHOUT running update-503 backward.

CONTROLS
    In a fresh process, load the testing-only snapshot and evaluate the same Q1
    regression with:
      (a) CPU native torch.nn.LSTM
      (b) NPU explicit one-layer LSTM recurrence using the saved Q1 weights
    No production trajectory is replayed in this phase.

No production source is modified. No environment rollout, formal training, or
training checkpoint is written.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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
SNAPSHOT_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-run-dir", required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--phase",
        required=True,
        choices=("reference", "snapshot", "controls"),
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=list(LAMBDAS),
    )
    parser.add_argument("--snapshot-dir")
    parser.add_argument("--output")
    return parser.parse_args()


def validate_lambdas(values):
    values = tuple(float(x) for x in values)
    if any(value not in LAMBDAS for value in values):
        raise ValueError(f"Supported lambdas are exactly {LAMBDAS}")
    return values


def sha256_bytes(data):
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_sha256(state_dict):
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


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


def parameter_grad_stats(module):
    bad_rows = []
    total_nonfinite = 0
    max_abs_finite = 0.0
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        stat = tensor_stats(parameter.grad)
        total_nonfinite += int(stat["nonfinite_count"])
        if stat["max_abs_finite"] is not None:
            max_abs_finite = max(
                max_abs_finite, float(stat["max_abs_finite"])
            )
        if stat["nonfinite_count"]:
            bad_rows.append({
                "name": name,
                "shape": list(parameter.shape),
                "nonfinite_count": int(stat["nonfinite_count"]),
                "total_count": int(parameter.numel()),
                "max_abs_finite": stat["max_abs_finite"],
            })
    return {
        "finite": total_nonfinite == 0,
        "nonfinite_scalar_count": int(total_nonfinite),
        "max_abs_finite": float(max_abs_finite),
        "nonfinite_parameters": bad_rows,
        "first_nonfinite_parameter": bad_rows[0] if bad_rows else None,
    }


def compare_tensors(left, right):
    a = left.detach().float().cpu()
    b = right.detach().float().cpu()
    if tuple(a.shape) != tuple(b.shape):
        return {
            "shape_match": False,
            "left_shape": list(a.shape),
            "right_shape": list(b.shape),
        }
    diff = (a - b).abs()
    return {
        "shape_match": True,
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
    }


def current_q_inputs(agent, sequence):
    values = agent._tensor_batch({
        "observations": sequence["observations"],
        "actions": sequence["actions"],
        "episode_steps": sequence["episode_steps"],
    })
    observations = values["observations"]
    actions = values["actions"]
    steps = values["episode_steps"]

    previous_actions = torch.zeros_like(actions)
    previous_actions[:, 1:] = actions[:, :-1]
    previous_actions = previous_actions.masked_fill(
        steps.eq(0).unsqueeze(-1), 0.0
    )
    progress = (
        steps.to(dtype=observations.dtype).unsqueeze(-1)
        / float(agent.config["horizon"])
    )
    return observations, previous_actions, progress, actions[:, -1]


def exact_anchored_target(agent, sequence, final):
    b = agent._tensor_batch(final)
    with torch.no_grad():
        target = agent.bellman_target(b, sequence)[0].detach()
    return target


def snapshot_paths(base_dir, lam):
    tag = str(lam).replace(".", "p")
    return (
        base_dir / f"mc_anchor_lstm_update502_lambda_{tag}.pt",
        base_dir / f"mc_anchor_lstm_update502_lambda_{tag}.json",
    )


def default_snapshot_dir(run):
    return run / "testing/stage2_vs_stage3_readiness/lstm_kernel_snapshots"


def default_output(run, phase_name):
    return (
        run
        / "testing/stage2_vs_stage3_readiness"
        / f"multi_mc_anchor_lstm_three_stage_{phase_name}.json"
    )


def run_reference(s, lambdas):
    """Run only the known lightweight reproducer at update 503."""
    branches = []
    for lam in lambdas:
        agent, start, first_target = firstbad.reproduce_prefix(s, lam)
        branch = firstbad.run_update_503(agent, s, lam)
        branch["initial"] = start
        branch["first_target_contract"] = first_target
        branches.append(branch)
        print(
            f"[REFERENCE lambda={lam:.2f}] "
            f"first_bad={branch['first_nonfinite_backward_event']} "
            f"boundary="
            f"{branch['q1_logical_backward_chain']['finite_to_nonfinite_boundary']}",
            flush=True,
        )
        del agent
        prior.cleanup_device(s["device"])

    reproduced = all(
        branch["first_nonfinite_backward_event"] is not None
        and branch["q1_logical_backward_chain"][
            "finite_to_nonfinite_boundary"
        ] is not None
        and branch["q1_logical_backward_chain"][
            "finite_to_nonfinite_boundary"
        ]["classification"]
        == "LSTM_BACKWARD_OUTPUT_GRAD_FINITE_TO_INPUT_GRAD_NONFINITE"
        for branch in branches
    )
    q2_finite = all(
        branch["q2_logical_backward_chain"][
            "first_bad_in_logical_chain"
        ]
        is None
        for branch in branches
    )
    return {
        "phase": "reference",
        "branches": branches,
        "summary": {
            "native_failure_reproduced_all": reproduced,
            "q2_control_finite_all": q2_finite,
        },
        "status": "PASS" if reproduced and q2_finite else "INCONCLUSIVE",
    }


def run_snapshot(s, lambdas, snapshot_dir):
    """Save exact update-502 Q1 state + update-503 regression data; no backward."""
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    branches = []

    for lam in lambdas:
        agent, start, first_target = firstbad.reproduce_prefix(s, lam)
        refs = s["schedule"][TARGET_UPDATE - 1]
        sequence, diagnostic = prior.prepare_training_batch(
            s["episodes"], refs, s["length"], s["returns"]
        )
        prior.install_diagnostic_target_data(agent, diagnostic)
        final = prior.final_transition(sequence)

        observations, previous, progress, action = current_q_inputs(
            agent, sequence
        )
        target = exact_anchored_target(agent, sequence, final)

        # Snapshot only after the full 1..502 trajectory is complete. This
        # phase never performs update-503 backward.
        q1_state = {
            key: value.detach().cpu().clone()
            for key, value in agent.critic.q1.state_dict().items()
        }
        payload = {
            "snapshot_version": SNAPSHOT_VERSION,
            "lambda": float(lam),
            "target_update": TARGET_UPDATE,
            "q1_state_dict": q1_state,
            "observations": observations.detach().cpu(),
            "previous_actions": previous.detach().cpu(),
            "progress": progress.detach().cpu(),
            "current_action": action.detach().cpu(),
            "anchored_target": target.detach().cpu(),
            "episode_steps": torch.as_tensor(
                sequence["episode_steps"], dtype=torch.long
            ),
            "refs": torch.as_tensor(refs, dtype=torch.long),
            "meta": {
                "seed": 20260926,
                "batch_size": 256,
                "schedule_sha256": original.digest_refs(s["schedule"]),
                "probe_sha256": original.digest_refs(s["probe"]),
                "initial_online_hash": start["online_hash"],
                "first_target_reconstruction_max_abs": float(
                    first_target["reconstruction_max_abs"]
                ),
            },
        }

        pt_path, json_path = snapshot_paths(snapshot_dir, lam)
        torch.save(payload, pt_path)

        manifest = {
            "snapshot_version": SNAPSHOT_VERSION,
            "lambda": float(lam),
            "target_update": TARGET_UPDATE,
            "snapshot_path": str(pt_path),
            "snapshot_size_bytes": int(pt_path.stat().st_size),
            "q1_state_sha256": state_sha256(q1_state),
            "observations_sha256": tensor_sha256(payload["observations"]),
            "previous_actions_sha256": tensor_sha256(
                payload["previous_actions"]
            ),
            "progress_sha256": tensor_sha256(payload["progress"]),
            "current_action_sha256": tensor_sha256(
                payload["current_action"]
            ),
            "anchored_target_sha256": tensor_sha256(
                payload["anchored_target"]
            ),
            "target_stats": tensor_stats(payload["anchored_target"]),
            "observations_stats": tensor_stats(payload["observations"]),
            "action_stats": tensor_stats(payload["current_action"]),
            "meta": payload["meta"],
            "update503_backward_executed": False,
            "optimizer_step_at_update503": 0,
            "gradient_clip_at_update503": 0,
            "polyak_update_at_update503": 0,
        }
        json_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

        branches.append(manifest)
        print(
            f"[SNAPSHOT lambda={lam:.2f}] saved={pt_path}",
            flush=True,
        )

        del agent
        prior.cleanup_device(s["device"])

    return {
        "phase": "snapshot",
        "snapshot_dir": str(snapshot_dir),
        "branches": branches,
        "status": "PASS",
    }


def clone_q1_from_snapshot(snapshot, template_q1, device):
    q1 = copy.deepcopy(template_q1).to(device)
    q1.load_state_dict(snapshot["q1_state_dict"], strict=True)
    q1.zero_grad(set_to_none=True)
    return q1


def run_native_q1(q1, observations, previous, progress, action, target, label):
    q1.zero_grad(set_to_none=True)

    token = q1.token_encoder(
        torch.cat((observations, previous, progress), dim=-1)
    )
    token.retain_grad()

    context, _ = q1.lstm(token)
    context.retain_grad()

    output = q1.q_head(
        torch.cat((context[:, -1], action), dim=-1)
    )
    output.retain_grad()

    loss = F.mse_loss(output, target)
    loss.backward()

    if target.device.type != "cpu":
        prior.sync(target.device)

    return {
        "implementation": label,
        "device": str(target.device),
        "q1": tensor_stats(output),
        "loss_q1": tensor_stats(loss),
        "q1_output_gradient": tensor_stats(output.grad),
        "lstm_output_gradient": tensor_stats(
            context.grad, per_timestep=True
        ),
        "lstm_input_gradient": tensor_stats(
            token.grad, per_timestep=True
        ),
        "parameter_gradients": parameter_grad_stats(q1),
    }, {
        "q1": output.detach(),
        "context": context.detach(),
        "token": token.detach(),
    }


def manual_lstm_sequence(lstm, token):
    if int(lstm.num_layers) != 1:
        raise ValueError("Manual recurrence requires one LSTM layer")
    if bool(lstm.bidirectional):
        raise ValueError("Manual recurrence requires unidirectional LSTM")
    if float(lstm.dropout) != 0.0:
        raise ValueError("Manual recurrence requires zero dropout")

    h = torch.zeros(
        token.shape[0],
        int(lstm.hidden_size),
        dtype=token.dtype,
        device=token.device,
    )
    c = torch.zeros_like(h)

    outputs = []
    steps = []
    for t in range(token.shape[1]):
        gates = (
            F.linear(
                token[:, t],
                lstm.weight_ih_l0,
                lstm.bias_ih_l0,
            )
            + F.linear(
                h,
                lstm.weight_hh_l0,
                lstm.bias_hh_l0,
            )
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
        steps.append({
            "t": int(t),
            "gates": gates,
            "input_gate": i,
            "forget_gate": f,
            "candidate_gate": g,
            "output_gate": o,
            "cell": c,
            "hidden": h,
        })

    return torch.stack(outputs, dim=1), steps


def summarize_manual_steps(steps):
    rows = []
    for step in steps:
        rows.append({
            "t": step["t"],
            "forward": {
                "gates": tensor_stats(step["gates"]),
                "input_gate": tensor_stats(step["input_gate"]),
                "forget_gate": tensor_stats(step["forget_gate"]),
                "candidate_gate": tensor_stats(
                    step["candidate_gate"]
                ),
                "output_gate": tensor_stats(step["output_gate"]),
                "cell": tensor_stats(step["cell"]),
                "hidden": tensor_stats(step["hidden"]),
            },
            "backward": {
                "gates_grad": tensor_stats(step["gates"].grad),
                "cell_grad": tensor_stats(step["cell"].grad),
                "hidden_grad": tensor_stats(step["hidden"].grad),
            },
        })
    return rows


def run_manual_q1(
    q1, observations, previous, progress, action, target, label
):
    q1.zero_grad(set_to_none=True)

    token = q1.token_encoder(
        torch.cat((observations, previous, progress), dim=-1)
    )
    token.retain_grad()

    context, steps = manual_lstm_sequence(q1.lstm, token)
    context.retain_grad()

    output = q1.q_head(
        torch.cat((context[:, -1], action), dim=-1)
    )
    output.retain_grad()

    loss = F.mse_loss(output, target)
    loss.backward()
    prior.sync(target.device)

    return {
        "implementation": label,
        "device": str(target.device),
        "q1": tensor_stats(output),
        "loss_q1": tensor_stats(loss),
        "q1_output_gradient": tensor_stats(output.grad),
        "lstm_output_gradient": tensor_stats(
            context.grad, per_timestep=True
        ),
        "lstm_input_gradient": tensor_stats(
            token.grad, per_timestep=True
        ),
        "parameter_gradients": parameter_grad_stats(q1),
        "manual_timestep_diagnostics": summarize_manual_steps(steps),
    }, {
        "q1": output.detach(),
        "context": context.detach(),
        "token": token.detach(),
    }


def run_controls(s, lambdas, snapshot_dir):
    """Fresh-process controls loaded from the testing-only snapshots."""
    branches = []

    # Build one template Q1 from the same Stage2/Stage3 architecture. No
    # production trajectory is replayed here; snapshot weights overwrite it.
    template_agent = original.make_agent(
        s["stage2"], s["payload"], s["device"], 0.0
    )

    for lam in lambdas:
        pt_path, json_path = snapshot_paths(snapshot_dir, lam)
        if not pt_path.exists():
            raise FileNotFoundError(
                f"Missing snapshot for lambda={lam}: {pt_path}"
            )
        if not json_path.exists():
            raise FileNotFoundError(
                f"Missing snapshot manifest for lambda={lam}: {json_path}"
            )

        snapshot = torch.load(pt_path, map_location="cpu")
        manifest = json.loads(json_path.read_text())

        if int(snapshot["snapshot_version"]) != SNAPSHOT_VERSION:
            raise RuntimeError("Snapshot version mismatch")
        if float(snapshot["lambda"]) != float(lam):
            raise RuntimeError("Snapshot lambda mismatch")
        if int(snapshot["target_update"]) != TARGET_UPDATE:
            raise RuntimeError("Snapshot target update mismatch")
        if state_sha256(snapshot["q1_state_dict"]) != manifest[
            "q1_state_sha256"
        ]:
            raise RuntimeError("Snapshot Q1 state hash mismatch")

        cpu_q1 = clone_q1_from_snapshot(
            snapshot, template_agent.critic.q1, torch.device("cpu")
        )
        npu_manual_q1 = clone_q1_from_snapshot(
            snapshot, template_agent.critic.q1, s["device"]
        )

        obs_cpu = snapshot["observations"].float()
        prev_cpu = snapshot["previous_actions"].float()
        progress_cpu = snapshot["progress"].float()
        action_cpu = snapshot["current_action"].float()
        target_cpu = snapshot["anchored_target"].float()

        cpu_result, cpu_tensors = run_native_q1(
            cpu_q1,
            obs_cpu,
            prev_cpu,
            progress_cpu,
            action_cpu,
            target_cpu,
            "cpu_native_torch_nn_LSTM",
        )

        obs_npu = obs_cpu.to(s["device"])
        prev_npu = prev_cpu.to(s["device"])
        progress_npu = progress_cpu.to(s["device"])
        action_npu = action_cpu.to(s["device"])
        target_npu = target_cpu.to(s["device"])

        manual_result, manual_tensors = run_manual_q1(
            npu_manual_q1,
            obs_npu,
            prev_npu,
            progress_npu,
            action_npu,
            target_npu,
            "npu_explicit_one_layer_lstm_recurrence",
        )

        comparisons = {
            "cpu_native_vs_npu_manual_q1": compare_tensors(
                cpu_tensors["q1"], manual_tensors["q1"]
            ),
            "cpu_native_vs_npu_manual_context": compare_tensors(
                cpu_tensors["context"], manual_tensors["context"]
            ),
            "cpu_native_vs_npu_manual_token": compare_tensors(
                cpu_tensors["token"], manual_tensors["token"]
            ),
        }

        cpu_finite = (
            cpu_result["lstm_input_gradient"]["finite"] is True
            and cpu_result["parameter_gradients"]["finite"] is True
        )
        manual_finite = (
            manual_result["lstm_input_gradient"]["finite"] is True
            and manual_result["parameter_gradients"]["finite"] is True
        )

        branch = {
            "lambda": float(lam),
            "snapshot_path": str(pt_path),
            "snapshot_manifest_path": str(json_path),
            "snapshot_manifest": manifest,
            "cpu_native": cpu_result,
            "npu_manual": manual_result,
            "forward_comparisons": comparisons,
            "summary": {
                "cpu_control_finite": cpu_finite,
                "npu_manual_control_finite": manual_finite,
            },
        }
        branches.append(branch)

        print(
            f"[CONTROLS lambda={lam:.2f}] "
            f"cpu_finite={cpu_finite} "
            f"manual_finite={manual_finite} "
            f"q_diff={comparisons['cpu_native_vs_npu_manual_q1'].get('max_abs')}",
            flush=True,
        )

        del cpu_q1, npu_manual_q1
        prior.cleanup_device(s["device"])

    del template_agent
    prior.cleanup_device(s["device"])

    completed = all(
        branch["summary"]["cpu_control_finite"]
        and branch["summary"]["npu_manual_control_finite"]
        for branch in branches
    )

    return {
        "phase": "controls",
        "snapshot_dir": str(snapshot_dir),
        "branches": branches,
        "summary": {
            "cpu_control_finite_all": all(
                branch["summary"]["cpu_control_finite"]
                for branch in branches
            ),
            "npu_manual_control_finite_all": all(
                branch["summary"]["npu_manual_control_finite"]
                for branch in branches
            ),
        },
        "status": "PASS" if completed else "INCONCLUSIVE",
    }


def main():
    args = parse_args()
    lambdas = validate_lambdas(args.lambdas)

    s = phase.setup(args)
    phase.s_global_payload = s["payload"]

    # Preserve the same one-time pre-run production contract work used by the
    # earlier reproducing diagnostics. Controls do not need this numerically,
    # but running it keeps loader/contract validation consistent.
    first_batch_contract = phase.prepare_original_start(s)

    snapshot_dir = (
        Path(args.snapshot_dir).resolve()
        if args.snapshot_dir
        else default_snapshot_dir(s["run"])
    )
    output_path = (
        Path(args.output).resolve()
        if args.output
        else default_output(s["run"], args.phase)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    common = {
        "experiment": "stage3_mc_anchor_lstm_three_stage_isolation",
        "phase": args.phase,
        "device": str(s["device"]),
        "seed": 20260926,
        "batch_size": 256,
        "target_update": TARGET_UPDATE,
        "lambdas": list(lambdas),
        "stage2_checkpoint": str(s["stage2"]),
        "step0_checkpoint": str(s["step0"]),
        "canonical_replay": str(s["replay"]),
        "schedule_sha256": original.digest_refs(s["schedule"]),
        "probe_sha256": original.digest_refs(s["probe"]),
        "first_batch_contract": first_batch_contract,
        "snapshot_dir": str(snapshot_dir),
        "safety": {
            "environment_steps": 0,
            "formal_training_steps": 0,
            "training_checkpoint_writes": 0,
            "production_source_modified": False,
        },
    }

    if args.phase == "reference":
        phase_result = run_reference(s, lambdas)
    elif args.phase == "snapshot":
        phase_result = run_snapshot(s, lambdas, snapshot_dir)
    else:
        phase_result = run_controls(s, lambdas, snapshot_dir)

    result = {**common, **phase_result}
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"[STATUS] {result['status']} [SAVED] {output_path}",
        flush=True,
    )

    if result["status"] == "INCONCLUSIVE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
