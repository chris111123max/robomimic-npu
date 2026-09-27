#!/usr/bin/env python3
"""Reproduce and phase-trace MC-anchor nonfinite without changing production."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import test_stage3_bootstrap_vs_oracle_mc_causal as prior
import test_stage3_moving_vs_fixed_qmean_replay_bootstrap as fixed
import test_stage3_production_mc_anchor_causal as original


LAMBDAS = (0.25, 0.50)
CHECKPOINTS = (0, 100, 250, 500)
ORIGINAL_JSON = (
    "/data/home/3220251075/lerobot_workspace/training_runs/"
    "Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3/"
    "stage3v5_h10_20260923_161337/testing/"
    "stage2_vs_stage3_readiness/multi_production_mc_anchor_causal.json"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage3-run-dir", required=True)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--phase", choices=("minimal", "trace", "pinpoint", "precursor"), required=True)
    p.add_argument("--output")
    return p.parse_args()


def setup(args):
    class Loader:
        stage3_run_dir = args.stage3_run_dir
        stage2_checkpoint = prior.DEFAULT_STAGE2
        diagnostic_replay = None
        updates = 2000
        batch_size = 256
        probe_size = 8192
        probe_batch_size = 1024
    run, stage2, step0, replay, payload, config = fixed.load_inputs(Loader)
    canonical, episodes = prior.load_canonical(replay)
    length = int(config["recurrent_replay"]["critic_context_length"])
    returns = prior.mc_return_cache(episodes, float(config["gamma"]))
    schedule = prior.make_reference_schedule(
        episodes, 1000, 256, length, 20260926)
    probe = prior.make_probe_refs(
        episodes, 8192, length, 20260926+1000003)
    probe_returns, labels, ids = prior.probe_reference_returns(
        episodes, probe, length, float(config["gamma"]))
    return dict(run=run, stage2=stage2, step0=step0, replay=replay,
                payload=payload, config=config, canonical=canonical,
                episodes=episodes, length=length, returns=returns,
                schedule=schedule, probe=probe, probe_returns=probe_returns,
                labels=labels, ids=ids, geometry_refs=probe[:2048],
                device=prior.resolve_device(args.device))


def prepare_original_start(s):
    # Mirror the original diagnostic's pre-run forward work and contracts.
    a = original.make_agent(s["stage2"], s["payload"], s["device"], 0.0)
    first = original.first_batch_contract(
        a, s["episodes"], s["schedule"][0], s["length"],
        s["returns"], float(s["config"]["gamma"]))
    del a
    prior.cleanup_device(s["device"])
    return first


def initial_fingerprint(agent):
    state = agent.critic_optimizer.state_dict()
    digest = hashlib.sha256()
    def visit(obj):
        if torch.is_tensor(obj):
            t = obj.detach().cpu().contiguous()
            digest.update(str(t.dtype).encode())
            digest.update(np.asarray(t.shape, dtype=np.int64).tobytes())
            digest.update(t.numpy().tobytes())
        elif isinstance(obj, dict):
            for k in sorted(obj, key=str):
                digest.update(str(k).encode()); visit(obj[k])
        elif isinstance(obj, (tuple, list)):
            for x in obj: visit(x)
        else:
            digest.update(repr(obj).encode())
    visit(state)
    return {
        "online_hash": prior.module_digest(agent.critic),
        "target_hash": prior.module_digest(agent.target_critic),
        "actor_hash": prior.module_digest(agent.actor),
        "target_actor_hash": prior.module_digest(agent.target_actor),
        "optimizer_sha256": digest.hexdigest(),
        "optimizer_matches_checkpoint": prior.nested_state_equal(
            state, s_global_payload["critic_optimizer"]),
    }


# Assigned in main solely so the fingerprint checks the exact loaded payload.
s_global_payload = None


def minimal_branch(s, lam):
    agent = original.make_agent(s["stage2"], s["payload"], s["device"], lam)
    start = initial_fingerprint(agent)
    first_target = original.initial_lambda_contract(
        agent, s["episodes"], s["schedule"][0], s["length"],
        s["returns"], lam)
    milestones = []
    for update in CHECKPOINTS:
        if update == 0:
            r = original.record(
                agent, s["episodes"], s["probe"], s["probe_returns"],
                s["labels"], s["ids"], s["geometry_refs"], s["returns"],
                s["device"], s["config"], s["length"], 1024, lam, 0)
            milestones.append({"update": 0,
                               "qmin_spearman": r["online"]["qmin"]["spearman"]})
    completed = 0
    first_nonfinite = None
    for index, refs in enumerate(s["schedule"]):
        seq, diag = prior.prepare_training_batch(
            s["episodes"], refs, s["length"], s["returns"])
        prior.install_diagnostic_target_data(agent, diag)
        final = prior.final_transition(seq)
        try:
            agent.critic_update(final, seq, collect_metrics=False)
            agent.polyak_update()
        except FloatingPointError as exc:
            first_nonfinite = {"update": index+1, "error": str(exc)}
            break
        completed = index+1
        if completed in CHECKPOINTS:
            r = original.record(
                agent, s["episodes"], s["probe"], s["probe_returns"],
                s["labels"], s["ids"], s["geometry_refs"], s["returns"],
                s["device"], s["config"], s["length"], 1024, lam, completed)
            milestones.append({"update": completed,
                               "qmin_spearman": r["online"]["qmin"]["spearman"]})
            print(f"[MINIMAL {lam}] update={completed} "
                  f"qmin={r['online']['qmin']['spearman']:.6f}", flush=True)
        if completed >= 510:
            break
    result = {"lambda": lam, "initial": start, "first_target": first_target,
              "updates_completed": completed, "first_nonfinite": first_nonfinite,
              "milestones": milestones,
              "actor_unchanged": start["actor_hash"] == prior.module_digest(agent.actor),
              "target_actor_unchanged": start["target_actor_hash"] ==
                                         prior.module_digest(agent.target_actor)}
    del agent
    prior.cleanup_device(s["device"])
    return result



def tensor_stats(tensor):
    a = tensor.detach().float().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    finite = np.isfinite(a)
    good = a[finite]
    return {"finite": bool(finite.all()), "nonfinite_count": int((~finite).sum()),
            "min": float(good.min()) if len(good) else None,
            "max": float(good.max()) if len(good) else None,
            "mean": float(good.mean()) if len(good) else None,
            "std": float(good.std()) if len(good) else None,
            "max_abs": float(np.abs(good).max()) if len(good) else None}


def array_stats(array):
    a = np.asarray(array, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a)
    good = a[finite]
    return {"finite": bool(finite.all()), "nonfinite_count": int((~finite).sum()),
            "min": float(good.min()) if len(good) else None,
            "max": float(good.max()) if len(good) else None,
            "mean": float(good.mean()) if len(good) else None,
            "std": float(good.std()) if len(good) else None,
            "max_abs": float(np.abs(good).max()) if len(good) else None}


def block_of(name):
    if name.startswith(("q1.token_encoder.", "q1.lstm.",
                        "q2.token_encoder.", "q2.lstm.")):
        return "recurrent_history_encoder"
    if name.startswith("q1.q_head."):
        return "q1_head"
    if name.startswith("q2.q_head."):
        return "q2_head"
    return "shared_representation"


def named_tensor_summary(named, accessor):
    groups = {}
    total_sq = 0.0
    max_abs = 0.0
    nonfinite_count = 0
    nonfinite_parameter_count = 0
    first = None
    bad_parameters = []
    for name, parameter in named:
        value = accessor(parameter)
        if value is None:
            continue
        a = value.detach().float().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
        mask = np.isfinite(a)
        bad = int((~mask).sum())
        if bad:
            detail = {"name": name, "shape": list(parameter.shape),
                      "nonfinite_count": bad, "total_count": int(a.size),
                      "finite_max_abs": float(np.abs(a[mask]).max()) if mask.any() else None}
            bad_parameters.append(detail)
            if first is None:
                first = {"name": name, "shape": list(parameter.shape)}
        nonfinite_count += bad
        nonfinite_parameter_count += int(bad > 0)
        good = a[mask]
        block = block_of(name)
        group = groups.setdefault(block, {"squared_norm": 0.0, "max_abs": 0.0,
                                          "nonfinite_count": 0})
        if len(good):
            sq = float(np.dot(good, good))
            peak = float(np.abs(good).max())
            total_sq += sq
            max_abs = max(max_abs, peak)
            group["squared_norm"] += sq
            group["max_abs"] = max(group["max_abs"], peak)
        group["nonfinite_count"] += bad
    return {
        "finite": nonfinite_count == 0, "global_l2_norm": float(np.sqrt(total_sq)),
        "max_abs": max_abs, "nonfinite_parameter_count": nonfinite_parameter_count,
        "nonfinite_scalar_count": nonfinite_count, "first_nonfinite_parameter": first,
        "nonfinite_parameters": bad_parameters,
        "blocks": {k: {"finite": v["nonfinite_count"] == 0,
                        "l2_norm": float(np.sqrt(v["squared_norm"])),
                        "max_abs": v["max_abs"],
                        "nonfinite_count": v["nonfinite_count"]}
                   for k, v in groups.items()}}


def optimizer_summary(agent):
    named = list(agent.critic.named_parameters())
    first = None
    nonfinite_count = 0
    max_avg = 0.0
    max_sq = 0.0
    step_max = 0.0
    for name, parameter in named:
        state = agent.critic_optimizer.state.get(parameter, {})
        for key in ("exp_avg", "exp_avg_sq", "step"):
            if key not in state:
                continue
            value = state[key]
            a = (value.detach().float().cpu().numpy().reshape(-1)
                 if torch.is_tensor(value) else np.asarray([value]))
            good = np.isfinite(a)
            bad = int((~good).sum())
            if bad and first is None:
                first = {"parameter": name, "shape": list(parameter.shape),
                         "state_key": key}
            nonfinite_count += bad
            if good.any():
                peak = float(np.abs(a[good]).max())
                if key == "exp_avg":
                    max_avg = max(max_avg, peak)
                elif key == "exp_avg_sq":
                    max_sq = max(max_sq, float(a[good].max()))
                else:
                    step_max = max(step_max, peak)
    return {"finite": nonfinite_count == 0, "nonfinite_scalar_count": nonfinite_count,
            "first_nonfinite_state": first, "exp_avg_max_abs": max_avg,
            "exp_avg_sq_max": max_sq, "step_max": step_max}


def residual_stats(residual):
    a = residual.detach().float().cpu().numpy().astype(np.float64).reshape(-1)
    good = a[np.isfinite(a)]
    absolute = np.abs(good)
    return {"finite": len(good) == len(a),
            "mae": float(absolute.mean()) if len(good) else None,
            "rmse": float(np.sqrt(np.square(good).mean())) if len(good) else None,
            "max_abs": float(absolute.max()) if len(good) else None,
            "p95_abs": float(np.percentile(absolute, 95)) if len(good) else None,
            "p99_abs": float(np.percentile(absolute, 99)) if len(good) else None}


def first_bad(row, phase, mapping):
    for name, stat in mapping.items():
        if not stat["finite"]:
            row["first_nonfinite"] = {"update": row["update"],
                                      "phase": phase, "tensor_or_parameter": name}
            return True
    return False


def traced_update(agent, sequence, diagnostic, final, update):
    import stage3_v5_agent as agent_module
    row = {"update": update}
    row["input"] = {
        "observation_history": array_stats(sequence["observations"]),
        "action_history": array_stats(sequence["actions"]),
        "next_observations": array_stats(sequence["next_observations"]),
        "replay_next_action": array_stats(diagnostic["replay_next_actions"]),
        "reward": array_stats(final["rewards"]),
        "terminal": array_stats(final["terminals"]),
        "mc_return": array_stats(diagnostic["mc_current"])}
    if first_bad(row, "input", row["input"]):
        return row
    row["online_parameter_entry"] = named_tensor_summary(
        list(agent.critic.named_parameters()), lambda p: p)
    row["target_parameter_entry"] = named_tensor_summary(
        list(agent.target_critic.named_parameters()), lambda p: p)
    row["optimizer_entry"] = optimizer_summary(agent)
    b = agent._tensor_batch(final)
    captured = {}
    original_component_mean_q = agent_module.component_mean_q
    def capture_component(*a, **kw):
        result = original_component_mean_q(*a, **kw)
        captured["q1_target"] = result[1]
        captured["q2_target"] = result[2]
        captured["min_target"] = torch.minimum(result[1], result[2])
        return result
    agent_module.component_mean_q = capture_component
    try:
        with torch.no_grad():
            production = agent_module.RecurrentGMMTD3.bellman_target(
                agent, b, sequence)
    finally:
        agent_module.component_mean_q = original_component_mean_q
    y_stage3, distribution, expected_next = production[:3]
    mc = torch.as_tensor(diagnostic["mc_current"], dtype=torch.float32,
                         device=agent.device).reshape(-1, 1)
    lam = float(agent._diagnostic_mc_lambda)
    y_anchor = (1.0-lam)*y_stage3 + lam*mc
    base = distribution.component_distribution.base_dist
    actor = {
        "logits": tensor_stats(distribution.mixture_distribution.logits),
        "probabilities": tensor_stats(distribution.mixture_distribution.probs),
        "component_means": tensor_stats(base.loc),
        "component_scale": tensor_stats(base.scale)}
    row["target_actor"] = actor
    if first_bad(row, "target_actor_forward", actor):
        return row
    target = {k: tensor_stats(v) for k, v in captured.items()}
    target["expected_next"] = tensor_stats(expected_next)
    row["target_critic"] = target
    if first_bad(row, "target_critic_forward", target):
        return row
    row["targets"] = {
        "production_td": tensor_stats(y_stage3),
        "mc_return": tensor_stats(mc),
        "anchored_td": tensor_stats(y_anchor)}
    if not row["targets"]["production_td"]["finite"]:
        row["first_nonfinite"] = {"update": update, "phase": "td_target",
                                  "tensor_or_parameter": "y_stage3"}
        return row
    if not row["targets"]["anchored_td"]["finite"]:
        row["first_nonfinite"] = {"update": update, "phase": "anchored_target",
                                  "tensor_or_parameter": "y_anchor"}
        return row
    contexts = agent._history_contexts(agent.critic, sequence)
    q1, q2 = agent.critic.q_from_context(
        (contexts[0][:, -1], contexts[1][:, -1]), b["actions"])
    row["online_q"] = {"q1": tensor_stats(q1), "q2": tensor_stats(q2)}
    if first_bad(row, "online_q_forward", row["online_q"]):
        return row
    e1, e2 = q1-y_anchor, q2-y_anchor
    row["residual"] = {"e1": residual_stats(e1), "e2": residual_stats(e2)}
    if first_bad(row, "online_q_forward", row["residual"]):
        return row
    loss_q1 = torch.nn.functional.mse_loss(q1, y_anchor)
    loss_q2 = torch.nn.functional.mse_loss(q2, y_anchor)
    loss = loss_q1+loss_q2
    row["loss"] = {"q1": tensor_stats(loss_q1), "q2": tensor_stats(loss_q2),
                   "total": tensor_stats(loss)}
    if first_bad(row, "loss", row["loss"]):
        return row
    agent.critic_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    named = list(agent.critic.named_parameters())
    row["grad_preclip"] = named_tensor_summary(named, lambda p: p.grad)
    if not row["grad_preclip"]["finite"]:
        first = row["grad_preclip"]["first_nonfinite_parameter"]
        row["first_nonfinite"] = {"update": update, "phase": "backward_gradient",
                                  "tensor_or_parameter": first}
    returned = torch.nn.utils.clip_grad_norm_(
        agent.critic.parameters(), float(agent.config["critic_max_grad_norm"]))
    row["clip_return"] = tensor_stats(returned)
    row["grad_postclip"] = named_tensor_summary(named, lambda p: p.grad)
    if not row["clip_return"]["finite"] or not row["grad_postclip"]["finite"]:
        first = row["grad_postclip"]["first_nonfinite_parameter"]
        row.setdefault("first_nonfinite", {
            "update": update, "phase": "grad_clip",
            "tensor_or_parameter": first or "clip_return"})
    row["optimizer_before"] = optimizer_summary(agent)
    if not row["optimizer_before"]["finite"]:
        row["first_nonfinite"] = {
            "update": update, "phase": "optimizer_state_before_step",
            "tensor_or_parameter": row["optimizer_before"]["first_nonfinite_state"]}
        return row
    row["parameter_before"] = named_tensor_summary(named, lambda p: p)
    if not row["parameter_before"]["finite"]:
        row["first_nonfinite"] = {
            "update": update, "phase": "optimizer_step_parameter",
            "tensor_or_parameter": row["parameter_before"]["first_nonfinite_parameter"],
            "note": "Already nonfinite before optimizer step"}
        return row
    agent.critic_optimizer.step()
    agent.critic_updates += 1
    agent.actor_enabled_critic_updates += int(agent.actor_gate_open)
    row["parameter_after"] = named_tensor_summary(named, lambda p: p)
    row["optimizer_after"] = optimizer_summary(agent)
    if not row["parameter_after"]["finite"] and "first_nonfinite" not in row:
        row["first_nonfinite"] = {
            "update": update, "phase": "optimizer_step_parameter",
            "tensor_or_parameter": row["parameter_after"]["first_nonfinite_parameter"]}
        return row
    if not row["optimizer_after"]["finite"] and "first_nonfinite" not in row:
        row["first_nonfinite"] = {
            "update": update, "phase": "optimizer_state_after_step",
            "tensor_or_parameter": row["optimizer_after"]["first_nonfinite_state"]}
        return row
    target_named = list(agent.target_critic.named_parameters())
    row["target_parameter_before_polyak"] = named_tensor_summary(
        target_named, lambda p: p)
    agent.polyak_update()
    row["target_parameter_after_polyak"] = named_tensor_summary(
        target_named, lambda p: p)
    if not row["target_parameter_after_polyak"]["finite"] and "first_nonfinite" not in row:
        row["first_nonfinite"] = {
            "update": update, "phase": "polyak_update",
            "tensor_or_parameter":
                row["target_parameter_after_polyak"]["first_nonfinite_parameter"]}
    return row


def trace_branch(s, lam, start_trace=480):
    agent = original.make_agent(s["stage2"], s["payload"], s["device"], lam)
    start = initial_fingerprint(agent)
    first_target = original.initial_lambda_contract(
        agent, s["episodes"], s["schedule"][0], s["length"],
        s["returns"], lam)
    original.record(
        agent, s["episodes"], s["probe"], s["probe_returns"],
        s["labels"], s["ids"], s["geometry_refs"], s["returns"],
        s["device"], s["config"], s["length"], 1024, lam, 0)
    snapshots, rows = [], []
    completed = 0
    failure = None
    for index, refs in enumerate(s["schedule"]):
        update = index+1
        seq, diag = prior.prepare_training_batch(
            s["episodes"], refs, s["length"], s["returns"])
        prior.install_diagnostic_target_data(agent, diag)
        final = prior.final_transition(seq)
        if update >= start_trace:
            row = traced_update(agent, seq, diag, final, update)
            rows.append(row)
            if row.get("first_nonfinite"):
                failure = row["first_nonfinite"]
                break
        else:
            try:
                agent.critic_update(final, seq, collect_metrics=False)
                if start_trace == 504 and update == 503:
                    snapshots.append({
                        "update": update, "phase": "after_optimizer_before_polyak",
                        "online_parameter": named_tensor_summary(
                            list(agent.critic.named_parameters()), lambda p: p),
                        "gradient_postclip": named_tensor_summary(
                            list(agent.critic.named_parameters()), lambda p: p.grad),
                        "target_parameter": named_tensor_summary(
                            list(agent.target_critic.named_parameters()), lambda p: p),
                        "optimizer": optimizer_summary(agent)})
                agent.polyak_update()
                if start_trace == 504 and update == 503:
                    snapshots.append({
                        "update": update, "phase": "after_polyak",
                        "online_parameter": named_tensor_summary(
                            list(agent.critic.named_parameters()), lambda p: p),
                        "target_parameter": named_tensor_summary(
                            list(agent.target_critic.named_parameters()), lambda p: p)})
            except FloatingPointError as exc:
                failure = {"update": update, "phase": "pre_trace_exception",
                           "tensor_or_parameter": str(exc)}
                break
        completed = update
        if start_trace < 504 and update in (1, 100, 250, 400, 450):
            snapshots.append({
                "update": update,
                "online_parameter": named_tensor_summary(
                    list(agent.critic.named_parameters()), lambda p: p),
                "target_parameter": named_tensor_summary(
                    list(agent.target_critic.named_parameters()), lambda p: p),
                "optimizer": optimizer_summary(agent)})
        if update in (100, 250, 500):
            original.record(
                agent, s["episodes"], s["probe"], s["probe_returns"],
                s["labels"], s["ids"], s["geometry_refs"], s["returns"],
                s["device"], s["config"], s["length"], 1024, lam, update)
        if update >= 504:
            break
    result = {"lambda": lam, "initial": start,
              "first_target": first_target,
              "completed_updates": completed, "first_nonfinite": failure,
              "snapshots": snapshots, "rows_480_504": rows,
              "final_online_hash": prior.module_digest(agent.critic),
              "final_target_hash": prior.module_digest(agent.target_critic),
              "actor_unchanged": start["actor_hash"] == prior.module_digest(agent.actor),
              "target_actor_unchanged": start["target_actor_hash"] ==
                                         prior.module_digest(agent.target_actor)}
    del agent
    prior.cleanup_device(s["device"])
    return result

def main():
    global s_global_payload
    args = parse_args()
    s = setup(args)
    s_global_payload = s["payload"]
    old = json.loads(Path(ORIGINAL_JSON).read_text())
    first = prepare_original_start(s)
    output_path = (Path(args.output).resolve() if args.output else
                   s["run"]/"testing/stage2_vs_stage3_readiness/"
                   "multi_mc_anchor_nonfinite_trace.json")
    if args.phase == "minimal":
        result = {
            "phase": "minimal_reproduction",
            "device": str(s["device"]),
            "stage2_checkpoint": str(s["stage2"]),
            "step0_checkpoint": str(s["step0"]),
            "replay": str(s["replay"]),
            "seed": 20260926, "batch_size": 256, "probe_size": 8192,
            "schedule_sha256": original.digest_refs(s["schedule"]),
            "probe_sha256": original.digest_refs(s["probe"]),
            "expected_schedule_sha256": old["schedule_sha256"],
            "expected_probe_sha256": old["probe_sha256"],
            "first_batch_contract": first,
            "original_first_nonfinite": {
                str(b["lambda"]): b["failure"]["first_nonfinite_update"]
                for b in old["branches"] if b["lambda"] in LAMBDAS},
            "branches": [],
            "environment_steps": 0, "formal_training_steps": 0,
            "training_checkpoint_writes": 0,
        }
        for lam in LAMBDAS:
            result["branches"].append(minimal_branch(s, lam))
        result["same_schedule_as_original"] = (
            result["schedule_sha256"] == old["schedule_sha256"])
        result["same_probe_as_original"] = (
            result["probe_sha256"] == old["probe_sha256"])
        result["reproduced_504_both"] = all(
            b["first_nonfinite"] is not None and
            b["first_nonfinite"]["update"] == 504
            for b in result["branches"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
        print(f"[MINIMAL SAVED] {output_path} reproduced={result['reproduced_504_both']}",
              flush=True)
    elif args.phase == "trace":
        result = json.loads(output_path.read_text())
        if not (result.get("reproduced_504_both") and
                result.get("same_schedule_as_original") and
                result.get("same_probe_as_original")):
            raise RuntimeError("Minimal reproduction contract did not pass")
        result["phase"] = "detailed_trace"
        result["traces"] = []
        for lam in LAMBDAS:
            traced = trace_branch(s, lam)
            result["traces"].append(traced)
            output_path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
            print(f"[TRACE {lam}] first={traced['first_nonfinite']}", flush=True)
        result["trace_complete"] = True
        result["trace_initial_matches_minimal"] = all(
            t["initial"] == m["initial"]
            for t, m in zip(result["traces"], result["branches"]))
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
        print(f"[TRACE SAVED] {output_path}", flush=True)

    else:
        result = json.loads(output_path.read_text())
        if not result.get("reproduced_504_both"):
            raise RuntimeError("Minimal reproduction did not pass")
        key = "pinpoint" if args.phase == "pinpoint" else "precursor"
        start_trace = 504 if args.phase == "pinpoint" else 503
        result[key] = []
        for lam in LAMBDAS:
            traced = trace_branch(s, lam, start_trace=start_trace)
            result[key].append(traced)
            output_path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
            print(f"[{key.upper()} {lam}] first={traced['first_nonfinite']}", flush=True)
        result[key+"_complete"] = True
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
        print(f"[{key.upper()} SAVED] {output_path}", flush=True)

if __name__ == "__main__":
    main()
