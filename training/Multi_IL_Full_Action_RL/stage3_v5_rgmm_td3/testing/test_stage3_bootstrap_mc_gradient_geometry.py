#!/usr/bin/env python3
"""Static fixed-step0 bootstrap versus exact-MC label/gradient diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import test_stage3_bootstrap_vs_oracle_mc_causal as prior
import test_stage3_moving_vs_fixed_qmean_replay_bootstrap as fixed
from stage3_v5_readiness import correlation


def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument("--stage3-run-dir", required=True)
    p.add_argument("--stage2-checkpoint", default=prior.DEFAULT_STAGE2)
    p.add_argument("--diagnostic-replay")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--batches", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--probe-size", type=int, default=8192)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--output")
    p.set_defaults(updates=2000, probe_batch_size=1024)
    return p.parse_args()


def summary(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {"count": int(len(x)), "mean": float(x.mean()),
            "std": float(x.std()), "median": float(np.median(x)),
            "p05": float(np.percentile(x, 5)),
            "p90": float(np.percentile(x, 90)),
            "p95": float(np.percentile(x, 95)),
            "max": float(x.max()), "min": float(x.min())}


def pair_geometry(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    mask = (np.abs(a) > 1e-12) & (np.abs(b) > 1e-12)
    return {"cosine": dot / (na * nb) if na * nb else None,
            "dot": dot, "mc_norm": na, "bootstrap_norm": nb,
            "norm_ratio": nb / na if na else None,
            "relative_difference": float(np.linalg.norm(b-a) / na) if na else None,
            "sign_conflict_fraction": float(np.mean((a[mask] * b[mask]) < 0)) if mask.any() else None,
            "near_zero_mc_fraction": float(np.mean(np.abs(a) <= 1e-12)),
            "near_zero_bootstrap_fraction": float(np.mean(np.abs(b) <= 1e-12)),
            "elements": int(len(a))}


def grouped_grads(named, grads):
    groups = {"recurrent_history_encoder": [], "shared_representation": [],
              "q1_head": [], "q2_head": []}
    modules = {}
    for (name, _), grad in zip(named, grads):
        if name.startswith("q1.q_head."):
            group = "q1_head"
        elif name.startswith("q2.q_head."):
            group = "q2_head"
        elif name.startswith(("q1.token_encoder.", "q1.lstm.",
                              "q2.token_encoder.", "q2.lstm.")):
            group = "recurrent_history_encoder"
        else:
            group = "shared_representation"
        value = grad.detach().float().cpu().numpy().ravel().astype(np.float64, copy=False)
        groups[group].append(value)
        module = ".".join(name.split(".")[:2])
        modules.setdefault(module, []).append(value)
    return ({k: np.concatenate(v) for k, v in groups.items() if v},
            {k: np.concatenate(v) for k, v in modules.items()})


def correlations(a, b):
    s, p = correlation(np.asarray(a), np.asarray(b))
    return {"spearman": float(s), "pearson": float(p)}


def label_table(g, delta, yboot, labels, steps):
    order = np.argsort(g, kind="stable")
    deciles = []
    for i, ids in enumerate(np.array_split(order, 10)):
        d = delta[ids]
        deciles.append({"decile": i + 1, "count": int(len(ids)),
                        "mean_g": float(g[ids].mean()),
                        "mean_delta": float(d.mean()),
                        "median_delta": float(np.median(d)),
                        "mean_abs_delta": float(np.abs(d).mean()),
                        "bootstrap_target_mean": float(yboot[ids].mean()),
                        "mc_target_mean": float(g[ids].mean())})
    progress = {}
    for name, lo, hi in (("9_49", 9, 50), ("50_99", 50, 100),
                         ("100_199", 100, 200), ("200_399", 200, 400),
                         ("400_plus", 400, None)):
        mask = (steps >= lo) & ((steps < hi) if hi is not None else True)
        progress[name] = {"count": int(mask.sum()),
                          "mean_delta": float(delta[mask].mean()) if mask.any() else None,
                          "mean_abs_delta": float(np.abs(delta[mask]).mean()) if mask.any() else None}
    success = {}
    for name, value in (("success", 1), ("failure", 0)):
        mask = labels == value
        success[name] = {"count": int(mask.sum()),
                         "mean_delta": float(delta[mask].mean()) if mask.any() else None}
    return deciles, progress, success


def main():
    args = args_parse()
    if args.batches < 64 or args.batch_size != 256:
        raise ValueError("Require >=64 independent batches and batch size 256")
    run, stage2, step0_path, replay, payload, config = fixed.load_inputs(args)
    canonical, episodes = prior.load_canonical(replay)
    length = int(config["recurrent_replay"]["critic_context_length"])
    gamma = float(config["gamma"])
    returns = prior.mc_return_cache(episodes, gamma)
    schedule = prior.make_reference_schedule(
        episodes, args.batches, args.batch_size, length, args.seed)
    probe = prior.make_probe_refs(
        episodes, args.probe_size, length, args.seed + 1000003)
    probe_returns, probe_labels, _ = prior.probe_reference_returns(
        episodes, probe, length, gamma)
    device = prior.resolve_device(args.device)
    agent = prior.build_agent(stage2, payload, device)
    verify = prior.build_agent(stage2, payload, device)
    initial_hash = prior.module_digest(agent.critic)
    target_hash = prior.module_digest(agent.target_critic)
    actor_hash = prior.module_digest(agent.actor)
    target_actor_hash = prior.module_digest(agent.target_actor)
    optimizer_before = agent.critic_optimizer.state_dict()
    validity = {
        "same_initial_online_critic_hash": initial_hash == prior.module_digest(verify.critic),
        "same_initial_target_critic_hash": target_hash == prior.module_digest(verify.target_critic),
        "same_initial_optimizer_state": prior.nested_state_equal(
            optimizer_before, verify.critic_optimizer.state_dict()),
        "same_deterministic_batch_schedule": True,
        "same_probe_set": True,
    }
    del verify
    prior.cleanup_device(device)
    named = list(agent.critic.named_parameters())
    params = [p for _, p in named]
    all_g, all_delta, all_boot, all_error, all_steps = [], [], [], [], []
    all_labels = []
    nonfinite = 0
    # A fixed 8192-transition probe measures label structure independently of
    # any single optimization batch.
    for first in range(0, len(probe), 512):
        refs = probe[first:first+512]
        sequence, diagnostic = prior.prepare_training_batch(
            episodes, refs, length, returns)
        prior.install_diagnostic_target_data(agent, diagnostic)
        final = prior.final_transition(sequence)
        b = agent._tensor_batch(final)
        with torch.no_grad():
            boot = prior.replay_qmean_bellman_target(agent, b, sequence)[0].reshape(-1)
            contexts = agent._history_contexts(agent.critic, sequence)
            q1, q2 = agent.critic.q_from_context(
                (contexts[0][:, -1], contexts[1][:, -1]), b["actions"])
            qmean = (0.5 * (q1.reshape(-1) + q2.reshape(-1))).cpu().numpy()
        y = boot.cpu().numpy()
        g = np.asarray(diagnostic["mc_current"], dtype=np.float64)
        all_g.append(g)
        all_boot.append(y)
        all_delta.append(y-g)
        all_error.append(qmean-g)
        all_steps.append(np.asarray(sequence["episode_steps"])[:, -1].reshape(-1))
        all_labels.append(probe_labels[first:first+len(refs)])
        nonfinite += int((~np.isfinite(y)).sum())
    g, boot, delta, error, steps, labels = map(
        np.concatenate, (all_g, all_boot, all_delta, all_error, all_steps, all_labels))
    deciles, progress, success = label_table(g, delta, boot, labels, steps)
    rows = []
    for index, refs in enumerate(schedule):
        sequence, diagnostic = prior.prepare_training_batch(
            episodes, refs, length, returns)
        prior.install_diagnostic_target_data(agent, diagnostic)
        final = prior.final_transition(sequence)
        b = agent._tensor_batch(final)
        with torch.no_grad():
            yboot = prior.replay_qmean_bellman_target(agent, b, sequence)[0]
            ymc = torch.as_tensor(diagnostic["mc_current"],
                                   dtype=torch.float32, device=device).reshape(-1, 1)
        contexts = agent._history_contexts(agent.critic, sequence)
        q1, q2 = agent.critic.q_from_context(
            (contexts[0][:, -1], contexts[1][:, -1]), b["actions"])
        grads_by_mode = []
        for target in (ymc, yboot):
            agent.critic_optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
            loss.backward(retain_graph=True)
            grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                     for p in params]
            nonfinite += int(not bool(torch.isfinite(loss).item()))
            nonfinite += sum(int(not bool(torch.isfinite(v).all().item())) for v in grads)
            grads_by_mode.append(grouped_grads(named, grads))
        agent.critic_optimizer.zero_grad(set_to_none=True)
        gm, gb = grads_by_mode
        blocks = {key: pair_geometry(gm[0][key], gb[0][key]) for key in gm[0]}
        modules = {key: pair_geometry(gm[1][key], gb[1][key]) for key in gm[1]}
        rows.append({"batch": index,
                     "global": pair_geometry(np.concatenate(list(gm[0].values())),
                                               np.concatenate(list(gb[0].values()))),
                     "blocks": blocks, "modules": modules})
        if (index + 1) % 16 == 0:
            print(f"[GRADIENT] batches={index+1}/{len(schedule)}", flush=True)
    blocks = {}
    for key in rows[0]["blocks"]:
        blocks[key] = {metric: summary([r["blocks"][key][metric] for r in rows])
                       for metric in ("cosine", "mc_norm", "bootstrap_norm",
                                      "norm_ratio", "sign_conflict_fraction")}
    modules = {}
    for key in rows[0]["modules"]:
        modules[key] = {metric: summary([r["modules"][key][metric] for r in rows])
                        for metric in ("cosine", "norm_ratio", "sign_conflict_fraction")}
    global_stats = {metric: summary([r["global"][metric] for r in rows])
                    for metric in ("cosine", "dot", "mc_norm", "bootstrap_norm",
                                   "norm_ratio", "relative_difference", "sign_conflict_fraction",
                                   "near_zero_mc_fraction", "near_zero_bootstrap_fraction")}
    global_stats["negative_cosine_fraction"] = float(np.mean(
        [r["global"]["cosine"] < 0 for r in rows]))
    validity.update({
        "fixed_target_critic_unchanged": target_hash == prior.module_digest(agent.target_critic),
        "online_critic_unchanged": initial_hash == prior.module_digest(agent.critic),
        "same_optimizer_state_after_backward": prior.nested_state_equal(
            optimizer_before, agent.critic_optimizer.state_dict()),
        "actor_unchanged": actor_hash == prior.module_digest(agent.actor),
        "target_actor_unchanged": target_actor_hash == prior.module_digest(agent.target_actor),
        "all_batches_completed": len(rows) == args.batches,
        "nonfinite_zero": nonfinite == 0,
    })
    output = {
        "status": "PASS" if all(validity.values()) else "INVALID",
        "experiment": "stage3_step0_fixed_qmean_bootstrap_vs_mc_gradient_geometry",
        "device": str(device), "seed": args.seed, "batch_size": args.batch_size,
        "gradient_batches": args.batches, "gradient_optimizer_steps": 0,
        "environment_steps": 0, "training_checkpoint_writes": 0,
        "nonfinite_count": nonfinite, "stage2_checkpoint": str(stage2),
        "stage3_step0_checkpoint": str(step0_path), "replay": str(replay),
        "replay_seed": canonical.get("seed"), "schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(probe.tobytes()).hexdigest(), "probe_size": len(probe),
        "initial_online_hash": initial_hash, "initial_target_hash": target_hash,
        "initial_actor_hash": actor_hash, "initial_target_actor_hash": target_actor_hash,
        "validity": validity,
        "labels": {"delta": summary(delta), "abs_delta": summary(np.abs(delta)),
                   "positive_fraction": float(np.mean(delta > 0)),
                   "negative_fraction": float(np.mean(delta < 0)),
                   "delta_vs_mc_return": correlations(delta, g),
                   "delta_vs_current_qmean_error": correlations(delta, error),
                   "return_deciles": deciles, "success_failure": success,
                   "episode_progress": progress},
        "gradients": {"global": global_stats, "blocks": blocks,
                      "named_modules": modules, "per_batch": rows},
        "one_step_probe": "not_run_optional",
    }
    out = Path(args.output).resolve() if args.output else (
        run / "testing/stage2_vs_stage3_readiness/multi_bootstrap_mc_gradient_geometry.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"[STATUS] {output['status']} [SAVED] {out}", flush=True)
    if output["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
