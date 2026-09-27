#!/usr/bin/env python3
"""Testing-only production Stage3 target versus empirical MC anchor."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import types
from pathlib import Path

import numpy as np
import torch

import test_stage3_bootstrap_vs_oracle_mc_causal as prior
import test_stage3_moving_vs_fixed_qmean_replay_bootstrap as fixed
import test_stage3_td_target_contract as contract
from stage3_v5_agent import RecurrentGMMTD3
from stage3_v5_history_critic import encode_replay_contexts
from stage3_v5_readiness import auc, correlation


LAMBDAS = (0.0, 0.25, 0.50)
MILESTONES = (0, 100, 250, 500, 1000)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage3-run-dir", required=True)
    p.add_argument("--stage2-checkpoint", default=prior.DEFAULT_STAGE2)
    p.add_argument("--diagnostic-replay")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--probe-size", type=int, default=8192)
    p.add_argument("--probe-batch-size", type=int, default=1024)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--output")
    return p.parse_args()


@torch.no_grad()
def anchored_bellman_target(self, b, sequence):
    # Use production's target construction verbatim: moving target Critic,
    # target Actor, component-mean categorical expectation, clipped twin min,
    # successor history and original terminal semantics.
    production = RecurrentGMMTD3.bellman_target(self, b, sequence)
    lam = float(self._diagnostic_mc_lambda)
    if lam == 0.0:
        return production
    mc = torch.as_tensor(self._diagnostic_mc_current,
                         dtype=torch.float32, device=self.device).reshape(-1, 1)
    target = (1.0 - lam) * production[0] + lam * mc
    return (target, *production[1:])


def make_agent(stage2, payload, device, lam):
    agent = prior.build_agent(stage2, payload, device)
    agent._diagnostic_mc_lambda = float(lam)
    agent.bellman_target = types.MethodType(anchored_bellman_target, agent)
    return agent


def digest_refs(refs):
    return hashlib.sha256(np.ascontiguousarray(refs).tobytes()).hexdigest()


def max_abs(a, b):
    return float((a - b).abs().max().item())


def dist_max(a, b):
    return max(contract.distribution_diffs(a, b).values())


@torch.no_grad()
def first_batch_contract(agent, episodes, refs, length, returns, gamma):
    sequence, diagnostic = prior.prepare_training_batch(
        episodes, refs, length, returns)
    prior.install_diagnostic_target_data(agent, diagnostic)
    final = prior.final_transition(sequence)
    b = agent._tensor_batch(final)
    production = RecurrentGMMTD3.bellman_target(agent, b, sequence)
    helper = agent.bellman_target(b, sequence)
    production_td, prod_dist, prod_q, _, _, prod_context = production
    helper_td, helper_dist, helper_q, _, _, helper_context = helper
    masks = []
    manual_context, manual_dist, manual_q, manual_td = [], [], [], []
    for row, (ep_index, start) in enumerate(np.asarray(refs, dtype=np.int64)):
        ep_index, start = int(ep_index), int(start)
        target = start + length - 1
        terminal = float(prior.episode_terminals(episodes[ep_index])[target])
        masks.append(abs(terminal - float(final["terminals"][row, 0])))
        if row >= 8:
            continue
        reference = contract.manual_reference(
            agent, episodes[ep_index], start, target,
            int(agent.config["horizon"]),
            int(agent.config["actor_source_contract"]["rnn_horizon"]))
        manual_context.append(max(
            max_abs(prod_context[0][row:row+1, -1],
                    reference["contexts"][0][:, -1]),
            max_abs(prod_context[1][row:row+1, -1],
                    reference["contexts"][1][:, -1])))
        base = prod_dist.component_distribution.base_dist
        ref_base = reference["distribution"].component_distribution.base_dist
        manual_dist.append(max(
            max_abs(base.loc[row:row+1], ref_base.loc),
            max_abs(base.scale[row:row+1], ref_base.scale),
            max_abs(prod_dist.mixture_distribution.logits[row:row+1],
                    reference["distribution"].mixture_distribution.logits)))
        manual_q.append(max_abs(
            prod_q[row:row+1].reshape(-1),
            reference["expected_next"].reshape(-1)))
        manual_td.append(max_abs(
            production_td[row:row+1].reshape(-1),
            reference["td_target"].reshape(-1)))
    terminal_mask = b["terminals"].reshape(-1) >= 0.5
    terminal_diff = (max_abs(production_td.reshape(-1)[terminal_mask],
                             b["rewards"].reshape(-1)[terminal_mask])
                     if bool(terminal_mask.any()) else 0.0)
    mc = torch.as_tensor(diagnostic["mc_current"],
                         dtype=torch.float32, device=agent.device).reshape(-1)
    mc_identity = prior.canonical_mc_identity(episodes, gamma)
    return {
        "production_target_max_abs_diff": max_abs(helper_td, production_td),
        "production_target_mean_abs_diff": float(
            (helper_td-production_td).abs().mean().item()),
        "production_terminal_mask_max_diff": float(max(masks)),
        "production_successor_context_max_diff": max(
            max_abs(prod_context[0], helper_context[0]),
            max_abs(prod_context[1], helper_context[1])),
        "production_actor_distribution_max_diff": dist_max(helper_dist, prod_dist),
        "production_next_q_max_diff": max_abs(helper_q, prod_q),
        "manual_successor_context_max_diff_first8": max(manual_context),
        "manual_actor_distribution_max_diff_first8": max(manual_dist),
        "manual_next_q_max_diff_first8": max(manual_q),
        "manual_td_max_diff_first8": max(manual_td),
        "terminal_target_equals_reward_max_abs": terminal_diff,
        "terminal_count_first_batch": int(terminal_mask.sum().item()),
        "mc_identity_max_abs_all_episodes": mc_identity["abs_max"],
        "mc_first_batch_count": int(mc.numel()),
    }


def initial_lambda_contract(agent, episodes, refs, length, returns, lam):
    sequence, diagnostic = prior.prepare_training_batch(
        episodes, refs, length, returns)
    prior.install_diagnostic_target_data(agent, diagnostic)
    final = prior.final_transition(sequence)
    b = agent._tensor_batch(final)
    with torch.no_grad():
        production = RecurrentGMMTD3.bellman_target(agent, b, sequence)[0]
        actual = agent.bellman_target(b, sequence)[0]
        mc = torch.as_tensor(diagnostic["mc_current"],
                             dtype=torch.float32, device=agent.device).reshape(-1, 1)
        expected = (1-lam)*production + lam*mc
    return {
        "lambda": lam,
        "reconstruction_max_abs": max_abs(actual, expected),
        "production_vs_mc_mae": float((production-mc).abs().mean().item()),
        "anchored_vs_mc_mae": float((actual-mc).abs().mean().item()),
    }


@torch.no_grad()
def evaluate_critic(critic, episodes, refs, returns, labels, episode_ids,
                    device, horizon, length, batch_size):
    critic.eval()
    q1_parts, q2_parts = [], []
    for first in range(0, len(refs), batch_size):
        seq = prior.stack_sequence_batch(episodes, refs[first:first+batch_size], length)
        obs = torch.as_tensor(seq["observations"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(seq["actions"], dtype=torch.float32, device=device)
        steps = torch.as_tensor(seq["episode_steps"], dtype=torch.long, device=device)
        contexts = encode_replay_contexts(critic, obs, actions, steps, horizon)
        a = actions[:, -1]
        q1, q2 = critic.q_from_context(
            (contexts[0][:, -1], contexts[1][:, -1]), a)
        q1_parts.append(q1.reshape(-1).cpu().numpy())
        q2_parts.append(q2.reshape(-1).cpu().numpy())
    q1 = np.concatenate(q1_parts).astype(np.float64)
    q2 = np.concatenate(q2_parts).astype(np.float64)
    qmin = np.minimum(q1, q2)
    qmean = 0.5*(q1+q2)
    returns = np.asarray(returns, dtype=np.float64)
    labels = np.asarray(labels)
    episode_ids = np.asarray(episode_ids)

    def metrics(q):
        spear, pearson = correlation(q, returns)
        diff = q-returns
        return {"spearman": float(spear), "pearson": float(pearson),
                "mae": float(np.abs(diff).mean()),
                "rmse": float(np.sqrt(np.square(diff).mean())),
                "signed_bias": float(diff.mean()),
                "mean": float(q.mean()), "std": float(q.std())}

    ids = np.unique(episode_ids)
    episode_q = np.asarray([qmin[episode_ids == ep].mean() for ep in ids])
    episode_labels = np.asarray([labels[np.flatnonzero(episode_ids == ep)[0]]
                                 for ep in ids])
    gap = np.abs(q1-q2)
    success = labels == 1
    failure = labels == 0
    return {
        "qmin": metrics(qmin), "qmean": metrics(qmean),
        "q1": metrics(q1), "q2": metrics(q2),
        "twin_abs_gap_mean": float(gap.mean()),
        "twin_abs_gap_median": float(np.median(gap)),
        "twin_abs_gap_p95": float(np.percentile(gap, 95)),
        "episode_auc_qmin": float(auc(episode_labels, episode_q))
            if len(np.unique(episode_labels)) == 2 else None,
        "success_qmin_mean": float(qmin[success].mean()) if success.any() else None,
        "failure_qmin_mean": float(qmin[failure].mean()) if failure.any() else None,
        "success_minus_failure_qmin": float(qmin[success].mean()-qmin[failure].mean())
            if success.any() and failure.any() else None,
    }


@torch.no_grad()
def target_geometry(agent, episodes, refs, returns, length, batch_size, lam):
    yprod_parts = []
    g_parts = []
    for first in range(0, len(refs), batch_size):
        seq, diag = prior.prepare_training_batch(
            episodes, refs[first:first+batch_size], length, returns)
        final = prior.final_transition(seq)
        b = agent._tensor_batch(final)
        y = RecurrentGMMTD3.bellman_target(agent, b, seq)[0].reshape(-1)
        yprod_parts.append(y.cpu().numpy())
        g_parts.append(diag["mc_current"])
    yprod = np.concatenate(yprod_parts).astype(np.float64)
    g = np.concatenate(g_parts).astype(np.float64)
    yanchor = (1-lam)*yprod + lam*g
    def metrics(y):
        spear, pearson = correlation(y, g)
        return {"spearman": float(spear), "pearson": float(pearson),
                "mae": float(np.abs(y-g).mean()),
                "signed_bias": float((y-g).mean()),
                "mean": float(y.mean()), "std": float(y.std())}
    order = np.argsort(g, kind="stable")
    deciles = []
    for i, indices in enumerate(np.array_split(order, 10)):
        deciles.append({
            "decile": i+1, "count": len(indices),
            "mean_mc": float(g[indices].mean()),
            "production_minus_mc_mean": float((yprod[indices]-g[indices]).mean()),
            "anchored_minus_mc_mean": float((yanchor[indices]-g[indices]).mean()),
        })
    return {"count": len(g), "production": metrics(yprod),
            "anchored": metrics(yanchor), "deciles": deciles}


def record(agent, episodes, probe, probe_returns, labels, episode_ids,
           geometry_refs, returns, device, config, length, batch_size, lam, update):
    prior.sync(device)
    online = evaluate_critic(
        agent.critic, episodes, probe, probe_returns, labels, episode_ids,
        device, int(config["horizon"]), length, batch_size)
    target = evaluate_critic(
        agent.target_critic, episodes, probe, probe_returns, labels, episode_ids,
        device, int(config["horizon"]), length, batch_size)
    geometry = target_geometry(
        agent, episodes, geometry_refs, returns, length, batch_size, lam)
    prior.sync(device)
    return {"update": update, "online": online, "target": target,
            "target_vs_mc_geometry": geometry}


def main():
    args = parse_args()
    if (args.seed != 20260926 or args.batch_size != 256 or
        args.probe_size != 8192 or args.updates != 1000):
        raise ValueError("Fixed seed/batch/probe/updates contract changed")
    if args.probe_batch_size not in (1024, 512, 256):
        raise ValueError("Probe batch size must be 1024, 512 or 256")
    loader_args = argparse.Namespace(**vars(args))
    loader_args.updates = 2000
    run, stage2, step0_path, replay, payload, config = fixed.load_inputs(loader_args)
    canonical, episodes = prior.load_canonical(replay)
    length = int(config["recurrent_replay"]["critic_context_length"])
    gamma = float(config["gamma"])
    returns = prior.mc_return_cache(episodes, gamma)
    schedule = prior.make_reference_schedule(
        episodes, args.updates, args.batch_size, length, args.seed)
    probe = prior.make_probe_refs(
        episodes, args.probe_size, length, args.seed+1000003)
    probe_returns, probe_labels, probe_ids = prior.probe_reference_returns(
        episodes, probe, length, gamma)
    geometry_refs = probe[:min(2048, len(probe))]
    device = prior.resolve_device(args.device)
    terminal = contract.validate_terminal_contract(episodes)
    first_agent = make_agent(stage2, payload, device, 0.0)
    first_contract = first_batch_contract(
        first_agent, episodes, schedule[0], length, returns, gamma)
    del first_agent
    prior.cleanup_device(device)
    first_pass = (
        first_contract["production_target_max_abs_diff"] <= 2e-6 and
        first_contract["production_terminal_mask_max_diff"] == 0 and
        first_contract["production_successor_context_max_diff"] <= 2e-6 and
        first_contract["production_actor_distribution_max_diff"] <= 2e-6 and
        first_contract["production_next_q_max_diff"] <= 2e-6 and
        first_contract["manual_successor_context_max_diff_first8"] <= 2e-4 and
        first_contract["manual_actor_distribution_max_diff_first8"] <= 2e-4 and
        first_contract["manual_next_q_max_diff_first8"] <= 2e-4 and
        first_contract["manual_td_max_diff_first8"] <= 2e-4 and
        first_contract["terminal_target_equals_reward_max_abs"] <= 2e-6 and
        first_contract["mc_identity_max_abs_all_episodes"] <= 2e-6 and
        terminal["mismatch_count"] == 0)
    initial = []
    branches = []
    nonfinite = 0
    if first_pass:
        for lam in LAMBDAS:
            agent = make_agent(stage2, payload, device, lam)
            start = {
                "online_hash": prior.module_digest(agent.critic),
                "target_hash": prior.module_digest(agent.target_critic),
                "actor_hash": prior.module_digest(agent.actor),
                "target_actor_hash": prior.module_digest(agent.target_actor),
                "optimizer_state": copy.deepcopy(agent.critic_optimizer.state_dict()),
            }
            initial.append(start)
            target_contract = initial_lambda_contract(
                agent, episodes, schedule[0], length, returns, lam)
            trajectory = [record(
                agent, episodes, probe, probe_returns, probe_labels, probe_ids,
                geometry_refs, returns, device, config, length,
                args.probe_batch_size, lam, 0)]
            completed = 0
            failure = None
            for index, refs in enumerate(schedule):
                sequence, diagnostic = prior.prepare_training_batch(
                    episodes, refs, length, returns)
                prior.install_diagnostic_target_data(agent, diagnostic)
                final = prior.final_transition(sequence)
                try:
                    agent.critic_update(final, sequence, collect_metrics=False)
                    agent.polyak_update()
                except FloatingPointError as exc:
                    nonfinite += 1
                    failure = {"first_nonfinite_update": index+1,
                               "exception": str(exc)}
                    break
                completed = index+1
                if completed in MILESTONES:
                    try:
                        trajectory.append(record(
                            agent, episodes, probe, probe_returns,
                            probe_labels, probe_ids, geometry_refs, returns,
                            device, config, length, args.probe_batch_size,
                            lam, completed))
                    except (FloatingPointError, ValueError) as exc:
                        nonfinite += 1
                        failure = {"first_nonfinite_update": completed,
                                   "exception": str(exc)}
                        break
                    row = trajectory[-1]
                    print(f"[LAMBDA {lam:.2f}] update={completed} "
                          f"qmin_spearman={row['online']['qmin']['spearman']:.6f} "
                          f"mae={row['online']['qmin']['mae']:.6f}", flush=True)
            branches.append({
                "lambda": lam, "updates_completed": completed,
                "polyak_updates_completed": completed,
                "failure": failure, "first_batch_target_contract": target_contract,
                "trajectory": trajectory,
                "initial_online_hash": start["online_hash"],
                "final_online_hash": prior.module_digest(agent.critic),
                "initial_target_hash": start["target_hash"],
                "final_target_hash": prior.module_digest(agent.target_critic),
                "initial_actor_hash": start["actor_hash"],
                "final_actor_hash": prior.module_digest(agent.actor),
                "initial_target_actor_hash": start["target_actor_hash"],
                "final_target_actor_hash": prior.module_digest(agent.target_actor),
                "optimizer_loaded": prior.nested_state_equal(
                    start["optimizer_state"], payload["critic_optimizer"]),
            })
            del agent
            prior.cleanup_device(device)
    same_initial = len(initial) == len(LAMBDAS) and all(
        x["online_hash"] == initial[0]["online_hash"] and
        x["target_hash"] == initial[0]["target_hash"] and
        x["actor_hash"] == initial[0]["actor_hash"] and
        x["target_actor_hash"] == initial[0]["target_actor_hash"] and
        prior.nested_state_equal(x["optimizer_state"], initial[0]["optimizer_state"])
        for x in initial)
    validity = {
        "first_batch_production_target_contract": first_pass,
        "same_initial_online_critic_hash": same_initial,
        "same_initial_target_critic_hash": same_initial,
        "same_initial_actor_hash": same_initial,
        "same_initial_target_actor_hash": same_initial,
        "same_initial_optimizer_state": same_initial,
        "same_deterministic_batch_schedule": True,
        "same_probe_set": True,
        "lambda_formula_first_batch": len(branches) == len(LAMBDAS) and all(
            b["first_batch_target_contract"]["reconstruction_max_abs"] <= 2e-6
            for b in branches),
        "production_polyak_each_update": all(
            b["polyak_updates_completed"] == b["updates_completed"]
            for b in branches),
        "actor_unchanged_all": all(
            b["initial_actor_hash"] == b["final_actor_hash"] for b in branches),
        "target_actor_unchanged_all": all(
            b["initial_target_actor_hash"] == b["final_target_actor_hash"]
            for b in branches),
        "optimizer_loaded_all": all(b["optimizer_loaded"] for b in branches),
        "all_branches_completed": len(branches) == len(LAMBDAS) and all(
            b["updates_completed"] == args.updates for b in branches),
        "nonfinite_zero": nonfinite == 0,
    }
    status = "INVALID" if not first_pass or not same_initial or not validity["lambda_formula_first_batch"] else (
        "PARTIAL" if nonfinite or not validity["all_branches_completed"] else
        "PASS" if all(validity.values()) else "INVALID")
    output = {
        "status": status,
        "experiment": "stage3_v5_production_td_mc_anchor_paired_causal",
        "device": str(device), "seed": args.seed, "batch_size": args.batch_size,
        "probe_size": len(probe), "probe_batch_size": args.probe_batch_size,
        "updates_per_branch": args.updates, "lambdas": LAMBDAS,
        "milestones": MILESTONES, "geometry_probe_size": len(geometry_refs),
        "schedule_sha256": digest_refs(schedule),
        "probe_sha256": digest_refs(probe),
        "geometry_probe_sha256": digest_refs(geometry_refs),
        "stage2_checkpoint": str(stage2), "step0_checkpoint": str(step0_path),
        "canonical_replay": str(replay), "canonical_seed": canonical.get("seed"),
        "first_batch_contract": first_contract, "terminal_contract": {
            k: terminal[k] for k in
            ("transition_count", "terminated_count", "truncated_count",
             "done_count", "mismatch_count")},
        "validity": validity, "branches": branches, "nonfinite_count": nonfinite,
        "environment_steps": 0, "formal_training_steps": 0,
        "training_checkpoint_writes": 0, "actor_optimizer_steps": 0,
        "target_actor_updates": 0,
    }
    out = Path(args.output).resolve() if args.output else (
        run/"testing/stage2_vs_stage3_readiness/multi_production_mc_anchor_causal.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True)+"\n")
    print(f"[STATUS] {status} [SAVED] {out}", flush=True)
    if status == "INVALID":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
