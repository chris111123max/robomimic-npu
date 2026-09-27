#!/usr/bin/env python3
"""Fixed step0 qmean bootstrap / exact MC target interpolation causal test."""
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


MILESTONES = (0, 100, 250, 500, 1000)
ALPHAS = (0.0, 0.25, 0.5, 1.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage3-run-dir", required=True)
    p.add_argument("--stage2-checkpoint", default=prior.DEFAULT_STAGE2)
    p.add_argument("--diagnostic-replay")
    p.add_argument("--device", default="npu:0")
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--probe-size", type=int, default=8192)
    p.add_argument("--probe-batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--output")
    return p.parse_args()


@torch.no_grad()
def interpolated_target(self, b, sequence):
    mc = torch.as_tensor(self._diagnostic_mc_current,
                         dtype=torch.float32, device=self.device).reshape(-1, 1)
    boot = prior.replay_qmean_bellman_target(self, b, sequence)[0]
    y = (1.0 - self._diagnostic_alpha) * mc + self._diagnostic_alpha * boot
    return y, None, boot.reshape(-1), None, None, None


def make_agent(stage2, payload, device, alpha):
    agent = prior.build_agent(stage2, payload, device)
    agent._diagnostic_alpha = float(alpha)
    agent.bellman_target = types.MethodType(interpolated_target, agent)
    return agent


def record(agent, episodes, refs, returns, labels, episode_ids,
           device, config, length, probe_batch_size, update):
    prior.sync(device)
    data = prior.evaluate_probe_twin(
        agent.critic, episodes, refs, returns, labels, episode_ids,
        device, int(config["horizon"]), length, probe_batch_size)
    prior.sync(device)
    qmin = data["qmin"]
    qmean = data["qmean"]
    return {"update": update,
            "qmin_spearman": qmin["spearman_q_return"],
            "qmean_spearman": qmean["spearman_q_return"],
            "qmin_mae": qmin["mae_q_return"],
            "qmin_signed_bias": qmin["signed_bias_q_return"],
            "qmin_mean": qmin["mean"], "qmin_std": qmin["std"]}


def main():
    args = parse_args()
    if args.updates != 1000 or args.batch_size != 256:
        raise ValueError("Use matched 1000 updates and batch size 256")
    # The existing loader's 2000-step constraint applies to the old paired
    # experiment, not to this 1000-step dose-response schedule.
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
        episodes, args.probe_size, length, args.seed + 1000003)
    probe_returns, probe_labels, probe_ids = prior.probe_reference_returns(
        episodes, probe, length, gamma)
    device = prior.resolve_device(args.device)
    schedule_hash = hashlib.sha256(schedule.tobytes()).hexdigest()
    probe_hash = hashlib.sha256(probe.tobytes()).hexdigest()
    first_sequence, first_diag = prior.prepare_training_batch(
        episodes, schedule[0], length, returns)
    first_final = prior.final_transition(first_sequence)
    initial = []
    branches = []
    nonfinite = 0
    for alpha in ALPHAS:
        agent = make_agent(stage2, payload, device, alpha)
        online_hash = prior.module_digest(agent.critic)
        target_hash = prior.module_digest(agent.target_critic)
        actor_hash = prior.module_digest(agent.actor)
        target_actor_hash = prior.module_digest(agent.target_actor)
        optimizer_state = copy.deepcopy(agent.critic_optimizer.state_dict())
        prior.install_diagnostic_target_data(agent, first_diag)
        b = agent._tensor_batch(first_final)
        with torch.no_grad():
            first_y = agent.bellman_target(b, first_sequence)[0].reshape(-1).cpu().numpy()
            reference_boot = prior.replay_qmean_bellman_target(
                agent, b, first_sequence)[0].reshape(-1).cpu().numpy()
        initial.append({"alpha": alpha, "online_hash": online_hash,
                        "target_hash": target_hash, "actor_hash": actor_hash,
                        "target_actor_hash": target_actor_hash,
                        "optimizer_state": optimizer_state,
                        "first_target": first_y, "first_boot": reference_boot})
        trajectory = [record(agent, episodes, probe, probe_returns, probe_labels,
                             probe_ids, device, config, length,
                             args.probe_batch_size, 0)]
        completed = 0
        failure = None
        for index, refs in enumerate(schedule):
            sequence, diagnostic = prior.prepare_training_batch(
                episodes, refs, length, returns)
            prior.install_diagnostic_target_data(agent, diagnostic)
            final = prior.final_transition(sequence)
            try:
                agent.critic_update(final, sequence, collect_metrics=False)
            except FloatingPointError as exc:
                nonfinite += 1
                failure = {"update": index+1, "exception": str(exc)}
                break
            completed = index + 1
            if completed in MILESTONES:
                trajectory.append(record(agent, episodes, probe, probe_returns,
                                         probe_labels, probe_ids, device, config,
                                         length, args.probe_batch_size, completed))
                row = trajectory[-1]
                print(f"[ALPHA {alpha:.2f}] update={completed} "
                      f"qmin_spearman={row['qmin_spearman']:.6f}", flush=True)
        branches.append({"alpha": alpha, "completed_updates": completed,
                         "failure": failure, "trajectory": trajectory,
                         "initial_online_hash": online_hash,
                         "final_online_hash": prior.module_digest(agent.critic),
                         "initial_target_hash": target_hash,
                         "final_target_hash": prior.module_digest(agent.target_critic),
                         "actor_unchanged": actor_hash == prior.module_digest(agent.actor),
                         "target_actor_unchanged": target_actor_hash == prior.module_digest(agent.target_actor),
                         "optimizer_initial_state_matches_loaded":
                             prior.nested_state_equal(optimizer_state,
                                                      payload["critic_optimizer"])})
        del agent
        prior.cleanup_device(device)
        if failure is not None:
            break
    same_initial = len(initial) == len(ALPHAS) and all(
        row["online_hash"] == initial[0]["online_hash"] and
        row["target_hash"] == initial[0]["target_hash"] and
        row["actor_hash"] == initial[0]["actor_hash"] and
        row["target_actor_hash"] == initial[0]["target_actor_hash"] and
        prior.nested_state_equal(row["optimizer_state"], initial[0]["optimizer_state"])
        for row in initial)
    mc = np.asarray(first_diag["mc_current"], dtype=np.float32)
    boot = initial[0]["first_boot"] if len(initial) == len(ALPHAS) else None
    endpoint_contract = bool(
        len(initial) == len(ALPHAS)
        and np.max(np.abs(initial[0]["first_target"] - mc)) <= 2e-6
        and np.max(np.abs(initial[-1]["first_target"] - boot)) <= 2e-6
        and all(np.max(np.abs(row["first_boot"] - boot)) <= 2e-6 for row in initial))
    interpolation_contract = bool(
        endpoint_contract and all(np.max(np.abs(
            row["first_target"] - ((1-row["alpha"])*mc + row["alpha"]*boot))) <= 2e-6
            for row in initial))
    validity = {
        "same_initial_online_critic_hash": same_initial,
        "same_initial_target_critic_hash": same_initial,
        "same_initial_optimizer_state": same_initial,
        "same_deterministic_batch_schedule": True,
        "same_probe_set": True,
        "fixed_teacher_all_branches": all(
            x["initial_target_hash"] == x["final_target_hash"] for x in branches),
        "actor_unchanged_all_branches": all(x["actor_unchanged"] for x in branches),
        "target_actor_unchanged_all_branches": all(
            x["target_actor_unchanged"] for x in branches),
        "optimizer_loaded_all_branches": all(
            x["optimizer_initial_state_matches_loaded"] for x in branches),
        "alpha_endpoints_and_interpolation_match": interpolation_contract,
        "all_branches_completed": len(branches) == len(ALPHAS) and all(
            x["completed_updates"] == args.updates for x in branches),
        "nonfinite_zero": nonfinite == 0,
    }
    if not all(validity.values()):
        status = "PARTIAL" if nonfinite else "INVALID"
    else:
        status = "PASS"
    output = {
        "status": status, "experiment": "stage3_fixed_qmean_bootstrap_mc_alpha_dose_response",
        "device": str(device), "seed": args.seed, "batch_size": args.batch_size,
        "updates_per_branch": args.updates, "alphas": ALPHAS,
        "stage2_checkpoint": str(stage2), "step0_checkpoint": str(step0_path),
        "replay": str(replay), "canonical_seed": canonical.get("seed"),
        "schedule_sha256": schedule_hash, "probe_sha256": probe_hash,
        "probe_size": len(probe), "fixed_teacher": True,
        "target_actor_used": False, "twin_min_used": False,
        "environment_steps": 0, "actor_optimizer_steps": 0,
        "training_checkpoint_writes": 0, "nonfinite_count": nonfinite,
        "validity": validity, "branches": branches,
        "first_batch_mc_vs_boot_mae": float(np.mean(np.abs(boot-mc))) if boot is not None else None,
    }
    out = Path(args.output).resolve() if args.output else (
        run / "testing/stage2_vs_stage3_readiness/multi_fixed_qmean_mc_bootstrap_interpolation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"[STATUS] {status} [SAVED] {out}", flush=True)
    if status == "INVALID":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
