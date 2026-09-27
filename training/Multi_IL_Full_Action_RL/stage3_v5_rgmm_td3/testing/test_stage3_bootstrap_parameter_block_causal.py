#!/usr/bin/env python3
"""Diagnostic fixed-teacher Critic update ablation by actual parameter blocks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import test_stage3_bootstrap_vs_oracle_mc_causal as prior
import test_stage3_mc_bootstrap_interpolation_causal as interpolation
import test_stage3_moving_vs_fixed_qmean_replay_bootstrap as fixed


MODES = ("full", "heads_only", "encoder_only")


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


def block_of(name):
    if name.startswith(("q1.q_head.", "q2.q_head.")):
        return "head"
    if name.startswith(("q1.token_encoder.", "q1.lstm.",
                        "q2.token_encoder.", "q2.lstm.")):
        return "encoder"
    raise RuntimeError(f"Unclassified Critic parameter: {name}")


def main():
    args = parse_args()
    if args.updates != 1000 or args.batch_size != 256:
        raise ValueError("Use matched 1000 updates and batch size 256")
    loader_args = argparse.Namespace(**vars(args))
    loader_args.updates = 2000
    run, stage2, step0_path, replay, payload, config = fixed.load_inputs(loader_args)
    canonical, episodes = prior.load_canonical(replay)
    length = int(config["recurrent_replay"]["critic_context_length"])
    returns = prior.mc_return_cache(episodes, float(config["gamma"]))
    schedule = prior.make_reference_schedule(
        episodes, args.updates, args.batch_size, length, args.seed)
    probe = prior.make_probe_refs(
        episodes, args.probe_size, length, args.seed + 1000003)
    probe_returns, probe_labels, probe_ids = prior.probe_reference_returns(
        episodes, probe, length, float(config["gamma"]))
    device = prior.resolve_device(args.device)
    branches = []
    states = []
    nonfinite = 0
    for mode in MODES:
        agent = interpolation.make_agent(stage2, payload, device, 1.0)
        original_hash = prior.module_digest(agent.critic)
        target_hash = prior.module_digest(agent.target_critic)
        actor_hash = prior.module_digest(agent.actor)
        target_actor_hash = prior.module_digest(agent.target_actor)
        optimizer_state = copy.deepcopy(agent.critic_optimizer.state_dict())
        named = list(agent.critic.named_parameters())
        block_names = {name: block_of(name) for name, _ in named}
        for name, param in named:
            block = block_names[name]
            param.requires_grad_(mode == "full" or
                                 (mode == "heads_only" and block == "head") or
                                 (mode == "encoder_only" and block == "encoder"))
        trainable = {"encoder": 0, "head": 0}
        for name, param in named:
            if param.requires_grad:
                trainable[block_names[name]] += param.numel()
        states.append({"online_hash": original_hash, "target_hash": target_hash,
                       "actor_hash": actor_hash, "target_actor_hash": target_actor_hash,
                       "optimizer": optimizer_state})
        trajectory = [interpolation.record(
            agent, episodes, probe, probe_returns, probe_labels,
            probe_ids, device, config, length, args.probe_batch_size, 0)]
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
            if completed in interpolation.MILESTONES:
                trajectory.append(interpolation.record(
                    agent, episodes, probe, probe_returns, probe_labels,
                    probe_ids, device, config, length, args.probe_batch_size,
                    completed))
                print(f"[{mode}] update={completed} "
                      f"qmin_spearman={trajectory[-1]['qmin_spearman']:.6f}",
                      flush=True)
        branches.append({"mode": mode, "trainable_elements": trainable,
                         "completed_updates": completed, "failure": failure,
                         "trajectory": trajectory,
                         "initial_online_hash": original_hash,
                         "initial_target_hash": target_hash,
                         "final_target_hash": prior.module_digest(agent.target_critic),
                         "actor_unchanged": actor_hash == prior.module_digest(agent.actor),
                         "target_actor_unchanged": target_actor_hash == prior.module_digest(agent.target_actor),
                         "optimizer_initial_state_matches_loaded": prior.nested_state_equal(
                             optimizer_state, payload["critic_optimizer"])})
        del agent
        prior.cleanup_device(device)
        if failure is not None:
            break
    same_start = len(states) == len(MODES) and all(
        x["online_hash"] == states[0]["online_hash"] and
        x["target_hash"] == states[0]["target_hash"] and
        x["actor_hash"] == states[0]["actor_hash"] and
        x["target_actor_hash"] == states[0]["target_actor_hash"] and
        prior.nested_state_equal(x["optimizer"], states[0]["optimizer"])
        for x in states)
    validity = {
        "same_initial_online_critic_hash": same_start,
        "same_initial_target_critic_hash": same_start,
        "same_initial_optimizer_state": same_start,
        "same_deterministic_batch_schedule": True,
        "same_probe_set": True,
        "fixed_target_critic_all": all(x["initial_target_hash"] == x["final_target_hash"]
                                   for x in branches),
        "actor_unchanged_all": all(x["actor_unchanged"] for x in branches),
        "target_actor_unchanged_all": all(x["target_actor_unchanged"] for x in branches),
        "optimizer_loaded_all": all(x["optimizer_initial_state_matches_loaded"] for x in branches),
        "all_completed": len(branches) == len(MODES) and all(
            x["completed_updates"] == args.updates for x in branches),
        "nonfinite_zero": nonfinite == 0,
    }
    status = "PASS" if all(validity.values()) else ("PARTIAL" if nonfinite else "INVALID")
    output = {
        "status": status, "experiment": "stage3_fixed_qmean_bootstrap_parameter_block_ablation",
        "device": str(device), "seed": args.seed, "batch_size": args.batch_size,
        "updates_per_branch": args.updates, "probe_size": len(probe),
        "schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(probe.tobytes()).hexdigest(),
        "stage2_checkpoint": str(stage2), "step0_checkpoint": str(step0_path),
        "replay": str(replay), "canonical_seed": canonical.get("seed"),
        "teacher_fixed": True, "target_actor_used": False,
        "twin_min_used": False, "environment_steps": 0,
        "actor_optimizer_steps": 0, "training_checkpoint_writes": 0,
        "nonfinite_count": nonfinite, "validity": validity, "branches": branches}
    out = Path(args.output).resolve() if args.output else (
        run / "testing/stage2_vs_stage3_readiness/multi_fixed_bootstrap_parameter_blocks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"[STATUS] {status} [SAVED] {out}", flush=True)
    if status == "INVALID":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
