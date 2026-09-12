#!/usr/bin/env python3
"""16-env Stage3-v3 recurrent GMM TD3-style trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OLD_STAGE3 = ROOT / "training" / "Multi_IL_Full_Action_RL" / "stage3_new_sac"
if str(OLD_STAGE3) not in sys.path:
    sys.path.insert(0, str(OLD_STAGE3))
from stage3_progressive_vector_env import StaggeredVectorEnv  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=("rnn_q", "multi_q"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--pair-run-dir", required=True)
    parser.add_argument("--critic-init-checkpoint", required=True)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--total-env-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")
    os.replace(temporary, path)


def log_jsonl(path, value):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def profile_clock(torch, device):
    """Return a wall clock after pending device work has completed.

    Only use this in a sampled profiling round. Synchronizing on every
    transition would itself reduce training throughput.
    """
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(name, torch):
    if name.startswith("npu"):
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but unavailable")
        torch.npu.set_device(name)
    return torch.device(name)


def seed_all(seed, torch):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(int(seed))


def rng_state(torch):
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "torch": torch.get_rng_state()}
    if hasattr(torch, "npu") and torch.npu.is_available():
        result["npu"] = torch.npu.get_rng_state()
    return result


def restore_rng(state, torch):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "npu" in state:
        torch.npu.set_rng_state(state["npu"])


def new_context(config, num_envs, env_id, generation):
    return {"episode_id": int(generation * num_envs + env_id),
            "seed": int(config["train_seed_base"] + generation * num_envs + env_id),
            "length": 0, "return": 0.0}


def aggregate(rows):
    result = {"env_steps": rows[-1]["env_steps"],
              "updates": rows[-1]["updates"],
              "actor_updates": rows[-1]["actor_updates"],
              "actor_gate_open": rows[-1]["actor_gate_open"]}
    for key in set().union(*(row.keys() for row in rows)):
        if key in result:
            continue
        values = [float(row[key]) for row in rows
                  if isinstance(row.get(key), (int, float))
                  and not isinstance(row.get(key), bool)
                  and math.isfinite(float(row[key]))]
        result[key] = float(np.mean(values)) if values else rows[-1].get(key)
    return result


def milestones(config, key, total):
    values = {int(value) for value in config.get(key, []) if int(value) <= int(total)}
    values.add(int(total)); return values


def checkpoint_payload(agent, config, group, env_steps, generations,
                       episodes, successes, online, torch):
    return {
        "stage": "stage3-v3", "group": group, "env_steps": int(env_steps),
        "updates": int(agent.critic_updates), "actor_updates": int(agent.actor_updates),
        "actor": agent.actor.state_dict(), "target_actor": agent.target_actor.state_dict(),
        "q1_q2": agent.critic.state_dict(), "target_q1_q2": agent.target_critic.state_dict(),
        "q1": agent.critic.q1.state_dict(), "q2": agent.critic.q2.state_dict(),
        "target_q1": agent.target_critic.q1.state_dict(),
        "target_q2": agent.target_critic.q2.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "actor_gate_open": bool(agent.actor_gate_open),
        "gate_open_step": agent.gate_open_step, "lambda_bc_state": {
            "schedule": config["bc_lambda_schedule"], "env_steps": int(env_steps)},
        "config": config, "rng_state": rng_state(torch),
        "generations": list(map(int, generations)), "episodes": int(episodes),
        "successes": int(successes), "online_replay_transitions": int(online.transitions),
        "resume_semantics": "partial vector episodes are discarded and reset",
    }


def save_checkpoint(path, agent, config, group, env_steps, generations,
                    episodes, successes, online, torch):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    replay_path = path.with_suffix(".sequences.npy")
    online.save(replay_path)
    payload = checkpoint_payload(agent, config, group, env_steps, generations,
                                 episodes, successes, online, torch)
    payload["online_sequence_replay"] = str(replay_path)
    torch.save(payload, path)
    torch.save(payload, path.parent / "latest.pth")


def restore_checkpoint(path, agent, config, torch, OnlineSequenceReplay):
    payload = torch.load(path, map_location=agent.device)
    if payload.get("stage") != "stage3-v3":
        raise RuntimeError("Resume checkpoint is not Stage3-v3")
    if payload["config"]["bc_rnn_checkpoint_sha256"] != config["bc_rnn_checkpoint_sha256"]:
        raise RuntimeError("Resume Actor source differs")
    agent.actor.load_state_dict(payload["actor"], strict=True)
    agent.target_actor.load_state_dict(payload["target_actor"], strict=True)
    agent.critic.load_state_dict(payload["q1_q2"], strict=True)
    agent.target_critic.load_state_dict(payload["target_q1_q2"], strict=True)
    agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
    agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    agent.critic_updates = int(payload["updates"]); agent.actor_updates = int(payload["actor_updates"])
    agent.actor_gate_open = bool(payload["actor_gate_open"])
    agent.gate_open_step = payload["gate_open_step"]
    online = OnlineSequenceReplay.load(payload["online_sequence_replay"])
    online.current = {}
    restore_rng(payload["rng_state"], torch)
    return payload, online


def main():
    args = arguments()
    pair = Path(args.pair_run_dir).resolve()
    config = read_json(pair / "shared" / "config_resolved.json")
    phase0 = read_json(pair / "shared" / "phase0_gate.json")
    transfer = read_json(pair / "shared" / "transfer_validation.json")
    if not transfer["equivalence_pass"]:
        raise RuntimeError("TRANSFER_EQUIVALENCE_FAIL")
    if not phase0["competence_pass"]:
        raise RuntimeError("TRANSFER_COMPETENCE_FAIL")
    fairness = read_json(pair / "shared" / "pair_fairness.json")
    if not fairness["actor_hashes_identical"]:
        raise RuntimeError("PAIR_FAIRNESS_FAIL")
    sources = read_json(pair / "shared" / "stage2_source_manifest.json")
    expected = str(Path(sources[args.group]["checkpoint"]).resolve())
    actual = str(Path(args.critic_init_checkpoint).resolve())
    if actual != expected:
        raise RuntimeError(f"PAIR_FAIRNESS_FAIL: Critic path {actual} != {expected}")

    parallel = config["parallel_env"]
    num_envs = int(args.num_envs or parallel["num_envs"])
    total = int(args.total_env_steps or config["total_env_steps"])
    if args.smoke:
        if num_envs > 2 or total > 12000:
            raise RuntimeError("Smoke requires <=2 envs and <=12000 aggregate steps")
    elif num_envs != 16:
        raise RuntimeError("Formal Stage3-v3 requires exactly 16 environments")
    config["resolved_num_envs"] = num_envs
    config["resolved_total_env_steps"] = total
    config["run_type"] = "SMOKE" if args.smoke else "FORMAL"

    group_dir = pair / args.group
    for name in ("checkpoints", "evaluations", "diagnostics"):
        (group_dir / name).mkdir(parents=True, exist_ok=True)
    vector = eval_env = None
    stop_requested = False

    def stop(signum, _frame):
        nonlocal stop_requested
        print(f"[STAGE3-V3] signal {signum}; stopping after vector round", flush=True)
        stop_requested = True

    previous_int = signal.signal(signal.SIGINT, stop)
    previous_term = signal.signal(signal.SIGTERM, stop)
    try:
        vector = StaggeredVectorEnv(
            config["expert_dataset"], num_envs, config["train_seed_base"],
            delay=float(parallel["env_startup_delay_sec"]),
            timeout=float(parallel["env_startup_timeout_sec"]),
            command_timeout=float(parallel["env_command_timeout_sec"]),
            start_method=parallel["multiprocessing_start_method"])

        import torch
        from stage3_v3_actor import BatchedGMMExecutor, load_exact_actor, module_hash, obs_to_flat
        from stage3_v3_agent import RecurrentGMMTD3, strict_stage2_load
        from stage3_v3_evaluation import build_env, close_env, evaluate_actor
        from stage3_v3_replay import (OfflineDemonstrations, OnlineSequenceReplay,
                                     final_transition, symmetric_sequence_batch)

        device = resolve_device(args.device, torch)
        seed_all(config["training_seed"], torch)
        actor, rollout, metadata = load_exact_actor(config["bc_rnn_checkpoint"], device)
        actor_payload = torch.load(pair / "shared" / "actor_init.pth", map_location=device)
        incompatible = actor.load_state_dict(actor_payload["actor_state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("Strict Actor load failed")
        if module_hash(actor) != fairness["actor_hash"]:
            raise RuntimeError("PAIR_FAIRNESS_FAIL: Actor hash mismatch")
        scale = torch.as_tensor(rollout.action_normalization_stats["actions"]["scale"],
                                dtype=torch.float32, device=device).reshape(1, 1, 1, 14)
        offset = torch.as_tensor(rollout.action_normalization_stats["actions"]["offset"],
                                 dtype=torch.float32, device=device).reshape(1, 1, 1, 14)
        critic, _ = strict_stage2_load(actual, device)
        agent = RecurrentGMMTD3(actor, critic, config, device, scale, offset)
        initial_actor_hash = module_hash(actor)
        offline = OfflineDemonstrations(config["expert_dataset"], config["training_seed"])
        fixed_diagnostics = offline.sample_sequences(
            64, int(config["recurrent_replay"]["burn_in"])
                + int(config["recurrent_replay"]["train_seq_len"]))
        online = OnlineSequenceReplay(config["online_sequence_capacity"], config["training_seed"])
        executor = BatchedGMMExecutor(actor, scale, offset, num_envs, 10)
        eval_env = build_env(config["expert_dataset"])

        observations = list(vector.initial_observations)
        generations = [0] * num_envs
        contexts = [new_context(config, num_envs, env_id, 0) for env_id in range(num_envs)]
        env_steps = episodes = successes = 0
        if args.resume:
            saved, online = restore_checkpoint(args.resume, agent, config, torch,
                                               OnlineSequenceReplay)
            env_steps = int(saved["env_steps"]); episodes = int(saved["episodes"])
            successes = int(saved["successes"])
            generations = [int(value) + 1 for value in saved["generations"]]
            contexts = [new_context(config, num_envs, i, generations[i]) for i in range(num_envs)]
            for env_id in range(num_envs):
                observations[env_id] = vector.reset(env_id, contexts[env_id]["seed"])
            executor.reset_indices(range(num_envs))

        write_json(group_dir / "runtime_audit.json", {
            "stage": "stage3-v3", "group": args.group, "run_type": config["run_type"],
            "device": str(device), "num_envs": num_envs,
            "env_steps_semantics": "aggregate environment transitions", "utd": 1,
            "policy_delay": int(config["policy_delay"]),
            "critic_batch": "128 offline sequences + 128 online sequences; final transition",
            "actor": metadata, "execution": config["exploration"]["execution"],
            "td_target": "r + gamma*(1-terminal)*sum_k p'_k*min(Q1',Q2')(s',mu'_k)",
            "actor_objective": "-sum_k p_k*Q1(s,mu_k) + lambda_bc*GMM_NLL",
            "online_cql": False, "critic_checkpoint": actual,
            "actor_checkpoint_sha256": file_hash(config["bc_rnn_checkpoint"]),
        })

        evaluation_steps = milestones(config, "evaluation_env_steps", total)
        checkpoint_steps = milestones(config, "checkpoint_env_steps", total)
        best_success = -1.0

        def run_evaluation(step):
            nonlocal best_success
            saved_rng = rng_state(torch)
            seed_all(config["training_seed"] + 7000000 + int(step), torch)
            try:
                report = evaluate_actor(
                    actor, scale, offset, eval_env, config["evaluation_seeds"],
                    config["horizon"], config["sim_error_handling"]["evaluation_retry_count"],
                    vector.action_low if config["exploration"]["clip_to_env_bounds"] else None,
                    vector.action_high if config["exploration"]["clip_to_env_bounds"] else None)
            finally:
                restore_rng(saved_rng, torch)
            report.update({"stage": "stage3-v3", "group": args.group,
                           "env_steps": int(step), "actor_gate_open": agent.actor_gate_open})
            write_json(group_dir / "evaluations" / f"step_{step:07d}.json", report)
            diagnostics = agent.gmm_diagnostics(fixed_diagnostics)
            diagnostics.update({"stage": "stage3-v3", "group": args.group,
                                "env_steps": int(step)})
            write_json(group_dir / "diagnostics" / f"gmm_step_{step:07d}.json",
                       diagnostics)
            if report["success_rate"] is not None and report["success_rate"] > best_success:
                best_success = report["success_rate"]
                save_checkpoint(group_dir / "checkpoints" / "best_success.pth", agent,
                                config, args.group, step, generations, episodes,
                                successes, online, torch)
            log_jsonl(group_dir / "gate_metrics.jsonl", {
                "env_steps": int(step), "eval_success_count": report["success_count"],
                "eval_success_rate": report["success_rate"], "competence_pass": True,
                "warmup_pass": int(step) >= config["actor_gate"]["warmup_env_steps"],
                "gate_open": agent.actor_gate_open})

        if env_steps == 0:
            save_checkpoint(group_dir / "checkpoints" / "step0_transfer.pth", agent,
                            config, args.group, 0, generations, episodes, successes,
                            online, torch)
            if 0 in evaluation_steps:
                run_evaluation(0)

        print(f"[STAGE3-V3] group={args.group} num_envs={num_envs} "
              f"total_aggregate_env_steps={total} UTD=1 "
              f"policy_delay={int(config['policy_delay'])} mode={config['run_type']}", flush=True)
        metric_rows = []
        active_cursor = 0
        fatal_counts = [0] * num_envs
        last_report_step, last_report_time = env_steps, time.monotonic()
        while env_steps < total and not stop_requested:
            boundaries = [step for step in evaluation_steps | checkpoint_steps |
                          {int(config["actor_gate"]["warmup_env_steps"]), total}
                          if step > env_steps]
            boundary = min(boundaries) if boundaries else total
            count = min(num_envs, total - env_steps, boundary - env_steps)
            profile_round = env_steps // 1000 != (env_steps + count) // 1000
            profile = {}
            active = [(active_cursor + index) % num_envs for index in range(count)]
            active_cursor = (active_cursor + count) % num_envs
            if profile_round:
                profile_started = profile_clock(torch, device)
            actions = executor.actions_for(
                active, [observations[index] for index in active],
                config["exploration"]["external_action_noise_std"],
                vector.action_low if config["exploration"]["clip_to_env_bounds"] else None,
                vector.action_high if config["exploration"]["clip_to_env_bounds"] else None)
            if profile_round:
                profile["actor_inference_ms"] = 1000 * (
                    profile_clock(torch, device) - profile_started)
            states = {env_id: obs_to_flat(observations[env_id]) for env_id in active}
            if profile_round:
                profile_started = time.perf_counter()
            results = vector.step(actions, active)
            if profile_round:
                profile["vector_env_step_ms"] = 1000 * (
                    time.perf_counter() - profile_started)
            action_by_env = dict(zip(active, actions))
            for env_id, message in results:
                context = contexts[env_id]
                if message[0] == "FATAL":
                    fatal_counts[env_id] += 1
                    log_jsonl(group_dir / "sim_fatal_errors.jsonl", {
                        "env_steps": env_steps, "env_id": env_id,
                        "episode_id": context["episode_id"], "seed": context["seed"],
                        "exception": message[1]})
                    online.abort(env_id); generations[env_id] += 1
                    context = new_context(config, num_envs, env_id, generations[env_id])
                    contexts[env_id] = context
                    observations[env_id] = vector.reset(env_id, context["seed"], rebuild=True)
                    executor.reset_indices([env_id])
                    if fatal_counts[env_id] >= config["sim_error_handling"]["max_consecutive_fatal_errors"]:
                        raise RuntimeError(f"env_id={env_id} fatal error limit")
                    continue
                if message[0] != "OK":
                    raise RuntimeError(f"Unexpected vector response: {message}")
                fatal_counts[env_id] = 0
                _, next_observation, reward, raw_done, won, _ = message
                step_in_episode = context["length"]
                context["length"] += 1; context["return"] += float(reward)
                truncated = bool(context["length"] >= config["horizon"] and not won)
                terminal = bool((config["terminate_on_success"] and won)
                                or (raw_done and not truncated))
                next_flat = obs_to_flat(next_observation)
                online.add(env_id, states[env_id], action_by_env[env_id], reward,
                           next_flat, terminal, step_in_episode)
                env_steps += 1

                if (not agent.actor_gate_open
                        and env_steps == int(config["actor_gate"]["warmup_env_steps"])
                        and module_hash(actor) != initial_actor_hash):
                    raise RuntimeError("Actor changed during the complete 0-10k frozen interval")
                opened = agent.maybe_open_gate(env_steps, True, True)
                if opened:
                    log_jsonl(group_dir / "gate_metrics.jsonl", {
                        "env_steps": env_steps,
                        "eval_success_count": phase0["eval_success_count"],
                        "eval_success_rate": phase0["eval_success_rate"],
                        "competence_pass": True, "warmup_pass": True,
                        "gate_open": True, "transition": "LATCHED_FALSE_TO_TRUE"})
                    save_checkpoint(group_dir / "checkpoints" / "gate_open.pth", agent,
                                    config, args.group, env_steps, generations, episodes,
                                    successes, online, torch)

                context_length = int(config["recurrent_replay"]["critic_context_length"])
                if (env_steps >= config["min_online_replay_size"]
                        and online.can_sample(context_length)):
                    profile_critic = profile_round and "critic_update_ms" not in profile
                    if profile_critic:
                        profile_started = profile_clock(torch, device)
                    critic_sequences = symmetric_sequence_batch(
                        offline, online, 256, context_length)
                    critic_batch = final_transition(critic_sequences)
                    if profile_critic:
                        profile["critic_replay_ms"] = 1000 * (
                            profile_clock(torch, device) - profile_started)
                        profile_started = profile_clock(torch, device)
                    collect_metrics = (
                        (agent.critic_updates + 1)
                        % int(config["train_metrics_interval_updates"]) == 0
                    )
                    metrics = agent.critic_update(
                        critic_batch, critic_sequences,
                        collect_metrics=collect_metrics)
                    if profile_critic:
                        profile["critic_update_ms"] = 1000 * (
                            profile_clock(torch, device) - profile_started)
                    actor_metrics = {}
                    actor_length = (int(config["recurrent_replay"]["burn_in"])
                                    + int(config["recurrent_replay"]["train_seq_len"]))
                    if (agent.actor_gate_open
                            and agent.critic_updates % int(config["policy_delay"]) == 0
                            and online.can_sample(actor_length)):
                        profile_actor = profile_round and "actor_update_ms" not in profile
                        if profile_actor:
                            profile_started = profile_clock(torch, device)
                        actor_sequences = symmetric_sequence_batch(
                            offline, online,
                            int(config["recurrent_replay"]["actor_sequence_batch_size"]),
                            actor_length)
                        if profile_actor:
                            profile["actor_replay_ms"] = 1000 * (
                                profile_clock(torch, device) - profile_started)
                            profile_started = profile_clock(torch, device)
                        actor_metrics = agent.actor_update(
                            actor_sequences, env_steps,
                            collect_metrics=collect_metrics)
                        if profile_actor:
                            profile["actor_update_ms"] = 1000 * (
                                profile_clock(torch, device) - profile_started)
                    profile_polyak = profile_critic
                    if profile_polyak:
                        profile_started = profile_clock(torch, device)
                    agent.polyak_update()
                    if profile_polyak:
                        profile["polyak_update_ms"] = 1000 * (
                            profile_clock(torch, device) - profile_started)
                    metrics.update(actor_metrics)
                    metrics.update({"env_steps": env_steps, "updates": agent.critic_updates,
                                    "actor_updates": agent.actor_updates,
                                    "actor_gate_open": agent.actor_gate_open,
                                    "gate_open_step": agent.gate_open_step,
                                    "offline_batch_fraction": 0.5,
                                    "online_batch_fraction": 0.5,
                                    "actual_utd": agent.critic_updates / max(
                                        1, env_steps - config["min_online_replay_size"] + 1)})
                    metric_rows.append(metrics)
                    if len(metric_rows) >= config["train_metrics_interval_updates"]:
                        log_jsonl(group_dir / "train_metrics.jsonl", aggregate(metric_rows))
                        metric_rows = []

                observations[env_id] = next_observation
                if terminal or truncated:
                    online.finish(env_id); episodes += 1; successes += int(bool(won))
                    log_jsonl(group_dir / "episode_metrics.jsonl", {
                        "env_steps": env_steps, "env_id": env_id,
                        "episode_id": context["episode_id"], "seed": context["seed"],
                        "length": context["length"], "return": context["return"],
                        "success": bool(won), "terminated": terminal,
                        "truncated": truncated, "sim_error": False})
                    generations[env_id] += 1
                    context = new_context(config, num_envs, env_id, generations[env_id])
                    contexts[env_id] = context
                    observations[env_id] = vector.reset(env_id, context["seed"])
                    executor.reset_indices([env_id])

                if env_steps in checkpoint_steps:
                    save_checkpoint(group_dir / "checkpoints" / f"step_{env_steps:07d}.pth",
                                    agent, config, args.group, env_steps, generations,
                                    episodes, successes, online, torch)
                if env_steps in evaluation_steps:
                    run_evaluation(env_steps)

            if profile_round:
                log_jsonl(group_dir / "stage_timing.jsonl", {
                    "env_steps": env_steps, "group": args.group,
                    "policy_delay": int(config["policy_delay"]),
                    "sampled_vector_envs": count,
                    "sampled_critic_updates": int("critic_update_ms" in profile),
                    "sampled_actor_updates": int("actor_update_ms" in profile),
                    "timing_semantics": "one sampled vector round; device synchronized at measured boundaries",
                    **profile,
                })
            if env_steps - last_report_step >= 1000:
                now = time.monotonic()
                speed = (env_steps - last_report_step) / max(now - last_report_time, 1e-9)
                print(f"{args.group} env_steps={env_steps}/{total} updates={agent.critic_updates} "
                      f"actor_updates={agent.actor_updates} gate={agent.actor_gate_open} "
                      f"steps_per_sec={speed:.2f}", flush=True)
                log_jsonl(group_dir / "throughput_metrics.jsonl", {
                    "env_steps": env_steps, "aggregate_env_steps_per_sec": speed,
                    "num_envs": num_envs})
                last_report_step, last_report_time = env_steps, now

        if metric_rows:
            log_jsonl(group_dir / "train_metrics.jsonl", aggregate(metric_rows))
        status = "INTERRUPTED" if stop_requested and env_steps < total else "COMPLETE"
        last_path = group_dir / "checkpoints" / "last.pth"
        save_checkpoint(last_path, agent, config, args.group, env_steps,
                        generations, episodes, successes, online, torch)
        if args.smoke:
            roundtrip = torch.load(last_path, map_location=device)
            replay_roundtrip = OnlineSequenceReplay.load(roundtrip["online_sequence_replay"])
            smoke_checks = {
                "checkpoint_load": roundtrip.get("stage") == "stage3-v3",
                "checkpoint_env_steps": int(roundtrip["env_steps"]) == env_steps,
                "actor_state_entries": len(roundtrip["actor"]) == len(actor.state_dict()),
                "target_actor_state_entries": len(roundtrip["target_actor"]) == len(agent.target_actor.state_dict()),
                "critic_state_entries": len(roundtrip["q1_q2"]) == len(critic.state_dict()),
                "sequence_replay_load": replay_roundtrip.transitions == online.transitions,
                "sequence_boundary_policy": config["recurrent_replay"]["boundary_policy"] == "same_episode_only",
                "actor_frozen_below_10k": (True if env_steps >= 10000 else module_hash(actor) == initial_actor_hash),
                "gate_contract": agent.actor_gate_open == (env_steps >= 10000),
                "finite_training": all(math.isfinite(float(value)) for value in
                                       (env_steps, agent.critic_updates, agent.actor_updates)),
            }
            write_json(group_dir / "smoke_validation.json", {
                "status": "PASS" if all(smoke_checks.values()) else "FAIL",
                "checks": smoke_checks, "resume_supported": True})
            if not all(smoke_checks.values()):
                raise RuntimeError("Stage3-v3 smoke roundtrip failed")
        write_json(group_dir / "summary.json", {
            "stage": "stage3-v3", "status": status, "run_type": config["run_type"],
            "group": args.group, "env_steps": env_steps,
            "updates": agent.critic_updates, "actor_updates": agent.actor_updates,
            "actor_gate_open": agent.actor_gate_open,
            "gate_open_step": agent.gate_open_step, "episodes": episodes,
            "successes": successes, "actor_hash_final": module_hash(actor),
            "online_sequence_transitions": online.transitions})
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
        if eval_env is not None:
            try: close_env(eval_env)
            except BaseException: pass
        if vector is not None:
            vector.close()


if __name__ == "__main__":
    main()
