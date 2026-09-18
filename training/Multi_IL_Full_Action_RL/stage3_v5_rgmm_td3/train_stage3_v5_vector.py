#!/usr/bin/env python3
"""16-env Stage3-v5 Critic-gated, gradual-handoff GMM TD3-style trainer."""
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

V3 = Path(__file__).resolve().parents[1] / "stage3_v3_rgmm_td3"
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))

ROOT = Path(__file__).resolve().parents[3]
OLD_STAGE3 = ROOT / "training" / "Multi_IL_Full_Action_RL" / "stage3_new_sac"
if str(OLD_STAGE3) not in sys.path:
    sys.path.insert(0, str(OLD_STAGE3))
from stage3_v5_vector_env import StaggeredVectorEnv  # noqa: E402


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
    parser.add_argument("--compile-backend", choices=("none", "torchair"), default=None)
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--benchmark-mode", choices=("A", "B", "C", "D"))
    parser.add_argument("--benchmark-warmup-steps", type=int, default=1024)
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
    # Checkpoints are loaded with map_location=agent.device for model and
    # optimizer tensors. CPU RNG state must nevertheless remain a CPU byte
    # tensor when passed to PyTorch (same for torch_npu RNG state).
    torch.set_rng_state(state["torch"].cpu())
    if "npu" in state:
        torch.npu.set_rng_state(state["npu"].cpu())


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
                       episodes, successes, online, torch, handoff=None, update_credit=0.0, offline=None):
    return {
        "stage": "stage3-v5", "group": group, "env_steps": int(env_steps),
        "updates": int(agent.critic_updates), "actor_updates": int(agent.actor_updates),
        "actor_enabled_critic_updates": int(agent.actor_enabled_critic_updates),
        "actor": agent.actor.state_dict(), "target_actor": agent.target_actor.state_dict(),
        "q1_q2": agent.critic.state_dict(), "target_q1_q2": agent.target_critic.state_dict(),
        "q1": agent.critic.q1.state_dict(), "q2": agent.critic.q2.state_dict(),
        "target_q1": agent.target_critic.q1.state_dict(),
        "target_q2": agent.target_critic.q2.state_dict(),
        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic_optimizer": agent.critic_optimizer.state_dict(),
        "actor_gate_open": bool(agent.actor_gate_open),
        "gate_open_step": agent.gate_open_step,
        "config": config, "rng_state": rng_state(torch),
        "action_normalization_stats": {
            "scale": agent.action_scale.detach().cpu().reshape(-1).tolist(),
            "offset": agent.action_offset.detach().cpu().reshape(-1).tolist()},
        "generations": list(map(int, generations)), "episodes": int(episodes),
        "successes": int(successes), "online_replay_transitions": int(online.transitions),
        "training_state": handoff.state.serialize() if handoff else None,
        "update_credit": float(update_credit),
        "pipeline_state": agent.pipeline.metrics() if hasattr(agent, "pipeline") else None,
        "rollout_snapshot_state": agent.rollout_executor.metrics() if hasattr(agent, "rollout_executor") else None,
        "offline_sampler_state": offline.state_dict() if offline and hasattr(offline, "state_dict") else None,
        "resume_semantics": "partial vector episodes are discarded and reset",
    }


def save_checkpoint(path, agent, config, group, env_steps, generations,
                    episodes, successes, online, torch, handoff=None, update_credit=0.0, offline=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    replay_path = path.with_suffix(".sequences.npy")
    online.save(replay_path)
    payload = checkpoint_payload(agent, config, group, env_steps, generations,
                                 episodes, successes, online, torch, handoff, update_credit, offline)
    payload["online_sequence_replay"] = str(replay_path)
    torch.save(payload, path)
    torch.save(payload, path.parent / "latest.pth")


def restore_checkpoint(path, agent, config, torch, OnlineSequenceReplay):
    payload = torch.load(path, map_location=agent.device)
    if payload.get("stage") != "stage3-v5":
        raise RuntimeError("Resume checkpoint is not Stage3-v5")
    if payload["config"].get("bc_rnn_checkpoint_sha256") != config.get("bc_rnn_checkpoint_sha256"):
        raise RuntimeError("Resume Actor source differs")
    for key in ("objective_revision", "rl_policy_expectation", "adaptive_bc_enabled", "bc_weight",
                "actor_q_scale_normalization", "boundary_aligned_sequence_sampling",
                "policy_delay", "utd",
                "recurrent_replay"):
        if payload["config"].get(key) != config.get(key):
            raise RuntimeError(f"Resume training objective differs: {key}")
    agent.actor.load_state_dict(payload["actor"], strict=True)
    agent.target_actor.load_state_dict(payload["target_actor"], strict=True)
    agent.critic.load_state_dict(payload["q1_q2"], strict=True)
    agent.target_critic.load_state_dict(payload["target_q1_q2"], strict=True)
    agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
    agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    agent.critic_updates = int(payload["updates"]); agent.actor_updates = int(payload["actor_updates"])
    agent.actor_enabled_critic_updates = int(payload.get("actor_enabled_critic_updates", 0))
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
    from prepare_stage3_v5_pair import validate_config
    validate_config(config)
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
    if args.smoke and args.benchmark_mode:
        raise RuntimeError("Smoke and benchmark are separate runs")
    if args.benchmark_mode and (num_envs not in (2,4,8,16) or total <= args.benchmark_warmup_steps):
        raise RuntimeError("Benchmark needs 2/4/8/16 envs and a measured window after warmup")
    if args.smoke:
        if num_envs > 2 or total > 12000:
            raise RuntimeError("Smoke requires <=2 envs and <=12000 aggregate steps")
    elif not args.benchmark_mode and num_envs != 16:
        raise RuntimeError("Formal Stage3-v5 requires exactly 16 environments")
    config["resolved_num_envs"] = num_envs
    config["resolved_total_env_steps"] = total
    config["run_type"] = "BENCHMARK" if args.benchmark_mode else ("SMOKE" if args.smoke else "FORMAL")
    if args.smoke:
        # Smoke alters only timing/thresholds to exercise all FSM edges; the
        # immutable JSON remains the formal configuration.
        config["critic_readiness"] = dict(config["critic_readiness"], min_online_steps=4,
                                           max_critic_only_steps=10000, check_interval_steps=2,
                                           min_completed_episodes=0, min_success_episodes=0,
                                           min_failure_episodes=0, min_spearman=-1.0, min_auc=0.0,
                                           max_twin_median=1.0, max_twin_p95=1.0,
                                           consecutive_passes=1, ood_max_excess_q=float("inf"))
        config["actor_warmup"] = dict(config["actor_warmup"])
        config["smoke_warmup_steps"] = 4

    group_dir = pair / args.group
    if args.benchmark_mode:
        group_dir = group_dir / "benchmarks" / f"{args.benchmark_mode}_{num_envs}_{time.time_ns()}"
    elif args.smoke:
        group_dir = group_dir / "smokes" / str(time.time_ns())
    for name in ("checkpoints", "evaluations", "diagnostics"):
        (group_dir / name).mkdir(parents=True, exist_ok=True)
    vector = eval_env = None
    stop_requested = False

    def stop(signum, _frame):
        nonlocal stop_requested
        print(f"[STAGE3-V5] signal {signum}; stopping after vector round", flush=True)
        stop_requested = True

    previous_int = signal.signal(signal.SIGINT, stop)
    previous_term = signal.signal(signal.SIGTERM, stop)
    try:
        vector = StaggeredVectorEnv(
            config["expert_dataset"], num_envs, config["train_seed_base"],
            delay=0.0,
            timeout=float(parallel["env_startup_timeout_sec"]),
            command_timeout=float(parallel["env_command_timeout_sec"]),
            start_method=parallel["multiprocessing_start_method"],
            startup_parallelism=int(parallel["startup_parallelism"]))

        import torch
        from stage3_v5_actor import load_exact_actor, module_hash, obs_to_flat
        from stage3_v5_agent import RecurrentGMMTD3, strict_stage2_load
        from stage3_v3_evaluation import build_env, close_env, evaluate_actor
        from stage3_v5_replay import (Stage1OfflineSequenceReplay, BalancedOfflineDemonstrations, OnlineSequenceReplay,
                                     final_transition, symmetric_sequence_batch,
                                     aligned_sequence_batch, _sample_aligned, source_sample_metrics)

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
        agent.profiler.enabled = bool(args.profile or args.smoke or args.benchmark_mode)
        profiler = agent.profiler
        if args.benchmark_mode in ("A", "B"):
            profiler.device = None
        from stage3_v5_execution import prepare_round_batches, enable_npu_compile
        from stage3_v5_schedule import CriticHandoff, HandoffState, TrainingState
        from stage3_v5_readiness import replay_metrics
        from stage3_v5_pipeline import TransitionCredit, overlap_burst, interval_overlap
        from stage3_v5_rollout import BoundarySnapshotExecutor
        from stage3_v5_diagnostics import isolated_training_rng
        handoff = CriticHandoff(config)
        critic_lr_ready = float(config["critic_lr"])
        update_credit = 0.0
        optimization = config.get("execution_optimization", {})
        prefetch = optimization.get("prefetch_minibatches", True) and not args.no_prefetch
        compile_backend = args.compile_backend or optimization.get("compile_backend", "none")
        if compile_backend == "torchair":
            enable_npu_compile(agent)
        initial_actor_hash = module_hash(actor)
        initial_critic_hash = module_hash(critic)
        if args.group == "rnn_q":
            offline = Stage1OfflineSequenceReplay(config["offline_sources"]["bc_rnn"], "rnn", config["training_seed"])
        else:
            paths = [config["offline_sources"][key] for key in ("bc_rnn", "bc_transformer", "bc_gmm")]
            if any(path is None for path in paths):
                raise RuntimeError("multi_q requires BC-RNN, BC-Transformer and BC-GMM replay sources")
            offline = BalancedOfflineDemonstrations(paths, config["training_seed"])
        with isolated_training_rng(offline=offline):
            fixed_diagnostics = _sample_aligned(
                offline, 64, int(config["recurrent_replay"]["train_seq_len"]), 10, purpose="diagnostic")
        online = OnlineSequenceReplay(config["online_sequence_capacity"], config["training_seed"])
        executor = BoundarySnapshotExecutor(actor, scale, offset, num_envs, 10)
        # Do not allocate a separate evaluation simulator until the Critic
        # handoff reaches JOINT_RL and an evaluation is actually due.
        eval_env = None

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
            observations_by_env = vector.reset_many(
                {env_id: contexts[env_id]["seed"] for env_id in range(num_envs)})
            observations = [observations_by_env[env_id] for env_id in range(num_envs)]
            executor.reset_indices(range(num_envs))
            handoff = CriticHandoff(config, HandoffState.restore(saved["training_state"]))
            update_credit = float(saved.get("update_credit", 0.0))
            if saved.get("offline_sampler_state") and hasattr(offline, "load_state_dict"):
                offline.load_state_dict(saved["offline_sampler_state"])

        credit = TransitionCredit(config["utd"], parallel["max_collector_lag_transitions"], update_credit)
        if args.resume and saved.get("pipeline_state"):
            for key in ("collector_transition_head", "learner_consumed_transition_equivalent",
                        "max_collector_lag_seen", "collector_throttle_count"):
                if key in saved["pipeline_state"]:
                    setattr(credit, key, saved["pipeline_state"][key])
        agent.pipeline = credit
        agent.rollout_executor = executor

        write_json(group_dir / "runtime_audit.json", {
            "stage": "stage3-v5", "group": args.group, "run_type": config["run_type"],
            "device": str(device), "num_envs": num_envs,
            "startup_parallelism": int(parallel["startup_parallelism"]),
            "env_steps_semantics": "aggregate environment transitions",
            "utd": float(config.get("utd", 0.25)),
            "policy_delay": int(config["policy_delay"]),
            "prefetch_minibatches": bool(prefetch), "compile_backend": compile_backend,
            "actor_update_schedule": "one Actor step per four Critic steps after readiness",
            "critic_batch": "128 offline + 128 online sequences; final transition",
            "handoff": "CRITIC_ONLY -> ACTOR_WARMUP -> JOINT_RL; evaluation only in JOINT_RL",
            "online_replay_capacity_configured": int(config["online_replay_capacity"]),
            "online_sequence_capacity_effective": int(config["online_sequence_capacity"]),
            "offline_sources": config.get("offline_source_metadata", config["offline_sources"]),
            "offline_sampler": ("RNN-only" if args.group == "rnn_q"
                                 else "balanced RNN/Transformer/GMM rotation"),
            "actor": metadata, "execution": config["exploration"]["execution"],
            "objective_revision": config["objective_revision"],
            "td_target": "r + gamma*(1-terminal)*sum_k p'_k*min(Q1',Q2')(s',mu'_k)",
            "actor_objective": "-mean[sum_k p_k*Q1(s,mu_k)]",
            "sampled_learned_std_q": "no_grad diagnostic only; excluded from losses",
            "actor_sequence_batch": int(config["recurrent_replay"]["actor_sequence_batch_size"]),
            "boundary_aligned_sequence_sampling": True,
            "boundary_evidence": "BatchedGMMExecutor resets hidden at episode timestep mod 10 == 0 and episode reset; replay stores contiguous episode_steps from 0; Actor windows begin at 0,10,... and never cross episodes",
            "rollout_std_contract": "BatchedGMMExecutor temporarily calls actor.eval(); checkpoint low_noise_eval=True fixes Gaussian component std to 1e-4; categorical mode is sampled.",
            "fixed_evaluation_std_contract": "evaluate_actor uses BatchedGMMExecutor, so eval mode and Gaussian std 1e-4 match online rollout.",
            "evaluation_seeds": list(config.get("smoke_evaluation_seeds", config["evaluation"]["seeds"][:2]) if args.smoke
                                     else config["evaluation"]["seeds"]),
            "smoke_evaluation_policy": ("2 seeds at final 12k only" if args.smoke
                                         else "10 fixed seeds at configured milestones"),
            "actor_update_std_contract": "Actor is in train mode and computes learned std, but raw-Q RL objective uses only categorical probabilities and component means; std head has no direct RL gradient.",
            "target_actor_contract": "Independent frozen Polyak-updated Actor remains eval with low_noise_eval=True; Bellman target enumerates component means under no_grad.",
            "replay_action_contract": "The exact denormalized action array returned by BatchedGMMExecutor is passed to vector.step and online.add; no external noise or clipping.",
            "rollout_snapshot_contract": "Separate rollout snapshots synchronize only at per-env recurrent reset boundaries; hidden and weights remain consistent for the entire block.",
            "online_cql": False, "critic_checkpoint": actual,
            "actor_checkpoint_sha256": file_hash(config["bc_rnn_checkpoint"]),
        })

        evaluation_steps = set()  # V5 gates formal evaluation through CriticHandoff.
        evaluation_seeds = (config.get("smoke_evaluation_seeds", config["evaluation"]["seeds"][:2])
                            if args.smoke else config["evaluation"]["seeds"])
        checkpoint_steps = set(range(int(config["checkpoint_interval_steps"]), total + 1,
                                     int(config["checkpoint_interval_steps"]))) | {total}
        if args.benchmark_mode:
            checkpoint_steps.add(int(args.benchmark_warmup_steps))
        best_success = -1.0

        def run_evaluation(step):
            nonlocal best_success, eval_env
            evaluation_started = time.monotonic()
            saved_rng = rng_state(torch)
            seed_all(config["training_seed"] + 7000000 + int(step), torch)
            try:
                if eval_env is None:
                    eval_env = build_env(config["expert_dataset"])
                report = evaluate_actor(
                    actor, scale, offset, eval_env, evaluation_seeds,
                    config["horizon"], config["sim_error_handling"]["evaluation_retry_count"],
                    vector.action_low if config["exploration"]["clip_to_env_bounds"] else None,
                    vector.action_high if config["exploration"]["clip_to_env_bounds"] else None)
            finally:
                restore_rng(saved_rng, torch)
            report.update({"stage": "stage3-v5", "group": args.group,
                           "env_steps": int(step), "actor_gate_open": agent.actor_gate_open})
            report["time_evaluation_sec"] = time.monotonic() - evaluation_started
            write_json(group_dir / "evaluations" / f"step_{step:07d}.json", report)
            diagnostics = agent.gmm_diagnostics(fixed_diagnostics)
            diagnostics.update({"stage": "stage3-v5", "group": args.group,
                                "env_steps": int(step)})
            write_json(group_dir / "diagnostics" / f"gmm_step_{step:07d}.json",
                       diagnostics)
            if report["success_rate"] is not None and report["success_rate"] > best_success:
                best_success = report["success_rate"]
                save_checkpoint(group_dir / "checkpoints" / "best_success.pth", agent,
                                config, args.group, step, generations, episodes,
                                successes, online, torch, handoff, credit.pending, offline)
            log_jsonl(group_dir / "gate_metrics.jsonl", {
                "env_steps": int(step), "eval_success_count": report["success_count"],
                "eval_success_rate": report["success_rate"], "competence_pass": True,
                "training_state": handoff.state.state.value,
                "gate_open": agent.actor_gate_open})

        if env_steps == 0 and not args.benchmark_mode:
            save_checkpoint(group_dir / "checkpoints" / "step0_transfer.pth", agent,
                            config, args.group, 0, generations, episodes, successes,
                            online, torch, handoff, credit.pending, offline)
            if 0 in evaluation_steps and not args.smoke:
                run_evaluation(0)

        print(f"[STAGE3-V5] group={args.group} num_envs={num_envs} "
              f"total_aggregate_env_steps={total} UTD={config['utd']} "
              f"policy_delay={int(config['policy_delay'])} mode={config['run_type']}", flush=True)
        metric_rows = []
        active_cursor = 0
        fatal_counts = [0] * num_envs
        last_report_step, last_report_time = env_steps, time.monotonic()
        last_report_critic_updates = agent.critic_updates
        last_report_actor_updates = agent.actor_updates
        prefetched_one = None
        def learn_once():
            nonlocal metric_rows, prefetched_one
            learn_once.last_device_interval = None
            length = int(config["recurrent_replay"]["critic_context_length"])
            if not online.can_sample(length):
                return False
            schedule = handoff.schedule(env_steps, critic_lr_ready)
            agent.set_learning_rates(schedule["actor_lr"], schedule["critic_lr"])
            agent.set_actor_training_enabled(schedule["actor_enabled"], env_steps)
            if args.benchmark_mode:
                agent.set_actor_training_enabled(False)
            actor_ready = agent.actor_gate_open and online.can_sample(10)
            # A single atomic burst cannot strand unused prefetched batches.
            with profiler.measure("learner_total_ms", device=True):
                if prefetched_one is not None:
                    critics, actors = prefetched_one
                    prefetched_one = None
                else:
                    with profiler.measure("batch_prefetch_ms", device=True):
                        critics, actors = prepare_round_batches(
                            offline, online, 0, config, device, actor_ready,
                            update_count=1, critic_updates=agent.critic_updates,
                            profiler=profiler, transfer=args.benchmark_mode != "B" and prefetch)
                if args.benchmark_mode == "B":
                    benchmark_counts["replay_updates"] += 1
                    return True
                profiler.synchronize()
                device_learner_started = time.perf_counter()
                collect_metrics = (agent.critic_updates + 1) % int(config["train_metrics_interval_updates"]) == 0
                with profiler.measure("critic_total_ms", device=True):
                    metrics = agent.critic_update(*critics[0], collect_metrics=collect_metrics)
                if 0 in actors:
                    with profiler.measure("actor_total_ms", device=True):
                        metrics.update(agent.actor_update(actors[0], env_steps,
                                                          collect_metrics=collect_metrics))
                with profiler.measure("polyak_ms", device=True):
                    agent.polyak_update()
                learn_once.last_device_interval = (device_learner_started, time.perf_counter())
            executor.train_policy_version = agent.actor_updates
            metrics.update({"env_steps": env_steps, "updates": agent.critic_updates,
                            "actor_updates": agent.actor_updates,
                            "actor_gate_open": agent.actor_gate_open,
                            "gate_open_step": agent.gate_open_step,
                            "offline_batch_fraction": 0.5, "online_batch_fraction": 0.5,
                            "actual_utd": agent.critic_updates / max(1, credit.collector_transition_head),
                            "online_samples": int(agent.critic_updates * 128),
                            **source_sample_metrics(offline), **credit.metrics(),
                            **executor.metrics()})
            if collect_metrics:
                metric_rows.append(metrics)
                log_jsonl(group_dir / "train_metrics.jsonl", aggregate(metric_rows))
                metric_rows = []
            return True

        def catch_up():
            credit.collector_throttle_count += 1
            with profiler.measure("collector_wait_ms", device=True):
                while credit.updates_due:
                    if not learn_once():
                        raise RuntimeError("Replay cannot satisfy pending credit; collector cannot advance safely")
                    credit.consume()

        benchmark_counts = {"replay_updates": 0}
        measured_overlap_ms = 0.0
        overlap_update_count = 0
        training_started = time.monotonic()
        training_start_steps = env_steps
        training_start_updates = agent.critic_updates
        training_start_actor_updates = agent.actor_updates
        benchmark_measuring = not args.benchmark_mode
        from stage3_v5_profile import cpu_snapshot, cpu_measurement
        cpu_start = cpu_snapshot(vector.processes)
        while env_steps < total and not stop_requested:
            round_started = time.perf_counter()
            boundaries = [step for step in evaluation_steps | checkpoint_steps | {total}
                          if step > env_steps]
            boundary = min(boundaries) if boundaries else total
            count = min(num_envs, total - env_steps, boundary - env_steps)
            if args.benchmark_mode != "A" and credit.must_throttle(count):
                catch_up()
            profile_round = profiler.enabled
            profile = {}
            active = [(active_cursor + index) % num_envs for index in range(count)]
            active_cursor = (active_cursor + count) % num_envs
            async_mode = args.benchmark_mode in (None, "D")
            if async_mode and prefetch and credit.updates_due and online.can_sample(11):
                schedule = handoff.schedule(env_steps, critic_lr_ready)
                agent.set_learning_rates(schedule["actor_lr"], schedule["critic_lr"])
                agent.set_actor_training_enabled(schedule["actor_enabled"] and not args.benchmark_mode, env_steps)
                # Prepare the guaranteed first atomic update before dispatch.
                # Its compute can start immediately while workers simulate,
                # rather than missing short env windows during sampling.
                with profiler.measure("batch_prefetch_ms", device=True):
                    prefetched_one = prepare_round_batches(
                        offline, online, 0, config, device,
                        agent.actor_gate_open and online.can_sample(10), update_count=1,
                        critic_updates=agent.critic_updates, profiler=profiler)
            with profiler.measure("actor_inference_ms", device=True):
                if args.benchmark_mode in ("A", "B"):
                    # A/B run a fixed, preloaded Stage1 action stream. No
                    # Actor or Critic device forward occurs in the timed loop.
                    episode = offline.episodes[0]
                    actions = [episode["actions"][contexts[i]["length"] % len(episode["actions"])]
                               for i in active]
                else:
                    actions = executor.actions_for(
                        active, [observations[index] for index in active],
                        config["exploration"]["external_action_noise_std"],
                        vector.action_low if config["exploration"]["clip_to_env_bounds"] else None,
                        vector.action_high if config["exploration"]["clip_to_env_bounds"] else None,
                        train_policy_version=agent.actor_updates)
            states = {env_id: obs_to_flat(observations[env_id]) for env_id in active}
            round_critic_start, round_actor_start = agent.critic_updates, agent.actor_updates
            with profiler.measure("env_dispatch_ms"):
                vector.step_async(actions, active)
            collector_dispatch_started = time.perf_counter()
            action_by_env = dict(zip(active, actions))
            learner_intervals = []
            if async_mode and credit.updates_due:
                used = overlap_burst(vector, credit, learn_once,
                                     max(1, int(math.ceil(count * config["utd"]))),
                                     learner_intervals)
                overlap_update_count += used
            elif async_mode:
                with profiler.measure("learner_wait_ms"):
                    pass
            with profiler.measure("env_wait_ms"):
                results = vector.step_wait(active)
            if profiler.enabled:
                wait_ms = profiler.totals.get("env_wait_ms", 0.0)
                profiler.totals["learner_wait_ms"] = wait_ms
                profiler.counts["learner_wait_ms"] = profiler.counts.get("env_wait_ms", 1)
            elapsed_env_ms = (time.perf_counter() - collector_dispatch_started) * 1000
            profiler.totals["vector_round_ms"] = profiler.totals.get("vector_round_ms", 0.0) + elapsed_env_ms
            profiler.counts["vector_round_ms"] = profiler.counts.get("vector_round_ms", 0) + 1
            # Device-complete timings are captured only by profile/smoke/benchmark.
            overlap_ms = interval_overlap(learner_intervals, results) if profiler.enabled else 0.0
            measured_overlap_ms += overlap_ms
            profile.update({"vector_env_step_ms": elapsed_env_ms,
                            "collector_learner_overlap_ms": overlap_ms,
                            "round_critic_updates": agent.critic_updates - round_critic_start})
            pending_resets = {}
            pending_rebuilds = {}
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
                    pending_rebuilds[env_id] = context["seed"]
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
                if args.benchmark_mode != "A":
                    with profiler.measure("online_replay_insert_ms"):
                        online.add(env_id, states[env_id], action_by_env[env_id], reward,
                                   next_flat, terminal, step_in_episode,
                                   terminated=terminal, truncated=truncated)
                env_steps += 1

                schedule = handoff.schedule(env_steps, critic_lr_ready)
                agent.set_learning_rates(schedule["actor_lr"], schedule["critic_lr"])
                agent.set_actor_training_enabled(schedule["actor_enabled"], env_steps)
                if args.benchmark_mode != "A" and env_steps >= config["min_online_replay_size"]:
                    credit.collect(1)


                observations[env_id] = next_observation
                if terminal or truncated:
                    online.finish(env_id, success=bool(won)); episodes += 1; successes += int(bool(won))
                    log_jsonl(group_dir / "episode_metrics.jsonl", {
                        "env_steps": env_steps, "env_id": env_id,
                        "episode_id": context["episode_id"], "seed": context["seed"],
                        "length": context["length"], "return": context["return"],
                        "success": bool(won), "terminated": terminal,
                        "truncated": truncated, "sim_error": False})
                    generations[env_id] += 1
                    context = new_context(config, num_envs, env_id, generations[env_id])
                    contexts[env_id] = context
                    pending_resets[env_id] = context["seed"]

                # Readiness metrics are defined on complete online episodes;
                # before the first one, defer the scheduled check rather than
                # treating a missing diagnostic sample as a training failure.
                if not args.benchmark_mode and handoff.readiness_due(env_steps) and online.episodes:
                    if credit.updates_due:
                        catch_up()
                    with profiler.measure("readiness_ms", device=True):
                        readiness = replay_metrics(agent, online, episodes, successes, config,
                                                   handoff.state.readiness_history, env_steps=env_steps, offline=offline)
                    record = handoff.submit_readiness(readiness)
                    log_jsonl(group_dir / "readiness_metrics.jsonl", record)
                    if record["critic_ready"]:
                        save_checkpoint(group_dir / "checkpoints" / "critic_ready.pth", agent,
                                        config, args.group, env_steps, generations, episodes,
                                        successes, online, torch, handoff, credit.pending, offline)
                if not args.benchmark_mode and handoff.fail_if_timed_out(env_steps):
                    save_checkpoint(group_dir / "checkpoints" / "critic_not_ready.pth", agent,
                                    config, args.group, env_steps, generations, episodes,
                                    successes, online, torch, handoff, credit.pending, offline)
                    stop_requested = True
                # Evaluations are prohibited until the warm-up completes.  The
                # transition itself schedules the first one immediately.
                if not args.benchmark_mode and handoff.evaluation_due(env_steps):
                    if credit.updates_due:
                        catch_up()
                    run_evaluation(env_steps)
                if not args.benchmark_mode and env_steps in checkpoint_steps:
                    save_checkpoint(group_dir / "checkpoints" / f"step_{env_steps:07d}.pth",
                                    agent, config, args.group, env_steps, generations,
                                    episodes, successes, online, torch, handoff, credit.pending, offline)

            # Reset all completed workers in one pipe round.  This avoids a
            # serial reset barrier when several of the 16 environments finish
            # in the same collector batch.
            reset_ids = []
            if pending_rebuilds:
                with profiler.measure("reset_many_ms"):
                    rebuilt = vector.reset_many(pending_rebuilds, rebuild=True)
                for env_id, observation in rebuilt.items():
                    observations[env_id] = observation
                reset_ids.extend(pending_rebuilds)
            if pending_resets:
                with profiler.measure("reset_many_ms"):
                    reset = vector.reset_many(pending_resets)
                for env_id, observation in reset.items():
                    observations[env_id] = observation
                reset_ids.extend(pending_resets)
            if reset_ids:
                executor.reset_indices(sorted(set(reset_ids)))

            if args.benchmark_mode in ("B", "C"):
                while credit.updates_due:
                    if not learn_once():
                        break
                    credit.consume()
            if args.benchmark_mode and not benchmark_measuring and env_steps >= args.benchmark_warmup_steps:
                if credit.updates_due:
                    catch_up()
                benchmark_measuring = True
                profiler.totals.clear(); profiler.counts.clear()
                measured_overlap_ms = 0.0; overlap_update_count = 0
                training_started = time.monotonic()
                training_start_steps, training_start_updates = env_steps, agent.critic_updates
                training_start_actor_updates = agent.actor_updates
                cpu_start = cpu_snapshot(vector.processes)

            if profile_round:
                log_jsonl(group_dir / "stage_timing.jsonl", {
                    "env_steps": env_steps, "group": args.group,
                    "policy_delay": int(config["policy_delay"]),
                    "sampled_vector_envs": count,
                    "sampled_critic_updates": agent.critic_updates - round_critic_start,
                    "sampled_actor_updates": agent.actor_updates - round_actor_start,
                    "round_critic_updates": agent.critic_updates - round_critic_start,
                    "round_actor_updates": agent.actor_updates - round_actor_start,
                    "round_wall_ms": 1000 * (time.perf_counter() - round_started),
                    "timing_semantics": "one sampled vector round; device synchronized at measured boundaries",
                    **credit.metrics(), **executor.metrics(),
                    "cumulative_profile": profiler.report(),
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
                    "critic_updates_per_sec": (agent.critic_updates - last_report_critic_updates)
                                              / max(now - last_report_time, 1e-9),
                    "actor_updates_per_sec": (agent.actor_updates - last_report_actor_updates)
                                             / max(now - last_report_time, 1e-9),
                    "phase": handoff.state.state.value,
                    "num_envs": num_envs})
                last_report_step, last_report_time = env_steps, now
                last_report_critic_updates = agent.critic_updates
                last_report_actor_updates = agent.actor_updates

        if args.benchmark_mode != "A" and credit.updates_due:
            catch_up()
        measured_seconds = time.monotonic() - training_started
        measured_steps = env_steps - training_start_steps
        measured_updates = agent.critic_updates - training_start_updates
        run_measurements = {
            "measurement_seconds": measured_seconds,
            "aggregate_steps_per_sec": measured_steps / max(measured_seconds, 1e-9),
            "collector_steps_per_sec": measured_steps / max(measured_seconds, 1e-9),
            "critic_updates_per_sec": measured_updates / max(measured_seconds, 1e-9),
            "actor_updates_per_sec": (agent.actor_updates - training_start_actor_updates) / max(measured_seconds, 1e-9),
            "effective_utd": agent.critic_updates / max(1, credit.collector_transition_head),
            "configured_policy_delay": int(config["policy_delay"]),
            "measured_overlap_ms": measured_overlap_ms,
            "overlap_critic_updates": overlap_update_count,
            "profile": profiler.report(), **credit.metrics(), **executor.metrics(),
            **source_sample_metrics(offline),
            "benchmark_mode": args.benchmark_mode, "num_envs": num_envs,
            "output_dir": str(group_dir),
            "measured_transitions": measured_steps,
            "action_stream": "preloaded Stage1" if args.benchmark_mode in ("A", "B") else "BC recurrent snapshot",
            "cpu": cpu_measurement(cpu_start, cpu_snapshot(vector.processes)),
            "effective_policy_delay": (agent.actor_enabled_critic_updates / agent.actor_updates
                                        if agent.actor_updates else None),
            "actor_enabled_critic_updates": agent.actor_enabled_critic_updates,
        }
        write_json(group_dir / "measurements.json", run_measurements)
        print(json.dumps({"measurements": run_measurements}), flush=True)
        if args.benchmark_mode:
            return
        if metric_rows:
            log_jsonl(group_dir / "train_metrics.jsonl", aggregate(metric_rows))
        status = "INTERRUPTED" if stop_requested and env_steps < total else "COMPLETE"
        last_path = group_dir / "checkpoints" / "last.pth"
        save_checkpoint(last_path, agent, config, args.group, env_steps,
                        generations, episodes, successes, online, torch, handoff, credit.pending, offline)
        if args.smoke:
            roundtrip = torch.load(last_path, map_location=device)
            replay_roundtrip = OnlineSequenceReplay.load(roundtrip["online_sequence_replay"])
            smoke_checks = {
                "checkpoint_load": roundtrip.get("stage") == "stage3-v5",
                "checkpoint_env_steps": int(roundtrip["env_steps"]) == env_steps,
                "actor_state_entries": len(roundtrip["actor"]) == len(actor.state_dict()),
                "target_actor_state_entries": len(roundtrip["target_actor"]) == len(agent.target_actor.state_dict()),
                "critic_state_entries": len(roundtrip["q1_q2"]) == len(critic.state_dict()),
                "sequence_replay_load": replay_roundtrip.transitions == online.transitions,
                "sequence_boundary_policy": config["recurrent_replay"]["boundary_policy"] == "same_episode_only",
                "actor_not_updated_before_handoff": (agent.actor_updates == 0 or agent.gate_open_step is not None),
                "critic_updated": module_hash(critic) != initial_critic_hash,
                "state_machine_present": handoff.state.state.value in {"CRITIC_ONLY", "ACTOR_WARMUP", "JOINT_RL", "CRITIC_NOT_READY"},
                "utd_contract": float(config["utd"]) == 0.25,
                "policy_delay_contract": int(config["policy_delay"]) == 4,
                "actual_cpu_learner_overlap": measured_overlap_ms > 0 and overlap_update_count > 0,
                "bounded_collector_lag": credit.max_collector_lag_seen <= credit.limit,
                "bounded_rollout_policy_lag": executor.max_policy_version_lag <= executor.max_policy_lag,
                "case_a_objective": config["objective_revision"] == "case-a-low-noise-component-mean-q",
                "finite_training": all(math.isfinite(float(value)) for value in
                                       (env_steps, agent.critic_updates, agent.actor_updates)),
            }
            write_json(group_dir / "smoke_validation.json", {
                "status": "PASS" if all(smoke_checks.values()) else "FAIL",
                "checks": smoke_checks, "resume_supported": True})
            if not all(smoke_checks.values()):
                raise RuntimeError("Stage3-v5 smoke roundtrip failed")
        write_json(group_dir / "summary.json", {
            "stage": "stage3-v5", "status": status, "run_type": config["run_type"],
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
