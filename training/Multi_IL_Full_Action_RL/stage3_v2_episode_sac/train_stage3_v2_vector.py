#!/usr/bin/env python3
"""16-environment Stage3-v2 trainer with episode-level behavior sources."""
from __future__ import annotations

import argparse
import copy
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
from stage3_v2_behavior import (  # noqa: E402
    bc_schedule,
    behavior_phase,
    new_episode_context,
    progressive_critic_schedule,
    validate_behavior_schedule,
)
from stage3_v2_replay import EpisodeReplay, SymmetricEpisodeSampler  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=("rnn_q", "multi_q"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--pair-run-dir", required=True)
    parser.add_argument("--critic-init-checkpoint", required=True)
    parser.add_argument("--total-env-steps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def log_jsonl(path, value):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contract_hash(config):
    ignored = {"resolved_device", "resolved_num_envs", "total_env_steps", "smoke"}
    payload = {key: value for key, value in config.items() if key not in ignored}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_config(config):
    validate_behavior_schedule(config["episode_behavior_schedule"])
    if int(config["batch_size"]) != 256 or int(config["utd"]) != 1:
        raise RuntimeError("Stage3-v2 requires batch_size=256 and UTD=1")
    if float(config["offline_fraction"]) != 0.5 or float(config["online_fraction"]) != 0.5:
        raise RuntimeError("Stage3-v2 replay must remain exactly 50/50")
    if config["cql"] != {
        "enabled": True, "lambda": 0.1, "num_random_actions": 10,
        "num_policy_actions": 1, "apply_to_expert": True,
        "apply_to_online": True, "detach_policy_actions": True,
        "source_diagnostic_interval_env_steps": 5000,
        "source_diagnostic_sample_size": 256,
    }:
        raise RuntimeError("Stage3-v2 CQL contract changed")
    progressive = config["progressive_critic_unfreeze"]
    if progressive != {
        "protected_until_env_steps": 10000,
        "unfreeze_end_env_steps": 30000,
        "critic_lr_schedule": "linear",
        "target_tau_schedule": "linear",
    }:
        raise RuntimeError("Stage3-v2 progressive Critic contract changed")


def apply_smoke_schedule(config):
    """Compress all algorithm phases into 2k steps without touching formal config."""
    config["formal_episode_behavior_schedule"] = copy.deepcopy(
        config["episode_behavior_schedule"]
    )
    config["formal_progressive_critic_unfreeze"] = copy.deepcopy(
        config["progressive_critic_unfreeze"]
    )
    config["formal_bc_regularization_schedule"] = copy.deepcopy(
        config["bc_regularization_schedule"]
    )
    config["episode_behavior_schedule"] = [
        {"name": "smoke_a_rnn_only", "start": 0, "end": 500, "rnn_fraction": 1.0},
        {"name": "smoke_b_half_rnn", "start": 500, "end": 1000, "rnn_fraction": 0.5},
        {"name": "smoke_c_quarter_rnn", "start": 1000, "end": 1500, "rnn_fraction": 0.25},
        {"name": "smoke_d_rl_only", "start": 1500, "end": None, "rnn_fraction": 0.0},
    ]
    config["progressive_critic_unfreeze"] = {
        "protected_until_env_steps": 500,
        "unfreeze_end_env_steps": 1000,
        "critic_lr_schedule": "linear",
        "target_tau_schedule": "linear",
    }
    config["bc_regularization_schedule"] = {
        **config["bc_regularization_schedule"],
        "bc_only_until_env_steps": 500,
        "decay_start_env_steps": 1000,
        "decay_end_env_steps": 1500,
    }
    config["smoke_phase_boundaries"] = {
        "protected_end": 500, "critic_full": 1000, "pure_sac": 1500
    }
    validate_behavior_schedule(config["episode_behavior_schedule"])


def resolve_device(name, torch):
    if name.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as error:
            raise RuntimeError("NPU requested but torch_npu is unavailable") from error
        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but torch.npu is unavailable")
        torch.npu.set_device(name)
    return torch.device(name)


def seed_all(seed, torch):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        npu.manual_seed_all(int(seed))


def rng_state(torch):
    result = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        result["npu"] = npu.get_rng_state()
    return result


def restore_rng(state, torch):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "npu" in state:
        torch.npu.set_rng_state(state["npu"])


def aggregate_metrics(rows):
    last = rows[-1]
    result = {
        "env_steps": last["env_steps"],
        "gradient_updates": last["gradient_updates"],
        "aggregation_updates": len(rows),
        "phase": last["phase"],
        "actor_objective": last["actor_objective"],
    }
    for key in set().union(*(row.keys() for row in rows)):
        if key in result:
            continue
        values = [
            float(row[key]) for row in rows
            if isinstance(row.get(key), (int, float))
            and not isinstance(row.get(key), bool)
            and math.isfinite(float(row[key]))
        ]
        if values:
            result[key] = float(np.mean(values))
        elif key in last and last[key] is None:
            result[key] = None
    return result


def milestones(config, key, total):
    result = {int(value) for value in config.get(key, []) if int(value) <= total}
    result.add(int(total))
    return result


def fixed_batch(source, count, seed):
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(int(source.size), min(int(count), int(source.size)), replace=False)
    return {key: value[indices].copy() for key, value in source.data.items()}


def behavior_report(window, cumulative, env_steps, config):
    phase = behavior_phase(config["episode_behavior_schedule"], env_steps)
    episode_total = window["rnn_episodes"] + window["rl_episodes"]
    transition_total = window["rnn_transitions"] + window["rl_transitions"]
    return {
        "env_steps": int(env_steps),
        "current_phase": phase["name"],
        "rnn_episodes_cumulative": int(cumulative["rnn_episodes"]),
        "rl_episodes_cumulative": int(cumulative["rl_episodes"]),
        "rnn_transitions_cumulative": int(cumulative["rnn_transitions"]),
        "rl_transitions_cumulative": int(cumulative["rl_transitions"]),
        "rnn_episode_fraction_window": (
            window["rnn_episodes"] / episode_total if episode_total else None
        ),
        "rl_episode_fraction_window": (
            window["rl_episodes"] / episode_total if episode_total else None
        ),
        "rnn_transition_fraction_window": (
            window["rnn_transitions"] / transition_total if transition_total else None
        ),
        "rl_transition_fraction_window": (
            window["rl_transitions"] / transition_total if transition_total else None
        ),
    }


def save_checkpoint(path, agent, replay, sampler, config, group, env_steps,
                    generations, counters, episodes_started, episodes_completed,
                    success_count, sim_error_aborts, torch):
    path = Path(path)
    replay_path = path.with_suffix(".replay.npz")
    replay.save(replay_path)
    payload = {
        "stage": "stage3-v2",
        "group": group,
        "env_steps": int(env_steps),
        "gradient_updates": int(agent.updates),
        "actor_state_dict": agent.actor.state_dict(),
        "critic_state_dict": agent.critic.state_dict(),
        "target_critic_state_dict": agent.target.state_dict(),
        "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
        "log_alpha": agent.log_alpha.detach().cpu(),
        "alpha_optimizer_state_dict": agent.alpha_optimizer.state_dict(),
        "config": config,
        "config_contract_hash": contract_hash(config),
        "replay_path": str(replay_path),
        "replay_rng_state": replay.rng.bit_generator.state,
        "sampler_rng_state": sampler.rng.bit_generator.state,
        "sampler_counts": sampler.counts,
        "offline_rng_state": sampler.offline.rng.bit_generator.state,
        "rng_state": rng_state(torch),
        "generations": list(map(int, generations)),
        "behavior_counters": counters,
        "episodes_started": int(episodes_started),
        "episodes_completed": int(episodes_completed),
        "success_count": int(success_count),
        "sim_error_aborts": int(sim_error_aborts),
        "resume_semantics": "restart all partial vector episodes with next generation",
    }
    torch.save(payload, path)
    torch.save(payload, path.parent / "latest.pth")


def restore_checkpoint(path, agent, config, offline, torch):
    payload = torch.load(path, map_location=agent.device)
    if payload.get("stage") != "stage3-v2":
        raise RuntimeError("Resume checkpoint is not Stage3-v2")
    if payload["config_contract_hash"] != contract_hash(config):
        raise RuntimeError("Resume checkpoint config contract mismatch")
    agent.actor.load_state_dict(payload["actor_state_dict"], strict=True)
    agent.critic.load_state_dict(payload["critic_state_dict"], strict=True)
    agent.target.load_state_dict(payload["target_critic_state_dict"], strict=True)
    agent.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
    agent.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
    agent.log_alpha.data.copy_(payload["log_alpha"].to(agent.device))
    agent.alpha_optimizer.load_state_dict(payload["alpha_optimizer_state_dict"])
    agent.updates = int(payload["gradient_updates"])
    replay = EpisodeReplay.load(payload["replay_path"])
    sampler = SymmetricEpisodeSampler(offline, replay, config["training_seed"])
    sampler.rng.bit_generator.state = payload["sampler_rng_state"]
    sampler.counts = payload["sampler_counts"]
    offline.rng.bit_generator.state = payload["offline_rng_state"]
    restore_rng(payload["rng_state"], torch)
    return payload, replay, sampler


def main():
    args = arguments()
    pair = Path(args.pair_run_dir).resolve()
    config = read_json(pair / "shared" / "config_resolved.json")
    validate_config(config)
    parallel = config["parallel_env"]
    num_envs = int(args.num_envs or parallel["num_envs"])
    total_env_steps = int(args.total_env_steps or config["total_env_steps"])
    if args.smoke:
        if num_envs > 2 or total_env_steps > 5000:
            raise RuntimeError("SMOKE requires --num-envs <=2 and --total-env-steps <=5000")
        config["smoke"] = True
        apply_smoke_schedule(config)
    elif num_envs != 16:
        raise RuntimeError("Formal Stage3-v2 requires exactly 16 environments")
    config["resolved_num_envs"] = num_envs
    config["total_env_steps"] = total_env_steps

    sources = read_json(pair / "shared" / "stage2_source_manifest.json")
    seed_manifest = read_json(pair / "shared" / "seed_manifest.json")
    expected_checkpoint = str(Path(sources[args.group]["checkpoint"]).resolve())
    actual_checkpoint = str(Path(args.critic_init_checkpoint).resolve())
    if actual_checkpoint != expected_checkpoint:
        raise RuntimeError(
            f"Critic checkpoint differs from prepared pair: {actual_checkpoint} != {expected_checkpoint}"
        )

    group_dir = pair / args.group
    for name in ("checkpoints", "evaluations", "probes", "actor_rnn_diagnostics"):
        (group_dir / name).mkdir(parents=True, exist_ok=True)

    vector = None
    eval_env = None
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        print(f"[STAGE3-V2] received signal {signum}; stopping after vector round", flush=True)
        stop_requested = True

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        vector = StaggeredVectorEnv(
            config["expert_dataset"], num_envs, config["train_seed_base"],
            delay=float(parallel["env_startup_delay_sec"]),
            timeout=float(parallel["env_startup_timeout_sec"]),
            start_method=parallel["multiprocessing_start_method"],
            command_timeout=float(parallel["env_command_timeout_sec"]),
        )

        import torch
        from stage3_new_dataset import ExpertDataset
        from stage3_new_evaluation import flatten
        from stage3_new_handoff import BatchedFrozenRNNProposer, FrozenRNNProposer
        from stage3_new_probe import load_fixed_probes, record_probe
        from stage3_v2_agent import Stage3V2SAC, build_actor, state_hash, strict_stage2_load
        from stage3_v2_evaluation import build_env, close_env, evaluate_frozen_rnn, evaluate_pure_actor

        device = resolve_device(args.device, torch)
        seed_all(config["training_seed"], torch)
        expert = ExpertDataset(
            config["expert_dataset"], config["training_seed"],
            config["expert_rnn_proposal_cache"],
        )
        actor = build_actor(config, device)
        actor_payload = torch.load(pair / "shared" / "actor_init.pth", map_location=device)
        actor.load_state_dict(actor_payload["actor_state_dict"], strict=True)
        if state_hash(actor) != actor_payload["actor_hash"]:
            raise RuntimeError("Shared Actor hash mismatch")
        pair_contract = read_json(pair / "shared" / "pair_contract.json")
        if not pair_contract["actor_hashes_identical"] or state_hash(actor) != pair_contract["actor_init"][f"{args.group}_hash"]:
            raise RuntimeError("Paired Actor initialization contract failed")
        critic, _ = strict_stage2_load(actual_checkpoint, device, config)
        agent = Stage3V2SAC(
            actor, critic, config, device, vector.action_low, vector.action_high
        )
        initial_actor_hash = state_hash(actor)
        initial_critic_hash = state_hash(critic)
        initial_target_hash = state_hash(agent.target)
        initial_alpha = float(agent.alpha.item())

        online = EpisodeReplay(
            config["online_replay_capacity"], 59, 14, config["training_seed"]
        )
        sampler = SymmetricEpisodeSampler(expert, online, config["training_seed"])
        proposer = BatchedFrozenRNNProposer(config["bc_rnn_checkpoint"], device, num_envs)
        eval_env = build_env(config["expert_dataset"])
        probes, _ = load_fixed_probes(config["stage2_run_dir"])
        fixed = fixed_batch(expert, 256, config["training_seed"] + 911)
        evaluation_seeds = (
            seed_manifest["evaluation_seeds"][:1]
            if args.smoke else seed_manifest["evaluation_seeds"]
        )

        evaluation_steps = milestones(config, "evaluation_env_steps", total_env_steps)
        checkpoint_steps = milestones(config, "checkpoint_env_steps", total_env_steps)
        probe_steps = milestones(config, "probe_env_steps", total_env_steps)
        actor_rnn_steps = milestones(config, "actor_rnn_diagnostic_env_steps", total_env_steps)
        behavior_interval = int(config["behavior_metrics_interval_env_steps"])
        behavior_steps = set(range(behavior_interval, total_env_steps + 1, behavior_interval))
        behavior_steps.add(total_env_steps)
        cql_interval = int(config["cql"]["source_diagnostic_interval_env_steps"])
        source_steps = set(range(cql_interval, total_env_steps + 1, cql_interval))
        source_steps.add(total_env_steps)
        progressive_boundaries = config["progressive_critic_unfreeze"]
        behavior_boundaries = {
            int(row["end"]) for row in config["episode_behavior_schedule"]
            if row.get("end") is not None
        }
        all_boundaries = (
            evaluation_steps | checkpoint_steps | probe_steps | actor_rnn_steps |
            behavior_steps | behavior_boundaries |
            {int(progressive_boundaries["protected_until_env_steps"]),
             int(progressive_boundaries["unfreeze_end_env_steps"]), total_env_steps}
        )

        observations = list(vector.initial_observations)
        generations = [0] * num_envs
        env_steps = 0
        episodes_started = num_envs
        episodes_completed = 0
        sim_error_aborts = 0
        success_count = 0
        counters = {
            "rnn_episodes": 0, "rl_episodes": 0,
            "rnn_transitions": 0, "rl_transitions": 0,
        }

        if args.resume:
            payload, online, sampler = restore_checkpoint(
                args.resume, agent, config, expert, torch
            )
            env_steps = int(payload["env_steps"])
            generations = [int(value) + 1 for value in payload["generations"]]
            counters = {key: int(value) for key, value in payload["behavior_counters"].items()}
            episodes_started = int(payload["episodes_started"]) + num_envs
            episodes_completed = int(payload["episodes_completed"])
            success_count = int(payload.get("success_count", 0))
            sim_error_aborts = int(payload["sim_error_aborts"])
            for env_id in range(num_envs):
                context = new_episode_context(config, num_envs, env_id, generations[env_id], env_steps)
                observations[env_id] = vector.reset(env_id, context["seed"])

        contexts = [
            new_episode_context(config, num_envs, env_id, generations[env_id], env_steps)
            for env_id in range(num_envs)
        ]
        for context in contexts:
            counters[f"{context['behavior_source']}_episodes"] += 1
        behavior_window = {key: 0 for key in counters}
        for context in contexts:
            behavior_window[f"{context['behavior_source']}_episodes"] += 1

        if args.group == "rnn_q" and not (pair / "shared" / "bc_rnn_baseline.json").exists():
            baseline_proposer = FrozenRNNProposer(config["bc_rnn_checkpoint"], device)
            saved = rng_state(torch)
            try:
                baseline = evaluate_frozen_rnn(
                    baseline_proposer, eval_env, evaluation_seeds,
                    config["horizon"], config["sim_error_handling"]["evaluation_retry_count"],
                )
            finally:
                restore_rng(saved, torch)
            baseline.update({"stage": "stage3-v2", "shared_once": True})
            write_json(pair / "shared" / "bc_rnn_baseline.json", baseline)
            del baseline_proposer

        write_json(group_dir / "runtime_audit.json", {
            "stage": "stage3-v2", "status": "SMOKE" if args.smoke else "FORMAL",
            "group": args.group, "device": str(device), "num_envs": num_envs,
            "total_aggregate_env_steps": total_env_steps, "utd": 1,
            "execution_policy": "episode_constant_rnn_or_stochastic_sac_actor",
            "td_target": "standard_sac_only",
            "formal_evaluation": "pure_deterministic_sac_actor",
            "stage2_checkpoint": actual_checkpoint,
            "actor_init_sha256": file_hash(pair / "shared" / "actor_init.pth"),
            "seed_manifest_sha256": file_hash(pair / "shared" / "seed_manifest.json"),
            "config_contract_hash": contract_hash(config),
        })
        print(
            f"[STAGE3-V2] group={args.group} num_envs={num_envs} "
            f"total_aggregate_env_steps={total_env_steps} UTD=1 "
            f"mode={'SMOKE' if args.smoke else 'FORMAL'}",
            flush=True,
        )

        def run_evaluation(step):
            saved = rng_state(torch)
            try:
                report = evaluate_pure_actor(
                    actor, eval_env, evaluation_seeds,
                    config["horizon"], device,
                    config["sim_error_handling"]["evaluation_retry_count"],
                )
            finally:
                restore_rng(saved, torch)
            report.update({"stage": "stage3-v2", "env_steps": int(step)})
            write_json(group_dir / "evaluations" / f"step_{step:07d}.json", report)

        def run_probe(step):
            summary = record_probe(
                actor, agent.critic, probes, device,
                group_dir / "probes" / f"step_{step:07d}.npz",
                vector.action_low, vector.action_high,
                int(config["cql"]["num_random_actions"]),
                config["training_seed"] + step,
            )
            write_json(group_dir / "probes" / f"step_{step:07d}.json", summary)

        if env_steps == 0:
            if 0 in evaluation_steps:
                run_evaluation(0)
            if 0 in probe_steps:
                run_probe(0)
            if 0 in actor_rnn_steps:
                write_json(
                    group_dir / "actor_rnn_diagnostics" / "step_0000000.json",
                    {"env_steps": 0, **agent.actor_vs_rnn_diagnostics(
                        fixed["observations"], fixed["action_rnn"]
                    )},
                )
            if 0 in checkpoint_steps:
                save_checkpoint(
                    group_dir / "checkpoints" / "step_0000000.pth", agent, online,
                    sampler, config, args.group, 0, generations, counters,
                    episodes_started, episodes_completed, success_count,
                    sim_error_aborts, torch,
                )

        consecutive_fatal = [0] * num_envs
        max_fatal = int(config["sim_error_handling"]["max_consecutive_fatal_errors"])
        metric_window = []
        latest_metrics = None
        active_cursor = 0
        protected_boundary = int(
            config["progressive_critic_unfreeze"]["protected_until_env_steps"]
        )
        protected_checked = env_steps >= protected_boundary
        started = time.monotonic()
        last_report_steps = env_steps
        last_report_time = started

        while env_steps < total_env_steps and not stop_requested:
            future = [value for value in all_boundaries if value > env_steps]
            boundary = min(future) if future else total_env_steps
            active_count = min(num_envs, total_env_steps - env_steps, boundary - env_steps)
            if active_count <= 0:
                active_count = min(num_envs, total_env_steps - env_steps)
            active = [(active_cursor + offset) % num_envs for offset in range(active_count)]
            active_cursor = (active_cursor + active_count) % num_envs
            states = np.stack([flatten(observations[env_id]) for env_id in active]).astype(np.float32)
            actions_by_env = {}
            teachers_by_env = {}
            rnn_ids = [env_id for env_id in active if contexts[env_id]["behavior_source"] == "rnn"]
            rl_ids = [env_id for env_id in active if contexts[env_id]["behavior_source"] == "rl"]
            if rnn_ids:
                rnn_actions = proposer.actions_for(rnn_ids, [observations[i] for i in rnn_ids])
                for env_id, action in zip(rnn_ids, rnn_actions):
                    actions_by_env[env_id] = action
                    teachers_by_env[env_id] = action
            if rl_ids:
                rl_states = torch.as_tensor(
                    np.stack([flatten(observations[i]) for i in rl_ids]),
                    dtype=torch.float32, device=device,
                )
                with torch.no_grad():
                    rl_actions = actor(
                        rl_states, reparameterize=True, return_log_prob=False
                    )[0].cpu().numpy()
                for env_id, action in zip(rl_ids, rl_actions):
                    actions_by_env[env_id] = action
                    teachers_by_env[env_id] = np.zeros(14, np.float32)
            actions = [actions_by_env[env_id] for env_id in active]
            results = vector.step(actions, active)
            state_by_env = {env_id: states[index] for index, env_id in enumerate(active)}

            for env_id, message in results:
                context = contexts[env_id]
                if message[0] == "FATAL":
                    consecutive_fatal[env_id] += 1
                    sim_error_aborts += 1
                    log_jsonl(group_dir / "sim_fatal_errors.jsonl", {
                        "aggregate_env_steps": env_steps, "env_id": env_id,
                        "episode_id": context["episode_id"],
                        "episode_seed": context["seed"],
                        "behavior_source": context["behavior_source"],
                        "exception": message[1], "status": "sim_error_abort",
                    })
                    if consecutive_fatal[env_id] >= max_fatal:
                        raise RuntimeError(f"env_id={env_id} reached fatal-error limit")
                    generations[env_id] += 1
                    context = new_episode_context(
                        config, num_envs, env_id, generations[env_id], env_steps
                    )
                    observations[env_id] = vector.reset(env_id, context["seed"], rebuild=True)
                    proposer.reset_indices([env_id])
                    contexts[env_id] = context
                    counters[f"{context['behavior_source']}_episodes"] += 1
                    behavior_window[f"{context['behavior_source']}_episodes"] += 1
                    episodes_started += 1
                    continue
                if message[0] != "OK":
                    raise RuntimeError(f"Unexpected env response: {message}")

                consecutive_fatal[env_id] = 0
                _, next_observation, reward, raw_done, won, _ = message
                if context["behavior_source"] not in ("rnn", "rl"):
                    raise RuntimeError("Episode behavior source changed or became invalid")
                context["length"] += 1
                context["return"] += float(reward)
                source = context["behavior_source"]
                counters[f"{source}_transitions"] += 1
                behavior_window[f"{source}_transitions"] += 1
                truncated = bool(context["length"] >= int(config["horizon"]) and not won)
                terminal = bool(
                    (config["terminate_on_success"] and won)
                    or (raw_done and not truncated)
                )
                online.add(
                    state_by_env[env_id], actions_by_env[env_id], reward,
                    flatten(next_observation), terminal,
                    {
                        "behavior_source": 0 if source == "rnn" else 1,
                        "action_rnn": teachers_by_env[env_id],
                        "env_id": env_id, "episode_id": context["episode_id"],
                        "episode_seed": context["seed"],
                        "behavior_phase": context["behavior_phase_index"],
                    },
                )
                env_steps += 1

                # Validate the complete [0, 10k) protected interval before
                # the env_steps=10000 update enables SAC Actor / alpha logic.
                if not protected_checked and env_steps == protected_boundary:
                    if state_hash(critic) != initial_critic_hash or state_hash(agent.target) != initial_target_hash:
                        raise RuntimeError("Phase A changed protected Critic or target Critic")
                    if not np.isclose(float(agent.alpha.item()), initial_alpha):
                        raise RuntimeError("Phase A changed fixed alpha")
                    if state_hash(actor) == initial_actor_hash:
                        raise RuntimeError("Phase A BC did not change Actor")
                    protected_checked = True

                if online.size >= int(config["min_online_replay_size"]):
                    metrics = agent.update(sampler.sample(256), env_steps)
                    fractions = sampler.fractions()
                    metrics.update({
                        "env_steps": env_steps,
                        "gradient_updates": agent.updates,
                        "online_replay_size": online.size,
                        "offline_batch_fraction": fractions["offline"],
                        "online_batch_fraction": fractions["online"],
                        "actual_utd_after_replay_ready": agent.updates / max(
                            1, env_steps - int(config["min_online_replay_size"]) + 1
                        ),
                    })
                    latest_metrics = metrics
                    metric_window.append(metrics)
                    if len(metric_window) >= int(config["train_metrics_interval_updates"]):
                        log_jsonl(group_dir / "train_metrics.jsonl", aggregate_metrics(metric_window))
                        metric_window = []

                observations[env_id] = next_observation
                if terminal or truncated:
                    episodes_completed += 1
                    success_count += int(bool(won))
                    log_jsonl(group_dir / "episode_metrics.jsonl", {
                        "aggregate_env_steps": env_steps, "env_id": env_id,
                        "episode_id": context["episode_id"],
                        "episode_seed": context["seed"],
                        "behavior_source": context["behavior_source"],
                        "behavior_phase_at_start": context["behavior_phase"],
                        "episode_length": context["length"],
                        "episode_return": context["return"],
                        "success": bool(won), "terminated": terminal,
                        "truncated": truncated,
                    })
                    generations[env_id] += 1
                    context = new_episode_context(
                        config, num_envs, env_id, generations[env_id], env_steps
                    )
                    observations[env_id] = vector.reset(env_id, context["seed"])
                    proposer.reset_indices([env_id])
                    contexts[env_id] = context
                    counters[f"{context['behavior_source']}_episodes"] += 1
                    behavior_window[f"{context['behavior_source']}_episodes"] += 1
                    episodes_started += 1

                if env_steps in behavior_steps:
                    log_jsonl(
                        group_dir / "behavior_metrics.jsonl",
                        behavior_report(behavior_window, counters, env_steps, config),
                    )
                    behavior_window = {key: 0 for key in behavior_window}
                if env_steps in actor_rnn_steps:
                    write_json(
                        group_dir / "actor_rnn_diagnostics" / f"step_{env_steps:07d}.json",
                        {"env_steps": env_steps, **agent.actor_vs_rnn_diagnostics(
                            fixed["observations"], fixed["action_rnn"]
                        )},
                    )
                if env_steps in probe_steps:
                    run_probe(env_steps)
                if env_steps in source_steps and online.size:
                    saved = rng_state(torch)
                    seed_all(config["training_seed"] + env_steps, torch)
                    try:
                        count = int(config["cql"]["source_diagnostic_sample_size"])
                        report = {
                            "env_steps": env_steps,
                            "expert": agent.source_diagnostics(fixed_batch(expert, count, env_steps)),
                            "online": agent.source_diagnostics(fixed_batch(online, count, env_steps)),
                        }
                    finally:
                        restore_rng(saved, torch)
                    log_jsonl(group_dir / "source_diagnostics.jsonl", report)
                if env_steps in checkpoint_steps:
                    save_checkpoint(
                        group_dir / "checkpoints" / f"step_{env_steps:07d}.pth",
                        agent, online, sampler, config, args.group, env_steps,
                        generations, counters, episodes_started, episodes_completed,
                        success_count, sim_error_aborts, torch,
                    )
                if env_steps in evaluation_steps:
                    run_evaluation(env_steps)

            if env_steps - last_report_steps >= 1000:
                now = time.monotonic()
                speed = (env_steps - last_report_steps) / max(now - last_report_time, 1e-9)
                schedule = progressive_critic_schedule(config, env_steps)
                bc_now = bc_schedule(config, env_steps)
                log_jsonl(group_dir / "throughput_metrics.jsonl", {
                    "env_steps": env_steps, "num_envs": num_envs,
                    "aggregate_env_steps_per_sec": speed,
                    "wall_time_sec": now - started,
                })
                print(
                    f"{args.group} env_steps={env_steps}/{total_env_steps} "
                    f"replay={online.size} updates={agent.updates} "
                    f"alpha={float(agent.alpha.item()):.6g} "
                    f"phase={schedule['phase']} bc_lambda={bc_now['lambda_bc']:.4g} "
                    f"steps_per_sec={speed:.2f}", flush=True,
                )
                last_report_steps = env_steps
                last_report_time = now

        if metric_window:
            log_jsonl(group_dir / "train_metrics.jsonl", aggregate_metrics(metric_window))
        status = "INTERRUPTED" if stop_requested and env_steps < total_env_steps else "COMPLETE"
        save_checkpoint(
            group_dir / "checkpoints" / "last.pth", agent, online, sampler,
            config, args.group, env_steps, generations, counters, episodes_started,
            episodes_completed, success_count, sim_error_aborts, torch,
        )
        write_json(group_dir / "summary.json", {
            "stage": "stage3-v2", "status": status,
            "run_type": "SMOKE" if args.smoke else "FORMAL",
            "group": args.group, "env_steps": env_steps,
            "gradient_updates": agent.updates,
            "episodes_started": episodes_started,
            "episodes_completed": episodes_completed,
            "success_count": success_count,
            "sim_error_aborts": sim_error_aborts,
            "behavior_counters": counters,
            "replay_size": online.size,
            "actor_hash_final": state_hash(actor),
            "critic_hash_final": state_hash(critic),
            "last_train_metrics": latest_metrics,
        })
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if eval_env is not None:
            try:
                close_env(eval_env)
            except BaseException:
                pass
        if vector is not None:
            vector.close()


if __name__ == "__main__":
    main()
