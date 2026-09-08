#!/usr/bin/env python3
"""16-environment trainer exclusively for Stage3 progressive RNN handoff.

Important invariants:
- 16 CPU MuJoCo workers are created with spawn, one-by-one with READY handshake.
- Torch / NPU models are imported only after all training workers are ready.
- env_steps counts valid aggregate transitions, not vector-loop iterations.
- UTD=1 means one Stage3SAC update call per valid transition after replay warmup.
- 10k / 30k progressive boundaries use aggregate env_steps.
- Every environment owns an independent BC-RNN recurrent-state row.
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np

from stage3_progressive_vector_env import StaggeredVectorEnv


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=("rnn_q", "multi_q"))
    parser.add_argument("--device", required=True)
    parser.add_argument("--pair-run-dir", required=True)
    parser.add_argument("--critic-init-checkpoint", required=True)
    parser.add_argument("--total-env-steps", type=int)
    parser.add_argument("--num-envs", type=int)
    return parser.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def episode_identity(base_seed, num_envs, env_id, generation):
    """Stable per-env seed stream, independent of episode completion order."""
    episode_id = int(env_id) + int(generation) * int(num_envs)
    episode_seed = int(base_seed) + episode_id
    return episode_id, episode_seed


def new_context(base_seed, num_envs, env_id, generation):
    episode_id, episode_seed = episode_identity(
        base_seed, num_envs, env_id, generation
    )
    return {
        "episode_id": episode_id,
        "seed": episode_seed,
        "generation": int(generation),
        "length": 0,
        "return": 0.0,
        "rnn": 0,
        "rl": 0,
    }


def next_phase_boundary(progressive, env_steps, total_env_steps):
    boundaries = [
        int(progressive["protected_until_env_steps"]),
        int(progressive["unfreeze_end_env_steps"]),
        int(total_env_steps),
    ]
    future = [value for value in boundaries if value > int(env_steps)]
    return min(future) if future else int(total_env_steps)


def save_checkpoint(
    path,
    agent,
    online,
    config,
    group,
    env_steps,
    episodes_started,
    episodes_completed,
    sim_error_aborts,
    stage2_checkpoint,
    vector_steps,
    seed_generations,
):
    import torch

    path = Path(path)
    replay = path.with_suffix(".replay.npz")
    online.save(replay)
    torch.save(
        {
            "env_steps": int(env_steps),
            "episodes": int(episodes_completed),
            "episodes_started": int(episodes_started),
            "episodes_completed": int(episodes_completed),
            "sim_error_aborts": int(sim_error_aborts),
            "vector_steps": int(vector_steps),
            "seed_generations": list(map(int, seed_generations)),
            "actor_state_dict": agent.actor.state_dict(),
            "critic_state_dict": agent.critic.state_dict(),
            "target_critic_state_dict": agent.target.state_dict(),
            "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(),
            "log_alpha": agent.log_alpha.detach().cpu(),
            "alpha_optimizer_state_dict": agent.alpha_optimizer.state_dict(),
            "gradient_updates": int(agent.updates),
            "config": config,
            "group": group,
            "stage2_checkpoint": stage2_checkpoint,
            "replay_path": str(replay),
        },
        path,
    )


def selector_report(window, cumulative, env_steps, schedule):
    qrl = np.asarray(window["q_rl"], dtype=np.float64)
    qrnn = np.asarray(window["q_rnn"], dtype=np.float64)
    margin = np.asarray(window["margin"], dtype=np.float64)
    selected = np.asarray(window["selected_rl"], dtype=np.float64)

    row = {
        "env_steps": int(env_steps),
        "window_transition_count": int(len(margin)),
        "phase": schedule["phase"],
        "critic_lr_effective": float(schedule["critic_lr_effective"]),
        "critic_lr_scale": float(schedule["critic_lr_scale"]),
        "target_tau_effective": float(schedule["target_tau_effective"]),
        "target_tau_scale": float(schedule["target_tau_scale"]),
        "rl_selected_count_cumulative": int(cumulative["rl"]),
        "rnn_selected_count_cumulative": int(cumulative["rnn"]),
    }
    if len(margin):
        row.update(
            {
                "q_select_rl_mean": float(qrl.mean()),
                "q_select_rnn_mean": float(qrnn.mean()),
                "q_select_margin_mean": float(margin.mean()),
                "q_select_margin_p10": float(np.percentile(margin, 10)),
                "q_select_margin_p50": float(np.percentile(margin, 50)),
                "q_select_margin_p90": float(np.percentile(margin, 90)),
                "rl_selected_fraction_window": float(selected.mean()),
                "rnn_selected_fraction_window": float(1.0 - selected.mean()),
            }
        )
    else:
        row.update(
            {
                "q_select_rl_mean": None,
                "q_select_rnn_mean": None,
                "q_select_margin_mean": None,
                "q_select_margin_p10": None,
                "q_select_margin_p50": None,
                "q_select_margin_p90": None,
                "rl_selected_fraction_window": None,
                "rnn_selected_fraction_window": None,
            }
        )
    return row


def clear_selector_window(window):
    for value in window.values():
        value.clear()


def validate_progressive_contract(cfg):
    progressive = cfg.get("progressive_unfreeze", {})
    parallel = cfg.get("parallel_env", {})

    if not progressive.get("enabled", False):
        raise RuntimeError(
            "train_stage3_progressive_vector.py is exclusive to the "
            "progressive-unfreeze experiment"
        )
    if int(progressive.get("protected_until_env_steps", -1)) != 10000:
        raise RuntimeError("Progressive protected boundary must be 10000")
    if int(progressive.get("unfreeze_end_env_steps", -1)) != 30000:
        raise RuntimeError("Progressive unfreeze boundary must be 30000")
    if progressive.get("critic_lr_schedule") != "linear":
        raise RuntimeError("Critic LR schedule must remain linear")
    if progressive.get("target_tau_schedule") != "linear":
        raise RuntimeError("Target tau schedule must remain linear")
    if progressive.get("phase_a_actor_objective") != "handoff_only":
        raise RuntimeError("Phase A Actor objective must remain handoff_only")
    if float(progressive.get("phase_a_alpha_fixed", -1.0)) != 0.01:
        raise RuntimeError("Phase A alpha must remain fixed at 0.01")
    if int(progressive.get("train_metrics_interval_updates", 0)) != 100:
        raise RuntimeError("train_metrics interval must remain 100 updates")

    if int(parallel.get("num_envs", -1)) <= 0:
        raise RuntimeError("parallel_env.num_envs must be positive")
    if not parallel.get("env_startup_stagger", False):
        raise RuntimeError("Staggered environment startup must remain enabled")
    if parallel.get("multiprocessing_start_method") != "spawn":
        raise RuntimeError("Progressive vector backend requires spawn")
    if float(parallel.get("env_startup_delay_sec", -1.0)) < 0:
        raise RuntimeError("env_startup_delay_sec must be non-negative")
    if float(parallel.get("env_startup_timeout_sec", 0.0)) <= 0:
        raise RuntimeError("env_startup_timeout_sec must be positive")


def main():
    args = arguments()
    pair = Path(args.pair_run_dir).resolve()
    cfg = read_json(pair / "shared" / "config_resolved.json")
    validate_progressive_contract(cfg)

    progressive = cfg["progressive_unfreeze"]
    parallel = cfg["parallel_env"]
    num_envs = int(
        args.num_envs
        if args.num_envs is not None
        else parallel["num_envs"]
    )
    if num_envs <= 0:
        raise ValueError("--num-envs must be positive")

    total_env_steps = int(
        args.total_env_steps
        if args.total_env_steps is not None
        else cfg["total_env_steps"]
    )
    if total_env_steps <= 0:
        raise ValueError("--total-env-steps must be positive")

    cfg["total_env_steps"] = total_env_steps
    cfg["resolved_num_envs"] = num_envs

    sources = read_json(pair / "shared" / "stage2_source_manifest.json")
    seeds = read_json(pair / "shared" / "seed_manifest.json")
    expected = str(Path(sources[args.group]["checkpoint"]).resolve())
    actual = str(Path(args.critic_init_checkpoint).resolve())
    if actual != expected:
        raise RuntimeError(
            f"Critic checkpoint differs from prepared pair:\n"
            f"expected={expected}\nactual={actual}"
        )

    group_dir = pair / args.group
    for name in ("checkpoints", "evaluations"):
        (group_dir / name).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Start CPU MuJoCo workers BEFORE importing torch / stage3 model code.
    # ------------------------------------------------------------------
    vector = None
    eval_env = None
    agent = None
    online = None
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        print(
            f"[STAGE3-VECTOR] received signal {signum}; "
            "will stop after the current vector round",
            flush=True,
        )
        stop_requested = True

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)

    try:
        vector = StaggeredVectorEnv(
            cfg["expert_dataset"],
            num_envs,
            cfg["train_seed_base"],
            delay=parallel["env_startup_delay_sec"],
            timeout=parallel["env_startup_timeout_sec"],
            start_method=parallel["multiprocessing_start_method"],
            command_timeout=parallel.get(
                "env_command_timeout_sec",
                parallel["env_startup_timeout_sec"],
            ),
        )

        # Heavy imports happen only after all environment workers are READY.
        import torch

        from stage3_new_agent import (
            Stage3SAC,
            build_actor,
            progressive_schedule,
            state_hash,
            strict_stage2_load,
        )
        from stage3_new_dataset import ExpertDataset
        from stage3_new_evaluation import build_env, close_env, flatten
        from stage3_new_handoff import (
            BatchedFrozenRNNProposer,
            FrozenRNNProposer,
            evaluate_handoff,
            select_actions,
        )
        from stage3_new_replay import SymmetricSampler, TransitionBuffer
        from train_stage3_new import (
            aggregate_metrics,
            device,
            file_hash,
            log,
            write,
        )

        # Verify the imported schedule implementation before any long run.
        checks = {
            0: ("protected_handoff", 0.0, 0.0),
            9999: ("protected_handoff", 0.0, 0.0),
            10000: ("progressive_unfreeze", 0.0, 0.0),
            20000: (
                "progressive_unfreeze",
                0.5 * float(cfg["critic_lr"]),
                0.5 * float(cfg["tau"]),
            ),
            30000: (
                "full_sac",
                float(cfg["critic_lr"]),
                float(cfg["tau"]),
            ),
        }
        for step, (phase, critic_lr, target_tau) in checks.items():
            schedule = progressive_schedule(cfg, step)
            if (
                schedule["phase"] != phase
                or not np.isclose(
                    schedule["critic_lr_effective"], critic_lr
                )
                or not np.isclose(
                    schedule["target_tau_effective"], target_tau
                )
            ):
                raise RuntimeError(
                    f"Progressive schedule contract failed at env_steps={step}: "
                    f"{schedule}"
                )

        dev = device(args.device)
        expert = ExpertDataset(
            cfg["expert_dataset"],
            cfg["training_seed"],
            cfg["expert_rnn_proposal_cache"],
        )
        actor = build_actor(cfg, dev)
        actor_payload = torch.load(
            pair / "shared" / "actor_init.pth",
            map_location=dev,
        )
        actor.load_state_dict(
            actor_payload["actor_state_dict"],
            strict=True,
        )
        if state_hash(actor) != actor_payload["actor_hash"]:
            raise RuntimeError("Shared Actor hash mismatch")

        critic, _stage2_payload = strict_stage2_load(actual, dev, cfg)
        agent = Stage3SAC(
            actor,
            critic,
            cfg,
            dev,
            action_low=vector.action_low,
            action_high=vector.action_high,
        )
        online = TransitionBuffer(
            cfg["online_replay_capacity"],
            cfg["obs_dim"],
            cfg["action_dim"],
            cfg["training_seed"],
        )
        sampler = SymmetricSampler(
            expert,
            online,
            cfg["training_seed"],
        )
        proposer = BatchedFrozenRNNProposer(
            cfg["bc_rnn_checkpoint"],
            dev,
            num_envs,
        )

        observations = list(vector.initial_observations)
        rnn_actions = proposer.actions(observations)

        generations = [0 for _ in range(num_envs)]
        contexts = [
            new_context(
                cfg["train_seed_base"],
                num_envs,
                env_id,
                generations[env_id],
            )
            for env_id in range(num_envs)
        ]
        episodes_started = num_envs
        episodes_completed = 0
        sim_error_aborts = 0
        success_count = 0

        consecutive_fatal = [0 for _ in range(num_envs)]
        max_consecutive_fatal = int(
            cfg["sim_error_handling"]["max_consecutive_fatal_errors"]
        )

        selector_counts = {"rnn": 0, "rl": 0}
        selector_window = {
            "q_rl": [],
            "q_rnn": [],
            "margin": [],
            "selected_rl": [],
        }

        started = time.monotonic()
        timing = {
            key: 0.0
            for key in (
                "rollout",
                "env",
                "actor",
                "rnn",
                "selector",
                "replay",
                "update",
            )
        }
        metric_window = []
        env_steps = 0
        last_timing_step = 0
        last_timing_time = started
        last_timing_vector_steps = 0
        latest_episode_success = None
        active_cursor = 0

        eval_env = build_env(cfg["expert_dataset"])
        eval_proposer = FrozenRNNProposer(
            cfg["bc_rnn_checkpoint"],
            dev,
        )

        evaluation_steps = set(map(int, cfg["evaluation_env_steps"]))
        evaluation_steps.add(total_env_steps)
        checkpoint_steps = set(map(int, cfg["checkpoint_env_steps"]))
        checkpoint_steps.add(total_env_steps)
        selector_steps = set(
            map(int, cfg["selector_diagnostic_env_steps"])
        )
        selector_steps.add(total_env_steps)

        write(
            group_dir / "runtime_audit.json",
            {
                "group": args.group,
                "device": str(dev),
                "num_envs": num_envs,
                "vector_backend": (
                    "spawn subprocess pipes; sequential STARTING/READY handshake"
                ),
                "actor_sha256": file_hash(
                    pair / "shared" / "actor_init.pth"
                ),
                "stage2_checkpoint": actual,
                "aggregate_env_steps": True,
                "utd": 1,
                "seed_protocol": (
                    "seed = train_seed_base + env_id + "
                    "generation * num_envs"
                ),
                "progressive_unfreeze": progressive,
                "parallel_env": parallel,
                "total_env_steps": total_env_steps,
            },
        )

        print(
            f"[STAGE3-VECTOR] group={args.group} "
            f"num_envs={num_envs} "
            f"total_aggregate_env_steps={total_env_steps} "
            f"UTD=1",
            flush=True,
        )

        # Preserve config semantics for step-0 checkpoint/evaluation.
        if 0 in checkpoint_steps:
            save_checkpoint(
                group_dir / "checkpoints" / "step_000000.pth",
                agent,
                online,
                cfg,
                args.group,
                env_steps,
                episodes_started,
                episodes_completed,
                sim_error_aborts,
                actual,
                vector.vector_steps,
                generations,
            )
        if 0 in evaluation_steps:
            report = evaluate_handoff(
                actor,
                agent.target,
                eval_proposer,
                eval_env,
                seeds["evaluation_seeds"],
                cfg["horizon"],
                dev,
                cfg["sim_error_handling"]["evaluation_retry_count"],
            )
            write(
                group_dir / "evaluations" / "step_000000.json",
                report,
            )

        phase_before = progressive_schedule(cfg, 0)["phase"]

        while env_steps < total_env_steps and not stop_requested:
            round_start = time.monotonic()

            # Do not let one vector action batch straddle the 10k / 30k
            # phase boundaries.  This removes an avoidable 16-step policy lag
            # exactly where the training objective changes.
            boundary = next_phase_boundary(
                progressive,
                env_steps,
                total_env_steps,
            )
            active_count = min(
                num_envs,
                total_env_steps - env_steps,
                boundary - env_steps,
            )
            if active_count <= 0:
                active_count = min(
                    num_envs,
                    total_env_steps - env_steps,
                )
            active = [
                (active_cursor + offset) % num_envs
                for offset in range(int(active_count))
            ]
            active_cursor = (active_cursor + int(active_count)) % num_envs
            active_pos = {
                env_id: local
                for local, env_id in enumerate(active)
            }

            states = np.stack(
                [flatten(observations[env_id]) for env_id in active]
            ).astype(np.float32, copy=False)
            states_t = torch.as_tensor(
                states,
                dtype=torch.float32,
                device=dev,
            )

            tick = time.monotonic()
            with torch.no_grad():
                rl_actions_t = actor(
                    states_t,
                    reparameterize=True,
                    return_log_prob=False,
                )[0]
            timing["actor"] += time.monotonic() - tick

            tick = time.monotonic()
            rnn_actions_t = torch.as_tensor(
                rnn_actions[active],
                dtype=torch.float32,
                device=dev,
            )
            timing["rnn"] += time.monotonic() - tick

            tick = time.monotonic()
            chosen_t, qrl_t, qrnn_t, rl_wins_t = select_actions(
                agent.target,
                states_t,
                rl_actions_t,
                rnn_actions_t,
            )
            timing["selector"] += time.monotonic() - tick

            actions = chosen_t.cpu().numpy()
            rl_actions_np = rl_actions_t.cpu().numpy()
            qrl_np = qrl_t.cpu().numpy()
            qrnn_np = qrnn_t.cpu().numpy()
            rl_wins_np = rl_wins_t.cpu().numpy()

            tick = time.monotonic()
            results = vector.step(actions, active)
            timing["env"] += time.monotonic() - tick

            valid = [
                (env_id, message)
                for env_id, message in results
                if message[0] == "OK"
            ]
            next_rnn_by_id = {}
            if valid:
                valid_ids = [env_id for env_id, _ in valid]
                next_obs_list = [message[1] for _, message in valid]
                tick = time.monotonic()
                next_rnn = proposer.actions_for(
                    valid_ids,
                    next_obs_list,
                )
                timing["rnn"] += time.monotonic() - tick
                next_rnn_by_id = {
                    env_id: action
                    for env_id, action in zip(valid_ids, next_rnn)
                }

            for env_id, message in results:
                if message[0] == "FATAL":
                    consecutive_fatal[env_id] += 1
                    sim_error_aborts += 1
                    ctx = contexts[env_id]
                    log(
                        group_dir / "sim_fatal_errors.jsonl",
                        {
                            "aggregate_env_steps": env_steps,
                            "env_id": env_id,
                            "episode_id": ctx["episode_id"],
                            "episode_seed": ctx["seed"],
                            "episode_length_before_abort": ctx["length"],
                            "consecutive_fatal_errors_for_env": (
                                consecutive_fatal[env_id]
                            ),
                            "exception": message[1],
                            "status": "sim_error_abort",
                        },
                    )
                    if consecutive_fatal[env_id] >= max_consecutive_fatal:
                        raise RuntimeError(
                            f"env_id={env_id} reached "
                            f"{consecutive_fatal[env_id]} consecutive "
                            "MuJoCo FatalErrors"
                        )

                    generations[env_id] += 1
                    new_ctx = new_context(
                        cfg["train_seed_base"],
                        num_envs,
                        env_id,
                        generations[env_id],
                    )
                    observations[env_id] = vector.reset(
                        env_id,
                        new_ctx["seed"],
                        rebuild=True,
                    )
                    proposer.reset_indices([env_id])
                    rnn_actions[env_id] = proposer.actions_for(
                        [env_id],
                        [observations[env_id]],
                    )[0]
                    contexts[env_id] = new_ctx
                    episodes_started += 1
                    continue

                if message[0] != "OK":
                    raise RuntimeError(
                        f"env_id={env_id} unexpected result: {message}"
                    )

                consecutive_fatal[env_id] = 0
                (
                    _,
                    next_observation,
                    reward,
                    raw_done,
                    won,
                    _info,
                ) = message

                local = active_pos[env_id]
                ctx = contexts[env_id]
                ctx["length"] += 1
                ctx["return"] += float(reward)

                selected_rl = int(bool(rl_wins_np[local]))
                source = "rl" if selected_rl else "rnn"
                ctx[source] += 1
                selector_counts[source] += 1

                selector_window["q_rl"].append(float(qrl_np[local]))
                selector_window["q_rnn"].append(float(qrnn_np[local]))
                selector_window["margin"].append(
                    float(qrl_np[local] - qrnn_np[local])
                )
                selector_window["selected_rl"].append(selected_rl)

                truncated = bool(
                    ctx["length"] >= int(cfg["horizon"]) and not won
                )
                terminal = bool(
                    (cfg["terminate_on_success"] and won)
                    or (raw_done and not truncated)
                )

                metadata = {
                    "action_exec": actions[local],
                    "action_rl": rl_actions_np[local],
                    "action_rnn": rnn_actions[env_id],
                    "rnn_next_actions": next_rnn_by_id[env_id],
                    "selected_source": [selected_rl],
                    "q_select_rl": [qrl_np[local]],
                    "q_select_rnn": [qrnn_np[local]],
                    "q_select_margin": [
                        qrl_np[local] - qrnn_np[local]
                    ],
                    "is_online": [1],
                    "env_id": [env_id],
                    "episode_id": [ctx["episode_id"]],
                    "episode_seed": [ctx["seed"]],
                }

                tick = time.monotonic()
                online.add(
                    states[local],
                    actions[local],
                    reward,
                    flatten(next_observation),
                    terminal,
                    metadata,
                )
                env_steps += 1
                timing["replay"] += time.monotonic() - tick

                if online.size >= int(cfg["min_online_replay_size"]):
                    tick = time.monotonic()
                    metrics = agent.update(
                        sampler.sample(cfg["batch_size"]),
                        env_steps=env_steps,
                    )
                    timing["update"] += time.monotonic() - tick

                    update_eligible_transitions = (
                        env_steps
                        - int(cfg["min_online_replay_size"])
                        + 1
                    )
                    fractions = sampler.fractions()
                    metrics.update(
                        {
                            "env_steps": env_steps,
                            "gradient_updates": agent.updates,
                            "online_replay_size": online.size,
                            "offline_batch_fraction": fractions["offline"],
                            "online_batch_fraction": fractions["online"],
                            "actual_utd_after_replay_ready": (
                                agent.updates
                                / max(1, update_eligible_transitions)
                            ),
                        }
                    )
                    metric_window.append(metrics)
                    if len(metric_window) >= int(
                        progressive["train_metrics_interval_updates"]
                    ):
                        log(
                            group_dir / "train_metrics.jsonl",
                            aggregate_metrics(metric_window),
                        )
                        metric_window = []

                observations[env_id] = next_observation
                rnn_actions[env_id] = next_rnn_by_id[env_id]

                if terminal or truncated:
                    episodes_completed += 1
                    success_count += int(bool(won))
                    latest_episode_success = bool(won)
                    log(
                        group_dir / "episode_metrics.jsonl",
                        {
                            "aggregate_env_steps": env_steps,
                            "env_id": env_id,
                            "episode_id": ctx["episode_id"],
                            "episode_seed": ctx["seed"],
                            "episode_length": ctx["length"],
                            "episode_return": ctx["return"],
                            "success": bool(won),
                            "terminated": bool(terminal),
                            "truncated": bool(truncated),
                            "rnn_selected_fraction": (
                                ctx["rnn"] / ctx["length"]
                            ),
                            "rl_selected_fraction": (
                                ctx["rl"] / ctx["length"]
                            ),
                        },
                    )

                    generations[env_id] += 1
                    new_ctx = new_context(
                        cfg["train_seed_base"],
                        num_envs,
                        env_id,
                        generations[env_id],
                    )
                    observations[env_id] = vector.reset(
                        env_id,
                        new_ctx["seed"],
                    )
                    proposer.reset_indices([env_id])
                    rnn_actions[env_id] = proposer.actions_for(
                        [env_id],
                        [observations[env_id]],
                    )[0]
                    contexts[env_id] = new_ctx
                    episodes_started += 1

                schedule_now = progressive_schedule(cfg, env_steps)
                if schedule_now["phase"] != phase_before:
                    print(
                        f"[STAGE3-PROGRESSIVE] env_steps={env_steps} "
                        f"{phase_before} -> {schedule_now['phase']} "
                        f"critic_lr={schedule_now['critic_lr_effective']:.8g} "
                        f"target_tau={schedule_now['target_tau_effective']:.8g}",
                        flush=True,
                    )
                    phase_before = schedule_now["phase"]

                if env_steps in selector_steps:
                    log(
                        group_dir / "selector_diagnostics.jsonl",
                        selector_report(
                            selector_window,
                            selector_counts,
                            env_steps,
                            schedule_now,
                        ),
                    )
                    clear_selector_window(selector_window)

                if env_steps in checkpoint_steps:
                    save_checkpoint(
                        group_dir
                        / "checkpoints"
                        / f"step_{env_steps:06d}.pth",
                        agent,
                        online,
                        cfg,
                        args.group,
                        env_steps,
                        episodes_started,
                        episodes_completed,
                        sim_error_aborts,
                        actual,
                        vector.vector_steps,
                        generations,
                    )

                if env_steps in evaluation_steps:
                    report = evaluate_handoff(
                        actor,
                        agent.target,
                        eval_proposer,
                        eval_env,
                        seeds["evaluation_seeds"],
                        cfg["horizon"],
                        dev,
                        cfg["sim_error_handling"][
                            "evaluation_retry_count"
                        ],
                    )
                    write(
                        group_dir
                        / "evaluations"
                        / f"step_{env_steps:06d}.json",
                        report,
                    )

            timing["rollout"] += time.monotonic() - round_start

            if env_steps - last_timing_step >= 1000:
                now = time.monotonic()
                delta_steps = env_steps - last_timing_step
                delta_vector_steps = (
                    vector.vector_steps - last_timing_vector_steps
                )
                interval_wall = max(
                    now - last_timing_time,
                    1e-9,
                )
                row = {
                    "num_envs": num_envs,
                    "vector_steps": vector.vector_steps,
                    "env_steps": env_steps,
                    "aggregate_env_steps_per_sec": (
                        delta_steps / interval_wall
                    ),
                    "vector_steps_per_sec": (
                        delta_vector_steps / interval_wall
                    ),
                    "interval_wall_sec": interval_wall,
                    "rollout_wall_ms_total": (
                        1000.0 * timing["rollout"]
                    ),
                    "env_step_wall_ms_total": (
                        1000.0 * timing["env"]
                    ),
                    "actor_inference_ms_total": (
                        1000.0 * timing["actor"]
                    ),
                    "rnn_inference_ms_total": (
                        1000.0 * timing["rnn"]
                    ),
                    "selector_q_ms_total": (
                        1000.0 * timing["selector"]
                    ),
                    "replay_insert_ms_total": (
                        1000.0 * timing["replay"]
                    ),
                    "rl_update_ms_total": (
                        1000.0 * timing["update"]
                    ),
                    "rollout_time_fraction": (
                        timing["rollout"] / interval_wall
                    ),
                    "update_time_fraction": (
                        timing["update"] / interval_wall
                    ),
                    "wall_time_sec": now - started,
                }
                log(
                    group_dir / "throughput_metrics.jsonl",
                    row,
                )
                success_text = (
                    "-"
                    if latest_episode_success is None
                    else str(latest_episode_success)
                )
                print(
                    f"{args.group} "
                    f"env_steps={env_steps}/{total_env_steps} "
                    f"replay={online.size} "
                    f"updates={agent.updates} "
                    f"alpha={float(agent.alpha.item()):.6g} "
                    f"phase={phase_before} "
                    f"steps_per_sec="
                    f"{row['aggregate_env_steps_per_sec']:.2f} "
                    f"success={success_text}",
                    flush=True,
                )
                latest_episode_success = None
                last_timing_step = env_steps
                last_timing_time = now
                last_timing_vector_steps = vector.vector_steps
                timing = {key: 0.0 for key in timing}

        if metric_window:
            log(
                group_dir / "train_metrics.jsonl",
                aggregate_metrics(metric_window),
            )

        status = (
            "INTERRUPTED"
            if stop_requested and env_steps < total_env_steps
            else "COMPLETE"
        )
        save_checkpoint(
            group_dir / "checkpoints" / "last.pth",
            agent,
            online,
            cfg,
            args.group,
            env_steps,
            episodes_started,
            episodes_completed,
            sim_error_aborts,
            actual,
            vector.vector_steps,
            generations,
        )
        write(
            group_dir / "summary.json",
            {
                "status": status,
                "group": args.group,
                "env_steps": env_steps,
                "vector_steps": vector.vector_steps,
                "num_envs": num_envs,
                "gradient_updates": agent.updates,
                "episodes": episodes_completed,
                "episodes_started": episodes_started,
                "episodes_completed": episodes_completed,
                "sim_error_aborts": sim_error_aborts,
                "success_count": success_count,
                "replay_size": online.size,
                "selector_counts": selector_counts,
                "actor_hash_final": state_hash(actor),
                "critic_hash_final": state_hash(critic),
            },
        )

    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)

        if eval_env is not None:
            try:
                from stage3_new_evaluation import close_env

                close_env(eval_env)
            except BaseException:
                pass

        if vector is not None:
            vector.close()


if __name__ == "__main__":
    main()
