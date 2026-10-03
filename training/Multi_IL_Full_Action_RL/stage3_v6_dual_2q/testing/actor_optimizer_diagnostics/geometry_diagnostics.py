"""Testing-only diagnostics for the real optimizer-geometry continuation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL_ROOT = HERE.parents[2]
V3 = RL_ROOT / "stage3_v3_rgmm_td3"
OLD = RL_ROOT / "stage3_new_sac"
V5 = RL_ROOT / "stage3_v5_rgmm_td3"
for folder in (HERE, V3, OLD, V5):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from run_optimizer_step_probe import actor_group, actor_outputs, policy_drift  # noqa: E402
from stage3_v3_actor import BatchedGMMExecutor  # noqa: E402
from stage3_new_evaluation import build_env, close_env, reset_seed, seed_all, success  # noqa: E402
from stage3_v5_actor import module_hash  # noqa: E402
from stage3_v5_replay import (  # noqa: E402
    BalancedOfflineDemonstrations,
    OnlineSequenceReplay,
    aligned_sequence_batch,
)


SEEDS = (20008, 20002, 20005, 20007)
CACHE = {}


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _parameter_drift(actor, reference):
    accum = {}
    for (name, parameter), (reference_name, reference_parameter) in zip(
        actor.named_parameters(), reference.named_parameters()
    ):
        if name != reference_name:
            raise RuntimeError("Actor/reference parameter ordering changed")
        key = actor_group(name)
        delta = (parameter.detach() - reference_parameter.detach()).float()
        entry = accum.setdefault(key, [0.0, 0])
        entry[0] += float(delta.square().sum().cpu())
        entry[1] += int(delta.numel())
    total_sq = sum(value[0] for value in accum.values())
    total_count = sum(value[1] for value in accum.values())
    result = {
        key: {
            "l2": value[0] ** 0.5,
            "rms": (value[0] / value[1]) ** 0.5,
            "numel": value[1],
        }
        for key, value in accum.items()
    }
    result["total"] = {
        "l2": total_sq ** 0.5,
        "rms": (total_sq / total_count) ** 0.5,
        "numel": total_count,
    }
    return result


def _fixed_observations(agent, config, source_checkpoint):
    cache_key = ("fixed_observations", str(source_checkpoint))
    if cache_key in CACHE:
        return CACHE[cache_key]

    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    offline = BalancedOfflineDemonstrations(
        [config["offline_sources"][key] for key in ("bc_rnn", "bc_transformer", "bc_gmm")],
        seed=int(config["training_seed"]),
    )
    if payload.get("offline_sampler_state") is not None:
        offline.load_state_dict(payload["offline_sampler_state"])
    online = OnlineSequenceReplay.load(payload["online_sequence_replay"])
    online.current = {}
    batch = aligned_sequence_batch(
        offline,
        online,
        count=int(config["recurrent_replay"]["actor_sequence_batch_size"]),
        length=int(config["actor_source_contract"]["rnn_horizon"]),
        horizon=int(config["actor_source_contract"]["rnn_horizon"]),
    )
    observations = torch.as_tensor(
        batch["observations"],
        dtype=torch.float32,
        device=agent.device,
    )
    reference_outputs = actor_outputs(
        agent.geometry_reference,
        observations,
        agent.device,
    )
    CACHE[cache_key] = (observations, reference_outputs)
    return CACHE[cache_key]


def _run_closed_loop(agent, step, out):
    """Four known-good seeds, production executor, no retries."""
    envs = []
    actor = agent.actor
    actor_mode = actor.training
    scale = agent.action_scale.reshape(14)
    offset = agent.action_offset.reshape(14)
    scale_np = scale.detach().cpu().numpy()
    reference = CACHE.get("closed_loop_reference")
    traces = [[] for _ in SEEDS]

    try:
        for index in range(len(SEEDS)):
            print(
                json.dumps(
                    {"event": "geometry_eval_env_start", "step": int(step), "index": index}
                ),
                flush=True,
            )
            envs.append(build_env(Path(agent.config["expert_dataset"])))

        observations = [
            reset_seed(env, seed)
            for env, seed in zip(envs, SEEDS)
        ]
        seed_all(SEEDS[-1])
        executor = BatchedGMMExecutor(actor, scale, offset, len(SEEDS), horizon=10)
        actor.eval()
        active = [True] * len(SEEDS)
        won = [False] * len(SEEDS)
        lengths = [0] * len(SEEDS)
        returns = [0.0] * len(SEEDS)
        errors = [None] * len(SEEDS)

        for timestep in range(700):
            # Always consume RNG for all four policy slots.
            actions = executor.actions_for(
                list(range(len(SEEDS))),
                observations,
                0.0,
                None,
                None,
            )
            for index, env in enumerate(envs):
                if not active[index]:
                    continue
                obs = observations[index]
                action = actions[index]
                row = {
                    "step": int(step),
                    "seed": int(SEEDS[index]),
                    "timestep": int(timestep),
                    "action": action.tolist(),
                    "eef0": np.asarray(obs["robot0_eef_pos"]).reshape(-1).tolist(),
                    "eef1": np.asarray(obs["robot1_eef_pos"]).reshape(-1).tolist(),
                }
                if reference is not None and timestep < len(reference[SEEDS[index]]):
                    old = reference[SEEDS[index]][timestep]
                    row["normalized_action_deviation"] = float(
                        np.linalg.norm(
                            (action - np.asarray(old["action"], dtype=np.float32))
                            / scale_np
                        )
                    )
                    row["arm0_eef_distance"] = float(
                        np.linalg.norm(
                            np.asarray(row["eef0"]) - np.asarray(old["eef0"])
                        )
                    )
                    row["arm1_eef_distance"] = float(
                        np.linalg.norm(
                            np.asarray(row["eef1"]) - np.asarray(old["eef1"])
                        )
                    )
                try:
                    nxt, reward, done, _ = env.step(action)
                    lengths[index] = timestep + 1
                    returns[index] += float(reward)
                    won[index] = bool(success(env))
                    observations[index] = nxt
                    row.update(
                        reward=float(reward),
                        success=bool(won[index]),
                        sim_error=False,
                    )
                    active[index] = not (
                        won[index] or done or lengths[index] >= 700
                    )
                except Exception as error:
                    errors[index] = repr(error)
                    active[index] = False
                    row.update(sim_error=True, error=repr(error))
                traces[index].append(row)
            if timestep % 100 == 0:
                print(
                    json.dumps(
                        {
                            "event": "geometry_eval_progress",
                            "step": int(step),
                            "timestep": int(timestep),
                            "active": int(sum(active)),
                        }
                    ),
                    flush=True,
                )
            if not any(active):
                break

        if reference is None:
            CACHE["closed_loop_reference"] = {
                seed: traces[index]
                for index, seed in enumerate(SEEDS)
            }

        episodes = []
        for index, seed in enumerate(SEEDS):
            action_divergence = [
                row["timestep"]
                for row in traces[index]
                if row.get("normalized_action_deviation", 0.0) > 1e-3
            ]
            state_divergence = [
                row["timestep"]
                for row in traces[index]
                if max(
                    row.get("arm0_eef_distance", 0.0),
                    row.get("arm1_eef_distance", 0.0),
                ) > 1e-3
            ]
            episodes.append(
                {
                    "seed": int(seed),
                    "success": bool(won[index]),
                    "length": int(lengths[index]),
                    "return": float(returns[index]),
                    "sim_error": errors[index],
                    "first_action_divergence_gt_1e_3": (
                        min(action_divergence) if action_divergence else None
                    ),
                    "first_eef_divergence_gt_1mm": (
                        min(state_divergence) if state_divergence else None
                    ),
                }
            )

        trace_path = out / f"trajectory_{int(step):07d}.jsonl"
        if trace_path.exists():
            raise FileExistsError(trace_path)
        with trace_path.open("x", encoding="utf-8") as handle:
            for rows in traces:
                for row in rows:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")

        return {
            "seeds": list(SEEDS),
            "success_count": int(sum(won)),
            "count": len(SEEDS),
            "mean_length": float(np.mean(lengths)),
            "sim_error_count": int(sum(error is not None for error in errors)),
            "episodes": episodes,
            "trajectory": str(trace_path),
            "reference_semantics": (
                "step-130K trajectory from the same branch process"
                if step != 130000
                else "this milestone establishes the step-130K reference"
            ),
        }
    finally:
        actor.train(actor_mode)
        for index, env in enumerate(envs):
            close_env(env)
            print(
                json.dumps(
                    {"event": "geometry_eval_env_closed", "step": int(step), "index": index}
                ),
                flush=True,
            )


def testing_milestone(agent, config, step, group_dir, torch_arg, source_checkpoint):
    out = Path(group_dir) / "geometry_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"step_{int(step):07d}.json"
    if target.exists():
        return

    import train_stage3_v6_vector as production_trainer

    rng = production_trainer.rng_state(torch_arg)
    actor_mode = agent.actor.training
    try:
        if module_hash(agent.geometry_reference) != agent.geometry_reference_hash:
            raise RuntimeError("Geometry reference Actor changed")

        observations, reference_outputs = _fixed_observations(
            agent, config, source_checkpoint
        )
        current_outputs = actor_outputs(agent.actor, observations, agent.device)
        drift = policy_drift(current_outputs, reference_outputs)
        parameter_drift = _parameter_drift(agent.actor, agent.geometry_reference)
        closed_loop = _run_closed_loop(agent, int(step), out)

        result = {
            "testing_only": True,
            "branch": agent.geometry_branch,
            "env_steps": int(step),
            "actor_updates": int(agent.actor_updates),
            "critic_updates": int(agent.critic_updates),
            "fixed_policy_probe": {
                "policy_drift": drift,
                "parameter_drift": parameter_drift,
            },
            "closed_loop": closed_loop,
            "reference_hash_unchanged": (
                module_hash(agent.geometry_reference)
                == agent.geometry_reference_hash
            ),
        }
        if closed_loop["sim_error_count"]:
            raise RuntimeError("Closed-loop diagnostic simulator error")
        if int(step) == 130000:
            if closed_loop["success_count"] != 4:
                raise RuntimeError(
                    "130K source Actor failed one of four known-good baseline seeds"
                )
            if drift["weighted_action_l2_mean"] != 0.0:
                raise RuntimeError("130K source Actor differs from reference Actor")

        _write_json(target, result)
        print(
            "GEOMETRY_MILESTONE "
            + json.dumps(
                {
                    "branch": agent.geometry_branch,
                    "env_steps": int(step),
                    "actor_updates": int(agent.actor_updates),
                    "success": closed_loop["success_count"],
                    "parameter_drift_l2": parameter_drift["total"]["l2"],
                    "sampled_action_drift": drift["sampled_action_l2_mean"],
                    "weighted_action_drift": drift["weighted_action_l2_mean"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        agent.actor.train(actor_mode)
        agent.geometry_reference.eval()
        production_trainer.restore_rng(rng, torch_arg)
