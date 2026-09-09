"""Deterministic episode behavior and BC schedules for Stage3-v2."""
from __future__ import annotations

from fractions import Fraction


def validate_behavior_schedule(schedule):
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("episode_behavior_schedule must be a non-empty list")
    previous_end = 0
    for index, row in enumerate(schedule):
        start = int(row["start"])
        end = row.get("end")
        end = None if end is None else int(end)
        fraction = float(row["rnn_fraction"])
        if start != previous_end:
            raise ValueError(f"Behavior schedule is not contiguous at row {index}")
        if end is not None and end <= start:
            raise ValueError(f"Behavior schedule has invalid end at row {index}")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"Invalid rnn_fraction at row {index}")
        rational = Fraction(str(fraction)).limit_denominator(100)
        if abs(float(rational) - fraction) > 1e-12:
            raise ValueError("rnn_fraction must have a stable periodic representation")
        if index + 1 < len(schedule) and end is None:
            raise ValueError("Only the final behavior phase may have end=null")
        previous_end = end
    if schedule[-1].get("end") is not None:
        raise ValueError("Final behavior phase must have end=null")


def behavior_phase(schedule, env_steps):
    validate_behavior_schedule(schedule)
    step = int(env_steps)
    if step < 0:
        raise ValueError("env_steps must be non-negative")
    for index, row in enumerate(schedule):
        start = int(row["start"])
        end = row.get("end")
        if step >= start and (end is None or step < int(end)):
            result = dict(row)
            result["index"] = index
            result["name"] = str(row.get("name", f"phase_{index}"))
            return result
    raise RuntimeError(f"No behavior phase covers env_steps={step}")


def assign_behavior_source(schedule, env_steps, episode_id):
    """Return an episode-constant source with no RNG dependency."""
    phase = behavior_phase(schedule, env_steps)
    fraction = Fraction(str(float(phase["rnn_fraction"]))).limit_denominator(100)
    period = int(fraction.denominator)
    rnn_slots = int(fraction.numerator)
    source = "rnn" if int(episode_id) % period < rnn_slots else "rl"
    return source, phase


def progressive_critic_schedule(config, env_steps):
    cfg = config["progressive_critic_unfreeze"]
    step = int(env_steps)
    protected = int(cfg["protected_until_env_steps"])
    end = int(cfg["unfreeze_end_env_steps"])
    if protected < 0 or end <= protected:
        raise ValueError("Invalid progressive Critic schedule")
    if step < protected:
        phase, scale = "critic_frozen", 0.0
    elif step < end:
        phase = "critic_progressive_unfreeze"
        scale = (step - protected) / float(end - protected)
    else:
        phase, scale = "critic_full", 1.0
    return {
        "phase": phase,
        "critic_lr_scale": scale,
        "critic_lr_effective": float(config["critic_lr"]) * scale,
        "target_tau_scale": scale,
        "target_tau_effective": float(config["tau"]) * scale,
        "critic_update_enabled": scale > 0.0,
        "sac_actor_enabled": step >= protected,
        "alpha_tuning_enabled": step >= protected,
    }


def bc_schedule(config, env_steps):
    cfg = config["bc_regularization_schedule"]
    step = int(env_steps)
    bc_only_end = int(cfg["bc_only_until_env_steps"])
    decay_start = int(cfg["decay_start_env_steps"])
    decay_end = int(cfg["decay_end_env_steps"])
    initial = float(cfg["lambda_bc"])
    if not (0 <= bc_only_end <= decay_start < decay_end) or initial < 0:
        raise ValueError("Invalid BC regularization schedule")
    if step < bc_only_end:
        return {"actor_objective": "bc_only", "lambda_bc": initial}
    if step < decay_start:
        return {"actor_objective": "sac_plus_bc", "lambda_bc": initial}
    if step < decay_end:
        scale = 1.0 - (step - decay_start) / float(decay_end - decay_start)
        return {"actor_objective": "sac_plus_bc", "lambda_bc": initial * scale}
    return {"actor_objective": "pure_sac", "lambda_bc": 0.0}


def episode_identity(base_seed, num_envs, env_id, generation):
    episode_id = int(env_id) + int(generation) * int(num_envs)
    return episode_id, int(base_seed) + episode_id


def new_episode_context(config, num_envs, env_id, generation, env_steps):
    episode_id, seed = episode_identity(
        config["train_seed_base"], num_envs, env_id, generation
    )
    source, phase = assign_behavior_source(
        config["episode_behavior_schedule"], env_steps, episode_id
    )
    return {
        "episode_id": episode_id,
        "seed": seed,
        "generation": int(generation),
        "behavior_source": source,
        "behavior_phase": phase["name"],
        "behavior_phase_index": int(phase["index"]),
        "behavior_start_env_steps": int(env_steps),
        "length": 0,
        "return": 0.0,
    }
