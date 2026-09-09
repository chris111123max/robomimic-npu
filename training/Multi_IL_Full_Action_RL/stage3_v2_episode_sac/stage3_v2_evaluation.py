"""Pure-policy evaluations for Stage3-v2."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OLD_STAGE3 = ROOT / "training" / "Multi_IL_Full_Action_RL" / "stage3_new_sac"
if str(OLD_STAGE3) not in sys.path:
    sys.path.insert(0, str(OLD_STAGE3))

from stage3_new_evaluation import (  # noqa: E402,F401
    KEYS,
    build_env,
    close_env,
    evaluate,
    mujoco_fatal_error_type,
    reset_seed,
    success,
)


def evaluate_pure_actor(actor, env, seeds, horizon, device, retries=1):
    """Formal Stage3-v2 metric; no BC-RNN and no Critic are accepted here."""
    result = evaluate(actor, env, seeds, horizon, device, retries)
    result["evaluation_policy"] = "pure_deterministic_sac_actor"
    result["pure_actor_success_count"] = sum(
        int(bool(row["success"])) for row in result["episodes"]
        if not row["sim_error"]
    )
    result["pure_actor_success_rate"] = result.pop("success_rate")
    return result


def evaluate_frozen_rnn(proposer, env, seeds, horizon, retries=1):
    rows = []
    for seed in seeds:
        fatal = None
        for attempt in range(int(retries) + 1):
            try:
                observation = reset_seed(env, seed)
                proposer.start_episode()
                episode_return = 0.0
                won = success(env)
                raw_done = False
                steps = 0
                for step in range(int(horizon)):
                    action = proposer.action(observation)
                    observation, reward, raw_done, _ = env.step(action)
                    episode_return += float(reward)
                    steps = step + 1
                    won = success(env)
                    if raw_done or won or steps >= int(horizon):
                        break
                truncated = bool(steps >= int(horizon) and not won)
                rows.append({
                    "seed": int(seed),
                    "return": episode_return,
                    "length": steps,
                    "success": bool(won),
                    "terminated": bool(won or (raw_done and not truncated)),
                    "truncated": truncated,
                    "sim_error": False,
                    "attempts": attempt + 1,
                })
                fatal = None
                break
            except mujoco_fatal_error_type() as error:
                fatal = error
        if fatal is not None:
            rows.append({
                "seed": int(seed), "return": None, "length": None,
                "success": None, "terminated": False, "truncated": False,
                "sim_error": True, "attempts": int(retries) + 1,
                "exception": str(fatal),
            })
    valid = [row for row in rows if not row["sim_error"]]
    return {
        "status": "BASELINE",
        "evaluation_policy": "pure_frozen_bc_rnn",
        "evaluation_seeds": list(map(int, seeds)),
        "episodes": rows,
        "valid_episodes": len(valid),
        "sim_error_episodes": len(rows) - len(valid),
        "success_count": sum(int(bool(row["success"])) for row in valid),
        "success_rate": float(np.mean([row["success"] for row in valid])) if valid else None,
        "mean_return": float(np.mean([row["return"] for row in valid])) if valid else None,
        "mean_length": float(np.mean([row["length"] for row in valid])) if valid else None,
    }
