"""Closed-loop evaluation of the current recurrent GMM Actor only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "training" / "Multi_IL_Full_Action_RL" / "stage3_new_sac"
if str(OLD) not in sys.path:
    sys.path.insert(0, str(OLD))
from stage3_new_evaluation import (build_env, close_env, mujoco_fatal_error_type,
                                   reset_seed, success)  # noqa: E402
from stage3_v3_actor import BatchedGMMExecutor


def evaluate_actor(actor, action_scale, action_offset, env, seeds, horizon,
                   retries=1, action_low=None, action_high=None):
    executor = BatchedGMMExecutor(actor, action_scale, action_offset, 1, horizon=10)
    rows = []
    for seed in seeds:
        fatal = None
        for attempt in range(int(retries) + 1):
            try:
                observation = reset_seed(env, int(seed))
                executor.reset_indices([0])
                total, steps, won, raw_done = 0.0, 0, bool(success(env)), False
                for index in range(int(horizon)):
                    action = executor.actions_for([0], [observation], 0.0,
                                                  action_low, action_high)[0]
                    observation, reward, raw_done, _ = env.step(action)
                    total += float(reward)
                    steps = index + 1
                    won = bool(success(env))
                    if raw_done or won or steps >= int(horizon):
                        break
                truncated = bool(steps >= int(horizon) and not won)
                terminated = bool(won or (raw_done and not truncated))
                rows.append({"seed": int(seed), "return": total, "length": steps,
                             "success": won, "terminated": terminated,
                             "truncated": truncated, "sim_error": False,
                             "attempts": attempt + 1})
                fatal = None
                break
            except mujoco_fatal_error_type() as error:
                fatal = error
        if fatal is not None:
            rows.append({"seed": int(seed), "return": None, "length": None,
                         "success": None, "terminated": False, "truncated": False,
                         "sim_error": True, "attempts": int(retries) + 1,
                         "exception": str(fatal)})
    valid = [row for row in rows if not row["sim_error"]]
    return {
        "evaluation_policy": "current_recurrent_gmm_actor_checkpoint_sampling_low_noise_eval",
        "external_exploration_noise": 0.0, "evaluation_seeds": list(map(int, seeds)),
        "episodes": rows, "valid_episodes": len(valid),
        "sim_error_episodes": len(rows) - len(valid),
        "success_count": sum(int(row["success"]) for row in valid),
        "success_rate": float(np.mean([row["success"] for row in valid])) if valid else None,
        "mean_return": float(np.mean([row["return"] for row in valid])) if valid else None,
        "mean_length": float(np.mean([row["length"] for row in valid])) if valid else None,
    }
