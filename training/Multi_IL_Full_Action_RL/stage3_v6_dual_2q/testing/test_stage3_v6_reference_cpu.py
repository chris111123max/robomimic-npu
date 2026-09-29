#!/usr/bin/env python3
"""No-environment CPU forward check for the Stage2-initial Qmean reference."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

V6 = Path(__file__).resolve().parents[1]
V5 = V6.parent / "stage3_v5_rgmm_td3"
for folder in (V6, V5):
    sys.path.insert(0, str(folder))

from stage3_v5_actor import load_exact_actor
from stage3_v6_agent import RecurrentGMMTD3V6, strict_stage2_load

OLD_RUN = Path("/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_multi_mean_random_formal_20260928/mean2q/multi_q/checkpoints")
STAGE2 = Path("/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/stage2_2_h10_multi_20260923_150749/multi_q/checkpoints/step_00005000.pth")


def main():
    torch.set_num_threads(4)
    payload = torch.load(OLD_RUN / "step0_transfer.pth", map_location="cpu", weights_only=False)
    actor, _, _ = load_exact_actor(payload["config"]["bc_rnn_checkpoint"], torch.device("cpu"))
    critic, stage2 = strict_stage2_load(STAGE2, torch.device("cpu"))
    assert int(stage2["checkpoint_step"]) == 5000
    scale = torch.tensor(payload["action_normalization_stats"]["scale"]).reshape(1, 1, 1, 14)
    offset = torch.tensor(payload["action_normalization_stats"]["offset"]).reshape(1, 1, 1, 14)
    agent = RecurrentGMMTD3V6(actor, critic, payload["config"], torch.device("cpu"), scale, offset)
    frozen = np.load(OLD_RUN / "step_0100000.sequences.npy", allow_pickle=True).item()["fixed_critic_diagnostic_set"]
    sequences = {key: value[:4] for key, value in frozen["sequences"].items()}
    initial = agent.initial_qmean_on_sequences(sequences)
    with torch.no_grad():
        actions = agent._tensor_batch({"actions": sequences["actions"][:, -1]})["actions"]
        contexts = agent._history_contexts(agent.critic, sequences)
        q1, q2 = agent.critic.q_from_context((contexts[0][:, -1], contexts[1][:, -1]), actions)
        direct = (0.5 * (q1 + q2)).cpu().numpy().reshape(-1)
    assert np.allclose(initial, direct, atol=1e-6, rtol=1e-6)
    assert np.isfinite(initial).all()
    with torch.no_grad():
        next(agent.critic.parameters()).add_(0.1)
    assert np.allclose(agent.initial_qmean_on_sequences(sequences), initial,
                       atol=1e-6, rtol=1e-6)
    print({"status": "PASS", "stage2_step": 5000, "sample_count": len(initial),
           "max_abs_reference_difference": float(np.max(np.abs(initial-direct))),
           "environment_steps": 0, "optimizer_steps": 0})


if __name__ == "__main__":
    main()
