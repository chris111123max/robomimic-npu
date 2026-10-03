#!/usr/bin/env python3
"""TEST-ONLY, offline V6 readiness metrics on the original frozen set."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V6 = HERE.parents[1]
V5 = V6.parent / "stage3_v5_rgmm_td3"
for folder in (V6, V5):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from stage3_v5_actor import load_exact_actor
from stage3_v5_readiness import correlation, discounted_returns, auc
from stage3_v6_agent import RecurrentGMMTD3V6, strict_stage2_load

RUN = Path("/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_multi_mean_random_formal_20260928")
STAGE2 = Path("/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/stage2_2_h10_multi_20260923_150749/multi_q/checkpoints/step_00005000.pth")


def frozen_hash(sequences):
    digest = hashlib.sha256()
    for key, value in sequences.items():
        digest.update(key.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def evaluate(mode, device_name):
    if device_name.startswith("npu"):
        import torch_npu
        torch.npu.set_device(device_name)
    device = torch.device(device_name)
    base = RUN / mode / "multi_q"
    checks = base / "checkpoints"
    checkpoints = [(0, "step0_transfer.pth"), (100000, "step_0100000.pth"),
                   (200000, "step_0200000.pth"), (300000, "step_0300000.pth")]
    replay = np.load(checks / "step_0100000.sequences.npy", allow_pickle=True).item()
    fixed = replay["fixed_critic_diagnostic_set"]
    sequence_hash = frozen_hash(fixed["sequences"])
    for step in (200000, 300000):
        other = np.load(checks / f"step_{step:07d}.sequences.npy", allow_pickle=True).item()
        assert frozen_hash(other["fixed_critic_diagnostic_set"]["sequences"]) == sequence_hash
    payload0 = torch.load(checks / checkpoints[0][1], map_location="cpu", weights_only=False)
    config = payload0["config"]
    assert config["critic_target_mode"] == mode
    actor, _, _ = load_exact_actor(config["bc_rnn_checkpoint"], device)
    critic, _ = strict_stage2_load(STAGE2, device)
    scale = torch.tensor(payload0["action_normalization_stats"]["scale"], dtype=torch.float32, device=device).reshape(1, 1, 1, 14)
    offset = torch.tensor(payload0["action_normalization_stats"]["offset"], dtype=torch.float32, device=device).reshape(1, 1, 1, 14)
    agent = RecurrentGMMTD3V6(actor, critic, config, device, scale, offset)
    result = {"mode": mode, "device": device_name, "frozen_sequence_sha256": sequence_hash,
              "fixed_episodes": len(fixed["episodes"]), "fixed_sequences": len(fixed["indices"]),
              "fixed_seed": fixed["seed"], "environment_steps_executed": 0,
              "optimizer_steps_executed": 0, "rows": []}
    for step, name in checkpoints:
        payload = torch.load(checks / name, map_location="cpu", weights_only=False)
        assert payload["env_steps"] == step and payload["critic_target_mode"] == mode
        agent.actor.load_state_dict(payload["actor"], strict=True)
        agent.target_actor.load_state_dict(payload["target_actor"], strict=True)
        agent.critic.load_state_dict(payload["q1_q2"], strict=True)
        agent.target_critic.load_state_dict(payload["target_q1_q2"], strict=True)
        agent.actor.eval(); agent.target_actor.eval(); agent.critic.eval(); agent.target_critic.eval()
        diagnostic = agent.fixed_td_diagnostic(fixed["sequences"])
        q1 = np.asarray(diagnostic["q1"], dtype=np.float64)
        q2 = np.asarray(diagnostic["q2"], dtype=np.float64)
        target = np.asarray(diagnostic["td_target"], dtype=np.float64)
        qmin, qmean = np.minimum(q1, q2), 0.5 * (q1 + q2)
        td = {"qmin_mae": float(np.abs(qmin - target).mean()),
              "qmean_mae": float(np.abs(qmean - target).mean()),
              "members_mae": float(0.5 * (np.abs(q1 - target).mean() + np.abs(q2 - target).mean())),
              "member_max_mae": float(max(np.abs(q1 - target).mean(), np.abs(q2 - target).mean())),
              "q1_mae": float(np.abs(q1 - target).mean()),
              "q2_mae": float(np.abs(q2 - target).mean()),
              "target_mean": float(target.mean()), "target_std": float(target.std())}
        def distribution_stats(values):
            return {"mean": float(values.mean()), "std": float(values.std()),
                    "abs_mean": float(np.abs(values).mean()),
                    "min": float(values.min()), "max": float(values.max())}
        distribution = {"qmean": distribution_stats(qmean),
                        "q1": distribution_stats(q1),
                        "q2": distribution_stats(q2),
                        "td_target": distribution_stats(target)}
        outputs = agent.q_values_for_episodes(fixed["episodes"])
        scores = {key: [] for key in ("qmin", "qmean", "q1", "q2")}
        returns = []; labels = []
        for episode, (ep_q1, ep_q2) in zip(fixed["episodes"], outputs):
            a, b = np.asarray(ep_q1), np.asarray(ep_q2)
            scores["qmin"].extend(np.minimum(a,b).tolist())
            scores["qmean"].extend((0.5*(a+b)).tolist())
            scores["q1"].extend(a.tolist()); scores["q2"].extend(b.tolist())
            returns.extend(discounted_returns(episode, config["gamma"]).tolist())
            labels.append(int(bool(episode.get("success", False))))
        ranking = {}
        for key, values in scores.items():
            s, p = correlation(values, returns)
            ep_means = []
            offset_idx = 0
            for episode in fixed["episodes"]:
                n = len(episode["actions"])
                ep_means.append(float(np.mean(values[offset_idx:offset_idx+n])))
                offset_idx += n
            ranking[key] = {"spearman": s, "pearson": p, "success_auc": auc(labels, ep_means)}
        row = {"env_steps": step, "critic_updates_in_checkpoint": int(payload["updates"]),
               "actor_updates_in_checkpoint": int(payload["actor_updates"]),
               "td": td, "distribution": distribution, "ranking": ranking,
               "finite": bool(all(np.isfinite(a).all() for a in (q1,q2,target))) }
        result["rows"].append(row)
        print(json.dumps({"mode": mode, "step": step, "td": td, "ranking": ranking}), flush=True)
    assert agent.target_selector.audit()["random_updates"] == 0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mean2q","random2q"), required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate(args.mode, args.device)
    output = Path(args.output) if args.output else HERE / f"checkpoint_metrics_{args.mode}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("OUTPUT", output, flush=True)

if __name__ == "__main__":
    main()
