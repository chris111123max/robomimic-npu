#!/usr/bin/env python3
"""Small paired closed-loop audit; serial env startup, persistent env reuse."""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V5 = ROOT / "stage3_v5_rgmm_td3"
V6 = ROOT / "stage3_v6_dual_2q"
V3 = ROOT / "stage3_v3_rgmm_td3"
OLD = ROOT / "stage3_new_sac"
for folder in (HERE, V5, V6, V3, OLD):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from run_offline import CHECKPOINTS, a3_forward, resolve_device
from stage3_v5_actor import load_exact_actor, obs_to_flat
from stage3_v3_actor import BatchedGMMExecutor
from stage3_v6_agent import strict_stage2_load
from stage3_new_evaluation import build_env, close_env, reset_seed, success

SEEDS = (20008, 20002, 20005, 20007)
ROUNDS = (("B1", "actor_init", "step280k"), ("B2", "critic_ready", "step200k"))

def stats(rows):
    return {
        "success": sum(int(x["success"]) for x in rows),
        "count": len(rows),
        "mean_length": float(np.mean([x["length"] for x in rows])),
        "mean_return": float(np.mean([x["return"] for x in rows])),
        "sim_errors": sum(int(x["sim_error"]) for x in rows),
    }

def make_model(run, name, device, critic_source):
    actor, rollout, _ = load_exact_actor(run / "shared/bc_rnn_gmm_source.pth", device)
    path = run / "mean2q/multi_q/checkpoints" / CHECKPOINTS[name]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor.load_state_dict(payload["actor"], strict=True)
    actor.eval()
    critic, _ = strict_stage2_load(critic_source, device)
    critic.load_state_dict(payload["q1_q2"], strict=True)
    critic.eval()
    s = torch.as_tensor(rollout.action_normalization_stats["actions"]["scale"], device=device, dtype=torch.float32)
    o = torch.as_tensor(rollout.action_normalization_stats["actions"]["offset"], device=device, dtype=torch.float32)
    return actor, critic, s, o, payload

def audit_trajectory(records, init_actor, actor, critic, device, scale, offset):
    # Group by recurrent-block prefix length to exactly preserve 10-step Actor reset.
    if not records:
        return
    for width in range(1, 11):
        selected = [i for i, r in enumerate(records) if r["timestep"] % 10 + 1 == width]
        for lo in range(0, len(selected), 32):
            positions = selected[lo:lo + 32]
            obs = []
            acts = []
            steps = []
            for index in positions:
                row = records[index]
                block = row["timestep"] - width + 1
                episode = row["episode"]
                prefix = [r for r in records if r["episode"] == episode and block <= r["timestep"] <= row["timestep"]]
                if len(prefix) != width:
                    raise RuntimeError("Trajectory prefix missing")
                obs.append(np.stack([r["observation_flat"] for r in prefix]))
                acts.append(np.stack([r["action"] for r in prefix]))
                steps.append(np.arange(block, block + width, dtype=np.int64))
            contexts = {
                "a3_observations": np.stack(obs).astype(np.float32),
                "a3_actions": np.stack(acts).astype(np.float32),
                "a3_episode_steps": np.stack(steps),
            }
            result = a3_forward(
                init_actor, actor, critic, contexts, device,
                scale.reshape(1, 1, 1, 14), offset.reshape(1, 1, 1, 14), 700
            )
            for j, pos in enumerate(positions):
                row = records[pos]
                ai = result["init_action_env"][j]
                ac = result["current_action_env"][j]
                q1i = float(result["q1_init_action"][j])
                q2i = float(result["q2_init_action"][j])
                q1c = float(result["q1_current_action"][j])
                q2c = float(result["q2_current_action"][j])
                row["actor_init_reference_action"] = ai.tolist()
                row["actor_current_argmax_mean_action"] = ac.tolist()
                row["action_drift_normalized_l2"] = float(np.linalg.norm((ac-ai)/scale.cpu().numpy()))
                row["q1_init"] = q1i
                row["q2_init"] = q2i
                row["q1_current"] = q1c
                row["q2_current"] = q2c
                row["q_objective_init"] = float(result["q_obj_init"][j])
                row["q_objective_current"] = float(result["q_obj_current"][j])
                row["delta_q_objective"] = row["q_objective_current"] - row["q_objective_init"]
                row["twin_disagreement_current"] = abs(q1c-q2c)
                row["twin_disagreement_init"] = abs(q1i-q2i)
                row["gmm_probs"] = result["current_probs"][j].tolist()
                row["gmm_entropy"] = float(-np.sum(result["current_probs"][j] * np.log(np.maximum(result["current_probs"][j], 1e-12))))
                row["gmm_argmax_mode"] = int(result["current_mode"][j])
                row["gmm_component_means"] = result["current_means_env"][j].tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "closed_loop.json"
    rows_path = output / "closed_loop_steps.jsonl"
    seeds_path = output / "closed_loop_seeds.json"
    for path in (results_path, rows_path, seeds_path):
        if path.exists():
            raise FileExistsError(path)
    prior = json.loads((run / "actor_collapse_diagnosis/actor_init.json").read_text())
    prior_rows = {int(r["seed"]): r for r in prior["evaluation"]["episodes"]}
    if not all(prior_rows[s]["success"] for s in SEEDS):
        raise RuntimeError("Selected seeds were not actor_init successes")
    seeds_path.write_text(json.dumps({
        "seeds": SEEDS,
        "selection": "Sorted initial-success episode lengths: 400, 444, 467, 511; short/mid/long coverage",
        "prior_lengths": {str(s): prior_rows[s]["length"] for s in SEEDS},
    }, indent=2) + "\n")
    source = json.loads((run / "shared/stage2_source_manifest.json").read_text())
    devices = [resolve_device("npu:0"), resolve_device("npu:1")]
    actor_init = []
    for device in devices:
        model, _, _, _, _ = make_model(run, "actor_init", device, source["multi_q"]["checkpoint"])
        actor_init.append(model)
    envs = []
    all_results = []
    try:
        dataset = Path("/data/home/3220251075/lerobot_workspace/datasets/transport/PH/low_dim_v15.hdf5")
        for index in range(8):
            print(json.dumps({"event":"env_start", "index":index}), flush=True)
            envs.append(build_env(dataset))
            print(json.dumps({"event":"env_ready", "index":index}), flush=True)
        for round_name, left, right in ROUNDS:
            names = (left, right)
            models = [make_model(run, name, devices[i], source["multi_q"]["checkpoint"]) for i, name in enumerate(names)]
            executors = [
                BatchedGMMExecutor(model[0], model[2], model[3], 4, horizon=10)
                for model in models
            ]
            current = []
            active = [True] * 8
            totals = [0.0] * 8
            lengths = [0] * 8
            won = [False] * 8
            traces = [[] for _ in range(8)]
            print(json.dumps({"event":"round_start", "round":round_name, "checkpoint_pair":names}), flush=True)
            for index, env in enumerate(envs):
                seed = SEEDS[index % 4]
                current.append(reset_seed(env, seed))
                executors[index // 4].reset_indices([index % 4])
            for timestep in range(700):
                actions = [None] * 8
                for group in range(2):
                    indexes = [j for j in range(4) if active[group * 4 + j]]
                    if not indexes:
                        continue
                    observations = [current[group * 4 + j] for j in indexes]
                    batch_actions = executors[group].actions_for(indexes, observations, 0.0, None, None)
                    for j, action in zip(indexes, batch_actions):
                        actions[group * 4 + j] = action
                for index, env in enumerate(envs):
                    if not active[index]:
                        continue
                    observation = current[index]
                    action = actions[index]
                    next_obs, reward, done, _ = env.step(action)
                    totals[index] += float(reward)
                    lengths[index] = timestep + 1
                    won[index] = bool(success(env))
                    traces[index].append({
                        "round":round_name,
                        "checkpoint":names[index // 4],
                        "seed":SEEDS[index % 4],
                        "episode":index,
                        "timestep":timestep,
                        "observation_flat":obs_to_flat(observation).tolist(),
                        "action":action.tolist(),
                        "reward":float(reward),
                        "success":won[index],
                        "robot0_eef_pos":np.asarray(observation["robot0_eef_pos"]).reshape(-1).tolist(),
                        "robot1_eef_pos":np.asarray(observation["robot1_eef_pos"]).reshape(-1).tolist(),
                        "robot0_gripper_qpos":np.asarray(observation["robot0_gripper_qpos"]).reshape(-1).tolist(),
                        "robot1_gripper_qpos":np.asarray(observation["robot1_gripper_qpos"]).reshape(-1).tolist(),
                        "object":np.asarray(observation["object"]).reshape(-1).tolist(),
                    })
                    current[index] = next_obs
                    active[index] = not (won[index] or done or lengths[index] >= 700)
                if timestep % 100 == 0:
                    print(json.dumps({"event":"progress", "round":round_name, "step":timestep, "active":sum(active)}), flush=True)
                if not any(active):
                    break
            episode_rows = [
                {
                    "checkpoint":names[i // 4], "seed":SEEDS[i % 4],
                    "success":won[i], "length":lengths[i],
                    "return":totals[i], "sim_error":False,
                }
                for i in range(8)
            ]
            for group in range(2):
                merged = [row for index in range(group*4, group*4+4) for row in traces[index]]
                audit_trajectory(merged, actor_init[group], models[group][0], models[group][1],
                                 devices[group], models[group][2], models[group][3])
                with rows_path.open("a") as handle:
                    for row in merged:
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
            result = {
                "round":round_name,
                "checkpoints":names,
                "episodes":episode_rows,
                "left":stats(episode_rows[:4]),
                "right":stats(episode_rows[4:]),
            }
            all_results.append(result)
            print(json.dumps({"event":"round_done", "result":result}), flush=True)
    finally:
        for index, env in enumerate(envs):
            close_env(env)
            print(json.dumps({"event":"env_closed", "index":index}), flush=True)
    results_path.write_text(json.dumps({
        "seeds":SEEDS, "horizon":700, "startup_parallelism":1,
        "env_count":8, "rounds":all_results,
        "step_rows":str(rows_path),
    }, indent=2) + "\n")
    print(json.dumps({"event":"complete", "output":str(results_path)}), flush=True)

if __name__ == "__main__":
    main()
