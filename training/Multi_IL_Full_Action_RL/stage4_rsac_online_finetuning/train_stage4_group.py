#!/usr/bin/env python3
"""Train one isolated Stage4 recurrent SAC group."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
for path in (HERE, PROJECT / "stage1_rollout_collection", PROJECT / "stage3_actor_initialization"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

from stage4_core import (GROUPS, OnlineSequenceReplay, Stage4SAC, append_csv, assert_phase_contract,
                         atomic_json, cpu_tree, initialize_models, model_audit, phase_at, read_json,
                         restore_rng, rng_state, seed_all, select_device, state_hash)  # noqa: E402
from common import extract_canonical_observation, try_seed_environment  # noqa: E402
from collect_multi_il_rollouts import progress_observation_schema  # noqa: E402
from evaluate_stage3_actor import (close_env, env_success, extract_transport_progress,
                                   load_initial_states, partial_progress_score)  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402


SHAPES = {"robot0_eef_pos": [3], "robot0_eef_quat": [4], "robot0_gripper_qpos": [2],
          "robot1_eef_pos": [3], "robot1_eef_quat": [4], "robot1_gripper_qpos": [2], "object": [41]}


def parse_args():
    parser = argparse.ArgumentParser(); parser.add_argument("--group", choices=GROUPS, required=True); parser.add_argument("--device", required=True)
    parser.add_argument("--config", default=str(HERE / "stage4_config.json")); parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int); parser.add_argument("--smoke-test", action="store_true"); parser.add_argument("--resume")
    return parser.parse_args()


def effective_config(path, args):
    config = read_json(path); config["group"] = args.group; config["device"] = args.device
    if args.seed is not None: config["seed"] = args.seed
    if args.smoke_test:
        config.update(total_env_steps=12, actor_freeze_steps=4, actor_warmup_end=8, evaluation_interval=5,
                      evaluation_seed_start=10080, evaluation_seed_end=10081, evaluation_horizon=5,
                      replay_capacity=1000, minimum_replay_size=2, batch_size=2, updates_per_env_step=0.5,
                      checkpoint_steps=[0, 4, 8, 12])
    return config


def build_environment(actor_payload):
    checkpoint = FileUtils.maybe_dict_from_checkpoint(ckpt_path=actor_payload["teacher_checkpoint"])
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint); ObsUtils.initialize_obs_utils_with_config(config)
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=checkpoint, render=False, render_offscreen=False, verbose=False)
    return env


def flatten(observation, keys):
    canonical = extract_canonical_observation(observation, keys, SHAPES)
    return np.concatenate([canonical[key].reshape(-1) for key in keys]).astype(np.float32), canonical


def evaluate(actor, actor_payload, config, device, env_step, output):
    saved_rng = rng_state(); saved_state = cpu_tree(actor._state); saved_counter = actor._counter; was_training = actor.training
    seeds = list(range(int(config["evaluation_seed_start"]), int(config["evaluation_seed_end"]) + 1)); initial = load_initial_states(config["initial_states"], seeds)
    env = build_environment(actor_payload); schema = progress_observation_schema(env, actor_payload["observation_keys"], SHAPES); rows=[]; actor.eval()
    try:
        for seed in seeds:
            seed_all(seed); observation = env.reset_to(copy.deepcopy(initial[seed])); seed_all(seed); actor.reset()
            episode_return=0.0; ever_trash=ever_payload=final_trash=final_payload=False; raw_done=False; success=env_success(env)
            for index in range(int(config["evaluation_horizon"])):
                state, _ = flatten(observation, actor_payload["observation_keys"]); tensor = torch.as_tensor(state[None], device=device)
                with torch.no_grad(): action = actor.act(tensor, deterministic=True)[0][0].cpu().numpy()
                observation, reward, raw_done, _ = env.step(action); episode_return += float(reward)
                _, canonical = flatten(observation, actor_payload["observation_keys"]); progress = extract_transport_progress(canonical, schema)
                final_trash = progress["trash_in_trash_bin"]; final_payload = progress["payload_in_target_bin"]; ever_trash |= final_trash; ever_payload |= final_payload; success = env_success(env)
                if raw_done or (config["terminate_on_success"] and success): break
            rows.append({"seed": seed, "success": int(success), "trash": int(ever_trash), "payload": int(ever_payload), "progress": partial_progress_score(success, ever_trash, ever_payload), "return": episode_return, "length": index + 1})
    finally:
        close_env(env); restore_rng(saved_rng); actor._state = None if saved_state is None else tuple(item.to(device) for item in saved_state); actor._counter = saved_counter; actor.train(was_training)
    count=len(rows); report={"env_step":int(env_step),"success_count":sum(row["success"] for row in rows),"success_rate":sum(row["success"] for row in rows)/count,
        "trash_ever_count":sum(row["trash"] for row in rows),"trash_ever_rate":sum(row["trash"] for row in rows)/count,
        "payload_ever_count":sum(row["payload"] for row in rows),"payload_ever_rate":sum(row["payload"] for row in rows)/count,
        "both_ever_count":sum(row["trash"] and row["payload"] for row in rows),"both_ever_rate":sum(row["trash"] and row["payload"] for row in rows)/count,
        "mean_progress":float(np.mean([row["progress"] for row in rows])),"mean_return":float(np.mean([row["return"] for row in rows])),"mean_episode_length":float(np.mean([row["length"] for row in rows])),"episodes":rows}
    atomic_json(output, report); return report


def save_replay(replay, group_dir, env_step):
    path = group_dir / "replay" / f"step_{int(env_step):08d}.npz"; path.parent.mkdir(exist_ok=True)
    marker=path.with_suffix(".json");identity={"transitions":replay.transitions,"episodes":replay.episodes}
    if path.exists() and marker.exists() and read_json(marker)==identity:return path
    temporary=path.with_name(path.stem+".tmp.npz");replay.save(temporary);os.replace(temporary,path);atomic_json(marker,identity)
    return path


def save_checkpoint(path, engine, replay, config, env_step, episode, actor_payload, training_context, evaluation=None):
    replay_path = save_replay(replay, Path(path).parents[1], env_step)
    payload={"format_version":"multi_il_full_action_rl.stage4.v1","group":config["group"],"env_step":int(env_step),"episode":int(episode),"config":config,
        "actor_state_dict":cpu_tree(engine.actor.state_dict()),"critic_state_dict":cpu_tree(engine.critic.state_dict()),"target_critic_state_dict":cpu_tree(engine.target.state_dict()),
        "actor_optimizer_state_dict":cpu_tree(engine.actor_optimizer.state_dict()),"critic_optimizer_state_dict":cpu_tree(engine.critic_optimizer.state_dict()),
        "log_alpha_entropy":cpu_tree(engine.algo.log_alpha_entropy) if engine.algo.automatic_entropy_tuning else None,
        "alpha_optimizer_state_dict":cpu_tree(engine.algo.alpha_entropy_optim.state_dict()) if engine.algo.automatic_entropy_tuning else None,
        "update_index":engine.update_index,"rng_state":cpu_tree(rng_state()),"replay_path":str(replay_path),"replay_metadata":{"transitions":replay.transitions,"episodes":replay.episodes},
        "training_context":cpu_tree(training_context),"evaluation":evaluation,"actor_architecture":actor_payload["architecture"]}
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".tmp"); torch.save(payload,temporary); os.replace(temporary,path)


def load_resume(path, engine, replay, device):
    payload=torch.load(path,map_location="cpu"); engine.actor.load_state_dict(payload["actor_state_dict"],strict=True);engine.critic.load_state_dict(payload["critic_state_dict"],strict=True);engine.target.load_state_dict(payload["target_critic_state_dict"],strict=True)
    if payload["group"]!=engine.config["group"]:raise RuntimeError(f"Resume checkpoint group mismatch: {payload['group']} != {engine.config['group']}")
    for key in ("sequence_length","gamma","tau","actor_target_lr","critic_joint_lr","critic_max_grad_norm"):
        if payload["config"][key]!=engine.config[key]:raise RuntimeError(f"Resume config mismatch for {key}")
    engine.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"]);engine.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"]);engine.update_index=int(payload["update_index"])
    if engine.algo.automatic_entropy_tuning:
        engine.algo.log_alpha_entropy.data.copy_(payload["log_alpha_entropy"].to(device));engine.algo.alpha_entropy_optim.load_state_dict(payload["alpha_optimizer_state_dict"]);engine.algo.alpha_entropy=float(engine.algo.log_alpha_entropy.exp().item())
    replay.load(payload["replay_path"]);restore_rng(payload["rng_state"]);return payload


def aggregate_updates(rows):
    if not rows:return {key:None for key in ("actor_loss","critic_loss","alpha","alpha_loss","entropy","Q1_mean","Q2_mean","target_q_mean","TD_error_mean","TD_error_std","critic_grad_norm","critic_fraction_clipped","actor_grad_norm")}
    keys=[key for key in rows[0] if key not in ("phase","actor_lr","critic_lr")]; result={}
    for key in keys:
        values=[row[key] for row in rows if row[key] is not None];result[key]=None if not values else float(np.mean(values))
    return result


def training_context(env, actor, obs, states, actions, rewards, dones, terminated, truncated, next_states, episode_return, episode_length):
    getter=getattr(env,"get_state",None)
    return {"active":bool(states),"env_state":None if not states or not callable(getter) else getter(),"observation":obs,
        "states":states,"actions":actions,"rewards":rewards,"dones":dones,"terminated":terminated,"truncated":truncated,"next_states":next_states,"episode_return":episode_return,"episode_length":episode_length,
        "actor_internal_state":cpu_tree(actor._state),"actor_counter":actor._counter}


def main():
    args=parse_args();config=effective_config(args.config,args);assert_phase_contract(config);device=select_device(args.device);seed_all(config["seed"])
    group_dir=Path(args.run_dir).resolve()/args.group;group_dir.mkdir(parents=True,exist_ok=True)
    for name in ("checkpoints","evaluations","debug_nan","replay"): (group_dir/name).mkdir(exist_ok=True)
    actor,actor_payload,critic,target,source=initialize_models(args.group,config,device);engine=Stage4SAC(actor,critic,target,config,device);replay=OnlineSequenceReplay(config)
    audit=model_audit(args.group,actor,critic,target,source,config);atomic_json(group_dir/"initialization_audit.json",audit);atomic_json(group_dir/"config.json",config)
    env=build_environment(actor_payload);schema=progress_observation_schema(env,actor_payload["observation_keys"],SHAPES)
    env_step=0;episode=1;completed_updates=0;best_key=(-1.0,-1.0,float("-inf"));evaluation_rows=[]
    obs=None;states=[];actions=[];rewards=[];dones=[];terminated=[];truncated=[];next_states=[];episode_return=0.0;episode_length=0
    try:
        if args.resume:
            payload=load_resume(args.resume,engine,replay,device);env_step=int(payload["env_step"]);episode=int(payload["episode"]);context=payload["training_context"]
            if context["active"]:
                if context["env_state"] is None:raise RuntimeError("Resume checkpoint lacks active environment state")
                obs=env.reset_to(context["env_state"]);states=context["states"];actions=context["actions"];rewards=context["rewards"];dones=context["dones"];terminated=context["terminated"];truncated=context["truncated"];next_states=context["next_states"];episode_return=float(context["episode_return"]);episode_length=int(context["episode_length"]);actor._counter=int(context["actor_counter"]);actor._state=None if context["actor_internal_state"] is None else tuple(item.to(device) for item in context["actor_internal_state"])
            completed_updates=engine.update_index
            metrics_path=group_dir/"evaluation_metrics.csv"
            if metrics_path.exists():
                with metrics_path.open(newline="",encoding="utf-8") as handle:
                    existing=list(csv.DictReader(handle))
                if existing:
                    best_key=max((float(row["success_rate"]),float(row["mean_progress"]),-int(row["env_step"])) for row in existing)
                    evaluation_rows=[{"env_step":int(existing[-1]["env_step"])}]
        evaluation_steps=set(range(0,int(config["total_env_steps"])+1,int(config["evaluation_interval"])))|{int(config["actor_freeze_steps"]),int(config["actor_warmup_end"]),int(config["total_env_steps"])}
        checkpoint_steps=set(map(int,config["checkpoint_steps"]))
        if env_step==0:
            replay_before=replay.transitions;report=evaluate(actor,actor_payload,config,device,0,group_dir/"evaluations"/"step_00000000.json");evaluation_rows.append(report)
            if replay.transitions!=replay_before:raise RuntimeError("Evaluation polluted online replay")
            append_csv(group_dir/"evaluation_metrics.csv",{key:value for key,value in report.items() if key!="episodes"})
            context=training_context(env,actor,obs,states,actions,rewards,dones,terminated,truncated,next_states,episode_return,episode_length)
            save_checkpoint(group_dir/"checkpoints"/"step_00000000.pth",engine,replay,config,0,episode,actor_payload,context,report)
            save_checkpoint(group_dir/"checkpoints"/"best_success.pth",engine,replay,config,0,episode,actor_payload,context,report);best_key=(report["success_rate"],report["mean_progress"],0)
        while env_step<int(config["total_env_steps"]):
            if obs is None:
                episode_seed=int(config["seed"])+episode;seed_all(episode_seed);try_seed_environment(env,episode_seed);obs=env.reset();seed_all(episode_seed);actor.reset()
            state,_=flatten(obs,actor_payload["observation_keys"]);tensor=torch.as_tensor(state[None],dtype=torch.float32,device=device)
            with torch.no_grad():action=actor.act(tensor,deterministic=False)[0][0].cpu().numpy()
            if not np.isfinite(action).all():raise RuntimeError("Non-finite environment action")
            next_obs,reward,raw_done,_=env.step(action);next_state,canonical=flatten(next_obs,actor_payload["observation_keys"]);progress=extract_transport_progress(canonical,schema);success=env_success(env)
            env_step+=1;episode_length+=1;episode_return+=float(reward);finished=bool(raw_done or (config["terminate_on_success"] and success) or episode_length>=int(config["evaluation_horizon"]) or env_step>=int(config["total_env_steps"]))
            states.append(state);actions.append(action);rewards.append(float(reward));dones.append(float(finished));terminated.append(float(finished and raw_done));truncated.append(float(finished and not raw_done));next_states.append(next_state);obs=next_obs
            if env_step in evaluation_steps:
                replay_before=replay.transitions;report=evaluate(actor,actor_payload,config,device,env_step,group_dir/"evaluations"/f"step_{env_step:08d}.json");evaluation_rows.append(report)
                if replay.transitions!=replay_before:raise RuntimeError("Evaluation polluted online replay")
                append_csv(group_dir/"evaluation_metrics.csv",{key:value for key,value in report.items() if key!="episodes"})
                key=(report["success_rate"],report["mean_progress"],-env_step);context=training_context(env,actor,obs,states,actions,rewards,dones,terminated,truncated,next_states,episode_return,episode_length)
                if key>best_key:save_checkpoint(group_dir/"checkpoints"/"best_success.pth",engine,replay,config,env_step,episode,actor_payload,context,report);best_key=key
            if env_step in checkpoint_steps:
                context=training_context(env,actor,obs,states,actions,rewards,dones,terminated,truncated,next_states,episode_return,episode_length);save_checkpoint(group_dir/"checkpoints"/f"step_{env_step:08d}.pth",engine,replay,config,env_step,episode,actor_payload,context,evaluation_rows[-1] if evaluation_rows and evaluation_rows[-1]["env_step"]==env_step else None)
            if finished:
                replay.add_episode(states,actions,rewards,dones,next_states,terminated,truncated);new_budget=max(0,int(np.floor(replay.transitions*float(config["updates_per_env_step"])))-completed_updates);update_rows=[]
                if replay.transitions>=int(config["minimum_replay_size"]):
                    for _ in range(new_budget):update_rows.append(engine.update(replay.sample(config["batch_size"],device),env_step,group_dir/"debug_nan"))
                    completed_updates+=new_budget
                phase=phase_at(env_step,config);metrics={"env_step":env_step,"episode":episode,"phase":phase["phase"],"actor_lr":phase["actor_lr"],"critic_lr":phase["critic_lr"],**aggregate_updates(update_rows),"buffer_size":replay.transitions,"episode_return":episode_return,"episode_length":episode_length,"success":int(success),"trash_ever":int(progress["trash_in_trash_bin"]),"payload_ever":int(progress["payload_in_target_bin"]),"updates_this_episode":len(update_rows),"gradient_updates":completed_updates}
                append_csv(group_dir/"training_metrics.csv",metrics);print(f"{args.group} env_step={env_step}/{config['total_env_steps']} episode={episode} phase={phase['phase']} success={int(success)} replay={replay.transitions} updates={completed_updates}",flush=True)
                episode+=1;obs=None;states=[];actions=[];rewards=[];dones=[];terminated=[];truncated=[];next_states=[];episode_return=0.0;episode_length=0
        final_report=evaluate(actor,actor_payload,config,device,env_step,group_dir/"evaluations"/f"step_{env_step:08d}.json");append_csv(group_dir/"evaluation_metrics.csv",{key:value for key,value in final_report.items() if key!="episodes"})
        context=training_context(env,actor,obs,states,actions,rewards,dones,terminated,truncated,next_states,episode_return,episode_length);save_checkpoint(group_dir/"checkpoints"/f"step_{env_step:08d}.pth",engine,replay,config,env_step,episode,actor_payload,context,final_report);save_checkpoint(group_dir/"checkpoints"/"last.pth",engine,replay,config,env_step,episode,actor_payload,context,final_report)
        atomic_json(group_dir/"status.json",{"status":"COMPLETE","group":args.group,"env_step":env_step,"gradient_updates":completed_updates,"actor_hash_initial":audit["actor_hash"],"actor_hash_final":state_hash(actor),"critic_hash_initial":audit["critic_hash"],"nan_inf_failure":False})
        print(f"STAGE4 GROUP COMPLETE: {args.group} env_step={env_step}",flush=True)
    except BaseException as error:
        atomic_json(group_dir/"status.json",{"status":"FAILED","group":args.group,"env_step":env_step,"error":repr(error),"nan_inf_failure":"finite" in str(error).lower()});raise
    finally:close_env(env)


if __name__=="__main__":main()
