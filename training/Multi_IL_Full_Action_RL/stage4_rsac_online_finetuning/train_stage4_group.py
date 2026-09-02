#!/usr/bin/env python3
"""Train one isolated Stage4 recurrent SAC group."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
for _name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_name]="1"
import random
import sys
import time
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
from stage4_parallel_env import Stage4ParallelEnvPool  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402


SHAPES = {"robot0_eef_pos": [3], "robot0_eef_quat": [4], "robot0_gripper_qpos": [2],
          "robot1_eef_pos": [3], "robot1_eef_quat": [4], "robot1_gripper_qpos": [2], "object": [41]}


def parse_args():
    parser = argparse.ArgumentParser(); parser.add_argument("--group", choices=GROUPS, required=True); parser.add_argument("--device", required=True)
    parser.add_argument("--config", default=str(HERE / "stage4_config.json")); parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int); parser.add_argument("--smoke-test", action="store_true"); parser.add_argument("--resume");parser.add_argument("--cpu-affinity-file")
    return parser.parse_args()


def effective_config(path, args):
    config = read_json(path); config["group"] = args.group; config["device"] = args.device
    if args.seed is not None: config["seed"] = args.seed
    if args.smoke_test:
        config.update(total_env_steps=40, actor_freeze_steps=10, actor_warmup_end=20, evaluation_interval=10,
                      evaluation_seed_start=10080, evaluation_seed_end=10081, evaluation_horizon=5,
                      replay_capacity=1000, minimum_replay_size=2, batch_size=2, updates_per_env_step=0.5,
                      checkpoint_steps=[0, 10, 20, 40])
    return config


def build_environment(actor_payload):
    checkpoint = FileUtils.maybe_dict_from_checkpoint(ckpt_path=actor_payload["teacher_checkpoint"])
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=checkpoint); ObsUtils.initialize_obs_utils_with_config(config)
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=checkpoint, render=False, render_offscreen=False, verbose=False)
    return env


def flatten(observation, keys):
    canonical = extract_canonical_observation(observation, keys, SHAPES)
    return np.concatenate([canonical[key].reshape(-1) for key in keys]).astype(np.float32), canonical


def unflatten(state, keys):
    result={};cursor=0
    for key in keys:
        size=int(np.prod(SHAPES[key]));result[key]=np.asarray(state[cursor:cursor+size]).reshape(SHAPES[key]);cursor+=size
    return result


@torch.no_grad()
def batched_actor_actions(actor, observations, hidden, counters, workers, device, deterministic=False):
    """One recurrent policy step for an arbitrary subset of environment workers."""
    indices=torch.as_tensor(workers,dtype=torch.long,device=device)
    selected=tuple(item.index_select(1,indices) for item in hidden)
    reset=torch.as_tensor([counters[worker] % actor.horizon == 0 for worker in workers],device=device)
    if bool(reset.any()):
        selected=tuple(item.masked_fill(reset.view(1,-1,1),0.0) for item in selected)
    states=torch.as_tensor(np.stack([observations[worker] for worker in workers]),dtype=torch.float32,device=device)
    output,new_hidden=actor.lstm(actor.observation_adapter(states).unsqueeze(1),selected)
    action=actor.policy(output[:,0],deterministic=deterministic,return_log_prob=False)[0]
    for part,new_part in zip(hidden,new_hidden):part.index_copy_(1,indices,new_part)
    for worker in workers:counters[worker]+=1
    return action.cpu().numpy()


def evaluate(actor, actor_payload, config, device, env_step, output, pool, schema, training_contexts):
    """Evaluate seeds concurrently on the existing persistent pool, then restore training envs."""
    saved_rng = rng_state();was_training=actor.training;active=sorted(training_contexts);saved_envs=pool.get_states(active) if active else {}
    seeds = list(range(int(config["evaluation_seed_start"]), int(config["evaluation_seed_end"]) + 1)); initial = load_initial_states(config["initial_states"], seeds)
    rows=[];actor.eval();pending=list(seeds);contexts={};hidden=(torch.zeros(2,pool.num_envs,400,device=device),torch.zeros(2,pool.num_envs,400,device=device));counters=np.zeros(pool.num_envs,dtype=np.int64)
    try:
        while pending or contexts:
            free=[w for w in range(min(pool.num_envs,int(config.get("evaluation_parallel_envs",pool.num_envs)))) if w not in contexts]
            assignments={}
            for worker in free:
                if not pending:break
                seed=pending.pop(0);contexts[worker]={"seed":seed,"observation":None,"return":0.0,"length":0,"success":False,"trash":False,"payload":False};assignments[worker]=copy.deepcopy(initial[seed])
            if assignments:
                observations=pool.reset_to(assignments,{w:0 for w in assignments})
                for w,o in observations.items():contexts[w]["observation"]=o
            workers=sorted(contexts);observations={w:contexts[w]["observation"] for w in workers};actions=batched_actor_actions(actor,observations,hidden,counters,workers,device,True);results=pool.step(workers,actions)
            for worker in list(workers):
                c=contexts[worker];state,reward,done,info=results[worker];c["observation"]=state;c["return"]+=float(reward);c["length"]+=1;c["success"]|=bool(info["success"]);progress=extract_transport_progress(unflatten(state,actor_payload["observation_keys"]),schema);c["trash"]|=bool(progress["trash_in_trash_bin"]);c["payload"]|=bool(progress["payload_in_target_bin"])
                if done or c["length"]>=int(config["evaluation_horizon"]):rows.append({"seed":c["seed"],"success":int(c["success"]),"trash":int(c["trash"]),"payload":int(c["payload"]),"progress":partial_progress_score(c["success"],c["trash"],c["payload"]),"return":c["return"],"length":c["length"]});del contexts[worker];counters[worker]=0;hidden[0][:,worker].zero_();hidden[1][:,worker].zero_()
    finally:
        if active:
            restored=pool.reset_to(saved_envs,{w:training_contexts[w]["length"] for w in active})
            for worker,state in restored.items():training_contexts[worker]["observation"]=state
        restore_rng(saved_rng);actor.train(was_training)
    rows.sort(key=lambda row:row["seed"])
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


def new_episode(episode, observation):
    return {"episode":int(episode),"observation":observation,"states":[],"actions":[],"rewards":[],"dones":[],
            "terminated":[],"truncated":[],"next_states":[],"return":0.0,"length":0,
            "success":False,"trash_ever":False,"payload_ever":False}


def training_context(pool, contexts, hidden, counters, next_episode):
    workers=sorted(contexts)
    return {"parallel_envs":pool.num_envs,"active_workers":workers,"env_states":pool.get_states(workers),
            "contexts":contexts,"actor_hidden":cpu_tree(hidden),"actor_counters":counters.tolist(),
            "next_episode":int(next_episode)}


def add_finished_episode(replay, context):
    replay.add_episode(context["states"],context["actions"],context["rewards"],context["dones"],
                       context["next_states"],context["terminated"],context["truncated"])

def cgroup_usage_usec():
    path=Path("/sys/fs/cgroup/cpu.stat")
    if not path.exists():return None
    values=dict(line.split() for line in path.read_text().splitlines() if len(line.split())==2)
    return int(values.get("usage_usec",0))


def main():
    args=parse_args();torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads()!=1:raise
    config=effective_config(args.config,args);assert_phase_contract(config);device=select_device(args.device);seed_all(config["seed"])
    cpu_ids=None
    if args.cpu_affinity_file:
        affinity=read_json(args.cpu_affinity_file);key="rnn_cpu_ids" if args.group=="rnn_only_critic" else "multi_cpu_ids";cpu_ids=affinity[key][:int(config["parallel_envs"])];config["cpu_affinity_file"]=str(Path(args.cpu_affinity_file).resolve());config["cpu_affinity_ids"]=cpu_ids
    group_dir=Path(args.run_dir).resolve()/args.group;group_dir.mkdir(parents=True,exist_ok=True)
    for name in ("checkpoints","evaluations","debug_nan","replay"): (group_dir/name).mkdir(exist_ok=True)
    actor,actor_payload,critic,target,source=initialize_models(args.group,config,device);engine=Stage4SAC(actor,critic,target,config,device);replay=OnlineSequenceReplay(config)
    audit=model_audit(args.group,actor,critic,target,source,config);atomic_json(group_dir/"initialization_audit.json",audit);atomic_json(group_dir/"config.json",config)
    schema_env=build_environment(actor_payload);schema=progress_observation_schema(schema_env,actor_payload["observation_keys"],SHAPES);close_env(schema_env)
    pool=Stage4ParallelEnvPool(actor_payload["teacher_checkpoint"],actor_payload["observation_keys"],SHAPES,
        config["parallel_envs"],config["evaluation_horizon"],config["terminate_on_success"],config["seed"],
        config["parallel_start_method"],config["parallel_startup_timeout_seconds"],config["parallel_step_timeout_seconds"],cpu_ids)
    env_step=0;next_episode=1;completed_updates=0;best_key=(-1.0,-1.0,float("-inf"));evaluation_rows=[];contexts={};perf={k:0.0 for k in ("rollout_seconds","policy_inference_seconds","replay_insert_seconds","critic_update_seconds","actor_update_seconds","evaluation_seconds","checkpoint_seconds")};perf_started=time.perf_counter();perf_step=0;cpu_started=time.process_time();resource_started=time.perf_counter();cgroup_started=cgroup_usage_usec()
    hidden=(torch.zeros(2,pool.num_envs,400,device=device),torch.zeros(2,pool.num_envs,400,device=device));counters=np.zeros(pool.num_envs,dtype=np.int64)
    try:
        if args.resume:
            payload=load_resume(args.resume,engine,replay,device);env_step=int(payload["env_step"]);context=payload["training_context"]
            if int(context["parallel_envs"])!=pool.num_envs:raise RuntimeError("Resume parallel_envs mismatch")
            contexts={int(key):value for key,value in context["contexts"].items()};workers=sorted(contexts)
            observations=pool.reset_to({int(key):value for key,value in context["env_states"].items()},{worker:contexts[worker]["length"] for worker in workers})
            for worker in workers:contexts[worker]["observation"]=observations[worker]
            hidden=tuple(item.to(device) for item in context["actor_hidden"]);counters=np.asarray(context["actor_counters"],dtype=np.int64);next_episode=int(context["next_episode"])
            completed_updates=engine.update_index
            metrics_path=group_dir/"evaluation_metrics.csv"
            if metrics_path.exists():
                with metrics_path.open(newline="",encoding="utf-8") as handle:
                    existing=list(csv.DictReader(handle))
                if existing:
                    best_key=max((float(row["success_rate"]),float(row["mean_progress"]),-int(row["env_step"])) for row in existing)
                    evaluation_rows=[{"env_step":int(existing[-1]["env_step"])}]
        evaluation_steps=set(range(0,int(config["total_env_steps"])+1,int(config["evaluation_interval"])))|{int(config["actor_freeze_steps"]),int(config["actor_warmup_end"]),int(config["total_env_steps"])}
        checkpoint_steps=set(map(int,config["checkpoint_steps"]));performance_steps=set(range(int(config.get("performance_log_interval",1000)),int(config["total_env_steps"])+1,int(config.get("performance_log_interval",1000))));resource_steps=set(range(int(config.get("resource_log_interval",5000)),int(config["total_env_steps"])+1,int(config.get("resource_log_interval",5000))))
        if env_step==0:
            replay_before=replay.transitions;t=time.perf_counter();report=evaluate(actor,actor_payload,config,device,0,group_dir/"evaluations"/"step_00000000.json",pool,schema,contexts);perf["evaluation_seconds"]+=time.perf_counter()-t;evaluation_rows.append(report)
            if replay.transitions!=replay_before:raise RuntimeError("Evaluation polluted online replay")
            append_csv(group_dir/"evaluation_metrics.csv",{key:value for key,value in report.items() if key!="episodes"})
            context={"parallel_envs":pool.num_envs,"active_workers":[],"env_states":{},"contexts":{},"actor_hidden":cpu_tree(hidden),"actor_counters":counters.tolist(),"next_episode":next_episode}
            save_checkpoint(group_dir/"checkpoints"/"step_00000000.pth",engine,replay,config,0,next_episode,actor_payload,context,report)
            save_checkpoint(group_dir/"checkpoints"/"best_success.pth",engine,replay,config,0,next_episode,actor_payload,context,report);best_key=(report["success_rate"],report["mean_progress"],0)
        while env_step<int(config["total_env_steps"]):
            empty=[worker for worker in range(pool.num_envs) if worker not in contexts]
            if empty:
                seeds=[]
                for worker in empty:seeds.append(int(config["seed"])+next_episode);contexts[worker]=new_episode(next_episode,None);next_episode+=1;counters[worker]=0;hidden[0][:,worker].zero_();hidden[1][:,worker].zero_()
                observations=pool.reset(empty,seeds)
                for worker in empty:contexts[worker]["observation"]=observations[worker]
            future=sorted(step for step in evaluation_steps|checkpoint_steps|performance_steps|resource_steps|{int(config["total_env_steps"])} if step>env_step);boundary=future[0]
            workers=sorted(contexts)[:min(len(contexts),boundary-env_step)]
            observations={worker:contexts[worker]["observation"] for worker in workers};t=time.perf_counter();batch_actions=batched_actor_actions(actor,observations,hidden,counters,workers,device);perf["policy_inference_seconds"]+=time.perf_counter()-t
            if not np.isfinite(batch_actions).all():raise RuntimeError("Non-finite environment action")
            t=time.perf_counter();results=pool.step(workers,batch_actions);perf["rollout_seconds"]+=time.perf_counter()-t;finished_contexts=[]
            for worker,action in zip(workers,batch_actions):
                context=contexts[worker];state=context["observation"];next_state,reward,finished,info=results[worker];canonical=unflatten(next_state,actor_payload["observation_keys"]);progress=extract_transport_progress(canonical,schema)
                context["states"].append(state);context["actions"].append(action);context["rewards"].append(float(reward));context["dones"].append(float(finished));context["terminated"].append(float(info["terminated"]));context["truncated"].append(float(info["truncated"]));context["next_states"].append(next_state)
                context["observation"]=next_state;context["return"]+=float(reward);context["length"]+=1;context["success"]|=bool(info["success"]);context["trash_ever"]|=bool(progress["trash_in_trash_bin"]);context["payload_ever"]|=bool(progress["payload_in_target_bin"]);env_step+=1
                if finished:finished_contexts.append(context);del contexts[worker];counters[worker]=0;hidden[0][:,worker].zero_();hidden[1][:,worker].zero_()
            if env_step==int(config["total_env_steps"]):
                for worker in sorted(contexts):
                    context=contexts[worker]
                    if context["states"]:
                        context["dones"][-1]=1.0;context["terminated"][-1]=0.0;context["truncated"][-1]=1.0;finished_contexts.append(context)
                contexts.clear()
            t=time.perf_counter()
            for context in finished_contexts:add_finished_episode(replay,context)
            perf["replay_insert_seconds"]+=time.perf_counter()-t
            new_budget=max(0,int(np.floor(replay.transitions*float(config["updates_per_env_step"])))-completed_updates);update_rows=[]
            if replay.transitions>=int(config["minimum_replay_size"]):
                for _ in range(new_budget):
                    update_rows.append(engine.update(replay.sample(config["batch_size"],device),env_step,group_dir/"debug_nan"));perf["critic_update_seconds"]+=engine.last_update_timing["critic_update_seconds"];perf["actor_update_seconds"]+=engine.last_update_timing["actor_update_seconds"]
                completed_updates+=new_budget
            summary=aggregate_updates(update_rows)
            for context in finished_contexts:
                phase=phase_at(env_step,config);metrics={"env_step":env_step,"episode":context["episode"],"phase":phase["phase"],"actor_lr":phase["actor_lr"],"critic_lr":phase["critic_lr"],**summary,"buffer_size":replay.transitions,"episode_return":context["return"],"episode_length":context["length"],"success":int(context["success"]),"trash_ever":int(context["trash_ever"]),"payload_ever":int(context["payload_ever"]),"updates_this_batch":len(update_rows),"gradient_updates":completed_updates}
                append_csv(group_dir/"training_metrics.csv",metrics);print(f"{args.group} env_step={env_step}/{config['total_env_steps']} episode={context['episode']} phase={phase['phase']} success={int(context['success'])} replay={replay.transitions} updates={completed_updates}",flush=True)
            if env_step in evaluation_steps:
                replay_before=replay.transitions;t=time.perf_counter();report=evaluate(actor,actor_payload,config,device,env_step,group_dir/"evaluations"/f"step_{env_step:08d}.json",pool,schema,contexts);perf["evaluation_seconds"]+=time.perf_counter()-t;evaluation_rows.append(report)
                if replay.transitions!=replay_before:raise RuntimeError("Evaluation polluted online replay")
                append_csv(group_dir/"evaluation_metrics.csv",{key:value for key,value in report.items() if key!="episodes"})
                key=(report["success_rate"],report["mean_progress"],-env_step);context=training_context(pool,contexts,hidden,counters,next_episode)
                if key>best_key:save_checkpoint(group_dir/"checkpoints"/"best_success.pth",engine,replay,config,env_step,next_episode,actor_payload,context,report);best_key=key
            if env_step in checkpoint_steps:
                t=time.perf_counter();context=training_context(pool,contexts,hidden,counters,next_episode);save_checkpoint(group_dir/"checkpoints"/f"step_{env_step:08d}.pth",engine,replay,config,env_step,next_episode,actor_payload,context,evaluation_rows[-1] if evaluation_rows and evaluation_rows[-1]["env_step"]==env_step else None);perf["checkpoint_seconds"]+=time.perf_counter()-t
            if env_step in performance_steps:
                elapsed=time.perf_counter()-perf_started;append_csv(group_dir/"performance_metrics.csv",{"env_step":env_step,"env_steps_per_second":(env_step-perf_step)/elapsed,**perf});perf={key:0.0 for key in perf};perf_started=time.perf_counter();perf_step=env_step
            if env_step in resource_steps:
                wall=time.perf_counter()-resource_started;usage=cgroup_usage_usec();append_csv(group_dir/"resource_metrics.csv",{"env_step":env_step,"process_cpu_percentage":100.0*(time.process_time()-cpu_started)/wall,"live_environment_workers":pool.live_worker_count(),"cgroup_cpu_usage_usec":usage,"cgroup_cpu_usage_delta_usec":None if usage is None or cgroup_started is None else usage-cgroup_started});cpu_started=time.process_time();resource_started=time.perf_counter();cgroup_started=usage
        final_report=evaluation_rows[-1]
        context=training_context(pool,contexts,hidden,counters,next_episode);save_checkpoint(group_dir/"checkpoints"/f"step_{env_step:08d}.pth",engine,replay,config,env_step,next_episode,actor_payload,context,final_report);save_checkpoint(group_dir/"checkpoints"/"last.pth",engine,replay,config,env_step,next_episode,actor_payload,context,final_report)
        atomic_json(group_dir/"status.json",{"status":"COMPLETE","group":args.group,"env_step":env_step,"gradient_updates":completed_updates,"actor_hash_initial":audit["actor_hash"],"actor_hash_final":state_hash(actor),"critic_hash_initial":audit["critic_hash"],"nan_inf_failure":False})
        print(f"STAGE4 GROUP COMPLETE: {args.group} env_step={env_step}",flush=True)
    except BaseException as error:
        atomic_json(group_dir/"status.json",{"status":"FAILED","group":args.group,"env_step":env_step,"error":repr(error),"nan_inf_failure":"finite" in str(error).lower()});raise
    finally:pool.close()


if __name__=="__main__":main()
