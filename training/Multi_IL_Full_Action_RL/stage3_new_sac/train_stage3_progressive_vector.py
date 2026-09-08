#!/usr/bin/env python3
"""Sixteen-environment trainer exclusively for progressive RNN handoff."""
from __future__ import annotations
import argparse,json,math,os,time
from pathlib import Path
import numpy as np,torch
from stage3_new_agent import Stage3SAC,build_actor,progressive_schedule,state_hash,strict_stage2_load
from stage3_new_dataset import ExpertDataset
from stage3_new_evaluation import build_env,close_env,flatten
from stage3_new_handoff import BatchedFrozenRNNProposer,FrozenRNNProposer,evaluate_handoff,select_actions
from stage3_new_replay import SymmetricSampler,TransitionBuffer
from stage3_progressive_vector_env import StaggeredVectorEnv
from train_stage3_new import aggregate_metrics,device,file_hash,log,read,write

def arguments():
    p=argparse.ArgumentParser();p.add_argument("--group",required=True,choices=("rnn_q","multi_q"));p.add_argument("--device",required=True);p.add_argument("--pair-run-dir",required=True);p.add_argument("--critic-init-checkpoint",required=True);p.add_argument("--total-env-steps",type=int,default=300000);p.add_argument("--num-envs",type=int);return p.parse_args()
def save_checkpoint(path,agent,online,config,group,env_steps,episodes,stage2,vector_steps):
    path=Path(path);replay=path.with_suffix(".replay.npz");online.save(replay);torch.save({"env_steps":env_steps,"episodes":episodes,"vector_steps":vector_steps,"actor_state_dict":agent.actor.state_dict(),"critic_state_dict":agent.critic.state_dict(),"target_critic_state_dict":agent.target.state_dict(),"actor_optimizer_state_dict":agent.actor_optimizer.state_dict(),"critic_optimizer_state_dict":agent.critic_optimizer.state_dict(),"log_alpha":agent.log_alpha.detach().cpu(),"alpha_optimizer_state_dict":agent.alpha_optimizer.state_dict(),"gradient_updates":agent.updates,"config":config,"group":group,"stage2_checkpoint":stage2,"replay_path":str(replay)},path)
def main():
    a=arguments();pair=Path(a.pair_run_dir).resolve();cfg=read(pair/"shared"/"config_resolved.json");progressive=cfg.get("progressive_unfreeze",{})
    if not progressive.get("enabled",False):raise RuntimeError("Vector trainer is exclusive to the progressive experiment")
    num_envs=int(a.num_envs or cfg["parallel_env"]["num_envs"]);cfg["total_env_steps"]=int(a.total_env_steps);cfg["resolved_num_envs"]=num_envs;parallel=cfg["parallel_env"]
    sources=read(pair/"shared"/"stage2_source_manifest.json");seeds=read(pair/"shared"/"seed_manifest.json");expected=str(Path(sources[a.group]["checkpoint"]).resolve());actual=str(Path(a.critic_init_checkpoint).resolve())
    if actual!=expected:raise RuntimeError("Critic checkpoint differs from prepared pair")
    d=device(a.device);group_dir=pair/a.group
    for name in ("checkpoints","evaluations"): (group_dir/name).mkdir(exist_ok=True)
    expert=ExpertDataset(cfg["expert_dataset"],cfg["training_seed"],cfg["expert_rnn_proposal_cache"]);actor=build_actor(cfg,d);payload=torch.load(pair/"shared"/"actor_init.pth",map_location=d);actor.load_state_dict(payload["actor_state_dict"],strict=True)
    if state_hash(actor)!=payload["actor_hash"]:raise RuntimeError("Shared Actor hash mismatch")
    critic,stage2_payload=strict_stage2_load(actual,d,cfg);agent=Stage3SAC(actor,critic,cfg,d);online=TransitionBuffer(cfg["online_replay_capacity"],59,14,cfg["training_seed"]);sampler=SymmetricSampler(expert,online,cfg["training_seed"]);proposer=BatchedFrozenRNNProposer(cfg["bc_rnn_checkpoint"],d,num_envs)
    vector=None;eval_env=None;started=time.monotonic();timing={key:0. for key in ("rollout","env","actor","rnn","selector","replay","update")};metric_window=[];last_timing_step=0;last_timing_time=started
    env_steps=0;global_episode_index=num_envs;episodes=num_envs;successes=0;selector_counts={"rnn":0,"rl":0}
    try:
        vector=StaggeredVectorEnv(cfg["expert_dataset"],num_envs,cfg["train_seed_base"],parallel["env_startup_delay_sec"],parallel["env_startup_timeout_sec"],parallel["multiprocessing_start_method"]);observations=list(vector.initial_observations);rnn_actions=proposer.actions(observations)
        contexts=[{"episode_id":i,"seed":cfg["train_seed_base"]+i,"length":0,"return":0.,"rnn":0,"rl":0} for i in range(num_envs)]
        write(group_dir/"runtime_audit.json",{"group":a.group,"device":str(d),"num_envs":num_envs,"vector_backend":"spawn subprocess pipes with staggered READY handshake","actor_sha256":file_hash(pair/"shared"/"actor_init.pth"),"stage2_checkpoint":actual,"aggregate_env_steps":True,"utd":1,"progressive_unfreeze":progressive,"total_env_steps":cfg["total_env_steps"]})
        print(f"[STAGE3-VECTOR] group={a.group} num_envs={num_envs} total_aggregate_env_steps={cfg['total_env_steps']} UTD=1",flush=True)
        eval_env=build_env(cfg["expert_dataset"]);eval_proposer=FrozenRNNProposer(cfg["bc_rnn_checkpoint"],d);evaluation_steps=set(map(int,cfg["evaluation_env_steps"]))|{cfg["total_env_steps"]};checkpoint_steps=set(map(int,cfg["checkpoint_env_steps"]))|{cfg["total_env_steps"]};selector_steps=set(map(int,cfg["selector_diagnostic_env_steps"]))|{cfg["total_env_steps"]}
        while env_steps<cfg["total_env_steps"]:
            round_start=time.monotonic();remaining=cfg["total_env_steps"]-env_steps;active=list(range(min(num_envs,remaining)));states=np.stack([flatten(observations[i]) for i in active]);t=time.monotonic()
            with torch.no_grad():rl=actor(torch.as_tensor(states,dtype=torch.float32,device=d),reparameterize=True,return_log_prob=False)[0]
            timing["actor"]+=time.monotonic()-t;t=time.monotonic();rnn=torch.as_tensor(rnn_actions[active],dtype=torch.float32,device=d);timing["rnn"]+=time.monotonic()-t;t=time.monotonic();chosen,qrl,qrnn,wins=select_actions(agent.target,torch.as_tensor(states,dtype=torch.float32,device=d),rl,rnn);timing["selector"]+=time.monotonic()-t
            actions=chosen.cpu().numpy();rl_np=rl.cpu().numpy();qrl_np=qrl.cpu().numpy();qrnn_np=qrnn.cpu().numpy();wins_np=wins.cpu().numpy();t=time.monotonic();results=vector.step(actions,active);timing["env"]+=time.monotonic()-t
            valid=[(env_id,message) for env_id,message in results if message[0]=="OK"]
            if valid:
                ids=[item[0] for item in valid];next_obs=[item[1][1] for item in valid];t=time.monotonic();next_rnn=proposer.actions_for(ids,next_obs);timing["rnn"]+=time.monotonic()-t;next_rnn_by_id={env_id:value for env_id,value in zip(ids,next_rnn)}
            for env_id,message in results:
                if message[0]=="FATAL":
                    log(group_dir/"sim_fatal_errors.jsonl",{"env_steps":env_steps,"env_id":env_id,"episode_id":contexts[env_id]["episode_id"],"episode_seed":contexts[env_id]["seed"],"exception":message[1]});seed=cfg["train_seed_base"]+global_episode_index;global_episode_index+=1;episodes+=1;observations[env_id]=vector.reset(env_id,seed,True);proposer.reset_indices([env_id]);rnn_actions[env_id]=proposer.actions_for([env_id],[observations[env_id]])[0];contexts[env_id]={"episode_id":global_episode_index-1,"seed":seed,"length":0,"return":0.,"rnn":0,"rl":0};continue
                _,next_ob,reward,raw_done,won,_=message;ctx=contexts[env_id];ctx["length"]+=1;ctx["return"]+=reward;selected=int(wins_np[active.index(env_id)]);source="rl" if selected else "rnn";ctx[source]+=1;selector_counts[source]+=1;truncated=bool(ctx["length"]>=cfg["horizon"] and not won);terminal=bool((cfg["terminate_on_success"] and won) or (raw_done and not truncated));index=active.index(env_id);metadata={"action_exec":actions[index],"action_rl":rl_np[index],"action_rnn":rnn_actions[env_id],"rnn_next_actions":next_rnn_by_id[env_id],"selected_source":[selected],"q_select_rl":[qrl_np[index]],"q_select_rnn":[qrnn_np[index]],"q_select_margin":[qrl_np[index]-qrnn_np[index]],"is_online":[1],"env_id":[env_id],"episode_id":[ctx["episode_id"]],"episode_seed":[ctx["seed"]]};t=time.monotonic();online.add(states[index],actions[index],reward,flatten(next_ob),terminal,metadata);env_steps+=1;timing["replay"]+=time.monotonic()-t
                if online.size>=cfg["min_online_replay_size"]:
                    t=time.monotonic();metrics=agent.update(sampler.sample(cfg["batch_size"]),env_steps=env_steps);timing["update"]+=time.monotonic()-t;metrics.update({"env_steps":env_steps,"gradient_updates":agent.updates,"online_replay_size":online.size,"actual_utd_after_replay_ready":agent.updates/max(1,env_steps-cfg["min_online_replay_size"]+1)});metric_window.append(metrics)
                    if len(metric_window)>=progressive["train_metrics_interval_updates"]:log(group_dir/"train_metrics.jsonl",aggregate_metrics(metric_window));metric_window=[]
                observations[env_id]=next_ob;rnn_actions[env_id]=next_rnn_by_id[env_id]
                if terminal or truncated:
                    successes+=int(won);log(group_dir/"episode_metrics.jsonl",{"aggregate_env_steps":env_steps,"env_id":env_id,"episode_id":ctx["episode_id"],"episode_seed":ctx["seed"],"episode_length":ctx["length"],"episode_return":ctx["return"],"success":bool(won),"terminated":terminal,"truncated":truncated,"rnn_selected_fraction":ctx["rnn"]/ctx["length"],"rl_selected_fraction":ctx["rl"]/ctx["length"]});seed=cfg["train_seed_base"]+global_episode_index;global_episode_index+=1;episodes+=1;observations[env_id]=vector.reset(env_id,seed);proposer.reset_indices([env_id]);rnn_actions[env_id]=proposer.actions_for([env_id],[observations[env_id]])[0];contexts[env_id]={"episode_id":global_episode_index-1,"seed":seed,"length":0,"return":0.,"rnn":0,"rl":0}
                if env_steps in checkpoint_steps:save_checkpoint(group_dir/"checkpoints"/f"step_{env_steps:06d}.pth",agent,online,cfg,a.group,env_steps,episodes,actual,vector.vector_steps)
                if env_steps in evaluation_steps:
                    report=evaluate_handoff(actor,agent.target,eval_proposer,eval_env,seeds["evaluation_seeds"],cfg["horizon"],d,cfg["sim_error_handling"]["evaluation_retry_count"]);write(group_dir/"evaluations"/f"step_{env_steps:06d}.json",report)
            timing["rollout"]+=time.monotonic()-round_start
            if env_steps-last_timing_step>=1000:
                now=time.monotonic();delta=env_steps-last_timing_step;wall=now-last_timing_time;row={"num_envs":num_envs,"vector_steps":vector.vector_steps,"env_steps":env_steps,"aggregate_env_steps_per_sec":delta/wall,"vector_steps_per_sec":1000/max(wall,1e-9)/num_envs,"rollout_wall_ms":1000*timing["rollout"],"env_step_wall_ms":1000*timing["env"],"actor_inference_ms":1000*timing["actor"],"rnn_inference_ms":1000*timing["rnn"],"selector_q_ms":1000*timing["selector"],"replay_insert_ms":1000*timing["replay"],"rl_update_ms":1000*timing["update"],"wall_time_sec":now-started};log(group_dir/"throughput_metrics.jsonl",row);print(f"{a.group} env_steps={env_steps}/{cfg['total_env_steps']} aggregate_env_steps_per_sec={row['aggregate_env_steps_per_sec']:.2f}",flush=True);last_timing_step=env_steps;last_timing_time=now;timing={key:0. for key in timing}
        if metric_window:log(group_dir/"train_metrics.jsonl",aggregate_metrics(metric_window))
        save_checkpoint(group_dir/"checkpoints"/"last.pth",agent,online,cfg,a.group,env_steps,episodes,actual,vector.vector_steps);write(group_dir/"summary.json",{"status":"COMPLETE","group":a.group,"env_steps":env_steps,"vector_steps":vector.vector_steps,"num_envs":num_envs,"gradient_updates":agent.updates,"episodes":episodes,"success_count":successes,"replay_size":online.size,"selector_counts":selector_counts,"actor_hash_final":state_hash(actor),"critic_hash_final":state_hash(critic)})
    finally:
        if eval_env is not None:close_env(eval_env)
        if vector is not None:vector.close()
if __name__=="__main__":main()
