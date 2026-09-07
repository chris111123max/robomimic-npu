#!/usr/bin/env python3
"""Run one member of the prepared Stage3-new paired SAC experiment."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random
from datetime import datetime
from pathlib import Path
import numpy as np, torch
from stage3_new_agent import Stage3SAC, build_actor, state_hash, strict_stage2_load
from stage3_new_anchor import AnchorData,geometry_diagnostics,probe_geometry
from stage3_new_dataset import ExpertDataset
from stage3_new_evaluation import build_env, close_env, evaluate, flatten, mujoco_fatal_error_type, reset_seed, seed_all, success
from stage3_new_probe import load_fixed_probes, record_probe
from stage3_new_replay import SymmetricSampler, TransitionBuffer

def read(path):
    with open(path,encoding="utf-8") as f:return json.load(f)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temp,"w",encoding="utf-8") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n")
    os.replace(temp,path)
def log(path,value):
    with open(path,"a",encoding="utf-8") as f:f.write(json.dumps(value,sort_keys=True)+"\n")
def contract_hash(config):
    ignored={"resolved_device","total_env_steps"};payload={k:v for k,v in config.items() if k not in ignored};return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
def file_hash(path):
    digest=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()
def environment_action_bounds(env):
    current=env;seen=set()
    while current is not None and id(current) not in seen:
        seen.add(id(current));spec=getattr(current,"action_spec",None)
        if spec is not None:
            low,high=spec;return np.asarray(low,np.float32),np.asarray(high,np.float32)
        current=getattr(current,"env",None)
    raise RuntimeError("Cannot obtain action_space low/high from robosuite action_spec")
def fixed_diagnostic_batch(source,count,seed):
    size=int(source.size);rng=np.random.default_rng(int(seed));indices=rng.choice(size,min(int(count),size),replace=False);return {key:value[indices].copy() for key,value in source.data.items()}
def device(name):
    if name.startswith("npu"):
        try:import torch_npu  # noqa:F401
        except ImportError as e:raise RuntimeError("NPU requested but torch_npu is unavailable") from e
        if not torch.npu.is_available():raise RuntimeError("NPU requested but torch.npu is unavailable")
        torch.npu.set_device(name)
    return torch.device(name)
def args():
    p=argparse.ArgumentParser();p.add_argument("--group",required=True,choices=("rnn_q","multi_q"));p.add_argument("--device",required=True);p.add_argument("--pair-run-dir",required=True);p.add_argument("--critic-init-checkpoint",required=True);p.add_argument("--resume");p.add_argument("--total-env-steps",type=int);return p.parse_args()
def rng_state():
    result={"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state()};npu=getattr(torch,"npu",None)
    if npu is not None and npu.is_available():result["npu"]=npu.get_rng_state()
    return result
def restore_rng(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch"])
    if "npu" in state:torch.npu.set_rng_state(state["npu"])
def save(path,agent,replay,sampler,offline,config,group,stage2,env_steps,episode_index,context,env,anchor_data=None,sim_fatal_error_count=0):
    path=Path(path);replay_path=path.with_suffix(".replay.npz");replay.save(replay_path)
    env_state=None
    if context is not None:
        fn=getattr(env,"get_state",None)
        if callable(fn):env_state=copy.deepcopy(fn())
    payload={"env_steps":env_steps,"episode_index":episode_index,"episode_context":context,"environment_state":env_state,"actor_state_dict":agent.actor.state_dict(),"critic_state_dict":agent.critic.state_dict(),"target_critic_state_dict":agent.target.state_dict(),"actor_optimizer_state_dict":agent.actor_optimizer.state_dict(),"critic_optimizer_state_dict":agent.critic_optimizer.state_dict(),"log_alpha":agent.log_alpha.detach().cpu(),"alpha_optimizer_state_dict":agent.alpha_optimizer.state_dict(),"gradient_updates":agent.updates,"config":config,"group":group,"stage2_checkpoint":stage2,"training_seed":config["training_seed"],"replay_path":str(replay_path),"rng_state":rng_state(),"offline_rng_state":offline.rng.bit_generator.state,"sampler_rng_state":sampler.rng.bit_generator.state,"sampler_turn":sampler.turn,"sampler_counts":sampler.counts,"anchor_sampler_state":anchor_data.state_dict() if anchor_data is not None else None,"sim_fatal_error_count":int(sim_fatal_error_count)}
    torch.save(payload,path);torch.save(payload,path.parent/"latest.pth")
def load_resume(path,agent,offline,config,env,anchor_data=None):
    p=torch.load(path,map_location=agent.device)
    if p["group"] not in ("rnn_q","multi_q") or contract_hash(p["config"])!=contract_hash(config):raise RuntimeError("Resume checkpoint is incompatible with pair config")
    agent.actor.load_state_dict(p["actor_state_dict"]);agent.critic.load_state_dict(p["critic_state_dict"]);agent.target.load_state_dict(p["target_critic_state_dict"]);agent.actor_optimizer.load_state_dict(p["actor_optimizer_state_dict"]);agent.critic_optimizer.load_state_dict(p["critic_optimizer_state_dict"]);agent.log_alpha.data.copy_(p["log_alpha"].to(agent.device));agent.alpha_optimizer.load_state_dict(p["alpha_optimizer_state_dict"]);agent.updates=int(p["gradient_updates"])
    replay=TransitionBuffer.load(p["replay_path"]);sampler=SymmetricSampler(offline,replay,config["training_seed"]);offline.rng.bit_generator.state=p["offline_rng_state"];sampler.rng.bit_generator.state=p["sampler_rng_state"];sampler.turn=int(p["sampler_turn"]);sampler.counts=p["sampler_counts"]
    context=p["episode_context"]
    if context is not None:
        if p["environment_state"] is None:raise RuntimeError("Cannot resume mid-episode without saved environment state")
        observation=env.reset_to(copy.deepcopy(p["environment_state"]));context["observation"]=observation
    if anchor_data is not None and p.get("anchor_sampler_state") is not None:anchor_data.load_state_dict(p["anchor_sampler_state"])
    restore_rng(p["rng_state"]);return p,replay,sampler,context
def preserved_evaluation(actor,env,seeds,horizon,device_,retries=0):
    state=rng_state()
    try:return evaluate(actor,env,seeds,horizon,device_,retries)
    finally:restore_rng(state)
def main():
    a=args();pair=Path(a.pair_run_dir).resolve();config=read(pair/"shared"/"config_resolved.json");sources=read(pair/"shared"/"stage2_source_manifest.json");seeds=read(pair/"shared"/"seed_manifest.json")
    if not config.get("automatic_entropy_tuning") or "alpha_init" not in config:raise RuntimeError("Prepared pair does not use the explicit automatic-alpha initialization contract")
    if a.total_env_steps is not None:config["total_env_steps"]=a.total_env_steps
    expected=str(Path(sources[a.group]["checkpoint"]).resolve());actual=str(Path(a.critic_init_checkpoint).resolve())
    if expected!=actual:raise RuntimeError(f"Critic checkpoint differs from prepared pair: {actual} != {expected}")
    d=device(a.device);group_dir=pair/a.group
    for name in ("checkpoints","evaluations","probes"): (group_dir/name).mkdir(exist_ok=True)
    actor_init_path=pair/"shared"/"actor_init.pth";seed_manifest_path=pair/"shared"/"seed_manifest.json";expert=ExpertDataset(config["expert_dataset"],config["training_seed"]);actor=build_actor(config,d);actor_payload=torch.load(actor_init_path,map_location=d);actor.load_state_dict(actor_payload["actor_state_dict"],strict=True)
    if state_hash(actor)!=actor_payload["actor_hash"]:raise RuntimeError("Shared Actor initialization hash mismatch")
    critic,stage2_payload=strict_stage2_load(actual,d,config);anchor_cfg=config.get("anchor",{});anchor_enabled=bool(anchor_cfg.get("enabled",False));teacher=None;anchor_data=None
    if anchor_enabled:
        teacher,_=strict_stage2_load(actual,d,config);anchor_data=AnchorData(config["stage2_run_dir"],a.group,int(anchor_cfg["rng_seed"]));teacher.eval();teacher.requires_grad_(False)
    env=build_env(config["expert_dataset"]);eval_env=build_env(config["expert_dataset"]);action_low,action_high=environment_action_bounds(env);agent=Stage3SAC(actor,critic,config,d,action_low,action_high,teacher);online=TransitionBuffer(config["online_replay_capacity"],59,14,config["training_seed"]);sampler=SymmetricSampler(expert,online,config["training_seed"])
    expected_alpha=float(config["alpha_init"]);actual_alpha=float(agent.alpha.item())
    if not math.isclose(actual_alpha,expected_alpha,rel_tol=1e-5,abs_tol=1e-8):raise RuntimeError(f"alpha step-0 sanity check failed: {actual_alpha} != {expected_alpha}")
    probes,probe_manifest=load_fixed_probes(config["stage2_run_dir"])
    env_steps=episode_index=0;context=None;terminated_count=truncated_count=success_count=0;sim_fatal_error_count=consecutive_sim_fatal_errors=0
    if a.resume:
        payload,online,sampler,context=load_resume(a.resume,agent,expert,config,env,anchor_data);env_steps=int(payload["env_steps"]);episode_index=int(payload["episode_index"]);sim_fatal_error_count=int(payload.get("sim_fatal_error_count",0))
    else:seed_all(config["training_seed"])
    current_alpha=float(agent.alpha.item())
    audit={"group":a.group,"device":str(d),"pair_contract_hash":contract_hash(config),"experiment_tag":config.get("experiment_tag"),"actor":{"class":"TanhGaussianPolicy","hidden_dims":config["hidden_dims"],"source":str(actor_init_path),"initial_hash":actor_payload["actor_hash"],"artifact_sha256":file_hash(actor_init_path),"pretrained":False,"random_shared":True},"critic":{"stage2_checkpoint":actual,"model_config":stage2_payload["model_config"],"online_equals_target_step0":not bool(a.resume)},"cql":config.get("cql",{"enabled":False}),"anchor":anchor_cfg,"anchor_data":anchor_data.config if anchor_data is not None else None,"teacher":{"checkpoint":actual,"frozen":anchor_enabled,"in_optimizer":False} if anchor_enabled else None,"action_low":action_low.tolist(),"action_high":action_high.tolist(),"cql_random_seed_protocol":"shared training seed; one vectorized device-local draw per update; fixed diagnostics restore training RNG","fresh_run":not bool(a.resume),"initial_state":{"env_steps":env_steps,"gradient_updates":agent.updates,"online_replay_size":online.size},"gamma":config["gamma"],"utd":config["utd"],"batch_size":config["batch_size"],"critic_weight_decay":agent.critic_optimizer.param_groups[0]["weight_decay"],"automatic_entropy_tuning":config["automatic_entropy_tuning"],"alpha_init":config["alpha_init"],"log_alpha_init":math.log(float(config["alpha_init"])),"alpha_at_step0":actual_alpha if not a.resume else None,"target_entropy":config["target_entropy"],"expert_dataset":expert.audit(),"online_replay_capacity":online.capacity,"min_online_replay_size":config["min_online_replay_size"],"total_env_steps":config["total_env_steps"],"seed_manifest_sha256":file_hash(seed_manifest_path),"evaluation_seeds":seeds["evaluation_seeds"],"train_seed_rule":seeds["train_seed_rule"]}
    peer=pair/("multi_q" if a.group=="rnn_q" else "rnn_q")/"runtime_audit.json"
    if peer.exists():
        peer_audit=read(peer)
        if peer_audit.get("pair_contract_hash")!=audit["pair_contract_hash"]:raise RuntimeError("Peer process uses a different shared SAC configuration")
        if not a.resume and int(peer_audit["total_env_steps"])!=int(audit["total_env_steps"]):raise RuntimeError("Fresh paired processes use different environment-step budgets")
        if peer_audit.get("actor",{}).get("artifact_sha256")!=audit["actor"]["artifact_sha256"] or peer_audit.get("seed_manifest_sha256")!=audit["seed_manifest_sha256"]:raise RuntimeError("Peer process does not use identical shared Actor and seeds")
    write(group_dir/"runtime_audit.json",audit)
    alpha_label="alpha_at_step0" if not a.resume else "alpha_at_resume"
    print(f"{a.group} automatic_entropy_tuning={config['automatic_entropy_tuning']} alpha_init={config['alpha_init']} log_alpha_init={math.log(float(config['alpha_init'])):.9f} {alpha_label}={current_alpha:.9f}",flush=True)
    print(f"{a.group} fresh_run={not bool(a.resume)} env_steps={env_steps} gradient_updates={agent.updates} online_replay_size={online.size} Actor_source={actor_init_path} Critic_source={actual}",flush=True)
    cql_cfg=config.get("cql",{});print(f"[STAGE3-CQL] group={a.group} actor_init=random_shared cql_enabled={bool(cql_cfg.get('enabled',False))} lambda_cql={cql_cfg.get('lambda',0.0)} num_random_actions={cql_cfg.get('num_random_actions',0)} num_policy_actions={cql_cfg.get('num_policy_actions',0)} apply_expert={cql_cfg.get('apply_to_expert',False)} apply_online={cql_cfg.get('apply_to_online',False)} detach_policy_actions={cql_cfg.get('detach_policy_actions',False)} alpha_init={config['alpha_init']} utd={config['utd']}",flush=True)
    if anchor_enabled:print(f"[STAGE3-ANCHOR] group={a.group} type=zscore_functional lambda_anchor={anchor_cfg['lambda']} batch_size={anchor_cfg['batch_size']} teacher_frozen=True sampling={anchor_data.config['sampling']}",flush=True)
    total=int(config["total_env_steps"]);eval_steps=set(map(int,config["evaluation_env_steps"]))|{total};probe_steps=set(map(int,config["probe_env_steps"]))|{total};checkpoint_steps=set(map(int,config["checkpoint_env_steps"]))|{total};diag_interval=int(cql_cfg.get("source_diagnostic_interval_env_steps",5000));diagnostic_steps=set(range(diag_interval,total+1,diag_interval))|({1000,total} if cql_cfg.get("enabled",False) else set());anchor_diagnostic_steps=set(map(int,anchor_cfg.get("diagnostic_env_steps",[])))|({total} if anchor_enabled else set());eval_retries=int(config.get("sim_error_handling",{}).get("evaluation_retry_count",0))
    if env_steps==0:
        before=online.size;report=preserved_evaluation(actor,eval_env,seeds["evaluation_seeds"],config["horizon"],d,eval_retries);assert online.size==before;write(group_dir/"evaluations"/"step_000000.json",report)
        summary=record_probe(actor,critic,probes,d,group_dir/"probes"/"step_000000.npz",action_low,action_high,int(cql_cfg.get("num_random_actions",0)),config["training_seed"]);write(group_dir/"probes"/"step_000000.json",summary)
        if anchor_enabled:
            step0=geometry_diagnostics(agent,anchor_data.fixed("validation",int(anchor_cfg["batch_size"]),int(anchor_cfg["rng_seed"])+1));step0.update(probe_geometry(critic,probes,d));write(group_dir/"step0_anchor_audit.json",step0)
            if step0["pearson_q1"]<=.999 or step0["pearson_q2"]<=.999 or step0["max_abs_diff_q1"]>1e-5 or step0["max_abs_diff_q2"]>1e-5:raise RuntimeError(f"Step-0 student/teacher anchor equality failed: {step0}")
            log(group_dir/"anchor_diagnostics.jsonl",{"env_steps":0,"train":geometry_diagnostics(agent,anchor_data.fixed("train",int(anchor_cfg["batch_size"]),int(anchor_cfg["rng_seed"]))),"validation":step0,"geometry":probe_geometry(critic,probes,d),"sampling_proportions":anchor_data.proportions()})
        save(group_dir/"checkpoints"/"step_000000.pth",agent,online,sampler,expert,config,a.group,actual,0,0,None,env,anchor_data,sim_fatal_error_count)
    try:
        while env_steps<int(config["total_env_steps"]):
            if context is None:
                train_seed=int(config["train_seed_base"])+episode_index;observation=reset_seed(env,train_seed);context={"seed":train_seed,"observation":observation,"return":0.0,"length":0,"success":False};episode_index+=1
            state=flatten(context["observation"]);action=agent.action(state,False)[0]
            try:next_obs,reward,raw_done,_=env.step(action)
            except mujoco_fatal_error_type() as error:
                sim_fatal_error_count+=1;consecutive_sim_fatal_errors+=1;log(group_dir/"sim_fatal_errors.jsonl",{"timestamp":datetime.now().isoformat(),"group":a.group,"env_steps":env_steps,"episode_length":context["length"],"episode_seed":context["seed"],"exception":str(error),"consecutive":consecutive_sim_fatal_errors});context=None
                if consecutive_sim_fatal_errors>=int(config.get("sim_error_handling",{}).get("max_consecutive_fatal_errors",5)):raise RuntimeError(f"Reached consecutive MuJoCo FatalError limit after {sim_fatal_error_count} total errors") from error
                continue
            consecutive_sim_fatal_errors=0;context["length"]+=1;context["return"]+=float(reward);won=success(env);context["success"]|=won
            truncated=bool(context["length"]>=int(config["horizon"]) and not won);terminal=bool((config["terminate_on_success"] and won) or (raw_done and not truncated));online.add(state,action,reward,flatten(next_obs),terminal);env_steps+=1
            metrics=None
            if online.size>=int(config["min_online_replay_size"]):
                try:metrics=agent.update(sampler.sample(config["batch_size"]),anchor_data.sample(anchor_cfg["batch_size"]) if anchor_enabled else None)
                except FloatingPointError as error:
                    write(group_dir/"numerical_failure.json",{"env_steps":env_steps,"gradient_updates":agent.updates,"error":str(error)});save(group_dir/"checkpoints"/"numerical_failure.pth",agent,online,sampler,expert,config,a.group,actual,env_steps,episode_index,context,env,anchor_data,sim_fatal_error_count);raise
                metrics.update({"env_steps":env_steps,"gradient_updates":agent.updates,"offline_batch_fraction":sampler.fractions()["offline"],"online_batch_fraction":sampler.fractions()["online"],"online_replay_size":online.size,"actual_utd_after_replay_ready":1.0})
            if terminal or truncated:
                terminated_count+=int(terminal);truncated_count+=int(truncated);success_count+=int(won);episode_row={"episode":episode_index-1,"episode_seed":context["seed"],"episode_return":context["return"],"episode_length":context["length"],"success":bool(won),"terminated":terminal,"truncated":truncated};context=None
                if metrics is None:metrics={"env_steps":env_steps,"gradient_updates":agent.updates,"online_replay_size":online.size}
                metrics.update(episode_row)
            else:context["observation"]=next_obs
            if metrics is not None:
                log(group_dir/"train_metrics.jsonl",metrics)
                if env_steps%1000==0 or "episode_return" in metrics:
                    print(f"{a.group} env_steps={env_steps}/{config['total_env_steps']} replay={online.size} updates={agent.updates} alpha={metrics.get('alpha',float(agent.alpha.item()))} success={metrics.get('success','-')}",flush=True)
            if env_steps in eval_steps:
                before=online.size;report=preserved_evaluation(actor,eval_env,seeds["evaluation_seeds"],config["horizon"],d,eval_retries)
                if online.size!=before:raise RuntimeError("Evaluation polluted online replay")
                write(group_dir/"evaluations"/f"step_{env_steps:06d}.json",report)
            if env_steps in probe_steps:
                summary=record_probe(actor,critic,probes,d,group_dir/"probes"/f"step_{env_steps:06d}.npz",action_low,action_high,int(cql_cfg.get("num_random_actions",0)),config["training_seed"]+env_steps);write(group_dir/"probes"/f"step_{env_steps:06d}.json",summary)
            if env_steps in diagnostic_steps and cql_cfg.get("enabled",False):
                saved_rng=rng_state();seed_all(int(config["training_seed"])+env_steps)
                try:
                    count=int(cql_cfg["source_diagnostic_sample_size"]);expert_diag=agent.source_diagnostics(fixed_diagnostic_batch(expert,count,int(config["training_seed"])+env_steps));online_diag=agent.source_diagnostics(fixed_diagnostic_batch(online,count,int(config["training_seed"])+env_steps));online_diag.update({"q_behavior_mean":online_diag["q_data_mean"],"policy_minus_behavior_q_mean":online_diag["policy_minus_data_q_mean"],"random_max_minus_behavior_q_mean":online_diag["random_max_minus_data_q_mean"],"policy_gt_behavior_fraction":online_diag["policy_gt_data_fraction"],"random_max_gt_behavior_fraction":online_diag["random_max_gt_data_fraction"]});log(group_dir/"source_diagnostics.jsonl",{"env_steps":env_steps,"expert":expert_diag,"online":online_diag})
                finally:restore_rng(saved_rng)
            if anchor_enabled and env_steps in anchor_diagnostic_steps:
                row={"env_steps":env_steps,"train":geometry_diagnostics(agent,anchor_data.fixed("train",int(anchor_cfg["batch_size"]),int(anchor_cfg["rng_seed"])+env_steps)),"validation":geometry_diagnostics(agent,anchor_data.fixed("validation",int(anchor_cfg["batch_size"]),int(anchor_cfg["rng_seed"])+env_steps+1)),"geometry":probe_geometry(critic,probes,d),"sampling_proportions":anchor_data.proportions()};log(group_dir/"anchor_diagnostics.jsonl",row)
            if env_steps in checkpoint_steps:save(group_dir/"checkpoints"/f"step_{env_steps:06d}.pth",agent,online,sampler,expert,config,a.group,actual,env_steps,episode_index,context,env,anchor_data,sim_fatal_error_count)
        save(group_dir/"checkpoints"/"last.pth",agent,online,sampler,expert,config,a.group,actual,env_steps,episode_index,context,env,anchor_data,sim_fatal_error_count);write(group_dir/"summary.json",{"status":"COMPLETE","group":a.group,"env_steps":env_steps,"gradient_updates":agent.updates,"episodes":episode_index,"terminated_count":terminated_count,"truncated_count":truncated_count,"success_count":success_count,"sim_fatal_error_count":sim_fatal_error_count,"replay_size":online.size,"sampling_fractions":sampler.fractions(),"anchor_sampling_proportions":anchor_data.proportions() if anchor_enabled else None,"actor_hash_final":state_hash(actor),"critic_hash_final":state_hash(critic)})
    finally:close_env(env);close_env(eval_env)
if __name__=="__main__":main()
