#!/usr/bin/env python3
"""Read-only Q/source decomposition for a completed Stage3-new checkpoint."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import h5py,numpy as np,torch
from stage3_new_agent import Stage3SAC,build_actor,build_critic,strict_stage2_load
from stage3_new_dataset import ExpertDataset
from stage3_new_replay import TransitionBuffer

def read_json(path):
    with open(path,encoding="utf-8") as f:return json.load(f)
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n")
def write_csv(path,rows):
    fields=list(rows[0]) if rows else ["sample_index"]
    with open(path,"w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def stats(x):
    x=np.asarray(x,float).reshape(-1);return {"mean":float(x.mean()),"std":float(x.std()),"median":float(np.median(x)),"p90":float(np.percentile(x,90)),"p95":float(np.percentile(x,95)),"max":float(x.max())}
def ranks(x):
    x=np.asarray(x);order=np.argsort(x,kind="mergesort");result=np.empty(len(x),float);values=x[order];start=0
    for end in range(1,len(x)+1):
        if end==len(x) or values[end]!=values[start]:result[order[start:end]]=(start+end-1)/2;start=end
    return result
def corr(a,b):
    a,b=np.asarray(a).reshape(-1),np.asarray(b).reshape(-1);return None if len(a)<2 or np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])
def set_seed(seed,device):
    np.random.seed(seed);torch.manual_seed(seed)
    if str(device).startswith("npu"):torch.npu.manual_seed_all(seed)
def torch_device(name):
    if name.startswith("npu"):
        import torch_npu  # noqa:F401
        torch.npu.set_device(name)
    return torch.device(name)
def q_values(critic,states,actions,device,batch_size=2048):
    q1=[];q2=[]
    with torch.no_grad():
        for start in range(0,len(states),batch_size):
            s=torch.as_tensor(states[start:start+batch_size],dtype=torch.float32,device=device);a=torch.as_tensor(actions[start:start+batch_size],dtype=torch.float32,device=device);x,y=critic(s,a);q1.append(x.cpu().numpy());q2.append(y.cpu().numpy())
    one=np.concatenate(q1).reshape(-1);two=np.concatenate(q2).reshape(-1);return one,two,np.minimum(one,two)
def policy_actions(actor,states,device,batch_size=2048):
    result=[]
    with torch.no_grad():
        for start in range(0,len(states),batch_size):result.append(actor(torch.as_tensor(states[start:start+batch_size],dtype=torch.float32,device=device),deterministic=True)[0].cpu().numpy())
    return np.concatenate(result)
def exact_target_components(agent,data,device,seed,batch_size=2048):
    set_seed(seed,device);parts={key:[] for key in ("target_qmin","next_logp","entropy_bonus","td_target")}
    for start in range(0,len(data["observations"]),batch_size):
        b={key:torch.as_tensor(value[start:start+batch_size],dtype=torch.float32,device=device) for key,value in data.items()};values=agent.target_components(b)
        for key in parts:parts[key].append(values[key].cpu().numpy().reshape(-1))
    return {key:np.concatenate(value) for key,value in parts.items()}
def expert_ids(path):
    result=[]
    with h5py.File(path,"r") as f:
        for episode in sorted(f["data"]):result.extend((episode,index) for index in range(len(f["data"][episode]["actions"])))
    return result
def subset(data,indices):return {key:value[indices].copy() for key,value in data.items()}
def analyze_source(agent,data,indices,ids,source,random_k,seed,device):
    states=data["observations"];behavior=data["actions"];policy=policy_actions(agent.actor,states,device);q1_data,q2_data,q_data=q_values(agent.critic,states,behavior,device);q1_pi,q2_pi,q_pi=q_values(agent.critic,states,policy,device);distance=np.linalg.norm(policy-behavior,axis=1);delta=q_pi-q_data;relative=delta/(np.abs(q_data)+1e-6)
    rng=np.random.default_rng(seed);random_actions=rng.uniform(-1,1,size=(len(states),random_k,behavior.shape[1])).astype(np.float32);flat_states=np.repeat(states,random_k,axis=0);_,_,random_q_flat=q_values(agent.critic,flat_states,random_actions.reshape(-1,behavior.shape[1]),device);random_q=random_q_flat.reshape(len(states),random_k);random_mean=random_q.mean(axis=1);random_max=random_q.max(axis=1)
    target=exact_target_components(agent,data,device,seed);summary={"source":source,"count":len(states),"q_behavior_or_data":stats(q_data),"q_policy":stats(q_pi),"q_random_mean":stats(random_mean),"max_q_random":stats(random_max),"policy_minus_behavior_or_data_q":stats(delta),"relative_policy_advantage":stats(relative),"fraction_q_policy_gt_q_data":float(np.mean(q_pi>q_data)),"fraction_q_policy_gt_q_data_plus_0p1":float(np.mean(q_pi>q_data+.1)),"fraction_q_policy_gt_q_data_plus_1p0":float(np.mean(q_pi>q_data+1)),"fraction_random_max_gt_q_data":float(np.mean(random_max>q_data)),"action_distance_policy_vs_behavior_or_data":stats(distance),"distance_advantage_pearson":corr(distance,delta),"distance_advantage_spearman":corr(ranks(distance),ranks(delta))}
    support={"behavior_or_data":{"fraction_abs_gt_0p9":float(np.mean(np.abs(behavior)>.9)),"fraction_abs_gt_0p95":float(np.mean(np.abs(behavior)>.95)),"fraction_abs_gt_0p99":float(np.mean(np.abs(behavior)>.99))},"current_policy":{"fraction_abs_gt_0p9":float(np.mean(np.abs(policy)>.9)),"fraction_abs_gt_0p95":float(np.mean(np.abs(policy)>.95)),"fraction_abs_gt_0p99":float(np.mean(np.abs(policy)>.99))},"l2_distance":stats(distance)}
    target_summary={"reward":stats(data["rewards"]),"target_qmin":stats(target["target_qmin"]),"next_log_pi":stats(target["next_logp"]),"policy_entropy":stats(-target["next_logp"]),"entropy_bonus":stats(target["entropy_bonus"]),"td_target":stats(target["td_target"]),"bootstrap_mask":stats(1-data["terminals"])}
    rows=[]
    for i in range(len(states)):
        identity=ids[i];rows.append({"sample_index":int(indices[i]),"episode_id":identity[0] if isinstance(identity,tuple) else "","episode_timestep":identity[1] if isinstance(identity,tuple) else "","reward":float(data["rewards"][i,0]),"terminal":float(data["terminals"][i,0]),"q1_behavior_or_data":float(q1_data[i]),"q2_behavior_or_data":float(q2_data[i]),"qmin_behavior_or_data":float(q_data[i]),"q1_policy":float(q1_pi[i]),"q2_policy":float(q2_pi[i]),"qmin_policy":float(q_pi[i]),"policy_minus_behavior_or_data_q":float(delta[i]),"relative_policy_advantage":float(relative[i]),"action_l2":float(distance[i]),"random_q_mean":float(random_mean[i]),"random_q_max":float(random_max[i]),"target_qmin":float(target["target_qmin"][i]),"next_log_pi":float(target["next_logp"][i]),"entropy_bonus":float(target["entropy_bonus"][i]),"td_target":float(target["td_target"][i])})
    return summary,support,target_summary,rows,random_actions
def load_agent(checkpoint,config,device,group):
    payload=torch.load(checkpoint,map_location=device)
    if payload.get("group")!=group:raise RuntimeError(f"Checkpoint group {payload.get('group')!r} != {group!r}")
    actor=build_actor(config,device);critic=build_critic(59,14,config["hidden_dims"],"relu",True,device);agent=Stage3SAC(actor,critic,config,device);agent.actor.load_state_dict(payload["actor_state_dict"],strict=True);agent.critic.load_state_dict(payload["critic_state_dict"],strict=True);agent.target.load_state_dict(payload["target_critic_state_dict"],strict=True);agent.log_alpha.data.copy_(payload["log_alpha"].to(device));return agent,payload
def resolve_replay(checkpoint,payload):
    candidates=[Path(payload.get("replay_path","")),Path(checkpoint).with_suffix(".replay.npz")]
    for path in candidates:
        if str(path) and path.is_file():return path
    raise RuntimeError("CURRENT RUN ONLINE REPLAY NOT RECOVERABLE: saved replay_path and checkpoint-adjacent replay are absent")
def main():
    p=argparse.ArgumentParser();p.add_argument("--pair-run-dir",required=True);p.add_argument("--group",required=True,choices=("rnn_q","multi_q"));p.add_argument("--checkpoint");p.add_argument("--expert-data");p.add_argument("--sample-size",type=int,default=4096);p.add_argument("--random-actions",type=int,default=10);p.add_argument("--seed",type=int,default=20260906);p.add_argument("--device",default="cpu");p.add_argument("--output-dir");a=p.parse_args();pair=Path(a.pair_run_dir).resolve();config=read_json(pair/"shared"/"config_resolved.json");checkpoint=Path(a.checkpoint) if a.checkpoint else pair/a.group/"checkpoints"/"last.pth";expert_path=Path(a.expert_data or config.get("expert_dataset") or "")
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if not expert_path.is_file():raise FileNotFoundError(f"Expert dataset does not exist: {expert_path}")
    device=torch_device(a.device);agent,payload=load_agent(checkpoint,config,device,a.group);replay_path=resolve_replay(checkpoint,payload);expert=ExpertDataset(expert_path,a.seed);online=TransitionBuffer.load(replay_path);n=min(int(a.sample_size),expert.size,online.size)
    if n<=0:raise RuntimeError("No transitions available for audit")
    rng=np.random.default_rng(a.seed);expert_indices=rng.choice(expert.size,n,replace=False);online_indices=np.random.default_rng(a.seed).choice(online.size,n,replace=False);all_ids=expert_ids(expert_path);expert_data=subset(expert.data,expert_indices);online_data=subset(online.data,online_indices);expert_identity=[all_ids[i] for i in expert_indices];online_identity=[int(i) for i in online_indices]
    expert_summary,expert_support,expert_target,expert_rows,expert_random=analyze_source(agent,expert_data,expert_indices,expert_identity,"expert",a.random_actions,a.seed,device);online_summary,online_support,online_target,online_rows,_=analyze_source(agent,online_data,online_indices,online_identity,"online",a.random_actions,a.seed,device)
    sources=read_json(pair/"shared"/"stage2_source_manifest.json");stage2_path=Path(sources[a.group]["checkpoint"]);stage2_baseline=None
    if stage2_path.is_file():
        stage2,_=strict_stage2_load(stage2_path,device,config);_,_,data_q=q_values(stage2,expert_data["observations"],expert_data["actions"],device);_,_,random_q=q_values(stage2,np.repeat(expert_data["observations"],a.random_actions,axis=0),expert_random.reshape(-1,14),device);stage2_baseline={"checkpoint":str(stage2_path),"q_expert_data":stats(data_q),"q_random_mean":stats(random_q.reshape(n,a.random_actions).mean(axis=1)),"max_q_random":stats(random_q.reshape(n,a.random_actions).max(axis=1))}
    out=Path(a.output_dir) if a.output_dir else pair/"audits"/"q_source_decomposition"/a.group;out.mkdir(parents=True,exist_ok=True);source_summary={"group":a.group,"checkpoint":str(checkpoint.resolve()),"checkpoint_env_steps":int(payload["env_steps"]),"alpha":float(agent.alpha.item()),"expert":expert_summary,"online":online_summary,"four_way_q_means":{"q_expert_data":expert_summary["q_behavior_or_data"]["mean"],"q_expert_state_policy_action":expert_summary["q_policy"]["mean"],"q_online_behavior_action":online_summary["q_behavior_or_data"]["mean"],"q_online_state_policy_action":online_summary["q_policy"]["mean"]},"stage2_baseline":stage2_baseline};write_json(out/"source_q_summary.json",source_summary);write_csv(out/"expert_transition_metrics.csv",expert_rows);write_csv(out/"online_transition_metrics.csv",online_rows);write_json(out/"action_support_summary.json",{"expert":expert_support,"online":online_support});write_json(out/"target_component_summary.json",{"expert":expert_target,"online":online_target});write_json(out/"sample_manifest.json",{"seed":a.seed,"sample_size":n,"random_actions_per_state":a.random_actions,"random_action_range":[-1,1],"expert_indices":expert_indices.tolist(),"expert_episode_ids":[{"episode_id":x[0],"timestep":x[1]} for x in expert_identity],"online_indices":online_indices.tolist(),"online_replay":str(replay_path.resolve())});print(json.dumps(source_summary,indent=2))
if __name__=="__main__":main()
