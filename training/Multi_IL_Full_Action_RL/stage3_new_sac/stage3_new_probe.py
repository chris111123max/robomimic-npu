"""Load Stage2 fixed probes and record Critic/Actor geometry."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[3]; S2=ROOT/"training"/"Multi_IL_Full_Action_RL"/"stage2_new_critic_pretraining"
if str(S2) not in sys.path: sys.path.insert(0,str(S2))
from stage2_new_dataset import load_split_datasets  # noqa:E402

def read_json(path):
    with open(path,encoding="utf-8") as f:return json.load(f)
def load_fixed_probes(stage2_run_dir):
    run=Path(stage2_run_dir); config=read_json(run/"config_resolved.json"); manifest=read_json(run/"probe_manifest.json")
    train=range(int(config["train_seed_start"]),int(config["train_seed_end"])+1);val=range(int(config["val_seed_start"]),int(config["val_seed_end"])+1)
    _,datasets=load_split_datasets(config["dataset_root"],train,val,config["gamma"]);result={}
    for name,description in manifest.items():
        rows=description["rows"];result[name]={"states":np.stack([datasets[r["policy"]].state[int(r["index"])] for r in rows]),"actions":np.stack([datasets[r["policy"]].action[int(r["index"])] for r in rows])}
    return result,manifest
def _stats(values):
    x=np.asarray(values,float);return {"mean":float(x.mean()),"std":float(x.std()),"median":float(np.median(x)),"p90":float(np.percentile(x,90)),"p95":float(np.percentile(x,95))}
def record_probe(actor,critic,probes,device,npz_path,action_low=None,action_high=None,num_random_actions=0,seed=0):
    arrays={};summary={}
    for name,data in probes.items():
        states=torch.as_tensor(data["states"],dtype=torch.float32,device=device);actions=torch.as_tensor(data["actions"],dtype=torch.float32,device=device).requires_grad_(True)
        q1,q2=critic(states,actions);g1=torch.autograd.grad(q1.sum(),actions,retain_graph=True)[0];g2=torch.autograd.grad(q2.sum(),actions)[0]
        with torch.no_grad():mu=actor(states,deterministic=True)[0]
        values={"q1":q1.detach().cpu().numpy(),"q2":q2.detach().cpu().numpy(),"qmin":torch.minimum(q1,q2).detach().cpu().numpy(),"mu":mu.cpu().numpy(),"grad_q1":g1.cpu().numpy(),"grad_q2":g2.cpu().numpy()}
        if int(num_random_actions)>0:
            low=np.asarray(action_low,np.float32);high=np.asarray(action_high,np.float32);rng=np.random.default_rng(int(seed));random_np=rng.uniform(low,high,size=(len(states),int(num_random_actions),14)).astype(np.float32);random_actions=torch.as_tensor(random_np,device=device);expanded=states[:,None,:].expand(-1,int(num_random_actions),-1).reshape(-1,59)
            with torch.no_grad():pq1,pq2=critic(states,mu);rq1,rq2=critic(expanded,random_actions.reshape(-1,14));q_policy=torch.minimum(pq1,pq2);q_random=torch.minimum(rq1,rq2).reshape(len(states),int(num_random_actions));q_random_max=q_random.max(dim=1,keepdim=True).values
            values.update({"q_policy":q_policy.cpu().numpy(),"q_random_max":q_random_max.cpu().numpy(),"policy_minus_reference_q":(q_policy-torch.minimum(q1,q2).detach()).cpu().numpy(),"random_max_minus_reference_q":(q_random_max-torch.minimum(q1,q2).detach()).cpu().numpy()})
        for key,value in values.items():arrays[f"{name}__{key}"]=value
        summary[name]={"count":len(states),"q1":_stats(values["q1"]),"q2":_stats(values["q2"]),"qmin":_stats(values["qmin"]),"actor_action_l2_norm":_stats(np.linalg.norm(values["mu"],axis=1)),"grad_q1_norm":_stats(np.linalg.norm(values["grad_q1"],axis=1)),"grad_q2_norm":_stats(np.linalg.norm(values["grad_q2"],axis=1))}
        if int(num_random_actions)>0:summary[name].update({"q_policy_action":_stats(values["q_policy"]),"q_random_max":_stats(values["q_random_max"]),"policy_minus_reference_q":_stats(values["policy_minus_reference_q"]),"random_max_minus_reference_q":_stats(values["random_max_minus_reference_q"])})
    np.savez_compressed(npz_path,**arrays);return summary
