#!/usr/bin/env python3
"""Testing-only, frozen-Critic paired virtual Actor Adam trajectories."""
from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
for folder in (ROOT/"stage3_v5_rgmm_td3",ROOT/"stage3_v6_dual_2q",
               ROOT/"stage3_v6_dual_2q/testing/actor_collapse_diagnosis",
               Path(__file__).resolve().parent):
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))
from stage3_v5_actor import load_exact_actor
from stage3_v5_history_critic import component_mean_q,encode_replay_contexts
from stage3_v6_agent import strict_stage2_load
from run_offline import resolve_device
from run_gradient_alignment import actor_distribution,module

def q_values(actor,critic,obs,contexts,scale,offset):
    dist=actor_distribution(actor,obs)
    _,q1,_,t,means_env=component_mean_q(critic,contexts,dist,scale,offset,twin_min=False)
    q2=critic.q2.q_from_context(contexts[1],means_env).squeeze(-1)
    p=t["probs"]
    values={"q1":(p*q1).sum(-1),"q2":(p*q2).sum(-1),
            "mean":(p*(q1+q2)*.5).sum(-1),
            "min":(p*torch.minimum(q1,q2)).sum(-1)}
    action=(p.unsqueeze(-1)*t["means_normalized"]).sum(-2)
    return values,action

def vector_metrics(actor,critic,data,contexts,scale,offset):
    actor.train()
    metrics={k:[] for k in ("q1","q2","mean","min","action","replay_distance")}
    with torch.no_grad():
        for sl in range(0,len(data["a3_success"]),64):
            ids=slice(sl,sl+64)
            obs=torch.as_tensor(data["a3_observations"][ids],device=scale.device)
            c=tuple(x[ids] for x in contexts)
            q,a=q_values(actor,critic,obs,c,scale,offset)
            for k,v in q.items(): metrics[k].append(v.cpu().numpy())
            metrics["action"].append(a.cpu().numpy())
            replay=torch.as_tensor(data["a3_actions"][ids,-1],dtype=torch.float32,device=scale.device)
            replay_normalized=(replay-offset.reshape(14))/scale.reshape(14)
            metrics["replay_distance"].append(torch.linalg.vector_norm(a-replay_normalized,dim=-1).cpu().numpy())
    return {k:np.concatenate(v) for k,v in metrics.items()}

def summarize(current,baseline):
    out={}
    for label,mask in (("all",slice(None)),("success",baseline["success"]),
                       ("failure",~baseline["success"])):
        row={}
        for k in ("q1","q2","mean","min","replay_distance"):
            v=current[k][mask]; b=baseline[k][mask]
            row[k]={"mean":float(np.mean(v)),"delta_mean":float(np.mean(v-b)),
                    "delta_median":float(np.median(v-b)),"delta_positive_fraction":float(np.mean(v-b>0))}
        diff=current["action"][mask]-baseline["action"][mask]
        row["normalized_action_drift_mean"]=float(np.linalg.norm(diff,axis=-1).mean())
        row["normalized_action_drift_p95"]=float(np.percentile(np.linalg.norm(diff,axis=-1),95))
        out[label]=row
    return out

def drift(actor,initial):
    groups={}
    for k,p in actor.named_parameters():
        d=(p.detach().cpu()-initial[k]).double()
        groups[module(k)]=groups.get(module(k),0)+float(d.square().sum())
    groups["all"]=sum(groups.values())
    return {k:v**.5 for k,v in groups.items()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--contexts",type=Path,required=True)
    ap.add_argument("--device",required=True)
    ap.add_argument("--steps",type=int,default=100)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not 1<=args.steps<=100: raise ValueError("1<=steps<=100")
    device=resolve_device(args.device)
    run=args.run.resolve(); data=np.load(args.contexts.resolve())
    manifest=json.loads((run/"shared/stage2_source_manifest.json").read_text())
    actor,rollout,_=load_exact_actor(run/"shared/bc_rnn_gmm_source.pth",device)
    ready=torch.load(run/"mean2q/multi_q/checkpoints/critic_ready.pth",map_location="cpu",weights_only=False)
    assert ready["env_steps"]==140000 and ready["actor_updates"]==0
    actor.load_state_dict(ready["actor"],strict=True); actor.train()
    critic,_=strict_stage2_load(manifest["multi_q"]["checkpoint"],device)
    critic.load_state_dict(ready["q1_q2"],strict=True)
    critic.eval().requires_grad_(False)
    scale=torch.as_tensor(rollout.action_normalization_stats["actions"]["scale"],dtype=torch.float32,device=device).reshape(1,1,1,14)
    offset=torch.as_tensor(rollout.action_normalization_stats["actions"]["offset"],dtype=torch.float32,device=device).reshape(1,1,1,14)
    with torch.no_grad():
        obs=torch.as_tensor(data["a3_observations"],device=device)
        actions=torch.as_tensor(data["a3_actions"],device=device)
        ep=torch.as_tensor(data["a3_episode_steps"],device=device)
        full=encode_replay_contexts(critic,obs,actions,ep,700)
        contexts=(full[0][:,-1].detach(),full[1][:,-1].detach())
    initial={k:p.detach().cpu().clone() for k,p in actor.named_parameters()}
    baseline=vector_metrics(actor,critic,data,contexts,scale,offset)
    baseline["success"]=data["a3_success"]
    # Fixed balanced batches: 32 success and 32 failure, all 256 covered every 4 updates.
    suc=np.flatnonzero(data["a3_success"]); fail=np.flatnonzero(~data["a3_success"])
    assert len(suc)==len(fail)==128
    batches=[np.concatenate((suc[j*32:(j+1)*32],fail[j*32:(j+1)*32])) for j in range(4)]
    result={"device":str(device),"checkpoint":"critic_ready","fixed_contexts":str(args.contexts.resolve()),
            "steps":args.steps,"batch_size":64,"batch_schedule":"4 deterministic balanced batches cycling; no replay resampling",
            "optimizer":"torch.optim.Adam, betas=(0.9,0.999), eps=1e-8, weight_decay=0; grad clip 10",
            "lr_rule":"2e-6*(env_steps-140000)/140000; assumed one Actor update per 16-env collector round",
            "first_env_steps_inferred":140016,"first_lr":2e-6*16/140000,
            "first_lr_note":"inferred from resumed collector vector round and policy delay; earliest exact update not separately logged",
            "branches":{}}
    for mode in ("q1","mean","min"):
        clone=copy.deepcopy(actor); clone.train()
        opt=torch.optim.Adam(clone.parameters(),lr=0.0,weight_decay=0.0)
        branch={"milestones":{}}
        def record(step):
            current=vector_metrics(clone,critic,data,contexts,scale,offset)
            branch["milestones"][str(step)]={"metrics":summarize(current,baseline),
                                               "parameter_drift":drift(clone,initial)}
        record(0)
        for step in range(1,args.steps+1):
            env_step=140000+16*step
            lr=2e-6*(env_step-140000)/140000
            for group in opt.param_groups: group["lr"]=lr
            ids=batches[(step-1)%4]
            obs=torch.as_tensor(data["a3_observations"][ids],device=device)
            c=tuple(x[ids] for x in contexts)
            values,_=q_values(clone,critic,obs,c,scale,offset)
            loss=-values[mode].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in clone.parameters()):
                raise FloatingPointError(f"{mode} step {step} nonfinite gradient")
            torch.nn.utils.clip_grad_norm_(clone.parameters(),10.0)
            opt.step()
            if step in (1,5,10,25,50,100) or step==args.steps:
                record(step)
                print(json.dumps({"mode":mode,"step":step,"lr":lr,
                                  "action_drift":branch["milestones"][str(step)]["metrics"]["all"]["normalized_action_drift_mean"]}),flush=True)
        result["branches"][mode]=branch
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(args.output,flush=True)
if __name__=="__main__": main()
