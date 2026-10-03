#!/usr/bin/env python3
"""Offline Q1 vs twin-mean/min counterfactual on identical frozen contexts."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from run_offline import CHECKPOINTS, final_distribution, resolve_device, scalar_stats, to_np
from stage3_v5_actor import distribution_tensors, environment_means, load_exact_actor
from stage3_v5_history_critic import encode_replay_contexts
from stage3_v6_agent import strict_stage2_load

NAMES = ("step200k", "step280k", "last")
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)

def summarize(values, distance_delta, success):
    values = np.asarray(values, np.float64)
    def part(mask):
        x = values[mask]
        return {
            "count":int(mask.sum()),
            "delta":scalar_stats(x),
            "fraction_positive":float(np.mean(x > 0)),
            "fraction_positive_and_more_ood":float(np.mean((x > 0) & (distance_delta[mask] > 0))),
        }
    return {
        "all":part(np.ones(len(values), bool)),
        "success":part(success),
        "failure":part(~success),
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="npu:0")
    args=parser.parse_args()
    run=args.run.resolve()
    out=args.output.resolve()
    if out.exists(): raise FileExistsError(out)
    data=np.load(args.contexts.resolve())
    n=len(data["a3_observations"])
    if n != 256: raise RuntimeError("Fixed context contract changed")
    dev=resolve_device(args.device)
    source=json.loads((run/"shared/stage2_source_manifest.json").read_text())
    actor, rollout, _=load_exact_actor(run/"shared/bc_rnn_gmm_source.pth",dev)
    init_state=torch.load(run/"shared/actor_init.pth",map_location="cpu",weights_only=False)["actor_state_dict"]
    actor.load_state_dict(init_state,strict=True)
    init_actor=actor.eval()
    current=copy.deepcopy(init_actor).to(dev).eval()
    critic,_=strict_stage2_load(source["multi_q"]["checkpoint"],dev)
    critic.eval()
    scale=torch.as_tensor(rollout.action_normalization_stats["actions"]["scale"],dtype=torch.float32,device=dev).reshape(1,1,1,14)
    offset=torch.as_tensor(rollout.action_normalization_stats["actions"]["offset"],dtype=torch.float32,device=dev).reshape(1,1,1,14)
    result={"question":"Does conservative twin Actor value still favor the OOD learned action?",
            "fixed_contexts":str(args.contexts.resolve()),
            "context_count":n,"success_count":int(data["a3_success"].sum()),
            "failure_count":int((~data["a3_success"]).sum()),
            "actual_actor_objective":"GMM-probability-weighted Q1 at component means",
            "interpolation":"single argmax-mode environment action; diagnostic only",
            "checkpoints":{}}
    for name in NAMES:
        path=run/"mean2q/multi_q/checkpoints"/CHECKPOINTS[name]
        payload=torch.load(path,map_location="cpu",weights_only=False)
        current.load_state_dict(payload["actor"],strict=True)
        current.eval()
        critic.load_state_dict(payload["q1_q2"],strict=True)
        critic.eval()
        raw={k:[] for k in ("q1_init","q1_current","qmean_init","qmean_current",
                            "qmin_init","qmin_current","replay_distance_change")}
        curves={str(lam):{"q1":[],"q2":[],"qmean":[],"qmin":[]} for lam in LAMBDAS}
        with torch.inference_mode():
            for lo in range(0,n,32):
                hi=lo+32
                obs=torch.as_tensor(data["a3_observations"][lo:hi],dtype=torch.float32,device=dev)
                acts=torch.as_tensor(data["a3_actions"][lo:hi],dtype=torch.float32,device=dev)
                steps=torch.as_tensor(data["a3_episode_steps"][lo:hi],dtype=torch.long,device=dev)
                context=encode_replay_contexts(critic,obs,acts,steps,700)
                context=(context[0][:,-1],context[1][:,-1])
                actions=[]
                for label,policy in (("init",init_actor),("current",current)):
                    dist=final_distribution(policy,obs)
                    tensors=distribution_tensors(dist)
                    probs=tensors["probs"]
                    means=environment_means(dist,scale,offset)
                    q1=critic.q1.q_from_context(context[0],means).squeeze(-1)
                    q2=critic.q2.q_from_context(context[1],means).squeeze(-1)
                    for key,value in (("q1",q1),("qmean",(q1+q2)/2),("qmin",torch.minimum(q1,q2))):
                        raw[f"{key}_{label}"].append(to_np((probs*value).sum(-1)))
                    mode=probs.argmax(-1)
                    action=means.gather(1,mode[:,None,None].expand(-1,1,14)).squeeze(1)
                    actions.append(action)
                replay=acts[:,-1]
                d0=torch.linalg.vector_norm((actions[0]-replay)/scale.reshape(1,14),dim=-1)
                d1=torch.linalg.vector_norm((actions[1]-replay)/scale.reshape(1,14),dim=-1)
                raw["replay_distance_change"].append(to_np(d1-d0))
                for lam in LAMBDAS:
                    action=actions[0]*(1-lam)+actions[1]*lam
                    q1,q2=critic.q_from_context(context,action)
                    q1=q1.squeeze(-1); q2=q2.squeeze(-1)
                    for key,value in (("q1",q1),("q2",q2),("qmean",(q1+q2)/2),("qmin",torch.minimum(q1,q2))):
                        curves[str(lam)][key].append(to_np(value))
        raw={k:np.concatenate(v) for k,v in raw.items()}
        curves={lam:{k:np.concatenate(v) for k,v in fields.items()} for lam,fields in curves.items()}
        success=data["a3_success"].astype(bool)
        row={
            "env_steps":int(payload["env_steps"]),
            "actor_updates":int(payload["actor_updates"]),
            "replay_distance_change":scalar_stats(raw["replay_distance_change"]),
            "objectives":{},
            "single_action_curve":{},
        }
        for objective in ("q1","qmean","qmin"):
            delta=raw[f"{objective}_current"]-raw[f"{objective}_init"]
            row["objectives"][objective]=summarize(delta,raw["replay_distance_change"],success)
        for lam,fields in curves.items():
            row["single_action_curve"][lam]={k:scalar_stats(x) for k,x in fields.items()}
        result["checkpoints"][name]=row
        print(json.dumps({"checkpoint":name,"q1_gain":row["objectives"]["q1"]["all"]["delta"]["mean"],
                          "qmin_gain":row["objectives"]["qmin"]["all"]["delta"]["mean"],
                          "qmin_positive":row["objectives"]["qmin"]["all"]["fraction_positive"]}),flush=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(out,flush=True)

if __name__=="__main__":
    main()
