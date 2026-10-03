#!/usr/bin/env python3
"""Testing-only Stage3-v6 Actor objective gradient alignment on fixed contexts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
for folder in (ROOT/"stage3_v5_rgmm_td3",ROOT/"stage3_v6_dual_2q",
               ROOT/"stage3_v6_dual_2q/testing/actor_collapse_diagnosis"):
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))
from stage3_v5_actor import flat_to_obs,load_exact_actor
from stage3_v5_history_critic import component_mean_q,encode_replay_contexts
from stage3_v6_agent import strict_stage2_load
from run_offline import CHECKPOINTS,resolve_device,scalar_stats

NAMES=("critic_ready","step200k","step280k")
Q_MARGIN=0.01

def actor_distribution(actor,obs):
    seq=actor.forward_train(flat_to_obs(obs),rnn_init_state=None,return_state=False)
    base=seq.component_distribution.base_dist
    return torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=seq.mixture_distribution.logits[:,-1]),
        torch.distributions.Independent(torch.distributions.Normal(base.loc[:,-1],base.scale[:,-1]),1))

def module(key):
    if key.startswith("nets.rnn.nets."): return "rnn"
    if key.startswith("nets.decoder.nets.mean."): return "gmm_mean"
    if key.startswith("nets.decoder.nets.logits."): return "gmm_logits"
    if key.startswith("nets.decoder.nets.scale."): return "gmm_std"
    if "encoder" in key: return "encoder"
    return "other"

def cosine(x,y):
    dot=sum(float((a.double()*b.double()).sum()) for a,b in zip(x,y))
    nx=sum(float(a.double().square().sum()) for a in x)
    ny=sum(float(b.double().square().sum()) for b in y)
    return None if nx<=0 or ny<=0 else dot/(nx*ny)**0.5

def summarize_grads(names,grads):
    groups={}
    for key,value in zip(names,grads):
        g=module(key)
        groups.setdefault(g,[]).append(torch.zeros(0) if value is None else value.detach().cpu())
    return groups

def compare(names,gradient_sets):
    result={}
    for group in ("all","rnn","gmm_mean","gmm_logits","gmm_std","encoder","other"):
        idx=[i for i,k in enumerate(names) if group=="all" or module(k)==group]
        if not idx:
            result[group]={"parameter_count":0,"norms":None,"cosines":None}
            continue
        vectors={mode:[gradient_sets[mode][i] if gradient_sets[mode][i] is not None else torch.zeros_like(params[i]).cpu() for i in idx]
                 for mode in ("q1","mean","min")}
        norms={mode:sum(float(v.double().square().sum()) for v in values)**0.5 for mode,values in vectors.items()}
        cosines={"q1_mean":cosine(vectors["q1"],vectors["mean"]),
                 "q1_min":cosine(vectors["q1"],vectors["min"]),
                 "mean_min":cosine(vectors["mean"],vectors["min"])}
        result[group]={"parameter_count":sum(params[i].numel() for i in idx),"norms":norms,"cosines":cosines}
    return result

def attribution(raw,mask):
    q1=raw["a3_q1_expected_current"]-raw["a3_q1_expected_init"]
    q2=raw["a3_q2_expected_current"]-raw["a3_q2_expected_init"]
    x=q1[mask]; y=q2[mask]
    return {"count":int(mask.sum()),"q1_gain":scalar_stats(x),"q2_gain":scalar_stats(y),
            "fraction_q1_positive":float(np.mean(x>0)),"fraction_q2_positive":float(np.mean(y>0)),
            "q1_only_advantage_fraction":float(np.mean((x>0)&(y<=0))),
            "both_positive_fraction":float(np.mean((x>0)&(y>0))),
            "q1_exceeds_q2_by_margin_fraction":float(np.mean(x-y>Q_MARGIN)),
            "q1_minus_q2_gain":scalar_stats(x-y)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",type=Path,required=True)
    ap.add_argument("--contexts",type=Path,required=True)
    ap.add_argument("--prior-results",type=Path,required=True)
    ap.add_argument("--device",required=True)
    ap.add_argument("--checkpoints",nargs="+",choices=NAMES,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    device=resolve_device(args.device); run=args.run.resolve()
    data=np.load(args.contexts.resolve())
    source=json.loads((run/"shared/stage2_source_manifest.json").read_text())
    actor,rollout,_=load_exact_actor(run/"shared/bc_rnn_gmm_source.pth",device)
    actor.train()
    critic,_=strict_stage2_load(source["multi_q"]["checkpoint"],device)
    critic.eval().requires_grad_(False)
    scale=torch.as_tensor(rollout.action_normalization_stats["actions"]["scale"],dtype=torch.float32,device=device).reshape(1,1,1,14)
    offset=torch.as_tensor(rollout.action_normalization_stats["actions"]["offset"],dtype=torch.float32,device=device).reshape(1,1,1,14)
    global params
    named=list(actor.named_parameters()); names=[n for n,_ in named]; params=[p for _,p in named]
    success=np.flatnonzero(data["a3_success"])
    failure=np.flatnonzero(~data["a3_success"])
    batches={"all":np.arange(len(data["a3_success"])),"success":success,"failure":failure}
    result={"device":str(device),"fixed_contexts":str(args.contexts.resolve()),
            "context_counts":{k:len(v) for k,v in batches.items()},
            "q1_minus_q2_meaningful_margin":Q_MARGIN,
            "margin_basis":"1% of sparse 0/1 return scale",
            "checkpoints":{}}
    for name in args.checkpoints:
        checkpoint=run/"mean2q/multi_q/checkpoints"/CHECKPOINTS[name]
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        actor.load_state_dict(payload["actor"],strict=True)
        critic.load_state_dict(payload["q1_q2"],strict=True)
        actor.train(); critic.eval().requires_grad_(False)
        row={"env_steps":int(payload["env_steps"]),"actor_updates":int(payload["actor_updates"]),"gradients":{}}
        for label,ids in batches.items():
            obs=torch.as_tensor(data["a3_observations"][ids],dtype=torch.float32,device=device)
            actions=torch.as_tensor(data["a3_actions"][ids],dtype=torch.float32,device=device)
            steps=torch.as_tensor(data["a3_episode_steps"][ids],dtype=torch.long,device=device)
            with torch.no_grad():
                contexts=encode_replay_contexts(critic,obs,actions,steps,700)
                final_contexts=(contexts[0][:,-1],contexts[1][:,-1])
            dist=actor_distribution(actor,obs)
            actual,q1,_,tensors,means_env=component_mean_q(
                critic,final_contexts,dist,scale,offset,twin_min=False)
            q2=critic.q2.q_from_context(final_contexts[1],means_env).squeeze(-1)
            probs=tensors["probs"]
            q1obj=(probs*q1).sum(-1)
            if not torch.allclose(actual,q1obj,atol=1e-6,rtol=1e-6):
                raise RuntimeError("Q1 production objective mismatch")
            objectives={"q1":q1obj,"mean":(probs*(q1+q2)*0.5).sum(-1),
                        "min":(probs*torch.minimum(q1,q2)).sum(-1)}
            gradients={}
            for mode,obj in objectives.items():
                actor.zero_grad(set_to_none=True)
                grads=torch.autograd.grad(-obj.mean(),params,retain_graph=True,allow_unused=True)
                if any(g is not None and not bool(torch.isfinite(g).all()) for g in grads):
                    raise FloatingPointError(f"Nonfinite gradient {name} {label} {mode}")
                gradients[mode]=[None if g is None else g.detach().cpu().clone() for g in grads]
            align=compare(names,gradients)
            std_abs={mode:sum(float(g.abs().sum()) for k,g in zip(names,gradients[mode]) if module(k)=="gmm_std" and g is not None)
                     for mode in objectives}
            row["gradients"][label]={"modules":align,"std_abs_gradient":std_abs,
                                     "objective_values":{mode:float(obj.detach().mean()) for mode,obj in objectives.items()}}
            if name=="critic_ready" and label=="all":
                actual_state=torch.load(run/"mean2q/multi_q/checkpoints/step_0200000.pth",
                                        map_location="cpu",weights_only=False)["actor"]
                delta=[(actual_state[k]-payload["actor"][k]).detach().cpu() for k in names]
                row["real_delta_alignment"]={
                    mode:{"global":cosine(delta,[-g if g is not None else torch.zeros_like(params[i]).cpu()
                                                   for i,g in enumerate(gradients[mode])]),
                          "rnn":cosine([delta[i] for i,k in enumerate(names) if module(k)=="rnn"],
                                       [-gradients[mode][i] if gradients[mode][i] is not None else torch.zeros_like(delta[i])
                                        for i,k in enumerate(names) if module(k)=="rnn"]),
                          "gmm_mean":cosine([delta[i] for i,k in enumerate(names) if module(k)=="gmm_mean"],
                                            [-gradients[mode][i] if gradients[mode][i] is not None else torch.zeros_like(delta[i])
                                             for i,k in enumerate(names) if module(k)=="gmm_mean"])}
                    for mode in objectives}
                row["real_delta_alignment_note"]="140K→200K net parameter delta includes ~3743 Adam steps, evolving Critic and batches; only directional context."
            del dist,actual,q1,q2,means_env,gradients
        raw=np.load(args.prior_results.resolve()/f"offline_{name}.npz")
        success_mask=raw["a3_success"].astype(bool)
        row["q1_vs_q2"]={"all":attribution(raw,np.ones(len(success_mask),bool)),
                         "success":attribution(raw,success_mask),
                         "failure":attribution(raw,~success_mask)}
        result["checkpoints"][name]=row
        a=row["gradients"]["all"]["modules"]["all"]
        print(json.dumps({"checkpoint":name,"cos_q1_mean":a["cosines"]["q1_mean"],
                          "cos_q1_min":a["cosines"]["q1_min"],
                          "q1_only_fraction":row["q1_vs_q2"]["all"]["q1_only_advantage_fraction"]}),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(args.output,flush=True)
if __name__=="__main__": main()
