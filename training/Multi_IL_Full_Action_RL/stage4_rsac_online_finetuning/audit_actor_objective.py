#!/usr/bin/env python3
"""Read-only Stage4 Actor objective decomposition and alpha sweep."""
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from audit_critic_washout import GROUPS,aligned,atomic_json,checkpoints,load_critic,read_json,select_device,seed_all  # noqa
from stage4_core import OnlineSequenceReplay  # noqa
from stage3_r_actor import load_actor  # noqa

ALPHAS=(1.0,.5,.3,.1,.05,.03,.01,.005,.003,.001,0.0)
MODULES=("lstm","mu_head","log_std_head","lstm_plus_mu")

def arguments():
    p=argparse.ArgumentParser();p.add_argument("--run-dir",default="/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage4_rsac_online_finetuning/20260901_212533");p.add_argument("--config",default=str(HERE/"stage4_config.json"));p.add_argument("--device",default="npu:0");p.add_argument("--gradient-batches",type=int,default=20);p.add_argument("--batch-size",type=int,default=32);return p.parse_args()

def vector(grads,params):return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(grads,params)]).detach()
def cosine(a,b):return float(torch.dot(a,b).item()/max(float(a.norm().item()*b.norm().item()),1e-30))
def relative(a,b):return float((a-b).norm().item()/max(float(a.norm().item()+b.norm().item()),1e-30))
def module_name(name):
    if name.startswith("lstm."):return "lstm"
    if name.startswith("policy.last_fc_log_std."):return "log_std_head"
    if name.startswith("policy.last_fc."):return "mu_head"
    return "other"
def subset(v,names,wanted):
    pieces=[];cursor=0
    for name,size in names:
        part=v[cursor:cursor+size];cursor+=size
        if module_name(name) in wanted:pieces.append(part)
    return torch.cat(pieces) if pieces else torch.zeros(1,device=v.device)

def fixed_batch(run,config):
    path=run/"diagnostics"/"critic_washout"/"fixed_diagnostic_sequences.pt"
    if path.exists():
        payload=torch.load(path,map_location="cpu");return payload["batch"],payload.get("identifiers",[]),{"kind":"critic_washout fixed sequences","path":str(path)}
    replay=run/"rnn_only_critic"/"replay"/"step_00030000.npz"
    if not replay.exists():raise RuntimeError("Neither fixed diagnostic sequences nor step30000 replay exists; rollout is forbidden")
    store=OnlineSequenceReplay(config);store.load(replay);np.random.seed(20260902);values=store.buffer.random_episodes(1024);batch={k:torch.as_tensor(v,dtype=torch.float32) for k,v in values.items()};return batch,[],{"kind":"Stage4 replay","path":str(replay)}

def policy_graph(actor,observs,seed):
    seed_all(seed);features=actor.forward_sequence(observs.transpose(0,1),reset_interval=True);action,mean,log_std,logp=actor.actions_from_features(features,deterministic=False,return_log_prob=True)
    return action.transpose(0,1),mean.transpose(0,1),log_std.transpose(0,1),logp.transpose(0,1)

def gradients(actor,critics,batch,device,seed):
    b,observs,prev,rewards=aligned(batch,device);mask=b["mask"];valid=mask.sum().clamp_min(1.0);actions,mean,log_std,logp=policy_graph(actor,observs,seed)
    params=[p for p in actor.parameters() if p.requires_grad];names=[(n,p.numel()) for n,p in actor.named_parameters() if p.requires_grad]
    q_losses=[]
    for critic in critics:
        q1,q2=critic(prev,rewards,observs,actions);q=torch.minimum(q1,q2)[:-1];q_losses.append(-(q*mask).sum()/valid)
    entropy=(logp[:-1]*mask).sum()/valid
    gq=[]
    for loss in q_losses:gq.append(vector(torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True),params))
    gh=vector(torch.autograd.grad(entropy,params,retain_graph=False,allow_unused=True),params)
    valid_logp=logp[:-1][mask.bool()];valid_logstd=log_std[:-1][mask.bool().expand_as(log_std[:-1])].reshape(-1,14)
    return gq,gh,names,{"log_pi":valid_logp.detach().cpu(),"log_std":valid_logstd.detach().cpu()}

def alpha_history(run,config):
    maps={g:checkpoints(run/g) for g in GROUPS};wanted=(0,30000,50000,100000,200000,500000);rows=[]
    for step in wanted:
        row={"env_step":step}
        for group in GROUPS:
            if step==0:value=float(config["initial_entropy_alpha"]);log=math.log(value)
            elif step in maps[group]:
                payload=torch.load(maps[group][step],map_location="cpu");raw=payload.get("log_alpha_entropy");log=float(torch.as_tensor(raw).item()) if raw is not None else math.log(float(config["initial_entropy_alpha"]));value=math.exp(log)
            else:value=log=None
            row[f"{group}_alpha"]=value;row[f"{group}_log_alpha"]=log
        rows.append(row)
    return rows

def write_csv(path,rows):
    with Path(path).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    a=arguments();run=Path(a.run_dir);config=read_json(a.config);device=select_device(a.device);output=run/"diagnostics"/"actor_objective_alpha_sweep";output.mkdir(parents=True,exist_ok=True)
    batch,identifiers,source=fixed_batch(run,config);actor,payload=load_actor(config["actor_checkpoint"],device);actor.train().requires_grad_(True)
    critics=[load_critic(config,Path(config["rnn_critic_checkpoint"]),device)[0],load_critic(config,Path(config["multi_critic_checkpoint"]),device)[0]]
    N=batch["obs"].shape[1];all_batches=[]
    with torch.no_grad():
        full_b,full_obs,full_prev,full_rewards=aligned(batch,device);_,_,full_logstd,full_logp=policy_graph(actor,full_obs,20260902);full_mask=full_b["mask"].bool();logp=full_logp[:-1][full_mask].cpu().numpy();logstd=full_logstd[:-1][full_mask.expand_as(full_logstd[:-1])].reshape(-1,14).cpu().numpy()
    for i in range(a.gradient_batches):
        idx=torch.arange(i*a.batch_size,i*a.batch_size+a.batch_size)%N;mini={k:v.index_select(1,idx) for k,v in batch.items()};gq,gh,names,_=gradients(actor,critics,mini,device,20260902+i);all_batches.append((gq,gh,names))
    sweep=[];module_rows=[]
    for alpha in ALPHAS:
        per=[]
        for index,(gq,gh,names) in enumerate(all_batches):
            totals=[g+alpha*gh for g in gq];qnorm=[float(g.norm()) for g in gq];hn=float(gh.norm());tn=[float(g.norm()) for g in totals]
            row={"alpha":alpha,"batch_index":index,"rnn_q_grad_norm":qnorm[0],"multi_q_grad_norm":qnorm[1],"entropy_grad_norm_base":hn,"entropy_grad_norm_scaled":alpha*hn,"rnn_entropy_q_ratio":alpha*hn/max(qnorm[0],1e-30),"multi_entropy_q_ratio":alpha*hn/max(qnorm[1],1e-30),"rnn_q_fraction":qnorm[0]/max(qnorm[0]+alpha*hn,1e-30),"multi_q_fraction":qnorm[1]/max(qnorm[1]+alpha*hn,1e-30),"total_grad_cosine":cosine(*totals),"relative_grad_difference":relative(*totals),"q_only_cosine":cosine(*gq),"critic_gradient_separation":1.0-cosine(*totals)}
            mu=[subset(x,names,{"mu_head"}) for x in totals];trunk_mu=[subset(x,names,{"lstm","mu_head"}) for x in totals];row["mu_only_total_grad_cosine"]=cosine(*mu);row["lstm_mu_total_grad_cosine"]=cosine(*trunk_mu);row["mu_only_q_grad_cosine"]=cosine(subset(gq[0],names,{"mu_head"}),subset(gq[1],names,{"mu_head"}));per.append(row)
            if alpha==1.0:
                for module,wanted in (("lstm",{"lstm"}),("mu_head",{"mu_head"}),("log_std_head",{"log_std_head"}),("lstm_plus_mu",{"lstm","mu_head"})):
                    qv=[subset(g,names,wanted) for g in gq];hv=subset(gh,names,wanted);tv=[g+hv for g in qv];hn=float(hv.norm());module_rows.append({"batch_index":index,"module":module,"rnn_q_grad_norm":float(qv[0].norm()),"multi_q_grad_norm":float(qv[1].norm()),"entropy_grad_norm":hn,"rnn_entropy_q_ratio":hn/max(float(qv[0].norm()),1e-30),"multi_entropy_q_ratio":hn/max(float(qv[1].norm()),1e-30),"q_gradient_cosine":cosine(*qv),"total_gradient_cosine":cosine(*tv)})
        result={k:(alpha if k=="alpha" else float(np.mean([r[k] for r in per]))) for k in per[0]};sweep.append(result)
    std=np.exp(logstd);entropy_error=float(-logp.mean()-float(config["target_entropy"]));entropy_summary={"mean_log_pi":float(logp.mean()),"std_log_pi":float(logp.std()),"mean_entropy_estimate":float(-logp.mean()),"mean_log_std":float(logstd.mean()),"min_log_std":float(logstd.min()),"max_log_std":float(logstd.max()),"mean_std":float(std.mean()),"median_std":float(np.median(std)),"min_std":float(std.min()),"max_std":float(std.max()),"per_dimension_std_mean":std.mean(0).tolist(),"log_std_clamp":[-20.0,2.0],"target_entropy":float(config["target_entropy"]),"entropy_error_policy_minus_target":entropy_error,"policy_relative_to_target":"too_stochastic" if entropy_error>0 else "too_deterministic" if entropy_error<0 else "at_target","automatic_tuning_direction_at_initial_actor":"decrease_alpha" if entropy_error>0 else "increase_alpha" if entropy_error<0 else "no_change"}
    history=alpha_history(run,config);write_csv(output/"alpha_sweep_gradient_similarity.csv",sweep);write_csv(output/"module_gradient_decomposition.csv",module_rows);write_csv(output/"alpha_history.csv",history);atomic_json(output/"policy_entropy_summary.json",entropy_summary)
    base=sweep[0];prior_path=run/"diagnostics"/"critic_washout"/"stage4_critic_washout_audit.json";reproduction={"status":"NOT_CHECKED"}
    if prior_path.exists():
        prior=read_json(prior_path);old=next(r for r in prior["rows"] if int(r["env_step"])==0);dq=abs(base["q_only_cosine"]-float(old["q_grad_cos_mean"]));ds=abs(base["total_grad_cosine"]-float(old["sac_grad_cos_mean"]));reproduction={"status":"PASS" if dq<=.05 and ds<=.001 else "FAIL","q_cosine_absolute_difference":dq,"sac_cosine_absolute_difference":ds,"tolerances":{"q":.05,"sac":.001}}
    if reproduction["status"]=="FAIL":case="REPRODUCTION_FAILED";conclusion="AUDIT REPRODUCTION FAILED"
    else:
        log_ratio=np.mean([max(r["rnn_entropy_q_ratio"],r["multi_entropy_q_ratio"]) for r in module_rows if r["module"]=="log_std_head"]);mu_cos=base["mu_only_total_grad_cosine"]
        if base["mu_only_q_grad_cosine"]>=.9:case="E3";conclusion="Critic differences do not materially change the action-mean optimization direction."
        elif base["total_grad_cosine"]>=.99 and mu_cos<.9 and log_ratio>3:case="E2";conclusion="Entropy domination is concentrated in policy variance parameters; critic-specific mean-action gradients remain distinct."
        elif max(base["rnn_entropy_q_ratio"],base["multi_entropy_q_ratio"])>3 and base["total_grad_cosine"]>=.99 and sweep[-1]["total_grad_cosine"]<base["total_grad_cosine"]-.1:case="E1";conclusion="Shared entropy gradient suppresses critic-specific policy-gradient differences."
        else:case="E4";conclusion="No clear entropy-gradient domination detected."
    transitions={"largest_alpha_cos_below_0.9":max((r["alpha"] for r in sweep if r["total_grad_cosine"]<.9),default=None),"largest_alpha_cos_below_0.75":max((r["alpha"] for r in sweep if r["total_grad_cosine"]<.75),default=None),"largest_alpha_cos_below_0.5":max((r["alpha"] for r in sweep if r["total_grad_cosine"]<.5),default=None)}
    for group in ("rnn","multi"):
        transitions[f"{group}_alpha_ratio_nearest_1"]=min(sweep,key=lambda r:abs(r[f"{group}_entropy_q_ratio"]-1))["alpha"];transitions[f"{group}_alpha_ratio_nearest_0.5"]=min(sweep,key=lambda r:abs(r[f"{group}_entropy_q_ratio"]-.5))["alpha"]
    implementation={"formula":"mean(mask * (alpha * log_pi - min(Q1,Q2)))","automatic_entropy_tuning":bool(config["automatic_entropy_tuning"]),"log_alpha_trainable":True,"alpha_lr":config["alpha_lr"],"target_entropy":config["target_entropy"],"target_entropy_source":"explicit Stage4 config (equals -action_dim)","action_dim":14,"log_pi_reduction":"sum over 14 action dimensions; masked mean over T x B","timesteps":"first T of T+1 recurrent outputs","burn_in":False,"tanh_log_prob_correction":True,"reparameterized_rsample":True,"critic_frozen_actor_gradient_preserved":True}
    minibatch_indices=[[(i*a.batch_size+j)%N for j in range(a.batch_size)] for i in range(a.gradient_batches)]
    report={"run_path":str(run),"fixed_data_source":source,"sequence_count":N,"gradient_minibatches":a.gradient_batches,"gradient_minibatch_indices":minibatch_indices,"implementation":implementation,"entropy":entropy_summary,"alpha_history":history,"alpha_sweep":sweep,"module_decomposition":module_rows,"transitions":transitions,"reproduction":reproduction,"mechanism_case":case,"conclusion":conclusion,"training_performed":False,"optimizer_step_called":False,"new_stage4_started":False};atomic_json(output/"stage4_actor_objective_audit.json",report)
    lines=["# Stage4 Actor Objective Decomposition & Alpha Sweep Audit","","## Current SAC entropy configuration","","| Formula | Automatic alpha | Alpha LR | Target entropy | Burn-in |","|---|---:|---:|---:|---:|",f"| `{implementation['formula']}` | {implementation['automatic_entropy_tuning']} | {implementation['alpha_lr']} | {implementation['target_entropy']} | {implementation['burn_in']} |","","## Step0 Actor gradient decomposition","",f"Q-only cosine: {base['q_only_cosine']:.8f}; full cosine: {base['total_grad_cosine']:.8f}; mu-only full cosine: {base['mu_only_total_grad_cosine']:.8f}.","","## Alpha sweep","","| Alpha | Total cosine | Mu-only cosine | RNN H/Q | Multi H/Q | Relative difference |","|---:|---:|---:|---:|---:|---:|"]
    for r in sweep:lines.append(f"| {r['alpha']:.3g} | {r['total_grad_cosine']:.8f} | {r['mu_only_total_grad_cosine']:.8f} | {r['rnn_entropy_q_ratio']:.5f} | {r['multi_entropy_q_ratio']:.5f} | {r['relative_grad_difference']:.5f} |")
    lines += ["","## Module-level gradient decomposition","","See `module_gradient_decomposition.csv` for all 20 minibatches.","",f"Mechanism classification: **{case}**", "",conclusion,"",f"Reproduction: **{reproduction['status']}**","","NO TRAINING PERFORMED  ","NO OPTIMIZER STEP  ","NO NEW STAGE4 STARTED"];(output/"stage4_actor_objective_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Output: {output}");print(f"Reproduction: {reproduction['status']}");print(f"Mechanism: {case} - {conclusion}");print("NO TRAINING PERFORMED\nNO OPTIMIZER STEP\nNO NEW STAGE4 STARTED")

if __name__=="__main__":main()
