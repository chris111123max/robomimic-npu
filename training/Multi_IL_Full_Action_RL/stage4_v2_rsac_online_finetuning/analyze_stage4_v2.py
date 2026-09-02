#!/usr/bin/env python3
"""Read-only final Stage4-v2 drift, preservation, and learning-curve analysis."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent;V1=PROJECT/"stage4_rsac_online_finetuning";sys.path.insert(0,str(V1));sys.path.insert(0,str(PROJECT/"stage3_r_bc_rnn_to_rsac"))
from audit_critic_washout import aligned,load_critic,q_values,similarity,grad_pair,read_json,select_device
from stage3_r_actor import load_actor

GROUPS=("rnn_only_critic","multi_il_critic");EVAL=(0,5000,10000,20000,30000,40000,50000,60000,80000,100000);DRIFT=(0,5000,10000,20000,30000,40000,50000,100000);CRIT=(0,5000,10000,20000)
def rows(path):
 with Path(path).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def write(path,data):
 with Path(path).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
def params(state,prefixes=None):
 values=[v.detach().cpu().reshape(-1).double() for k,v in state.items() if prefixes is None or any(k.startswith(x) for x in prefixes)];return torch.cat(values)
def actor_diag(initial,current,batch,device):
 actor,_=load_actor(initial["path"],device);actor.load_state_dict(current,strict=True);actor.eval();base=initial["state"];out={}
 for name,prefix in (("full",None),("lstm",("lstm.",)),("mu_head",("policy.last_fc.",)),("log_std_head",("policy.last_fc_log_std.",))):
  a=params(base,prefix);b=params(current,prefix);out[f"{name}_drift_l2"]=float((b-a).norm());out[f"{name}_normalized_drift"]=float((b-a).norm()/a.norm().clamp_min(1e-30))
 obs=batch["obs"].to(device).transpose(0,1);mask=batch["mask"].to(device).transpose(0,1).bool().expand(-1,-1,14)
 with torch.no_grad():
  initial["actor"].eval();a0=initial["actor"].deterministic_sequence(obs);at=actor.deterministic_sequence(obs);features=actor.forward_sequence(obs);_,_,ls,lp=actor.actions_from_features(features,False,True);delta=(at-a0)[mask];out.update(action_mse_vs_initial=float((delta**2).mean()),action_mae_vs_initial=float(delta.abs().mean()),action_mean_cosine=float(torch.nn.functional.cosine_similarity(at[mask].reshape(-1,14),a0[mask].reshape(-1,14),dim=-1).mean()),mean_log_std=float(ls[mask].mean()),mean_std=float(ls[mask].exp().mean()),mean_log_pi=float(lp.squeeze(-1)[batch["mask"].to(device).transpose(0,1).squeeze(-1).bool()].mean()));out["estimated_entropy"]=-out["mean_log_pi"]
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",required=True);p.add_argument("--config",required=True);p.add_argument("--device",default="npu:0");a=p.parse_args();run=Path(a.run_dir);cfg=read_json(a.config);device=select_device(a.device);comp=run/"comparison";comp.mkdir(exist_ok=True)
 fixed=torch.load(cfg["diagnostic_sequences"],map_location="cpu");batch=fixed["batch"];initial_actor,payload=load_actor(cfg["actor_checkpoint"],device);initial={"path":cfg["actor_checkpoint"],"actor":initial_actor,"state":{k:v.cpu() for k,v in initial_actor.state_dict().items()}}
 summaries={};curves=[];drifts={}
 for g in GROUPS:
  ev={int(r["env_step"]):r for r in rows(run/g/"evaluation_metrics.csv")};curve={s:{"success_rate":float(ev[s]["success_rate"]),"mean_progress":float(ev[s]["mean_progress"])} for s in EVAL};summaries[g]={"curve":curve,"step0_sr":curve[0]["success_rate"],"minimum_sr_0_40k":min(curve[s]["success_rate"] for s in EVAL if s<=40000)};summaries[g]["degradation_0_40k"]=summaries[g]["step0_sr"]-summaries[g]["minimum_sr_0_40k"];best=max(EVAL,key=lambda s:(curve[s]["success_rate"],curve[s]["mean_progress"],-s));summaries[g].update(best_sr=curve[best]["success_rate"],best_step=best,final_sr=curve[100000]["success_rate"])
  drifts[g]={}
  for step in DRIFT:
   path=run/g/"checkpoints"/f"step_{step:08d}.pth";state=initial["state"] if step==0 else torch.load(path,map_location="cpu")["actor_state_dict"];drifts[g][str(step)]=actor_diag(initial,state,batch,device)
  summaries[g]["actor_diagnostics"]=drifts[g]
  diagnostic_dir=run/g/"diagnostics";diagnostic_dir.mkdir(exist_ok=True);write(diagnostic_dir/"actor_action_entropy_drift.csv",[{"env_step":int(step),**values} for step,values in drifts[g].items()])
 for s in EVAL:curves.append({"env_step":s,"rnn_success_rate":summaries[GROUPS[0]]["curve"][s]["success_rate"],"multi_success_rate":summaries[GROUPS[1]]["curve"][s]["success_rate"],"rnn_mean_progress":summaries[GROUPS[0]]["curve"][s]["mean_progress"],"multi_mean_progress":summaries[GROUPS[1]]["curve"][s]["mean_progress"]})
 write(comp/"stage4_v2_learning_curves.csv",curves)
 critic_rows=[]
 for step in CRIT:
  paths=[Path(cfg["rnn_critic_checkpoint"]),Path(cfg["multi_critic_checkpoint"])] if step==0 else [run/g/"checkpoints"/f"step_{step:08d}.pth" for g in GROUPS];critics=[load_critic(cfg,x,device)[0] for x in paths];b,obs,prev,rew=aligned(batch,device);mask=b["mask"].bool()
  with torch.no_grad():values=[q_values(c,prev,rew,obs,b["act"])[mask].cpu().numpy() for c in critics]
  sim=similarity(*values);gr=[]
  for i in range(20):idx=torch.arange(i*32,i*32+32)%batch["obs"].shape[1];mini={k:v.index_select(1,idx) for k,v in batch.items()};gr.append(grad_pair(critics,initial["state"],cfg["actor_checkpoint"],mini,device,20260902+i,.001)["q_cos"])
  critic_rows.append({"env_step":step,"q_pearson":sim["pearson"],"q_spearman":sim["spearman"],"q_nrmse":sim["normalized_rmse"],"q_grad_cosine":float(np.mean(gr)),"q_grad_cosine_std":float(np.std(gr))})
 write(comp/"critic_difference_over_time.csv",critic_rows)
 collapse=all(summaries[g]["minimum_sr_0_40k"]==0 for g in GROUPS);separation=any(abs(summaries[GROUPS[0]]["curve"][s]["success_rate"]-summaries[GROUPS[1]]["curve"][s]["success_rate"])>.05 or abs(summaries[GROUPS[0]]["curve"][s]["mean_progress"]-summaries[GROUPS[1]]["curve"][s]["mean_progress"])>.05 for s in EVAL)
 result={"stage":"Stage4-v2","run_path":str(run),"config":cfg,"config_diff":read_json(run/"config_diff.json"),"groups":summaries,"critic_difference":critic_rows,"synchronized_collapse_to_zero_0_40k":collapse,"behavioral_separation_detected":separation,"random_group_run":False,"extended_to_1m":False,"stage5_started":False,"extra_hyperparameter_tuning":False};(comp/"stage4_v2_comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
 lines=["# Stage4-v2 Comparison","","| Metric | RNN-only | Multi-IL |","|---|---:|---:|"]
 for label,key in (("Step0 SR","step0_sr"),("Min SR 0-40k","minimum_sr_0_40k"),("Degradation 0-40k","degradation_0_40k"),("Best SR","best_sr"),("Best step","best_step"),("Final SR","final_sr")):lines.append(f"| {label} | {summaries[GROUPS[0]][key]} | {summaries[GROUPS[1]][key]} |")
 lines += ["",f"Synchronized 30%→0% collapse: **{collapse}**",f"Behavioral separation detected: **{separation}**","","NO RANDOM GROUP RUN  ","NO 1M EXTENSION  ","NO STAGE5 STARTED  ","NO EXTRA HYPERPARAMETER TUNING"];(comp/"stage4_v2_comparison.md").write_text("\n".join(lines)+"\n")
 print("Stage4-v2 comparison:",comp/"stage4_v2_comparison.json")
if __name__=="__main__":main()
