#!/usr/bin/env python3
"""Frozen-scoring-function Actor temporal causal test. No production mutation."""
import argparse,copy,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
for f in (ROOT/"stage3_v5_rgmm_td3",ROOT/"stage3_v6_dual_2q",ROOT/"stage3_v6_dual_2q/testing/actor_collapse_diagnosis"):
 sys.path.insert(0,str(f))
from stage3_v5_actor import load_exact_actor,flat_to_obs,module_hash,distribution_tensors
from stage3_v5_replay import Stage1OfflineSequenceReplay
from stage3_v5_history_critic import encode_replay_contexts,component_mean_q
from stage3_v6_agent import strict_stage2_load
from stage3_v5_schedule import CriticHandoff,HandoffState,TrainingState
from run_offline import resolve_device
SEED=20261003
BRANCHES=("CURRENT","ALL_ALIGNED","FINAL_ONLY_FREEZE_RNN")
MILESTONES=(0,1,10,25,50,100,250,500,1000)

def dump(p,x):
 if p.exists(): raise FileExistsError(p)
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+"\n")

def group(n):
 if n.startswith("nets.rnn.nets."): return "rnn"
 if n.startswith("nets.decoder.nets.mean."): return "mean"
 if n.startswith("nets.decoder.nets.logits."): return "logits"
 if n.startswith("nets.decoder.nets.scale."): return "std"
 return "encoder" if "encoder" in n else "other"

def prepare(run,out):
 cfg=json.loads((run/"shared/config_resolved.json").read_text())
 frozen_path=run/"mean2q/multi_q/checkpoints/last.sequences.npy"
 frozen=np.load(frozen_path,allow_pickle=True).item()["fixed_critic_diagnostic_set"]
 offline=Stage1OfflineSequenceReplay(cfg["offline_sources"]["bc_rnn"],"rnn",seed=SEED)
 rng=np.random.default_rng(SEED)
 selections=[]; os=[]; ac=[]; es=[]; labels=[]
 for label,episodes,want_success in (("offline_success",offline.episodes,True),("online_success",frozen["episodes"],True),("online_failure",frozen["episodes"],False)):
  eligible=[(i,s) for i,e in enumerate(episodes) if bool(e.get("success",False))==want_success
            for s in range(10,len(e["actions"])-9,10)]
  if len(eligible)<64: raise RuntimeError((label,len(eligible)))
  chosen=rng.choice(len(eligible),64,replace=False)
  for j in chosen:
   i,s=eligible[int(j)]; e=episodes[i]; lo=s-9; hi=s+10
   os.append(np.asarray(e["observations"][lo:hi],np.float32))
   ac.append(np.asarray(e["actions"][lo:hi],np.float32))
   es.append(np.asarray(e["episode_steps"][lo:hi],np.int64))
   labels.append(label)
   selections.append({"source":label,"episode_index":int(i),"actor_start":int(s),"history_start":int(lo),"stop_exclusive":int(hi),"diagnostic_episode_id":int(e["diagnostic_episode_id"]) if e.get("diagnostic_episode_id") is not None else None})
 data={"observations":np.stack(os),"actions":np.stack(ac),"episode_steps":np.stack(es),"labels":np.array(labels)}
 assert data["observations"].shape==(192,19,59)
 assert np.all(data["episode_steps"][:,9]%10==0)
 np.savez_compressed(out/"fixed_temporal_contexts.npz",**data)
 dump(out/"sampling_manifest.json",{"seed":SEED,"source":str(frozen_path),"offline_source":cfg["offline_sources"]["bc_rnn"],"selections":selections,"actor_history_length":10,"critic_window_length":10,"combined_history_length":19})
 return data,cfg

def forward(actor,obs,capture=False):
 store=[]
 hook=actor.nets["rnn"].nets.register_forward_hook(lambda m,i,o:store.append(o[0].detach())) if capture else None
 try: dist=actor.forward_train(flat_to_obs(obs),rnn_init_state=None,return_state=False)
 finally:
  if hook: hook.remove()
 return dist,(store[0] if capture else None)

def final_distribution(dist):
 base=dist.component_distribution.base_dist
 return torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=dist.mixture_distribution.logits[:,-1]),torch.distributions.Independent(torch.distributions.Normal(base.loc[:,-1],base.scale[:,-1]),1))

def objective(actor,critic,obs,contexts,scale,offset,branch):
 dist,_=forward(actor,obs)
 if branch=="ALL_ALIGNED":
  q,_,_,_,_=component_mean_q(critic,contexts,dist,scale,offset,twin_min=False)
 else:
  q,_,_,_,_=component_mean_q(critic,tuple(c[:,-1] for c in contexts),final_distribution(dist),scale,offset,twin_min=False)
 return -q.mean()

def sampled_stream(actor,obs):
 # Direct production distribution.sample() semantics, low_noise_eval, zero initial state.
 cpu_rng=torch.get_rng_state(); npu_rng=torch.npu.get_rng_state()
 try:
  torch.manual_seed(SEED); torch.npu.manual_seed(SEED)
  state=None; samples=[]
  actor.eval()
  for t in range(10):
   dist,state=actor.forward_train_step(flat_to_obs(obs[:,t]),rnn_state=state)
   samples.append(dist.sample())
  return torch.stack(samples,dim=1)
 finally:
  torch.set_rng_state(cpu_rng); torch.npu.set_rng_state(npu_rng)

def evaluate(actor,obs,critic,contexts,scale,offset):
 result={k:[] for k in ("means","probs","logits","hidden","sampled","q")}
 actor.eval()
 with torch.no_grad():
  for lo in range(0,len(obs),64):
   x=obs[lo:lo+64]; dist,h=forward(actor,x,True); tensors=distribution_tensors(dist)
   q,_,_,_,_=component_mean_q(critic,tuple(c[lo:lo+64] for c in contexts),dist,scale,offset,twin_min=False)
   for k,v in (("means",tensors["means_normalized"]),("probs",tensors["probs"]),("logits",tensors["logits"]),("hidden",h),("sampled",sampled_stream(actor,x)),("q",q)):
    result[k].append(v.cpu().numpy())
 return {k:np.concatenate(v) for k,v in result.items()}

def temporal_stats(d):
 a=np.asarray(d,np.float64)
 return {"l2_mean_by_timestep":np.linalg.norm(a,axis=-1).mean(axis=0).tolist(),
         "abs_mean_by_timestep":np.abs(a).mean(axis=(0,2)).tolist(),
         "abs_max_by_timestep":np.abs(a).max(axis=(0,2)).tolist(),
         "dimension_abs_mean_by_timestep":np.abs(a).mean(axis=0).tolist()}

def parameter_drift(actor,initial):
 accum={}
 for name,p in actor.named_parameters():
  k=group(name); x=p.detach().cpu().double(); b=initial[name].double(); d=x-b
  a=accum.setdefault(k,[0.,0.,0]); a[0]+=float(d.square().sum()); a[1]+=float(b.square().sum()); a[2]+=p.numel()
 total=[sum(x[j] for x in accum.values()) for j in range(3)]; accum["total"]=total
 return {k:{"l2":v[0]**.5,"rms":(v[0]/v[2])**.5,"relative_l2":(v[0]/v[1])**.5 if v[1] else None,"numel":v[2]} for k,v in accum.items()}

def metrics(current,baseline,actor,initial):
 mean=(current["probs"][...,None]*current["means"]).sum(-2)
 old=(baseline["probs"][...,None]*baseline["means"]).sum(-2)
 structural=temporal_stats(mean-old)
 sampled=temporal_stats(current["sampled"]-baseline["sampled"])
 components=temporal_stats((current["means"]-baseline["means"]).reshape(len(mean),10,-1))
 hidden=temporal_stats(current["hidden"]-baseline["hidden"])
 v=np.array(structural["l2_mean_by_timestep"])
 ranks=np.argsort(-current["probs"],axis=-1); oldr=np.argsort(-baseline["probs"],axis=-1)
 entropy=-(current["probs"]*np.log(np.maximum(current["probs"],1e-12))).sum(-1)
 return {"weighted_mean_drift":structural,"sampled_action_drift":sampled,"component_means_drift":components,"hidden_drift":hidden,
         "early":float(v[:3].mean()),"mid":float(v[3:7].mean()),"late":float(v[7:9].mean()),"final":float(v[9]),
         "early_to_final_drift_ratio":float(v[:9].mean()/v[9]) if v[9]>0 else None,
         "gmm":{"probs_abs_mean_by_timestep":np.abs(current["probs"]-baseline["probs"]).mean(axis=(0,2)).tolist(),
                "logits_rms":float(np.sqrt(np.mean((current["logits"]-baseline["logits"])**2))),
                "entropy_mean_by_timestep":entropy.mean(axis=0).tolist(),
                "top1_change_fraction_by_timestep":np.mean(ranks[...,0]!=oldr[...,0],axis=0).tolist(),
                "rank_change_fraction_by_timestep":np.any(ranks!=oldr,axis=-1).mean(axis=0).tolist(),
                "top1_top2_swap_fraction_by_timestep":np.mean((ranks[...,0]==oldr[...,1])&(ranks[...,1]==oldr[...,0]),axis=0).tolist()},
         "parameter_drift":parameter_drift(actor,initial),"frozen_Q1_mean_by_timestep":current["q"].mean(axis=0).tolist(),
         "frozen_Q1_gain_by_timestep":(current["q"]-baseline["q"]).mean(axis=0).tolist()}

def grad_stats(actor):
 out={}
 for name,p in actor.named_parameters():
  k=group(name); a=out.setdefault(k,[0.,0.,0]); a[1]+=float(p.detach().cpu().double().square().sum()); a[2]+=p.numel()
  if p.grad is not None: a[0]+=float(p.grad.detach().cpu().double().square().sum())
 return {k:{"norm":v[0]**.5,"rms":(v[0]/v[2])**.5,"grad_parameter_norm_ratio":(v[0]/v[1])**.5 if v[1] else None,"numel":v[2]} for k,v in out.items()}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--run",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--device",default="npu:0"); ap.add_argument("--updates",type=int,default=1000); ap.add_argument("--hold",action="store_true")
 args=ap.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
 if (out/"offline_results.json").exists(): raise FileExistsError(out)
 device=resolve_device(args.device); run=args.run.resolve(); data,cfg=prepare(run,out)
 obs=torch.as_tensor(data["observations"][:,9:],device=device); acts=torch.as_tensor(data["actions"][:,9:],device=device); steps=torch.as_tensor(data["episode_steps"][:,9:],device=device)
 actor,rollout,_=load_exact_actor(run/"shared/bc_rnn_gmm_source.pth",device)
 ready=torch.load(run/"mean2q/multi_q/checkpoints/critic_ready.pth",map_location="cpu",weights_only=False)
 assert ready["env_steps"]==140000 and ready["actor_updates"]==0
 actor.load_state_dict(ready["actor"],strict=True)
 manifest=json.loads((run/"shared/stage2_source_manifest.json").read_text())
 critic,_=strict_stage2_load(manifest["multi_q"]["checkpoint"],device); critic.load_state_dict(ready["q1_q2"],strict=True); critic.eval().requires_grad_(False)
 critic_hash=module_hash(critic)
 scale=torch.as_tensor(rollout.action_normalization_stats["actions"]["scale"],dtype=torch.float32,device=device).reshape(1,1,1,14)
 offset=torch.as_tensor(rollout.action_normalization_stats["actions"]["offset"],dtype=torch.float32,device=device).reshape(1,1,1,14)
 contexts=[]
 with torch.no_grad():
  for t in range(10):
   cs=encode_replay_contexts(critic,torch.as_tensor(data["observations"][:,t:t+10],device=device),torch.as_tensor(data["actions"][:,t:t+10],device=device),torch.as_tensor(data["episode_steps"][:,t:t+10],device=device),700)
   contexts.append(tuple(x[:,-1] for x in cs))
  aligned=tuple(torch.stack([c[j] for c in contexts],dim=1).detach() for j in range(2))
  current=encode_replay_contexts(critic,obs,acts,steps,700)
  assert all(torch.equal(c[:,-1],a[:,-1]) for c,a in zip(current,aligned))
 initial={k:p.detach().cpu().clone() for k,p in actor.named_parameters()}
 baseline=evaluate(actor,obs,critic,aligned,scale,offset)
 np.savez_compressed(out/"baseline_outputs.npz",**baseline)
 clones={b:copy.deepcopy(actor) for b in BRANCHES}
 baseline_checks={}
 for branch,clone in clones.items():
  test=evaluate(clone,obs,critic,aligned,scale,offset)
  check={"hash":module_hash(clone),"same_parameter_hash":module_hash(clone)==module_hash(actor),"max_differences":{k:float(np.max(np.abs(test[k]-baseline[k]))) for k in baseline}}
  assert check["same_parameter_hash"] and all(v==0 for v in check["max_differences"].values()),check
  baseline_checks[branch]=check
 dump(out/"baseline_checks.json",baseline_checks)
 # 50% offline and 50% online, exactly matching production Actor batch composition.
 batches=[np.concatenate((np.arange(32)+(j%2)*32,np.arange(16)+64+(j%4)*16,np.arange(16)+128+(j%4)*16)) for j in range(4)]
 rows=[json.loads(l) for l in (run/"mean2q/multi_q/train_metrics.jsonl").read_text().splitlines()]
 anchors=[(0,140000)]+[(int(r["actor_updates"]),int(r["env_steps"])) for r in rows if r.get("actor_updates",0)>0 and r["env_steps"]>=140000]
 anchors=sorted(set(anchors)); monotonic=[]
 for u,e in anchors:
  if not monotonic or (u>monotonic[-1][0] and e>=monotonic[-1][1]): monotonic.append((u,e))
 envs=np.rint(np.interp(np.arange(1,args.updates+1),[x[0] for x in monotonic],[x[1] for x in monotonic])).astype(int)
 scheduler=CriticHandoff(cfg,HandoffState(state=TrainingState.ACTOR_WARMUP,critic_ready=True,critic_ready_step=140000,actor_warmup_steps=140000,joint_rl_start_step=280000))
 lrs=[scheduler.schedule(int(e),cfg["critic_lr"])["actor_lr"] for e in envs]
 dump(out/"schedule.json",{"kind":"production scheduler with log-calibrated update/env mapping","limitation":"per-optimizer-step env timestamps are not saved; linearly interpolated between actual logged cumulative Actor update counts; all branch schedules identical","anchors":monotonic,"env_steps":envs.tolist(),"actor_lrs":lrs})
 result={"run":str(run),"device":str(device),"updates":args.updates,"baseline_checks":baseline_checks,"sampling_manifest":str(out/"sampling_manifest.json"),"gradient_stats":{},"branches":{},"critic_frozen_hash":critic_hash,"production_source":"V6 inherits V5 final distribution, Q1-only, full BPTT, zero-init horizon10","closed_loop":"pending decision"}
 for branch,clone in clones.items():
  if branch=="FINAL_ONLY_FREEZE_RNN":
   for n,p in clone.named_parameters():
    if group(n)=="rnn": p.requires_grad_(False)
  optimizer=torch.optim.Adam(clone.parameters(),lr=0.,weight_decay=0.)
  milestones={"0":metrics(baseline,baseline,clone,initial)}
  for u in range(1,args.updates+1):
   clone.train(); ids=batches[(u-1)%4]
   for pg in optimizer.param_groups: pg["lr"]=lrs[u-1]
   loss=objective(clone,critic,obs[ids],tuple(c[ids] for c in aligned),scale,offset,branch)
   optimizer.zero_grad(set_to_none=True); loss.backward()
   if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in clone.parameters()): raise FloatingPointError((branch,u))
   if u==1: result["gradient_stats"][branch]=grad_stats(clone)
   torch.nn.utils.clip_grad_norm_(clone.parameters(),float(cfg["actor_max_grad_norm"])); optimizer.step()
   if u in MILESTONES or u==args.updates:
    evaluated=evaluate(clone,obs,critic,aligned,scale,offset)
    np.savez_compressed(out/f"{branch}_outputs_{u:04d}.npz",**evaluated)
    milestones[str(u)]=metrics(evaluated,baseline,clone,initial)
    m=milestones[str(u)]
    print(json.dumps({"branch":branch,"updates":u,"lr":lrs[u-1],"early":m["early"],"final":m["final"],"sampled_early":np.mean(m["sampled_action_drift"]["l2_mean_by_timestep"][:3])}),flush=True)
  result["branches"][branch]={"milestones":milestones}
 assert module_hash(critic)==critic_hash
 result["critic_hash_unchanged"]=True
 dump(out/"offline_results.json",result)
 print("OFFLINE_COMPLETE_IN_MEMORY_CLONES_HELD",flush=True)
 if args.hold:
  deadline=time.monotonic()+900
  decision=out/"closed_loop_decision.txt"
  while not decision.exists() and time.monotonic()<deadline: time.sleep(1)
  if decision.exists() and decision.read_text().strip()=="RUN":
   from run_temporal_closed_loop import run_closed_loop
   run_closed_loop(run,out,actor,clones,scale.reshape(14),offset.reshape(14),device)
  else: print("CLOSED_LOOP_SKIPPED",flush=True)
 print("COMPLETE; FORMAL TRAINING REMAINS STOPPED",flush=True)
if __name__=="__main__": main()
