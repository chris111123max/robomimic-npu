#!/usr/bin/env python3
"""Read-only Stage4 Critic-initialization washout audit."""
from __future__ import annotations

import argparse, copy, csv, hashlib, json, math, sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent
for path in (HERE,PROJECT/"stage3_r_bc_rnn_to_rsac",PROJECT/"stage2_r_recurrent_critic_pretraining"):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from stage4_core import OnlineSequenceReplay, Stage3RSACAdapter, architecture, make_pair, read_json, select_device, seed_all  # noqa
from stage3_r_actor import load_actor  # noqa

GROUPS=("rnn_only_critic","multi_il_critic")
FIELDS=("obs","obs2","act","rew","term","mask")

def args():
    p=argparse.ArgumentParser();p.add_argument("--run-dir");p.add_argument("--root",default="/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage4_rsac_online_finetuning");p.add_argument("--config",default=str(HERE/"stage4_config.json"));p.add_argument("--device",default="npu:0");p.add_argument("--sequences",type=int,default=1024);p.add_argument("--gradient-batches",type=int,default=20);p.add_argument("--gradient-batch-size",type=int,default=32);return p.parse_args()

def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8");tmp.replace(path)

def discover(root):
    candidates=[]
    for item in Path(root).iterdir():
        if not item.is_dir() or item.name.startswith("smoke_"):continue
        if all((item/g/"checkpoints").is_dir() for g in GROUPS):candidates.append(item)
    if not candidates:raise RuntimeError(f"No formal Stage4 run containing {GROUPS} under {root}")
    return max(candidates,key=lambda p:p.stat().st_mtime)

def checkpoints(directory):
    found={}
    for path in sorted((directory/"checkpoints").glob("*.pth")):
        try:payload=torch.load(path,map_location="cpu")
        except Exception:continue
        if "env_step" in payload:
            step=int(payload["env_step"])
            if step>0 and (step not in found or path.name.startswith("step_")):found[step]=path
    return found

def sequence_hash(batch,index):
    h=hashlib.sha256()
    for key in FIELDS:h.update(np.ascontiguousarray(batch[key][:,index]).tobytes())
    return h.hexdigest()

def from_replay(path,config,count):
    replay=OnlineSequenceReplay(config);replay.load(path);np.random.seed(20260902);batch=replay.buffer.random_episodes(min(int(count),max(1,int(replay.buffer._valid_starts.sum()))))
    ids=[{"index":i,"sha256":sequence_hash(batch,i),"source_replay":str(path)} for i in range(batch["obs"].shape[1])]
    return {key:torch.as_tensor(batch[key],dtype=torch.float32) for key in FIELDS},ids,{"kind":"Stage4 replay","path":str(path),"transitions":replay.transitions}

def from_stage1(config,count,actor,device):
    from stage2_r_data import load_episodes, attach_actor_actions, validation_sequences
    cfg=read_json(PROJECT/"stage2_r_v2_recurrent_critic_pretraining"/"stage2_r_v2_config.json")
    episodes=[]
    for policy,path in cfg["datasets"].items():episodes.extend(load_episodes(policy,path,range(10080,10100)))
    attach_actor_actions(episodes,actor,device);records=validation_sequences(episodes,config["sequence_length"])[:count]
    batch={key:torch.as_tensor(np.stack([r[key] for r in records],axis=1),dtype=torch.float32) for key in FIELDS}
    ids=[{"index":i,"policy":r["source"],"seed":int(r["seed"]),"start":int(r["start"]),"valid":int(r["valid"]),"sha256":sequence_hash(batch,i)} for i,r in enumerate(records)]
    return batch,ids,{"kind":"Stage1 validation","seeds":"10080-10099","policies":list(cfg["datasets"])}

def aligned(batch,device):
    b={k:v.to(device) for k,v in batch.items()};obs,obs2,act,rew=b["obs"],b["obs2"],b["act"],b["rew"];B=obs.shape[1]
    za=torch.zeros(1,B,14,device=device);zr=torch.zeros(1,B,1,device=device)
    return b,torch.cat((obs[[0]],obs2),0),torch.cat((za,act),0),torch.cat((zr,rew),0)

def load_critic(config,path,device,initial=False,group=None):
    critic,_=make_pair(config,device);payload=torch.load(path,map_location=device)
    state=payload["critic_state_dict"]
    critic.load_state_dict(state,strict=True);critic.eval().requires_grad_(False);return critic,payload

def q_values(critic,prev_actions,rewards,observs,current_actions):
    q1,q2=critic(prev_actions,rewards,observs,current_actions);return torch.minimum(q1,q2)

def ranks(x):
    order=np.argsort(x,kind="mergesort");result=np.empty(len(x),float);i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]:j+=1
        result[order[i:j]]=(i+j-1)/2;i=j
    return result

def corr(a,b):
    if len(a)<2 or np.std(a)==0 or np.std(b)==0:return 0.0
    return float(np.corrcoef(a,b)[0,1])

def similarity(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float);delta=a-b;rmse=float(np.sqrt(np.mean(delta**2)));pooled=float(np.sqrt((np.var(a)+np.var(b))/2));
    return {"q_rnn_mean":float(a.mean()),"q_multi_mean":float(b.mean()),"absolute_q_difference_mean":float(np.mean(np.abs(delta))),"rmse":rmse,"normalized_rmse":rmse/max(pooled,1e-12),"pearson":corr(a,b),"spearman":corr(ranks(a),ranks(b)),"sign_agreement":float(np.mean((a>0)==(b>0))),"pooled_q_std":pooled}

def flatten_grads(grads,parameters):
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g,p in zip(grads,parameters)])

def grad_pair(critics,actor_state,actor_path,batch,device,seed,alpha):
    actor,_=load_actor(actor_path,device);actor.load_state_dict(actor_state,strict=True);actor.train();actor.requires_grad_(True);parameters=[p for p in actor.parameters() if p.requires_grad]
    b,observs,prev,rewards=aligned(batch,device);seed_all(seed);actions,logp=Stage3RSACAdapter(actor)(prev,rewards,observs);mask=b["mask"];valid=mask.sum().clamp_min(1.0)
    losses=[]
    for critic in critics:
        q=q_values(critic,prev,rewards,observs,actions)[:-1];q_loss=-(q*mask).sum()/valid;full=(float(alpha)*logp[:-1]*mask).sum()/valid+q_loss;losses.append((q_loss,full))
    vectors=[]
    for index,(q_loss,full) in enumerate(losses):
        qg=flatten_grads(torch.autograd.grad(q_loss,parameters,retain_graph=True,allow_unused=True),parameters)
        fg=flatten_grads(torch.autograd.grad(full,parameters,retain_graph=index==0,allow_unused=True),parameters);vectors.append((qg.detach(),fg.detach()))
    def compare(x,y):
        nx=float(x.norm().item());ny=float(y.norm().item());cos=float(torch.dot(x,y).item()/max(nx*ny,1e-30));return cos,nx,ny,float((x-y).norm().item()/max(nx+ny,1e-30))
    q=compare(vectors[0][0],vectors[1][0]);s=compare(vectors[0][1],vectors[1][1]);return {"q_cos":q[0],"rnn_norm":q[1],"multi_norm":q[2],"norm_ratio":q[1]/max(q[2],1e-30),"relative_difference":q[3],"sac_cos":s[0]}

def alpha_of(payload,config):
    value=payload.get("log_alpha_entropy")
    return float(config["initial_entropy_alpha"]) if value is None else float(torch.as_tensor(value).exp().item())

def parameter_distance(a,b):
    va=torch.cat([p.detach().cpu().reshape(-1).double() for p in a.parameters()]);vb=torch.cat([p.detach().cpu().reshape(-1).double() for p in b.parameters()]);d=float((va-vb).norm());return d,d/max(float(va.norm()+vb.norm()),1e-30)

def evaluation_table(run):
    result={}
    for group in GROUPS:
        path=run/group/"evaluation_metrics.csv"
        if not path.exists():result[group]="deterministic evaluation data unavailable";continue
        with path.open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
        result[group]=[{"env_step":int(r["env_step"]),"success_rate":float(r["success_rate"]),"mean_progress":float(r["mean_progress"])} for r in rows]
    return result

def main():
    a=args();config=read_json(a.config);device=select_device(a.device);seed_all(20260902);run=Path(a.run_dir).resolve() if a.run_dir else discover(a.root);output=run/"diagnostics"/"critic_washout";output.mkdir(parents=True,exist_ok=True)
    maps={g:checkpoints(run/g) for g in GROUPS};common=sorted(set(maps[GROUPS[0]])&set(maps[GROUPS[1]]));steps=[0]+common
    actor,actor_payload=load_actor(config["actor_checkpoint"],device);actor_state=copy.deepcopy(actor.state_dict())
    replay=run/"rnn_only_critic"/"replay"/"step_00030000.npz"
    if replay.exists():batch,identifiers,source=from_replay(replay,config,a.sequences)
    else:batch,identifiers,source=from_stage1(config,a.sequences,actor,device)
    torch.save({"format_version":"stage4.critic_washout.fixed.v1","sequence_length":10,"seed":20260902,"source":source,"identifiers":identifiers,"batch":batch},output/"fixed_diagnostic_sequences.pt")
    initial={"rnn_only_critic":Path(config["rnn_critic_checkpoint"]),"multi_il_critic":Path(config["multi_critic_checkpoint"])};rows=[];gradient_rows=[];function={}
    for step in steps:
        paths=initial if step==0 else {g:maps[g][step] for g in GROUPS};loaded=[load_critic(config,paths[g],device,step==0,g) for g in GROUPS];critics=[x[0] for x in loaded];payloads=[x[1] for x in loaded]
        b,observs,prev,rewards=aligned(batch,device);mask=b["mask"].bool()
        with torch.no_grad():
            behavior=[q_values(c,prev,rewards,observs,b["act"])[mask].cpu().numpy() for c in critics]
            seed_all(20260902);pi_actions,_=Stage3RSACAdapter(actor)(prev,rewards,observs);policy=[q_values(c,prev,rewards,observs,pi_actions)[:-1][mask].cpu().numpy() for c in critics]
        bs=similarity(*behavior);ps=similarity(*policy);pd,pdn=parameter_distance(*critics);alphas=[alpha_of(x,config) for x in payloads];gr=[]
        size=batch["obs"].shape[1]
        for i in range(a.gradient_batches):
            start=(i*a.gradient_batch_size)%size;idx=torch.arange(start,start+a.gradient_batch_size)%size;mini={k:v.index_select(1,idx) for k,v in batch.items()};item=grad_pair(critics,actor_state,config["actor_checkpoint"],mini,device,20260902+i,float(config["initial_entropy_alpha"]));item.update(env_step=step,batch_index=i);gr.append(item);gradient_rows.append(item)
        gc=np.asarray([x["q_cos"] for x in gr]);sc=np.asarray([x["sac_cos"] for x in gr]);rd=np.asarray([x["relative_difference"] for x in gr])
        row={"env_step":step,"behavior_q_rmse":bs["rmse"],"behavior_q_nrmse":bs["normalized_rmse"],"behavior_q_pearson":bs["pearson"],"behavior_q_spearman":bs["spearman"],"policy_q_rmse":ps["rmse"],"policy_q_nrmse":ps["normalized_rmse"],"policy_q_pearson":ps["pearson"],"policy_q_spearman":ps["spearman"],"q_grad_cos_mean":float(gc.mean()),"q_grad_cos_std":float(gc.std()),"q_grad_cos_min":float(gc.min()),"q_grad_cos_max":float(gc.max()),"q_grad_relative_diff":float(rd.mean()),"sac_grad_cos_mean":float(sc.mean()),"sac_grad_cos_std":float(sc.std()),"parameter_distance":pd,"normalized_parameter_distance":pdn,"rnn_alpha":alphas[0],"multi_alpha":alphas[1]};rows.append(row);function[str(step)]={"behavior_action":bs,"policy_action":ps}
    def write_csv(path,items):
        with Path(path).open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    write_csv(output/"critic_washout_over_time.csv",rows);write_csv(output/"actor_gradient_similarity.csv",gradient_rows);atomic_json(output/"critic_function_similarity.json",function)
    base=rows[0];at30=next((r for r in rows if r["env_step"]==30000),None)
    if at30 is None:case="UNDETERMINED";conclusion="step30k checkpoint unavailable; washout classification cannot be made."
    else:
        pearson_gain=at30["behavior_q_pearson"]-base["behavior_q_pearson"];spear_gain=at30["behavior_q_spearman"]-base["behavior_q_spearman"];nrmse_drop=(base["behavior_q_nrmse"]-at30["behavior_q_nrmse"])/max(base["behavior_q_nrmse"],1e-12);grad_gain=at30["q_grad_cos_mean"]-base["q_grad_cos_mean"]
        if (pearson_gain>=.05 or spear_gain>=.05) and nrmse_drop>=.10 and grad_gain>=.10:case="A";conclusion="Evidence supports critic-initialization washout during the critic-only phase."
        elif base["q_grad_cos_mean"]>=.90 and abs(grad_gain)<.10:case="B";conclusion="Stage2-R critic differences do not substantially alter the Actor gradient direction even before online adaptation."
        else:case="C";conclusion="Critic initialization differences remain functionally significant at Actor unfreezing."
    evaluations=evaluation_table(run);report={"run_path":str(run),"available_checkpoint_steps":steps,"checkpoint_paths":{str(s):({g:str(initial[g]) for g in GROUPS} if s==0 else {g:str(maps[g][s]) for g in GROUPS}) for s in steps},"fixed_diagnostic_dataset":source,"sequence_count":batch["obs"].shape[1],"sequence_length":10,"rows":rows,"deterministic_evaluation":evaluations,"mechanism_case":case,"conclusion":conclusion,"thresholds":{"case_a":{"correlation_gain":.05,"nrmse_relative_drop":.10,"gradient_cosine_gain":.10},"case_b":{"initial_gradient_cosine":.90,"max_absolute_change":.10}},"training_performed":False,"optimizer_step_called":False,"new_stage4_started":False};atomic_json(output/"stage4_critic_washout_audit.json",report)
    lines=["# Stage4 Critic Initialization Washout Audit","","| Step | Q NRMSE | Q Pearson | Q Spearman | Policy-Q Pearson | Q-gradient cosine | Relative grad diff |","|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:lines.append(f"| {r['env_step']} | {r['behavior_q_nrmse']:.6f} | {r['behavior_q_pearson']:.6f} | {r['behavior_q_spearman']:.6f} | {r['policy_q_pearson']:.6f} | {r['q_grad_cos_mean']:.6f} ± {r['q_grad_cos_std']:.6f} | {r['q_grad_relative_diff']:.6f} |")
    lines += ["",f"Mechanism classification: **Case {case}**", "",conclusion,"","NO TRAINING PERFORMED  ","NO OPTIMIZER STEP  ","NO NEW STAGE4 STARTED"];(output/"stage4_critic_washout_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Stage4 run: {run}");print(f"Available checkpoint steps: {steps}");print(f"Fixed diagnostic source: {source['kind']}");print(f"Mechanism: Case {case} - {conclusion}");print("NO TRAINING PERFORMED\nNO OPTIMIZER STEP\nNO NEW STAGE4 STARTED")

if __name__=="__main__":main()
