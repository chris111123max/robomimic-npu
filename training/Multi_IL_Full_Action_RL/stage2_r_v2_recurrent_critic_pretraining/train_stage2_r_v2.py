#!/usr/bin/env python3
"""Stage2-R-v2: lower-LR, globally clipped recurrent critic pretraining."""
from __future__ import annotations

import argparse,copy,csv,json,random,sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent;V1=PROJECT/"stage2_r_recurrent_critic_pretraining";S3R=PROJECT/"stage3_r_bc_rnn_to_rsac";VENDOR=PROJECT/"third_party"/"pomdp_baselines"
for path in (HERE,V1,S3R,VENDOR):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from buffers.seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer  # noqa: E402
from stage3_r_actor import load_actor  # noqa: E402
from stage2_r_critic import architecture,make_pair,set_device,tensor_batch  # noqa: E402
from stage2_r_data import POLICIES,BalancedSampler,NativeSequenceSource,attach_actor_actions,load_episodes,validation_sequences  # noqa: E402
from stage2_r_evaluation import evaluate  # noqa: E402
from train_stage2_r import actor_sanity  # noqa: E402
from stage2_r_v2_critic import NonFiniteGradientError,update_v2  # noqa: E402
import torchkit.pytorch_utils as ptu  # noqa: E402


def read(path):
    with open(path,encoding="utf-8") as handle:return json.load(handle)
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as handle:json.dump(value,handle,indent=2,ensure_ascii=False)
def write_csv(path,rows):
    if not rows:return
    with open(path,"w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def device_of(name):
    if name.startswith("npu"):
        import torch_npu  # noqa
        if not torch.npu.is_available():raise RuntimeError("Ascend NPU unavailable")
        torch.npu.set_device(name)
    return torch.device(name)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if hasattr(torch,"npu") and torch.npu.is_available():torch.npu.manual_seed_all(seed)


def checkpoint(critic,target,optimizer,index,config,validation,group):
    return {"format_version":"multi_il_full_action_rl.stage2_r_v2.critic.v1","stage":"Stage2-R-v2","group":group,
            "update":int(index),"critic_state_dict":critic.state_dict(),"target_critic_state_dict":target.state_dict(),
            "optimizer_state_dict":None if optimizer is None else optimizer.state_dict(),"architecture":architecture(config),
            "critic_lr":config["critic_lr"],"max_gradient_norm":config["max_gradient_norm"],"gradient_norm_type":config["gradient_norm_type"],
            "gamma":config["gamma"],"tau":config["tau"],"target_update_interval":config["target_update_interval"],
            "sequence_length":config["sequence_length"],"frozen_actor_checkpoint":config["actor_checkpoint"],
            "pomdp_baselines_commit":config["pomdp_baselines_commit"],"validation":validation,
            "stage4_optimization_contract":{"critic_lr":config["critic_lr"],"max_gradient_norm":config["max_gradient_norm"]}}


def gradient_interval(values,threshold):
    array=np.asarray(values,dtype=np.float64)
    return {"grad_norm_mean":float(array.mean()),"grad_norm_median":float(np.median(array)),
            "grad_norm_p95":float(np.quantile(array,0.95)),"grad_norm_p99":float(np.quantile(array,0.99)),
            "grad_norm_max":float(array.max()),"fraction_grad_clipped":float(np.mean(array>float(threshold)))}


def critic_sanity_v2(config,source,device,directory):
    critic,target=make_pair(config,device);optimizer=torch.optim.Adam(critic.parameters(),lr=float(config["critic_lr"]))
    batch_np,_=BalancedSampler({source.source:source},config["random_seed"]+99).sample(2);batch=tensor_batch(batch_np,device)
    before=[parameter.detach().clone() for parameter in target.parameters()]
    stats=update_v2(critic,target,optimizer,batch,config,1,"sanity",directory,ptu.soft_update_from_to)
    target_changed=any(not torch.equal(old,new) for old,new in zip(before,target.parameters()))
    result={"status":"PASS","state_sequence_shape":[batch["obs"].shape[0],batch["obs"].shape[1],59],
            "action_sequence_shape":list(batch["act"].shape),"reward_sequence_shape":list(batch["rew"].shape),
            "next_state_sequence_shape":[batch["obs2"].shape[0],batch["obs2"].shape[1],59],
            "mask_sequence_shape":list(batch["mask"].shape),"target_updated":target_changed,
            "critic_lr":config["critic_lr"],"max_gradient_norm":config["max_gradient_norm"],**stats}
    if not target_changed:raise RuntimeError("Stage2-R-v2 target-network sanity update made no change")
    return result


def train_group(name,sampler,records,base_state,config,device,directory,max_updates):
    directory.mkdir(parents=True,exist_ok=False);(directory/"checkpoints").mkdir();(directory/"debug_nan").mkdir()
    critic,target=make_pair(config,device);critic.load_state_dict(base_state);target.load_state_dict(base_state);target.requires_grad_(False)
    optimizer=torch.optim.Adam(critic.parameters(),lr=float(config["critic_lr"]));train_rows=[];validation_rows=[]
    acc=defaultdict(float);norms=[];interval=0;best=float("inf");best_update=0
    for index in range(1,max_updates+1):
        batch_np,counts=sampler.sample(config["sequence_batch_size"]);batch=tensor_batch(batch_np,device)
        try:stats=update_v2(critic,target,optimizer,batch,config,index,name,directory/"debug_nan",ptu.soft_update_from_to)
        except NonFiniteGradientError:
            write_csv(directory/"training_metrics.csv",train_rows);write_csv(directory/"validation_metrics.csv",validation_rows)
            write(directory/"summary.json",{"stage":"Stage2-R-v2","group":name,"status":"FAILED_NONFINITE_GRADIENT","failed_update":index,
                                              "last_completed_update":index-1,"stage4_started":False})
            raise
        interval+=1;norms.append(stats["preclip_grad_norm"])
        for key,value in stats.items():
            if key not in ("preclip_grad_norm","gradient_clipped"):acc[key]+=value
        for source,count in counts.items():acc[f"{source}_sequences"]+=count
        if index%int(config["validation_every"])==0 or index==max_updates:
            row={"update":index,**{key:value/interval for key,value in acc.items() if not key.endswith("_sequences")},
                 **{key:int(value) for key,value in acc.items() if key.endswith("_sequences")},
                 **gradient_interval(norms,config["max_gradient_norm"])}
            train_rows.append(row);acc=defaultdict(float);norms=[];interval=0
            validation=evaluate(critic,target,records,config,device);validation_row={"update":index,**{key:value for key,value in validation.items() if key not in ("trajectory_scores","pairwise_details","per_timestep_td_mse")},
                                                                                     **{key:row[key] for key in ("grad_norm_mean","grad_norm_median","grad_norm_p95","grad_norm_p99","grad_norm_max","fraction_grad_clipped")}}
            validation_rows.append(validation_row)
            if validation["val_td_mse"]<best:
                best=validation["val_td_mse"];best_update=index;torch.save(checkpoint(critic,target,optimizer,index,config,validation,name),directory/"checkpoints"/"best.pth")
            if index%int(config["checkpoint_every"])==0:torch.save(checkpoint(critic,target,optimizer,index,config,validation,name),directory/"checkpoints"/f"update_{index}.pth")
            write_csv(directory/"training_metrics.csv",train_rows);write_csv(directory/"validation_metrics.csv",validation_rows)
            print(f"{name} update {index}/{max_updates} loss={row['critic_loss']:.6f} val_td_mse={validation['val_td_mse']:.6f} ranking={validation['pairwise_ranking_accuracy']} grad_p95={row['grad_norm_p95']:.6f} clipped={row['fraction_grad_clipped']:.4f}",flush=True)
    final=evaluate(critic,target,records,config,device);torch.save(checkpoint(critic,target,optimizer,max_updates,config,final,name),directory/"checkpoints"/"last.pth")
    best_payload=torch.load(directory/"checkpoints"/"best.pth",map_location=device);critic.load_state_dict(best_payload["critic_state_dict"]);target.load_state_dict(best_payload["target_critic_state_dict"])
    result=evaluate(critic,target,records,config,device);result["best_update"]=best_update
    sequence_counts={source:sum(int(row.get(f"{source}_sequences",0)) for row in train_rows) for source in sampler.sources}
    result["training_sampling"]={source:{"episodes":len(value.episodes),"sampled_sequences":sequence_counts[source],"effective_timesteps":sequence_counts[source]*int(config["sequence_length"])} for source,value in sampler.sources.items()}
    result["training_sequences"]=sum(sequence_counts.values());result["training_effective_timesteps"]=result["training_sequences"]*int(config["sequence_length"])
    result["gradient_statistics_over_intervals"]=[{key:row[key] for key in ("update","grad_norm_mean","grad_norm_median","grad_norm_p95","grad_norm_p99","grad_norm_max","fraction_grad_clipped")} for row in train_rows]
    result.update(status="COMPLETE_50000",updates_completed=max_updates);write(directory/"summary.json",result);return result


def compact(value):
    keys=("best_update","val_td_mse","pairwise_ranking_accuracy","success_q","failure_q","q_gap","q_mean","q_std","q_min","q_max","training_sequences","training_effective_timesteps","status","updates_completed")
    return {key:value.get(key) for key in keys}


def markdown(summary):
    groups=("random_critic","rnn_only_critic","multi_il_critic");lines=["# Stage2-R-v2 summary","",f"Frozen Actor: `{summary['frozen_actor_checkpoint']}`",f"Critic LR: `{summary['critic_lr']}`",f"Max gradient norm: `{summary['max_gradient_norm']}`","","| metric | Random | RNN-only | Multi-IL |","|---|---:|---:|---:|"]
    for key in ("val_td_mse","pairwise_ranking_accuracy","success_q","failure_q","q_gap","q_mean","q_std","best_update","training_sequences","training_effective_timesteps"):
        lines.append("| "+key+" | "+" | ".join(str(summary[group].get(key)) for group in groups)+" |")
    lines += ["","Stage3-R Actor was not modified. Stage4 was not started."]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default=str(HERE/"stage2_r_v2_config.json"));parser.add_argument("--device",default="npu:0");parser.add_argument("--smoke-test",action="store_true");parser.add_argument("--run-id");args=parser.parse_args()
    config=read(args.config)
    if float(config["stage2_r_v1_critic_lr"])!=3e-4 or float(config["critic_lr"])!=1e-4:raise RuntimeError("Stage2-R-v2 LR contract violated")
    if float(config["max_gradient_norm"])!=1.0:raise RuntimeError("Stage2-R-v2 clipping contract violated")
    device=device_of(args.device);set_device(device);seed_all(config["random_seed"])
    run=Path(config["output_root"])/(args.run_id or (("smoke_" if args.smoke_test else "")+datetime.now().strftime("%Y%m%d_%H%M%S")));run.mkdir(parents=True,exist_ok=False);(run/"sanity").mkdir()
    actor,payload=load_actor(config["actor_checkpoint"],device);actor.requires_grad_(False)
    if int(payload["epoch"])!=150:raise RuntimeError(f"Expected frozen Stage3-R epoch150, got {payload['epoch']}")
    train_seeds=list(range(config["train_seed_start"],config["train_seed_end"]+1));val_seeds=list(range(config["validation_seed_start"],config["validation_seed_end"]+1))
    if args.smoke_test:train_seeds=train_seeds[:5];val_seeds=val_seeds[:2]
    train={};validation={}
    for source in POLICIES:
        train[source]=load_episodes(source,config["datasets"][source],train_seeds);validation[source]=load_episodes(source,config["datasets"][source],val_seeds)
        attach_actor_actions(train[source],actor,device);attach_actor_actions(validation[source],actor,device)
    all_episodes=[episode for source in POLICIES for episode in train[source]+validation[source]]
    write(run/"sanity"/"stage3_r_actor_replay.json",actor_sanity(actor,all_episodes,device))
    sources={source:NativeSequenceSource(train[source],config["sequence_length"],config["sample_weight_baseline"],RAMEfficient_SeqReplayBuffer) for source in POLICIES}
    write(run/"sanity"/"critic_sequence.json",critic_sanity_v2(config,sources["bc_rnn"],device,run/"sanity"/"debug_nan"))
    records=validation_sequences([episode for source in POLICIES for episode in validation[source]],config["sequence_length"])
    effective=copy.deepcopy(config);max_updates=20 if args.smoke_test else int(config["max_updates"])
    if args.smoke_test:effective.update(sequence_batch_size=3,validation_every=10,checkpoint_every=10,validation_sequence_batch_size=32)
    write(run/"resolved_config.json",{**effective,"device":str(device),"train_seeds":train_seeds,"validation_seeds":val_seeds,"architecture":architecture(effective),"stage4_must_inherit":{"critic_lr":effective["critic_lr"],"max_gradient_norm":effective["max_gradient_norm"]}})
    base,target=make_pair(effective,device);base_state=copy.deepcopy(base.state_dict());random_dir=run/"random_critic";random_dir.mkdir()
    random_eval=evaluate(base,target,records,effective,device);random_eval.update(best_update=0,training_sequences=0,training_effective_timesteps=0,status="UNTRAINED_BASELINE",updates_completed=0)
    torch.save(checkpoint(base,target,None,0,effective,random_eval,"random_critic"),random_dir/"model_init.pth");write(random_dir/"summary.json",random_eval);del base,target
    rnn=train_group("rnn_only_critic",BalancedSampler({"bc_rnn":sources["bc_rnn"]},effective["random_seed"]+1),records,base_state,effective,device,run/"rnn_only_critic",max_updates)
    multi=train_group("multi_il_critic",BalancedSampler(sources,effective["random_seed"]+2),records,base_state,effective,device,run/"multi_il_critic",max_updates)
    summary={"stage":"Stage2-R-v2","status":"SMOKE_TEST_PASSED" if args.smoke_test else "STAGE2_R_V2_COMPLETE","run_directory":str(run),"frozen_actor_checkpoint":effective["actor_checkpoint"],"actor_frozen":all(not parameter.requires_grad for parameter in actor.parameters()),"architecture":architecture(effective),"critic_lr":effective["critic_lr"],"max_gradient_norm":effective["max_gradient_norm"],"gamma":effective["gamma"],"tau":effective["tau"],"train_seeds":train_seeds,"validation_seeds":val_seeds,"random_critic":compact(random_eval),"rnn_only_critic":compact(rnn),"multi_il_critic":compact(multi),"stage3_r_actor_modified":False,"stage4_started":False,"stage4_optimization_contract":{"critic_lr":effective["critic_lr"],"max_gradient_norm":effective["max_gradient_norm"]},"stage4_initialization_checkpoints":{"random":str(random_dir/"model_init.pth"),"rnn_only":str(run/"rnn_only_critic"/"checkpoints"/"best.pth"),"multi_il":str(run/"multi_il_critic"/"checkpoints"/"best.pth")}}
    write(run/"stage2_r_v2_summary.json",summary);(run/"stage2_r_v2_summary.md").write_text(markdown(summary),encoding="utf-8");print(summary["status"],"Run directory:",run,"Stage4 NOT started.",flush=True)
if __name__=="__main__":main()
