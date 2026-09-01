#!/usr/bin/env python3
"""Stage2-R recurrent SAC-native critic pretraining; Actor remains frozen."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent;S3R=PROJECT/"stage3_r_bc_rnn_to_rsac";VENDOR=PROJECT/"third_party"/"pomdp_baselines"
for path in (HERE,S3R,VENDOR):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from buffers.seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer  # noqa: E402
from stage3_r_actor import load_actor  # noqa: E402
from stage2_r_critic import architecture,make_pair,predictions,set_device,tensor_batch,update  # noqa: E402
from stage2_r_data import POLICIES,BalancedSampler,NativeSequenceSource,attach_actor_actions,load_episodes,validation_sequences  # noqa: E402
from stage2_r_evaluation import evaluate  # noqa: E402


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


def actor_sanity(actor, episodes, device, tolerance=1e-5):
    rows=[]
    for seed in (10000,10001):
        episode=next(item for item in episodes if item.source=="bc_rnn" and item.seed==seed)
        all_states=np.concatenate((episode.state[:1],episode.next_state),axis=0)
        states=torch.as_tensor(all_states,dtype=torch.float32,device=device)
        with torch.no_grad():sequence=actor.deterministic_sequence(states.unsqueeze(0))[0]
        actor.reset();step=[]
        with torch.no_grad():
            for state in states:step.append(actor.act(state.unsqueeze(0),deterministic=True)[0][0])
        stepped=torch.stack(step);error=float((sequence-stepped).abs().max().item())
        rows.append({"seed":seed,"steps":len(states),"max_sequence_vs_evaluator_error":error,
                     "finite":bool(torch.isfinite(sequence).all()),"deterministic":True})
        if error>tolerance or not rows[-1]["finite"]:raise RuntimeError(f"Stage3-R replay sanity failed: {rows[-1]}")
    return {"status":"PASS","actor_checkpoint_semantics":"Stage3-R evaluator act() with 10-step hidden reset","episodes":rows}


def critic_sanity(config,source,device):
    critic,target=make_pair(config,device);optimizer=torch.optim.Adam(critic.parameters(),lr=config["critic_lr"])
    batch_np,_=BalancedSampler({source.source:source},config["random_seed"]+99).sample(2)
    batch=tensor_batch(batch_np,device);T,B,_=batch["obs"].shape
    before=[value.detach().clone() for value in target.parameters()]
    stats=update(critic,target,optimizer,batch,config,1)
    changed=any(not torch.equal(a,b) for a,b in zip(before,target.parameters()))
    result={"status":"PASS","state_sequence_shape":[T,B,59],"action_sequence_shape":list(batch["act"].shape),
            "reward_sequence_shape":list(batch["rew"].shape),"next_state_sequence_shape":[T,B,59],
            "mask_sequence_shape":list(batch["mask"].shape),"target_updated":changed,**stats}
    if not changed:raise RuntimeError("Target-network sanity update made no change")
    return result


def checkpoint(critic,target,optimizer,index,config,validation,group):
    return {"format_version":"multi_il_full_action_rl.stage2_r.critic.v1","stage":"Stage2-R","group":group,
            "update":int(index),"critic_state_dict":critic.state_dict(),"target_critic_state_dict":target.state_dict(),
            "optimizer_state_dict":None if optimizer is None else optimizer.state_dict(),"architecture":architecture(config),
            "gamma":config["gamma"],"tau":config["tau"],"target_update_interval":config["target_update_interval"],
            "sequence_length":config["sequence_length"],"frozen_actor_checkpoint":config["actor_checkpoint"],
            "pomdp_baselines_commit":config["pomdp_baselines_commit"],"validation":validation,"stage4_compatible":True}


def compact(metrics):
    keys=("best_update","val_td_mse","pairwise_ranking_accuracy","success_q","failure_q","q_gap","q_mean","q_std","q_min","q_max","hidden_norm_mean","hidden_norm_std","sequence_start_q","sequence_end_q","nan_inf_count","training_sequences","training_effective_timesteps")
    return {key:metrics.get(key) for key in keys}


def train_group(name,sampler,records,base_state,config,device,directory,max_updates):
    directory.mkdir(parents=True,exist_ok=False);(directory/"checkpoints").mkdir()
    sampling_start={key:(value.sampled_sequences,value.effective_timesteps) for key,value in sampler.sources.items()}
    critic,target=make_pair(config,device);critic.load_state_dict(base_state);target.load_state_dict(base_state);target.requires_grad_(False)
    optimizer=torch.optim.Adam(critic.parameters(),lr=config["critic_lr"])
    train_rows=[];validation_rows=[];acc=defaultdict(float);best=float("inf");best_update=0;interval=0
    for index in range(1,max_updates+1):
        batch_np,counts=sampler.sample(config["sequence_batch_size"]);batch=tensor_batch(batch_np,device)
        stats=update(critic,target,optimizer,batch,config,index);interval+=1
        for key,value in stats.items():acc[key]+=value
        for source,count in counts.items():acc[f"{source}_sequences"]+=count
        if index%int(config["validation_every"])==0 or index==max_updates:
            train_row={"update":index,**{key:value/interval for key,value in acc.items() if not key.endswith("_sequences")}}
            train_row.update({key:int(value) for key,value in acc.items() if key.endswith("_sequences")});train_rows.append(train_row);acc=defaultdict(float);interval=0
            validation=evaluate(critic,target,records,config,device);validation_row={"update":index,**{k:v for k,v in validation.items() if k not in ("trajectory_scores","pairwise_details","per_timestep_td_mse")}};validation_rows.append(validation_row)
            if validation["val_td_mse"]<best:
                best=validation["val_td_mse"];best_update=index
                torch.save(checkpoint(critic,target,optimizer,index,config,validation,name),directory/"checkpoints"/"best.pth")
            if index%int(config["checkpoint_every"])==0:torch.save(checkpoint(critic,target,optimizer,index,config,validation,name),directory/"checkpoints"/f"update_{index}.pth")
            write_csv(directory/"training_metrics.csv",train_rows);write_csv(directory/"validation_metrics.csv",validation_rows)
            print(f"{name} update {index}/{max_updates} loss={train_row['critic_loss']:.6f} val_td_mse={validation['val_td_mse']:.6f} ranking={validation['pairwise_ranking_accuracy']}",flush=True)
    final=evaluate(critic,target,records,config,device);torch.save(checkpoint(critic,target,optimizer,max_updates,config,final,name),directory/"checkpoints"/"last.pth")
    best_payload=torch.load(directory/"checkpoints"/"best.pth",map_location=device);critic.load_state_dict(best_payload["critic_state_dict"]);target.load_state_dict(best_payload["target_critic_state_dict"])
    result=evaluate(critic,target,records,config,device);result["best_update"]=best_update;result["training_sampling"]={key:{"episodes":len(value.episodes),"sampled_sequences":value.sampled_sequences-sampling_start[key][0],"effective_timesteps":value.effective_timesteps-sampling_start[key][1]} for key,value in sampler.sources.items()}
    result["training_sequences"]=sum(item["sampled_sequences"] for item in result["training_sampling"].values());result["training_effective_timesteps"]=sum(item["effective_timesteps"] for item in result["training_sampling"].values())
    write(directory/"summary.json",result);return result


def markdown(summary):
    groups=("random_critic","rnn_only_critic","multi_il_critic");labels=("Random","RNN-only","Multi-IL")
    lines=["# Stage2-R summary","",f"Frozen Actor: `{summary['frozen_actor_checkpoint']}`","", "| metric | Random | RNN-only | Multi-IL |","|---|---:|---:|---:|"]
    for key in ("val_td_mse","pairwise_ranking_accuracy","success_q","failure_q","q_gap","q_mean","q_std","best_update","training_sequences","training_effective_timesteps"):
        lines.append("| "+key+" | "+" | ".join(str(summary[group].get(key)) for group in groups)+" |")
    lines += ["",f"Multi > RNN-only > Random: **{summary['multi_gt_rnn_gt_random']}**","","Stage3-R Actor was not modified. Stage4 was not started."]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",default=str(HERE/"stage2_r_config.json"));parser.add_argument("--device",default="npu:0");parser.add_argument("--smoke-test",action="store_true");parser.add_argument("--run-id");args=parser.parse_args()
    config=read(args.config);device=device_of(args.device);set_device(device);seed_all(config["random_seed"])
    run=Path(config["output_root"])/(args.run_id or (("smoke_" if args.smoke_test else "")+datetime.now().strftime("%Y%m%d_%H%M%S")));run.mkdir(parents=True,exist_ok=False);(run/"sanity").mkdir()
    if not Path(config["actor_checkpoint"]).is_file():raise FileNotFoundError(config["actor_checkpoint"])
    actor,payload=load_actor(config["actor_checkpoint"],device);actor.requires_grad_(False)
    if int(payload["epoch"])!=150:raise RuntimeError(f"Stage2-R requires Stage3-R epoch 150, got {payload['epoch']}")
    train_seeds=list(range(config["train_seed_start"],config["train_seed_end"]+1));val_seeds=list(range(config["validation_seed_start"],config["validation_seed_end"]+1))
    if args.smoke_test:train_seeds=train_seeds[:5];val_seeds=val_seeds[:2]
    train={};validation={}
    for source in POLICIES:
        train[source]=load_episodes(source,config["datasets"][source],train_seeds);validation[source]=load_episodes(source,config["datasets"][source],val_seeds)
        attach_actor_actions(train[source],actor,device);attach_actor_actions(validation[source],actor,device)
    all_episodes=[episode for source in POLICIES for episode in train[source]+validation[source]]
    write(run/"sanity"/"stage3_r_actor_replay.json",actor_sanity(actor,all_episodes,device))
    sources={source:NativeSequenceSource(train[source],config["sequence_length"],config["sample_weight_baseline"],RAMEfficient_SeqReplayBuffer) for source in POLICIES}
    write(run/"sanity"/"critic_sequence.json",critic_sanity(config,sources["bc_rnn"],device))
    records=validation_sequences([episode for source in POLICIES for episode in validation[source]],config["sequence_length"])
    max_updates=20 if args.smoke_test else int(config["max_updates"]);effective=copy.deepcopy(config);effective["max_updates"]=max_updates
    if args.smoke_test:effective.update(sequence_batch_size=3,validation_every=10,checkpoint_every=10,validation_sequence_batch_size=32)
    write(run/"resolved_config.json",{**effective,"device":str(device),"train_seeds":train_seeds,"validation_seeds":val_seeds,"architecture":architecture(effective),"actor_checkpoint_epoch":payload["epoch"]})
    base,target=make_pair(effective,device);base_state=copy.deepcopy(base.state_dict());random_dir=run/"random_critic";random_dir.mkdir();(random_dir/"checkpoints").mkdir()
    random_eval=evaluate(base,target,records,effective,device);random_eval.update(best_update=0,training_sequences=0,training_effective_timesteps=0)
    torch.save(checkpoint(base,target,None,0,effective,random_eval,"random_critic"),random_dir/"model_init.pth");write(random_dir/"summary.json",random_eval)
    del base,target
    rnn=train_group("rnn_only_critic",BalancedSampler({"bc_rnn":sources["bc_rnn"]},effective["random_seed"]+1),records,base_state,effective,device,run/"rnn_only_critic",max_updates)
    multi=train_group("multi_il_critic",BalancedSampler(sources,effective["random_seed"]+2),records,base_state,effective,device,run/"multi_il_critic",max_updates)
    ranking=lambda value: -1.0 if value is None else float(value)
    relation=(multi["val_td_mse"]<rnn["val_td_mse"]<random_eval["val_td_mse"] and ranking(multi["pairwise_ranking_accuracy"])>ranking(rnn["pairwise_ranking_accuracy"])>ranking(random_eval["pairwise_ranking_accuracy"]))
    summary={"stage":"Stage2-R","status":"SMOKE_TEST_PASSED" if args.smoke_test else "STAGE2_R_COMPLETE","run_directory":str(run),"pomdp_baselines_commit":effective["pomdp_baselines_commit"],"frozen_actor_checkpoint":effective["actor_checkpoint"],"actor_checkpoint_epoch":payload["epoch"],"actor_frozen":all(not parameter.requires_grad for parameter in actor.parameters()),"architecture":architecture(effective),"gamma":effective["gamma"],"tau":effective["tau"],"target_update_interval":effective["target_update_interval"],"train_seeds":train_seeds,"validation_seeds":val_seeds,"random_critic":compact(random_eval),"rnn_only_critic":compact(rnn),"multi_il_critic":compact(multi),"training_sampling":{"rnn_only":rnn["training_sampling"],"multi_il":multi["training_sampling"]},"multi_gt_rnn_gt_random":relation,"stage3_r_actor_modified":False,"stage4_started":False,"stage4_initialization_checkpoints":{"random":str(random_dir/"model_init.pth"),"rnn_only":str(run/"rnn_only_critic"/"checkpoints"/"best.pth"),"multi_il":str(run/"multi_il_critic"/"checkpoints"/"best.pth")}}
    write(run/"stage2_r_summary.json",summary);(run/"stage2_r_summary.md").write_text(markdown(summary),encoding="utf-8")
    print("STAGE2-R",summary["status"],"Run directory:",run,"Stage4 NOT started.",flush=True)
if __name__=="__main__":main()
