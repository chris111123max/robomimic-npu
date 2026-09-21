#!/usr/bin/env python3
"""Train isolated Stage2.2 history-aware rnn_q and multi_q variants."""
from __future__ import annotations
import argparse,copy,json,random,time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from evaluation import evaluate
from history_critic import build_critic,checkpoint_payload
from sequence_dataset import load_splits
from sequence_sampler import SequenceSampler
HERE=Path(__file__).resolve().parent

def args():
    p=argparse.ArgumentParser();p.add_argument("--config",default=str(HERE/"stage2_2_config.json"));p.add_argument("--dataset-root");p.add_argument("--output-root");p.add_argument("--device",default="cpu");p.add_argument("--mode",choices=("rnn_q","multi_q","both"),default="both");p.add_argument("--run-id");p.add_argument("--max-updates",type=int);return p.parse_args()
def write(path,value): path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def append(path,value):
    with path.open("a") as f:f.write(json.dumps(value,sort_keys=True)+"\n")
def synchronize(device):
    if device.type=="npu" and hasattr(torch,"npu"): torch.npu.synchronize(device)
def main():
    a=args();config=json.loads(Path(a.config).read_text());
    for k,v in (("dataset_root",a.dataset_root),("output_root",a.output_root),("max_updates",a.max_updates)):
        if v is not None:config[k]=v
    if config["sequence_batch_size"]*config["learning_sequence_length"]!=256:raise RuntimeError("effective supervised batch must equal Stage2.1 batch 256")
    seed=config["training_seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);device=torch.device(a.device)
    train,val=load_splits(config["dataset_root"],range(config["train_seed_start"],config["train_seed_end"]+1),range(config["val_seed_start"],config["val_seed_end"]+1),config["gamma"])
    run=Path(config["output_root"])/(a.run_id or f"stage2_2_history_critic_{datetime.now():%Y%m%d_%H%M%S}")
    if run.exists():raise FileExistsError(run)
    (run/"shared").mkdir(parents=True);write(run/"shared"/"config_resolved.json",config)
    write(run/"shared"/"dataset_manifest.json",{"train_seeds":[config["train_seed_start"],config["train_seed_end"]],"validation_seeds":[config["val_seed_start"],config["val_seed_end"]],"sources":{p:train[p].path for p in train}})
    base=build_critic(config,device);base_state=copy.deepcopy(base.state_dict());torch.save(base_state,run/"shared"/"initial_state.pth")
    for label in (("rnn_q","multi_q") if a.mode=="both" else (a.mode,)):
        out=run/label;(out/"checkpoints").mkdir(parents=True);model=build_critic(config,device);model.load_state_dict(copy.deepcopy(base_state));opt=torch.optim.AdamW(model.parameters(),lr=config["critic_lr"],weight_decay=config["weight_decay"])
        sampler=SequenceSampler(train,config["burn_in_length"],config["learning_sequence_length"],config["horizon"],seed,label=="multi_q");best=float("inf")
        for step in range(1,config["max_updates"]+1):
            started=time.perf_counter();b=sampler.sample(config["sequence_batch_size"]);sample_ms=(time.perf_counter()-started)*1000
            tensors={k:torch.as_tensor(b[k],device=device) for k in ("observations","previous_actions","progress","actions","returns")};synchronize(device);started=time.perf_counter()
            q1,q2=model.forward_sequence(tensors["observations"],tensors["previous_actions"],tensors["progress"],tensors["actions"],config["burn_in_length"]);target=tensors["returns"][:,config["burn_in_length"]:]
            synchronize(device);forward_ms=(time.perf_counter()-started)*1000;loss1=torch.nn.functional.mse_loss(q1,target);loss2=torch.nn.functional.mse_loss(q2,target);loss=loss1+loss2;started=time.perf_counter();opt.zero_grad(set_to_none=True);loss.backward();synchronize(device);backward_ms=(time.perf_counter()-started)*1000;started=time.perf_counter();opt.step();synchronize(device);optimizer_ms=(time.perf_counter()-started)*1000
            total_ms=sample_ms+forward_ms+backward_ms+optimizer_ms
            row={"step":step,"q1_loss":float(loss1),"q2_loss":float(loss2),"sample_ms":sample_ms,"forward_ms":forward_ms,"backward_ms":backward_ms,"optimizer_ms":optimizer_ms,"effective_timesteps":256,"effective_timesteps_per_second":256000.0/total_ms,"sampling":sampler.proportions()}
            if step%config["eval_interval"]==0 or step==config["max_updates"]:
                ev=evaluate(model,val,device,config["horizon"]);metric=ev["bc_rnn"]["twin_mean_mse"] if label=="rnn_q" else ev["balanced_aggregate"]["twin_mean_mse"];row["validation"]=ev
                if metric<best:best=metric;torch.save(checkpoint_payload(model,opt,config,step,metric),out/"checkpoints"/"best.pth")
            append(out/"train_metrics.jsonl",row)
        torch.save(checkpoint_payload(model,opt,config,config["max_updates"],best),out/"checkpoints"/"last.pth")
        best_payload=torch.load(out/"checkpoints"/"best.pth",map_location=device);model.load_state_dict(best_payload["critic_state_dict"],strict=True)
        write(out/"final_validation.json",evaluate(model,val,device,config["horizon"]));write(out/"sampling_audit.json",{"counts":sampler.counts,"ratios":sampler.proportions()})
        peak=(torch.npu.max_memory_allocated(device) if device.type=="npu" and hasattr(torch,"npu") else 0)
        write(out/"performance.json",{"q1_parameters":sum(p.numel() for p in model.q1.parameters()),"q2_parameters":sum(p.numel() for p in model.q2.parameters()),"total_parameters":sum(p.numel() for p in model.parameters()),"peak_device_memory_bytes":int(peak),"burn_in_length":config["burn_in_length"],"learning_sequence_length":config["learning_sequence_length"],"effective_training_timesteps_per_batch":256})
    print(json.dumps({"status":"COMPLETE","run_dir":str(run)},indent=2))
if __name__=="__main__":main()
