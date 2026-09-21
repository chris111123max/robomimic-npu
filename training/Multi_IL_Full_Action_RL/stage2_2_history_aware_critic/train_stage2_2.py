#!/usr/bin/env python3
"""Train Stage2.2 with full-prefix semantics and mandatory finite guards."""
from __future__ import annotations
import argparse,copy,json,random,time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from evaluation import evaluate
from finite_diagnostics import (dump_failure,gradient_statistics,numpy_batch_stats,
    optimizer_non_finite,parameter_statistics,tensor_stats)
from history_critic import build_critic,checkpoint_payload
from sequence_dataset import load_splits
from sequence_sampler import SequenceSampler
HERE=Path(__file__).resolve().parent

def arguments():
    p=argparse.ArgumentParser();p.add_argument("--config",default=str(HERE/"stage2_2_config.json"));p.add_argument("--dataset-root");p.add_argument("--output-root");p.add_argument("--device",default="cpu")
    p.add_argument("--mode",choices=("rnn_q","multi_q","both","matched_rnn_q","matched_multi_q","matched_both"),default="both");p.add_argument("--run-id");p.add_argument("--max-updates",type=int);return p.parse_args()
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def append(path,value):
    with path.open("a") as f:f.write(json.dumps(value,sort_keys=True)+"\n")
def select_device(name):
    name=str(name)
    if name.startswith("npu"):
        try:import torch_npu  # noqa: F401
        except ImportError as exc:raise RuntimeError("NPU requested but torch_npu cannot be imported") from exc
        if not hasattr(torch,"npu") or not torch.npu.is_available():raise RuntimeError("NPU requested but unavailable")
        torch.npu.set_device(name)
    return torch.device(name)
def synchronize(device):
    if device.type=="npu":torch.npu.synchronize(device)
def finite_or_fail(stage,named,batch,out,step,model,opt,extra=None):
    bad=[name for name,value in named if value is not None and not torch.isfinite(value).all()]
    if bad:dump_failure(out,step,stage,batch,{"non_finite_names":bad,"tensor_statistics":{n:tensor_stats(v) for n,v in named if v is not None},**(extra or {})},model,opt)
def finite_metrics(value,path="validation"):
    if isinstance(value,dict):
        for key,item in value.items():finite_metrics(item,f"{path}.{key}")
    elif isinstance(value,(float,int)) and not np.isfinite(value):raise FloatingPointError(f"non-finite {path}: {value}")
def labels(mode):
    if mode=="both":return ("rnn_q","multi_q")
    if mode=="matched_both":return ("matched_rnn_q","matched_multi_q")
    return (mode,)
def critic_type(label):return "matched_memoryless_twin_q" if label.startswith("matched_") else "history_aware_twin_q"

def main():
    a=arguments();config=json.loads(Path(a.config).read_text())
    for key,value in (("dataset_root",a.dataset_root),("output_root",a.output_root),("max_updates",a.max_updates)):
        if value is not None:config[key]=value
    if config["history_semantics"]!="full_episode_prefix_unroll_learning_mask":raise RuntimeError("unsupported history semantics")
    device=select_device(a.device);seed=int(config["training_seed"]);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device.type=="npu":torch.npu.manual_seed_all(seed)
    train,val=load_splits(config["dataset_root"],range(config["train_seed_start"],config["train_seed_end"]+1),range(config["val_seed_start"],config["val_seed_end"]+1),config["gamma"])
    run=Path(config["output_root"])/(a.run_id or f"stage2_2_history_critic_{datetime.now():%Y%m%d_%H%M%S}")
    if run.exists():raise FileExistsError(run)
    (run/"shared").mkdir(parents=True);write(run/"shared"/"config_resolved.json",config)
    write(run/"shared"/"normalization.json",config["normalization"]);write(run/"shared"/"architecture_contract.json",{"history_semantics":config["history_semantics"],"token_schema":config["token_schema"],"current_action_enters_recurrence":False})
    write(run/"shared"/"dataset_manifest.json",{"train_seeds":[config["train_seed_start"],config["train_seed_end"]],"validation_seeds":[config["val_seed_start"],config["val_seed_end"]],"sources":{p:train[p].path for p in train}})
    initial_states={}
    for kind in sorted({critic_type(label) for label in labels(a.mode)}):
        torch.manual_seed(seed)
        if device.type=="npu":torch.npu.manual_seed_all(seed)
        initial_states[kind]=copy.deepcopy(build_critic(config,device,kind).state_dict())
        torch.save(initial_states[kind],run/"shared"/f"initial_state_{kind}.pth")
    for label in labels(a.mode):
        kind=critic_type(label);out=run/label;(out/"checkpoints").mkdir(parents=True);model=build_critic(config,device,kind);model.load_state_dict(copy.deepcopy(initial_states[kind]))
        opt=torch.optim.AdamW(model.parameters(),lr=config["critic_lr"],weight_decay=config["weight_decay"])
        sampler=SequenceSampler(train,config["legacy_replay_burn_in_length"],config["learning_sequence_length"],config["horizon"],seed,"multi_q" in label)
        best=float("inf");last_metric=None
        for step in range(1,int(config["max_updates"])+1):
            model.train()
            started=time.perf_counter();batch=sampler.sample(config["sequence_batch_size"]);sample_ms=(time.perf_counter()-started)*1000
            stats=numpy_batch_stats(batch)
            if not all(item["isfinite"] for item in stats.values()):dump_failure(out,step,"input",batch,{"tensor_statistics":stats},model,opt)
            tensors={key:torch.as_tensor(batch[key],device=device) for key in ("observations","previous_actions","progress","actions","returns","learning_mask")}
            synchronize(device);started=time.perf_counter();(q1,q2),(d1,d2)=model.diagnostic_forward(tensors["observations"],tensors["previous_actions"],tensors["progress"],tensors["actions"]);synchronize(device);forward_ms=(time.perf_counter()-started)*1000
            finite_or_fail("q1_token_encoder_output",[("q1.token_embedding",d1["token_embedding"])],batch,out,step,model,opt)
            finite_or_fail("q2_token_encoder_output",[("q2.token_embedding",d2["token_embedding"])],batch,out,step,model,opt)
            finite_or_fail("q1_prefix_lstm_state",[("q1.final_hidden",d1["final_hidden"]),("q1.final_cell",d1["final_cell"])],batch,out,step,model,opt)
            finite_or_fail("q2_prefix_lstm_state",[("q2.final_hidden",d2["final_hidden"]),("q2.final_cell",d2["final_cell"])],batch,out,step,model,opt)
            finite_or_fail("q1_learning_recurrent_output",[("q1.context",d1["context"])],batch,out,step,model,opt)
            finite_or_fail("q2_learning_recurrent_output",[("q2.context",d2["context"])],batch,out,step,model,opt)
            finite_or_fail("q_output",[("q1",q1),("q2",q2)],batch,out,step,model,opt)
            mask=tensors["learning_mask"].bool();target=tensors["returns"];effective=int(mask.sum().item())
            loss1=torch.nn.functional.mse_loss(q1[mask],target[mask]);loss2=torch.nn.functional.mse_loss(q2[mask],target[mask]);loss=loss1+loss2
            finite_or_fail("loss",[("loss1",loss1),("loss2",loss2),("total_loss",loss)],batch,out,step,model,opt)
            started=time.perf_counter();opt.zero_grad(set_to_none=True);loss.backward();synchronize(device);backward_ms=(time.perf_counter()-started)*1000
            grad=gradient_statistics(model)
            if grad["non_finite_gradient_parameter_names"]:dump_failure(out,step,"backward_gradient",batch,grad,model,opt)
            started=time.perf_counter();opt.step();synchronize(device);optimizer_ms=(time.perf_counter()-started)*1000
            parameters=parameter_statistics(model)
            if not parameters["q1_parameters_finite"] or not parameters["q2_parameters_finite"]:dump_failure(out,step,"optimizer_step_parameter",batch,parameters,model,opt)
            bad_state=optimizer_non_finite(opt)
            if bad_state:dump_failure(out,step,"optimizer_state",batch,{"non_finite_optimizer_state":bad_state},model,opt)
            total_ms=sample_ms+forward_ms+backward_ms+optimizer_ms
            row={"step":step,"model_training":bool(model.training),"q1_loss":float(loss1),"q2_loss":float(loss2),"sample_ms":sample_ms,"forward_ms":forward_ms,"backward_ms":backward_ms,"optimizer_ms":optimizer_ms,"effective_timesteps":effective,"effective_timesteps_per_second":effective*1000.0/total_ms,"sampling":sampler.proportions(),**grad,**parameters}
            if step%int(config["eval_interval"])==0 or step==int(config["max_updates"]):
                evaluation=evaluate(model,val,device,config["horizon"])
                try:finite_metrics(evaluation)
                except FloatingPointError as exc:dump_failure(out,step,"validation",batch,{"error":str(exc),"validation":evaluation},model,opt)
                metric=evaluation["bc_rnn"]["twin_mean_mse"] if "rnn_q" in label and "multi_q" not in label else evaluation["balanced_aggregate"]["twin_mean_mse"]
                last_metric=float(metric);row["validation"]=evaluation
                if metric<best:
                    best=float(metric);torch.save(checkpoint_payload(model,opt,config,step,metric,best,kind),out/"checkpoints"/"best.pth")
            append(out/"train_metrics.jsonl",row)
        torch.save(checkpoint_payload(model,opt,config,config["max_updates"],last_metric,best,kind),out/"checkpoints"/"last.pth")
        best_payload=torch.load(out/"checkpoints"/"best.pth",map_location=device);model.load_state_dict(best_payload["critic_state_dict"],strict=True)
        final=evaluate(model,val,device,config["horizon"]);finite_metrics(final);write(out/"final_validation.json",final);write(out/"sampling_audit.json",{"counts":sampler.counts,"ratios":sampler.proportions()})
        peak=torch.npu.max_memory_allocated(device) if device.type=="npu" else 0
        write(out/"performance.json",{"q1_parameters":sum(p.numel() for p in model.q1.parameters()),"q2_parameters":sum(p.numel() for p in model.q2.parameters()),"total_parameters":sum(p.numel() for p in model.parameters()),"peak_device_memory_bytes":int(peak),"history_semantics":config["history_semantics"],"burn_in_length":"variable_episode_prefix_from_step_0","legacy_replay_burn_in_length":config["legacy_replay_burn_in_length"],"learning_sequence_length":config["learning_sequence_length"]})
    print(json.dumps({"status":"COMPLETE","run_dir":str(run)},indent=2))
if __name__=="__main__":main()
