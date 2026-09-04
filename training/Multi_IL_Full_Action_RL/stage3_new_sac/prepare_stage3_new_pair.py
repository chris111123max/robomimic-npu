#!/usr/bin/env python3
"""Create immutable shared state for a paired RNN-Q versus Multi-Q run."""
from __future__ import annotations
import argparse, hashlib, json, random
from datetime import datetime
from pathlib import Path
import numpy as np, torch
from stage3_new_agent import build_actor, state_hash, strict_stage2_load
from stage3_new_dataset import ExpertDataset

HERE=Path(__file__).resolve().parent
def read(path):
    with open(path,encoding="utf-8") as f:return json.load(f)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"x",encoding="utf-8") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n")
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()
def args():
    p=argparse.ArgumentParser();p.add_argument("--config",default=str(HERE/"stage3_new_config.json"));p.add_argument("--output-root");p.add_argument("--run-id");p.add_argument("--expert-dataset");p.add_argument("--expert-checkpoint");p.add_argument("--stage2-run-dir",required=True);p.add_argument("--rnn-q-checkpoint",required=True);p.add_argument("--multi-q-checkpoint",required=True);return p.parse_args()
def resolve_expert(a):
    if a.expert_dataset:return str(Path(a.expert_dataset).resolve()),"explicit CLI path"
    if not a.expert_checkpoint:raise ValueError("Provide --expert-dataset or --expert-checkpoint")
    import robomimic.utils.file_utils as FileUtils
    checkpoint=FileUtils.maybe_dict_from_checkpoint(ckpt_path=a.expert_checkpoint);cfg,_=FileUtils.config_from_checkpoint(ckpt_dict=checkpoint);value=cfg.train.data
    if isinstance(value,(list,tuple)):
        if len(value)!=1:raise RuntimeError(f"Expert checkpoint contains {len(value)} datasets; pass --expert-dataset explicitly")
        value=value[0]
    if isinstance(value,dict):value=value.get("path")
    if not value:raise RuntimeError("Cannot resolve train.data from expert checkpoint")
    return str(Path(value).resolve()),f"train.data from {Path(a.expert_checkpoint).resolve()}"
def main():
    a=args();c=read(a.config);c["expert_dataset"],c["expert_dataset_provenance"]=resolve_expert(a);c["stage2_run_dir"]=str(Path(a.stage2_run_dir).resolve())
    if a.output_root:c["output_root"]=a.output_root
    if c["gamma"]!=.99 or c["utd"]!=1 or c["min_online_replay_size"]!=1000 or c["offline_fraction"]!=.5 or c["online_fraction"]!=.5 or c["critic_weight_decay"]!=1e-4 or c["target_update_interval"]!=1 or not c["critic_layer_norm"] or not c["automatic_entropy_tuning"]:raise RuntimeError("Fixed Stage3-new contract changed")
    expert=ExpertDataset(c["expert_dataset"],c["training_seed"]);run=Path(c["output_root"])/(a.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    if run.exists():raise FileExistsError(run)
    shared=run/"shared";shared.mkdir(parents=True);(run/"rnn_q").mkdir();(run/"multi_q").mkdir()
    random.seed(c["training_seed"]);np.random.seed(c["training_seed"]);torch.manual_seed(c["training_seed"]);actor=build_actor(c,"cpu")
    actor_payload={"actor_state_dict":actor.state_dict(),"actor_hash":state_hash(actor),"architecture":{"obs_dim":59,"action_dim":14,"hidden_dims":c["hidden_dims"],"class":"rlkit TanhGaussianPolicy"},"training_seed":c["training_seed"]};torch.save(actor_payload,shared/"actor_init.pth")
    sources={}
    for group,path in (("rnn_q",a.rnn_q_checkpoint),("multi_q",a.multi_q_checkpoint)):
        _,payload=strict_stage2_load(path,"cpu",c);sources[group]={"checkpoint":str(Path(path).resolve()),"sha256":sha(path),"validation_metric":payload.get("validation_metric"),"model_config":payload.get("model_config"),"gamma":payload.get("gamma")}
    write(shared/"config_resolved.json",c);write(shared/"seed_manifest.json",{"training_seed":c["training_seed"],"train_seed_rule":"train_seed_base + episode_index","train_seed_base":c["train_seed_base"],"evaluation_seeds":list(range(c["evaluation_seed_start"],c["evaluation_seed_start"]+c["evaluation_episodes"]))});write(shared/"stage2_source_manifest.json",sources);write(shared/"expert_dataset_audit.json",expert.audit())
    print(str(run.resolve()))
if __name__=="__main__":main()
