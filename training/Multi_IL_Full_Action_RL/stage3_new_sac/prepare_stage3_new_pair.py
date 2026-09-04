#!/usr/bin/env python3
"""Create immutable shared state for a paired RNN-Q versus Multi-Q run."""
from __future__ import annotations
import argparse, hashlib, json, math, random, shutil
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
    p=argparse.ArgumentParser();p.add_argument("--config",default=str(HERE/"stage3_new_config.json"));p.add_argument("--output-root");p.add_argument("--run-id");p.add_argument("--expert-dataset");p.add_argument("--expert-checkpoint");p.add_argument("--stage2-run-dir",required=True);p.add_argument("--rnn-q-checkpoint",required=True);p.add_argument("--multi-q-checkpoint",required=True);p.add_argument("--reference-pair-run-dir");p.add_argument("--alpha-init",type=float,default=0.01);return p.parse_args()
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
    a=args();c=read(a.config);c["alpha_init"]=float(a.alpha_init);c["experiment_tag"]="alpha_init_0p01" if c["alpha_init"]==0.01 else f"alpha_init_{c['alpha_init']:g}";c["expert_dataset"],c["expert_dataset_provenance"]=resolve_expert(a);c["stage2_run_dir"]=str(Path(a.stage2_run_dir).resolve())
    if a.output_root:c["output_root"]=a.output_root
    if not math.isfinite(c["alpha_init"]) or c["alpha_init"]<=0:raise ValueError("--alpha-init must be finite and positive")
    if c["gamma"]!=.99 or c["utd"]!=1 or c["min_online_replay_size"]!=1000 or c["offline_fraction"]!=.5 or c["online_fraction"]!=.5 or c["critic_weight_decay"]!=1e-4 or c["target_update_interval"]!=1 or not c["critic_layer_norm"] or not c["automatic_entropy_tuning"]:raise RuntimeError("Fixed Stage3-new contract changed")
    reference=None
    if a.reference_pair_run_dir:
        reference_pair=Path(a.reference_pair_run_dir).resolve();actor_source=reference_pair/"shared"/"actor_init.pth";seed_source=reference_pair/"shared"/"seed_manifest.json"
        if not actor_source.is_file() or not seed_source.is_file():raise FileNotFoundError("Reference pair must contain shared/actor_init.pth and shared/seed_manifest.json")
        actor_payload=torch.load(actor_source,map_location="cpu");actor=build_actor(c,"cpu");actor.load_state_dict(actor_payload["actor_state_dict"],strict=True);actual_actor_hash=state_hash(actor)
        if actual_actor_hash!=actor_payload.get("actor_hash"):raise RuntimeError("Reference Actor payload hash mismatch")
        seed_manifest=read(seed_source);required={"training_seed","train_seed_rule","train_seed_base","evaluation_seeds"}
        if not required.issubset(seed_manifest):raise RuntimeError(f"Reference seed manifest misses {sorted(required-set(seed_manifest))}")
        c["training_seed"]=int(seed_manifest["training_seed"]);c["train_seed_base"]=int(seed_manifest["train_seed_base"]);c["evaluation_episodes"]=len(seed_manifest["evaluation_seeds"])
        reference={"reference_pair_run_dir":str(reference_pair),"actor_init_source":str(actor_source),"actor_init_sha256":sha(actor_source),"seed_manifest_source":str(seed_source),"seed_manifest_sha256":sha(seed_source),"new_alpha_init":c["alpha_init"]}
    expert=ExpertDataset(c["expert_dataset"],c["training_seed"]);run=Path(c["output_root"])/(a.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    if run.exists():raise FileExistsError(run)
    shared=run/"shared";shared.mkdir(parents=True);(run/"rnn_q").mkdir();(run/"multi_q").mkdir()
    if reference is not None:
        shutil.copyfile(actor_source,shared/"actor_init.pth");shutil.copyfile(seed_source,shared/"seed_manifest.json")
        reference.update({"copied_actor_init_sha256":sha(shared/"actor_init.pth"),"copied_seed_manifest_sha256":sha(shared/"seed_manifest.json")})
        if reference["actor_init_sha256"]!=reference["copied_actor_init_sha256"] or reference["seed_manifest_sha256"]!=reference["copied_seed_manifest_sha256"]:raise RuntimeError("Reference shared artifact copy is not byte-exact")
    else:
        random.seed(c["training_seed"]);np.random.seed(c["training_seed"]);torch.manual_seed(c["training_seed"]);actor=build_actor(c,"cpu")
        actor_payload={"actor_state_dict":actor.state_dict(),"actor_hash":state_hash(actor),"architecture":{"obs_dim":59,"action_dim":14,"hidden_dims":c["hidden_dims"],"class":"rlkit TanhGaussianPolicy"},"training_seed":c["training_seed"]};torch.save(actor_payload,shared/"actor_init.pth")
        write(shared/"seed_manifest.json",{"training_seed":c["training_seed"],"train_seed_rule":"train_seed_base + episode_index","train_seed_base":c["train_seed_base"],"evaluation_seeds":list(range(c["evaluation_seed_start"],c["evaluation_seed_start"]+c["evaluation_episodes"]))})
        reference={"reference_pair_run_dir":None,"actor_init_source":"generated from training_seed","actor_init_sha256":sha(shared/"actor_init.pth"),"seed_manifest_source":"generated from config","seed_manifest_sha256":sha(shared/"seed_manifest.json"),"new_alpha_init":c["alpha_init"]}
    sources={}
    for group,path in (("rnn_q",a.rnn_q_checkpoint),("multi_q",a.multi_q_checkpoint)):
        _,payload=strict_stage2_load(path,"cpu",c);sources[group]={"checkpoint":str(Path(path).resolve()),"sha256":sha(path),"validation_metric":payload.get("validation_metric"),"model_config":payload.get("model_config"),"gamma":payload.get("gamma")}
    write(shared/"config_resolved.json",c);write(shared/"reference_pair_manifest.json",reference);write(shared/"stage2_source_manifest.json",sources);write(shared/"expert_dataset_audit.json",expert.audit())
    print(str(run.resolve()))
if __name__=="__main__":main()
