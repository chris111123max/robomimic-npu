#!/usr/bin/env python3
"""Create immutable shared state for a paired RNN-Q versus Multi-Q run."""
from __future__ import annotations
import argparse, hashlib, json, math, random, shutil
from datetime import datetime
from pathlib import Path
import numpy as np, torch
from stage3_new_agent import build_actor, state_hash, strict_stage2_load
from stage3_new_dataset import ExpertDataset
from stage3_new_handoff import build_expert_cache

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
    p=argparse.ArgumentParser();p.add_argument("--config",default=str(HERE/"stage3_new_config.json"));p.add_argument("--output-root");p.add_argument("--run-id");p.add_argument("--expert-dataset");p.add_argument("--expert-checkpoint");p.add_argument("--bc-rnn-checkpoint");p.add_argument("--expert-rnn-cache");p.add_argument("--expert-cache-workers",type=int,default=16);p.add_argument("--stage2-run-dir",required=True);p.add_argument("--rnn-q-checkpoint",required=True);p.add_argument("--multi-q-checkpoint",required=True);p.add_argument("--reference-pair-run-dir");p.add_argument("--alpha-init",type=float,default=0.01);return p.parse_args()
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
    a=args();c=read(a.config);c["alpha_init"]=float(a.alpha_init);cql=c.get("cql",{"enabled":False});anchor=c.get("anchor",{"enabled":False});handoff=c.get("handoff",{"enabled":False});progressive=c.get("progressive_unfreeze",{"enabled":False});c["experiment_tag"]="rnn_handoff_progressive_unfreeze" if progressive.get("enabled",False) else ("rnn_handoff_cql_lamH1p0" if handoff.get("enabled",False) else ("cqllite_anchor_zscore_lam0p1" if anchor.get("enabled",False) else ("cqllite_lam0p1" if cql.get("enabled",False) else ("alpha_init_0p01" if c["alpha_init"]==0.01 else f"alpha_init_{c['alpha_init']:g}"))));c["expert_dataset"],c["expert_dataset_provenance"]=resolve_expert(a);c["stage2_run_dir"]=str(Path(a.stage2_run_dir).resolve())
    if a.output_root:c["output_root"]=a.output_root
    if not math.isfinite(c["alpha_init"]) or c["alpha_init"]<=0:raise ValueError("--alpha-init must be finite and positive")
    if c["gamma"]!=.99 or c["utd"]!=1 or c["min_online_replay_size"]!=1000 or c["offline_fraction"]!=.5 or c["online_fraction"]!=.5 or c["critic_weight_decay"]!=1e-4 or c["target_update_interval"]!=1 or not c["critic_layer_norm"] or not c["automatic_entropy_tuning"]:raise RuntimeError("Fixed Stage3-new contract changed")
    if cql.get("enabled",False) and (float(cql.get("lambda",-1))!=.1 or int(cql.get("num_random_actions",-1))!=10 or int(cql.get("num_policy_actions",-1))!=1 or not cql.get("apply_to_expert") or not cql.get("apply_to_online") or not cql.get("detach_policy_actions")):raise RuntimeError("Fixed Stage3-new CQL-lite contract changed")
    if anchor.get("enabled",False) and (anchor.get("type")!="zscore_functional" or float(anchor.get("lambda",-1))!=.1 or int(anchor.get("batch_size",-1))!=256 or not anchor.get("teacher_frozen") or not anchor.get("every_critic_update")):raise RuntimeError("Fixed Stage3-new frozen z-score anchor contract changed")
    if handoff.get("enabled",False):
        if not a.bc_rnn_checkpoint:raise ValueError("Handoff preparation requires --bc-rnn-checkpoint")
        if anchor.get("enabled",False) or handoff.get("proposer")!="bc_rnn" or handoff.get("selector")!="target_twin_q_min" or float(handoff.get("margin",-1))!=0 or handoff.get("tie_break")!="rnn" or not handoff.get("bootstrap_proposal") or float(handoff.get("lambda_handoff",-1))!=1:raise RuntimeError("Fixed RNN handoff contract changed")
        c["bc_rnn_checkpoint"]=str(Path(a.bc_rnn_checkpoint).resolve())
    if progressive.get("enabled",False):
        if not handoff.get("enabled",False) or anchor.get("enabled",False) or int(progressive.get("protected_until_env_steps",-1))!=10000 or int(progressive.get("unfreeze_end_env_steps",-1))!=30000 or progressive.get("critic_lr_schedule")!="linear" or progressive.get("target_tau_schedule")!="linear" or progressive.get("phase_a_actor_objective")!="handoff_only" or float(progressive.get("phase_a_alpha_fixed",-1))!=.01 or int(progressive.get("train_metrics_interval_updates",0))!=100:raise RuntimeError("Fixed progressive-unfreeze contract changed")
    sim=c.get("sim_error_handling",{})
    if anchor.get("enabled",False) and (not sim.get("enabled") or int(sim.get("max_consecutive_fatal_errors",0))!=5):raise RuntimeError("Anchored run requires bounded MuJoCo FatalError recovery")
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
    handoff_cache=None
    if handoff.get("enabled",False) and a.expert_rnn_cache:
        source_cache=Path(a.expert_rnn_cache).resolve();source_manifest=source_cache.with_suffix(".manifest.json")
        if not source_cache.is_file() or not source_manifest.is_file():raise FileNotFoundError("Reusable expert RNN cache or manifest is missing")
        cached=read(source_manifest)
        if Path(cached["source_dataset_path"]).resolve()!=Path(c["expert_dataset"]).resolve() or Path(cached["bc_rnn_checkpoint_path"]).resolve()!=Path(c["bc_rnn_checkpoint"]).resolve() or cached["bc_rnn_checkpoint_sha256"]!=sha(c["bc_rnn_checkpoint"]) or cached["cache_sha256"]!=sha(source_cache) or int(cached["transition_count"])!=int(expert.size):raise RuntimeError("Reusable expert RNN cache provenance mismatch")
        destination=shared/"expert_rnn_proposal_cache.npz";shutil.copyfile(source_cache,destination);handoff_cache=dict(cached);handoff_cache.update({"cache_path":str(destination.resolve()),"cache_sha256":sha(destination),"reused_from":str(source_cache)});write(destination.with_suffix(".manifest.json"),handoff_cache);c["expert_rnn_proposal_cache"]=handoff_cache["cache_path"]
    elif handoff.get("enabled",False):
        if a.expert_cache_workers<1:raise ValueError("--expert-cache-workers must be positive")
        handoff_cache=build_expert_cache(c["expert_dataset"],c["bc_rnn_checkpoint"],shared/"expert_rnn_proposal_cache.npz","cpu",a.expert_cache_workers);c["expert_rnn_proposal_cache"]=handoff_cache["cache_path"]
    sources={}
    for group,path in (("rnn_q",a.rnn_q_checkpoint),("multi_q",a.multi_q_checkpoint)):
        _,payload=strict_stage2_load(path,"cpu",c);sources[group]={"checkpoint":str(Path(path).resolve()),"sha256":sha(path),"validation_metric":payload.get("validation_metric"),"model_config":payload.get("model_config"),"gamma":payload.get("gamma")}
    write(shared/"config_resolved.json",c);write(shared/"reference_pair_manifest.json",reference);write(shared/"stage2_source_manifest.json",sources);write(shared/"expert_dataset_audit.json",expert.audit());seeds=read(shared/"seed_manifest.json");manifest={"experiment_name":c["stage"],"experiment_tag":c["experiment_tag"],"timestamp":datetime.now().isoformat(),"actor_init_path":str((shared/"actor_init.pth").resolve()),"actor_init_hash":torch.load(shared/"actor_init.pth",map_location="cpu")["actor_hash"],"actor_init_sha256":sha(shared/"actor_init.pth"),"rnn_q_checkpoint":sources["rnn_q"]["checkpoint"],"multi_q_checkpoint":sources["multi_q"]["checkpoint"],"teacher_and_student_init_same_checkpoint":True,"teacher_checkpoints":{"rnn_q":{"teacher_checkpoint_path":sources["rnn_q"]["checkpoint"],"teacher_checkpoint_hash":sources["rnn_q"]["sha256"],"student_init_checkpoint_path":sources["rnn_q"]["checkpoint"]},"multi_q":{"teacher_checkpoint_path":sources["multi_q"]["checkpoint"],"teacher_checkpoint_hash":sources["multi_q"]["sha256"],"student_init_checkpoint_path":sources["multi_q"]["checkpoint"]}},"handoff":handoff,"progressive_unfreeze":progressive,"expert_rnn_proposal_cache":handoff_cache,"cql":cql,"anchor":anchor,"sim_error_handling":sim,"cql_random_seed_protocol":"shared training seed and identical vectorized draw schedule per gradient update","batch_size":c["batch_size"],"expert_fraction":c["offline_fraction"],"online_fraction":c["online_fraction"],"gamma":c["gamma"],"tau":c["tau"],"actor_lr":c["actor_lr"],"critic_lr":c["critic_lr"],"alpha_lr":c["alpha_lr"],"alpha_init":c["alpha_init"],"automatic_entropy_tuning":c["automatic_entropy_tuning"],"critic_layernorm":c["critic_layer_norm"],"critic_weight_decay":c["critic_weight_decay"],"seed":seeds["training_seed"],"eval_seeds":seeds["evaluation_seeds"],"step0_arbitration_seeds":c.get("step0_arbitration_seeds")};write(shared/"run_manifest.json",manifest)
    print(str(run.resolve()))
if __name__=="__main__":main()
