#!/usr/bin/env python3
"""Stage3C-v2 fixed 300-epoch component-mean distillation."""
from __future__ import annotations
import argparse, copy, csv, json, random, sys
from pathlib import Path
import h5py, numpy as np, torch

HERE = Path(__file__).resolve().parent
ACTOR_DIR = HERE.parent / "stage3_actor_initialization"
for path in (HERE, ACTOR_DIR):
    sys.path.insert(0, str(path)) if str(path) not in sys.path else None
from actor_network import build_actor, load_actor_checkpoint, stochastic_action_and_log_prob  # noqa
from component_targets import CachedTargets, backbone_sanity, build_cache, load_policy, transfer_backbone  # noqa

def read(path):
    with open(path, encoding="utf-8") as f: return json.load(f)
def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(value, f, indent=2, ensure_ascii=False)
def device_of(name):
    if name.startswith("npu"):
        import torch_npu  # noqa
        torch.npu.set_device(torch.device(name))
    return torch.device(name)
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available(): torch.npu.manual_seed_all(seed)
def evaluate(actor, data, batch, device):
    actor.eval(); sq = ab = 0.; maximum = 0.; count = 0
    with torch.no_grad():
        for start in range(0, len(data), batch):
            state = torch.as_tensor(data.states[start:start+batch], device=device)
            target = torch.as_tensor(data.targets[start:start+batch], device=device)
            diff = actor(state, deterministic=True)[0] - target
            sq += float(diff.square().sum().cpu()); ab += float(diff.abs().sum().cpu())
            maximum = max(maximum, float(diff.abs().max().cpu())); count += target.numel()
    return {"val_mse": sq/count, "val_mae": ab/count, "val_max_abs_error": maximum}
def payload(actor, optimizer, epoch, cfg, values):
    return {"format_version":"multi_il_full_action_rl.stage3c_v2.actor.v1", "stage":"stage3c_v2_bc_gmm_component_distillation",
            "epoch":epoch, "actor_state_dict":copy.deepcopy(actor.state_dict()), "optimizer_state_dict":copy.deepcopy(optimizer.state_dict()),
            "architecture":copy.deepcopy(cfg["architecture"]), "action_distribution":{"initial_log_std":cfg["initial_log_std"], "freeze_log_std_during_distillation":True},
            "action_convention":{"student_deterministic":"tanh(mu)", "target":"inferred sampled-component clean post-tanh mean"},
            "observation_keys":["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","robot1_eef_pos","robot1_eef_quat","robot1_gripper_qpos","object"],
            "observation_shapes":{"robot0_eef_pos":[3],"robot0_eef_quat":[4],"robot0_gripper_qpos":[2],"robot1_eef_pos":[3],"robot1_eef_quat":[4],"robot1_gripper_qpos":[2],"object":[41]},
            "teacher_checkpoint":cfg["teacher_checkpoint"], "source_dataset":cfg["dataset"], "validation":copy.deepcopy(values),
            "evaluation_horizon":cfg["evaluation_horizon"], "terminate_on_success":cfg["terminate_on_success"],
            "backbone_transferred":True, "log_std_frozen":True}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default=str(HERE/"stage3c_v2_config.json")); p.add_argument("--run-dir",required=True); p.add_argument("--device",default="npu:0"); p.add_argument("--smoke-test",action="store_true"); a=p.parse_args()
    cfg=read(a.config); run=Path(a.run_dir); ckpts=run/"checkpoints"; match=run/"component_matching"; ckpts.mkdir(parents=True,exist_ok=False); match.mkdir(parents=True,exist_ok=False); write(run/"resolved_config.json",cfg)
    device=device_of(a.device); seed_all(cfg["random_seed"]); policy,gmm=load_policy(cfg["teacher_checkpoint"],device)
    actor=build_actor(59,14,(1024,1024),cfg["initial_log_std"],True,device); teacher_order=transfer_backbone(gmm,actor)
    with h5py.File(cfg["dataset"],"r") as h:
        group=next(iter(h["episodes"].values())); parts=[np.asarray(group["obs"][k][:100],np.float32).reshape(-1,int(np.prod(group["obs"][k].shape[1:]))) for k in json.loads(h.attrs["canonical_observation_keys"])]
    sample=torch.as_tensor(np.concatenate(parts,axis=1),device=device); sanity=backbone_sanity(gmm,actor,sample,teacher_order); write(match/"backbone_transfer_sanity.json",sanity)
    if sanity["max_abs_error"]>cfg["backbone_max_abs_tolerance"]: raise RuntimeError(f"Backbone transfer failed: {sanity}")
    cache=match/"component_targets.hdf5"; seeds=(cfg["train_seeds"][:5]+cfg["validation_seeds"][:2]) if a.smoke_test else (cfg["train_seeds"]+cfg["validation_seeds"])
    stats=build_cache(cfg["dataset"],cache,policy,gmm,device,seeds,cfg["component_batch_size"]); write(match/"component_matching_statistics.json",stats)
    if stats["mean_min_distance"]>cfg["component_max_mean_l2_distance"] or stats["p99_min_distance"]>cfg["component_max_p99_l2_distance"]: raise RuntimeError(f"Component action-space sanity failed: {stats}")
    train_seeds=cfg["train_seeds"][:5] if a.smoke_test else cfg["train_seeds"]; val_seeds=cfg["validation_seeds"][:2] if a.smoke_test else cfg["validation_seeds"]
    train=CachedTargets(cache,train_seeds); val=CachedTargets(cache,val_seeds); epochs=2 if a.smoke_test else 300; fixed={1,2} if a.smoke_test else set(cfg["checkpoint_epochs"])
    optimizer=torch.optim.Adam([{"params":actor.fcs.parameters(),"lr":cfg["backbone_lr"]},{"params":actor.last_fc.parameters(),"lr":cfg["head_lr"]}])
    for param in actor.fcs.parameters(): param.requires_grad_(False)
    rng=np.random.default_rng(cfg["random_seed"]); history=[]; best=float("inf"); best_epoch=None
    print(f"Stage3C-v2 train={train.statistics} validation={val.statistics} epochs={epochs}")
    for epoch in range(1,epochs+1):
        if epoch==51:
            for param in actor.fcs.parameters(): param.requires_grad_(True)
        phase="A_head_only" if epoch<=50 else "B_backbone_finetune"; actor.train(); order=rng.permutation(len(train)); sq=ab=0.; elements=0
        for start in range(0,len(order),cfg["batch_size"]):
            idx=order[start:start+cfg["batch_size"]]; state=torch.as_tensor(train.states[idx],device=device); target=torch.as_tensor(train.targets[idx],device=device); out=actor(state,deterministic=True)[0]; diff=out-target; loss=diff.square().mean()
            if not torch.isfinite(loss): raise RuntimeError("non-finite loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if not all(x.grad is None or bool(torch.isfinite(x.grad).all()) for x in actor.parameters()): raise RuntimeError("non-finite gradient")
            optimizer.step(); sq+=float(diff.detach().square().sum().cpu()); ab+=float(diff.detach().abs().sum().cpu()); elements+=target.numel()
        values=evaluate(actor,val,cfg["batch_size"],device); row={"epoch":epoch,"phase":phase,"train_mse":sq/elements,"train_mae":ab/elements,**values,"backbone_lr":0.0 if epoch<=50 else cfg["backbone_lr"],"head_lr":cfg["head_lr"]}
        if values["val_mse"]<best: best=values["val_mse"]; best_epoch=epoch; torch.save(payload(actor,optimizer,epoch,cfg,values),ckpts/"bc_gmm_component_best_val_mse.pth")
        row.update({"best_val_epoch":best_epoch,"best_val_mse":best}); history.append(row); print("epoch {epoch:03d}/{epochs} phase={phase} train_mse={train_mse:.8f} val_mse={val_mse:.8f} val_mae={val_mae:.8f}".format(epochs=epochs,**row))
        if epoch in fixed: torch.save(payload(actor,optimizer,epoch,cfg,values),ckpts/f"bc_gmm_component_epoch_{epoch}.pth")
    with open(run/"training_metrics.csv","w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
    loaded,_=load_actor_checkpoint(ckpts/"bc_gmm_component_best_val_mse.pth",device,freeze_log_std=True); x=torch.as_tensor(val.states[:8],device=device)
    with torch.no_grad(): deterministic=loaded(x,deterministic=True)[0]; stochastic,log_prob=stochastic_action_and_log_prob(loaded,x)
    smoke={"status":"PASS","component_means_shape":["B",5,14],"saved_action_shape":["B",14],"target_shape":["B",14],"student_shape":list(deterministic.shape),"backbone":sanity,"loss_finite":True,"gradients_finite":True,"checkpoint_reload":True,"stochastic_finite":bool(torch.isfinite(stochastic).all()),"log_prob_finite":bool(torch.isfinite(log_prob).all()),"log_std_frozen":not any(p.requires_grad for p in loaded.last_fc_log_std.parameters())}; write(run/"smoke_test.json",smoke)
    summary={"status":"SMOKE_TEST_PASSED" if a.smoke_test else "TRAINING_COMPLETE_300_EPOCHS","run_directory":str(run),"train":train.statistics,"validation":val.statistics,"backbone_transfer_sanity":sanity,"component_matching":stats,"epochs_completed":epochs,"best_val_epoch":best_epoch,"best_val_mse":best,"best_val_mae":history[best_epoch-1]["val_mae"],"fixed_checkpoints":sorted(p.name for p in ckpts.glob("bc_gmm_component_epoch_*.pth")),"stage4_started":False}; write(run/"training_summary.json",summary); print(f"STAGE3C_V2_TRAINING_COMPLETE run_dir={run} best_epoch={best_epoch}")
if __name__=="__main__": main()
