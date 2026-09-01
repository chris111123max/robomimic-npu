#!/usr/bin/env python3
"""Stage3-R: transfer BC-RNN LSTM and initialize pomdp-baselines SAC head."""
from __future__ import annotations
import argparse, copy, csv, json, random, sys
from collections import OrderedDict
from pathlib import Path
import h5py, numpy as np, torch

HERE=Path(__file__).resolve().parent; PROJECT=HERE.parent
for path in (HERE, PROJECT/"stage3c_v2_bc_gmm_component_distillation"):
    if str(path) not in sys.path: sys.path.insert(0,str(path))
from stage3_r_actor import (BC_RNN_KEYS,CANONICAL_KEYS,SHAPES,Stage3RActor,checkpoint_payload,
                            set_vendor_device,transfer_lstm)  # noqa
from component_targets import component_means_to_environment_space  # noqa
import robomimic.utils.file_utils as FileUtils  # noqa

def read(p):
    with open(p,encoding="utf-8") as f:return json.load(f)
def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:json.dump(v,f,indent=2,ensure_ascii=False)
def device_of(name):
    if name.startswith("npu"):
        import torch_npu  # noqa
        if not torch.npu.is_available(): raise RuntimeError("NPU unavailable")
        torch.npu.set_device(name)
    return torch.device(name)
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if hasattr(torch,"npu") and torch.npu.is_available():torch.npu.manual_seed_all(seed)
def seed_of(g):return int(g.attrs.get("initial_seed",g["initial_seed"][0]))

def load_teacher(path,device):
    rollout,_=FileUtils.policy_from_checkpoint(ckpt_path=path,device=device,verbose=False);rollout.start_episode()
    teacher=rollout.policy.nets["policy"];teacher.eval()
    for p in teacher.parameters():p.requires_grad_(False)
    return rollout,teacher

def canonical_flat(group):
    n=len(group["actions"]);return np.concatenate([np.asarray(group[f"obs/{k}"],np.float32).reshape(n,-1) for k in CANONICAL_KEYS],axis=1)
def canonical_to_dict(x):
    out=OrderedDict();cursor=0
    for k in CANONICAL_KEYS:
        width=SHAPES[k];out[k]=x[...,cursor:cursor+width].reshape((*x.shape[:-1],width));cursor+=width
    return out

def teacher_features(teacher,canonical):
    adapted=torch.cat([canonical_to_dict(canonical)[k] for k in BC_RNN_KEYS],dim=-1)
    return teacher.nets["rnn"].nets(adapted)[0]

def build_cache(cfg,rollout,teacher,actor,device,seeds,path):
    states=[];targets=[];masks=[];seed_rows=[];distances=[];selected=[];sanity=[];full_sanity={}
    with h5py.File(cfg["dataset"],"r") as h:
        lookup={seed_of(g):g for g in h["episodes"].values()}
        for seed in seeds:
            g=lookup[seed]; flat=canonical_flat(g); acts=np.asarray(g["actions"],np.float32)
            episode_teacher=[];episode_student=[]
            for start in range(0,len(flat),10):
                stop=min(start+10,len(flat));valid=stop-start
                state=np.zeros((10,59),np.float32);state[:valid]=flat[start:stop]
                mask=np.zeros((10,1),np.float32);mask[:valid]=1
                x=torch.as_tensor(state[None],device=device); saved=torch.as_tensor(acts[start:stop],device=device)
                with torch.no_grad():
                    tf=teacher_features(teacher,x)[:,:valid];sf=actor.recurrent_features(x)[0][:,:valid]
                    diff=sf-tf;sanity.append(diff.cpu().numpy().reshape(-1,400));episode_teacher.append(tf.cpu());episode_student.append(sf.cpu())
                    outputs=teacher.nets["decoder"](tf);means=torch.tanh(outputs["mean"])
                    means=component_means_to_environment_space(rollout,means.reshape(-1,5,14))
                    d=torch.linalg.vector_norm(means-saved[:,None,:],dim=2);choice=d.argmin(1);idx=torch.arange(valid,device=device)
                    target=means[idx,choice];minimum=d[idx,choice]
                target_pad=np.zeros((10,14),np.float32);target_pad[:valid]=target.cpu().numpy()
                states.append(state);targets.append(target_pad);masks.append(mask);seed_rows.append(seed)
                distances.append(minimum.cpu().numpy());selected.append(choice.cpu().numpy())
            if seed in (10000,10001):
                a=torch.cat(episode_teacher,1);b=torch.cat(episode_student,1);d=b-a
                full_sanity[str(seed)]={"transitions":len(flat),"mse":float(d.square().mean()),"mae":float(d.abs().mean()),"max_abs_error":float(d.abs().max())}
    sanity=np.concatenate(sanity);d=np.concatenate(distances);sel=np.concatenate(selected)
    sanity_report={"random_sequence_count":min(100,len(states)),"all_sequence_count":len(states),"mse":float(np.square(sanity).mean()),"mae":float(np.abs(sanity).mean()),"max_abs_error":float(np.abs(sanity).max()),"full_episodes":full_sanity,"strict_parameter_mapping":True}
    stats={"num_transitions":len(d),"mean_min_distance":float(d.mean()),"median_min_distance":float(np.median(d)),"p90_min_distance":float(np.quantile(d,.9)),"p95_min_distance":float(np.quantile(d,.95)),"p99_min_distance":float(np.quantile(d,.99)),"max_min_distance":float(d.max()),"fraction_distance_below_1e-4":float((d<1e-4).mean()),"fraction_distance_below_1e-3":float((d<1e-3).mean()),"fraction_distance_below_1e-2":float((d<1e-2).mean()),"selected_component_histogram":{f"mode{i}":int((sel==i).sum()) for i in range(5)},"target":"nearest clean post-tanh GMM component mean in environment action space"}
    with h5py.File(path,"w") as h:
        h.create_dataset("states",data=np.stack(states),compression="gzip");h.create_dataset("targets",data=np.stack(targets),compression="gzip");h.create_dataset("masks",data=np.stack(masks),compression="gzip");h.create_dataset("seeds",data=np.asarray(seed_rows))
    return sanity_report,stats

class Data:
    def __init__(self,path,seeds):
        with h5py.File(path,"r") as h:
            take=np.isin(h["seeds"][:],seeds);self.x=h["states"][:][take];self.y=h["targets"][:][take];self.m=h["masks"][:][take]
        self.episodes=len(seeds);self.transitions=int(self.m.sum())
    def __len__(self):return len(self.x)

@torch.no_grad()
def evaluate(actor,data,batch,device):
    actor.eval();sq=ab=0.;maximum=0.;elements=0
    for s in range(0,len(data),batch):
        x=torch.as_tensor(data.x[s:s+batch],device=device);y=torch.as_tensor(data.y[s:s+batch],device=device);m=torch.as_tensor(data.m[s:s+batch],device=device)
        out=actor.deterministic_sequence(x);diff=(out-y)*m;sq+=float(diff.square().sum().cpu());ab+=float(diff.abs().sum().cpu());maximum=max(maximum,float(diff.abs().max().cpu()));elements+=int(m.sum().item())*14
    return {"val_mse":sq/elements,"val_mae":ab/elements,"val_max_abs_error":maximum}

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default=str(HERE/"stage3_r_config.json"));p.add_argument("--run-dir",required=True);p.add_argument("--device",default="npu:0");p.add_argument("--smoke-test",action="store_true");a=p.parse_args()
    cfg=read(a.config);run=Path(a.run_dir);ck=run/"checkpoints";san=run/"sanity";match=run/"component_matching"
    for d in (ck,san,match):d.mkdir(parents=True,exist_ok=False)
    write(run/"resolved_config.json",cfg);device=device_of(a.device);set_vendor_device(device);seed_all(cfg["random_seed"])
    rollout,teacher=load_teacher(cfg["teacher_checkpoint"],device)
    actual_order=list(teacher.nets["encoder"].nets["obs"].obs_shapes)
    if actual_order!=BC_RNN_KEYS:raise RuntimeError(f"BC-RNN observation order changed: {actual_order}")
    if int(rollout.policy._rnn_horizon)!=10 or bool(rollout.policy._rnn_is_open_loop):raise RuntimeError("BC-RNN recurrent semantics are not horizon=10, open_loop=False")
    if bool(teacher.use_tanh) or not bool(teacher.low_noise_eval) or int(teacher.num_modes)!=5:raise RuntimeError("Unexpected BC-RNN GMM action semantics")
    actor=Stage3RActor(cfg["initial_log_std"],10).to(device);mapping=transfer_lstm(teacher,actor)
    for p0 in actor.policy.last_fc_log_std.parameters():p0.requires_grad_(False)
    train_seeds=list(range(cfg["train_seed_start"],cfg["train_seed_end"]+1));val_seeds=list(range(cfg["validation_seed_start"],cfg["validation_seed_end"]+1))
    used=(train_seeds[:5]+val_seeds[:2]) if a.smoke_test else (train_seeds+val_seeds);cache=match/"stage3_r_targets.hdf5"
    sanity,stats=build_cache(cfg,rollout,teacher,actor,device,used,cache);sanity["parameter_mapping"]=mapping;write(san/"lstm_transfer_sanity.json",sanity);write(match/"component_matching_statistics.json",stats)
    if sanity["max_abs_error"]>cfg["lstm_transfer_max_abs_tolerance"]:raise RuntimeError(f"LSTM transfer sanity failed: {sanity}")
    if stats["mean_min_distance"]>cfg["component_max_mean_l2_distance"] or stats["p99_min_distance"]>cfg["component_max_p99_l2_distance"]:raise RuntimeError(f"component matching failed: {stats}")
    tr=Data(cache,train_seeds[:5] if a.smoke_test else train_seeds);va=Data(cache,val_seeds[:2] if a.smoke_test else val_seeds);epochs=2 if a.smoke_test else 300;fixed={1,2} if a.smoke_test else set(cfg["checkpoint_epochs"])
    opt=torch.optim.Adam([{"params":actor.lstm.parameters(),"lr":cfg["lstm_lr"]},{"params":actor.policy.last_fc.parameters(),"lr":cfg["head_lr"]}]);
    for q in actor.lstm.parameters():q.requires_grad_(False)
    rng=np.random.default_rng(cfg["random_seed"]);history=[];best=float("inf");best_epoch=0
    for epoch in range(1,epochs+1):
        if epoch==51:
            for q in actor.lstm.parameters():q.requires_grad_(True)
        phase="A_head_only" if epoch<=50 else "B_lstm_finetune";order=rng.permutation(len(tr));actor.train();sq=ab=0.;elements=0
        for start in range(0,len(order),cfg["batch_sequences"]):
            ids=order[start:start+cfg["batch_sequences"]];x=torch.as_tensor(tr.x[ids],device=device);y=torch.as_tensor(tr.y[ids],device=device);m=torch.as_tensor(tr.m[ids],device=device)
            out=actor.deterministic_sequence(x);diff=(out-y)*m;loss=diff.square().sum()/(m.sum()*14).clamp_min(1)
            if not torch.isfinite(loss):raise RuntimeError("non-finite loss")
            opt.zero_grad(set_to_none=True);loss.backward()
            if not all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in actor.parameters()):raise RuntimeError("non-finite gradient")
            opt.step();sq+=float(diff.detach().square().sum().cpu());ab+=float(diff.detach().abs().sum().cpu());elements+=int(m.sum().item())*14
        values=evaluate(actor,va,cfg["batch_sequences"],device);row={"epoch":epoch,"phase":phase,"train_mse":sq/elements,"train_mae":ab/elements,**values,"lstm_lr":0.0 if epoch<=50 else cfg["lstm_lr"],"head_lr":cfg["head_lr"]}
        if values["val_mse"]<best:best=values["val_mse"];best_epoch=epoch;torch.save(checkpoint_payload(actor,epoch,cfg,values,opt),ck/"stage3_r_best_val_mse.pth")
        row.update({"best_val_epoch":best_epoch,"best_val_mse":best});history.append(row)
        with open(run/"training_metrics.csv","w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
        if epoch in fixed:torch.save(checkpoint_payload(actor,epoch,cfg,values,opt),ck/f"stage3_r_epoch_{epoch}.pth")
        print(f"epoch {epoch:03d}/{epochs} {phase} train_mse={row['train_mse']:.8f} val_mse={values['val_mse']:.8f}",flush=True)
    summary={"status":"SMOKE_TEST_PASSED" if a.smoke_test else "TRAINING_COMPLETE_PENDING_CANDIDATE_EVALUATION","epochs_completed":epochs,"train":{"episodes":tr.episodes,"transitions":tr.transitions},"validation":{"episodes":va.episodes,"transitions":va.transitions},"best_val_epoch":best_epoch,"best_val_mse":best,"lstm_transfer_sanity":sanity,"component_matching":stats,"stage2_r_started":False,"stage4_started":False};write(run/"training_summary.json",summary)
    if a.smoke_test:
        sample=torch.as_tensor(tr.x[:2],device=device);act,_,_,logp=actor.actions_from_features(actor.forward_sequence(sample),False,True)
        if not torch.isfinite(act).all() or not torch.isfinite(logp).all():raise RuntimeError("stochastic SAC sanity failed")
        reloaded=Stage3RActor(cfg["initial_log_std"],10).to(device);reloaded.load_state_dict(torch.load(ck/"stage3_r_epoch_2.pth",map_location=device)["actor_state_dict"],strict=True)
        write(run/"smoke_test.json",{"status":"PASS","loss_finite":True,"gradients_finite":True,"checkpoint_reload":True,"stochastic_action_finite":True,"log_prob_finite":True,"log_std_frozen":True,"npu":str(device).startswith("npu"),"stage2_r_started":False,"stage4_started":False})
if __name__=="__main__":main()
