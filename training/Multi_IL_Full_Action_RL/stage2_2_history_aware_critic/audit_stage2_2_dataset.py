#!/usr/bin/env python3
"""Read-only audit of the real Stage1 episode data required by Stage2.2."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import h5py
import numpy as np
from sequence_dataset import KEYS,POLICIES,EpisodeDataset,load_splits,scalar

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default=str(Path(__file__).with_name("stage2_2_config.json")));p.add_argument("--dataset-root");p.add_argument("--output");a=p.parse_args()
    c=json.loads(Path(a.config).read_text());root=Path(a.dataset_root or c["dataset_root"])
    train,val=load_splits(root,range(c["train_seed_start"],c["train_seed_end"]+1),range(c["val_seed_start"],c["val_seed_end"]+1),c["gamma"])
    report={"status":"PASS","read_only":True,"canonical_observation_keys":list(KEYS),"non_finite_count":0,"non_finite_issues":[],"sources":{}}
    width=c["legacy_replay_burn_in_length"]+c["learning_sequence_length"]
    for policy in POLICIES:
        path=root/policy/"transitions.hdf5"
        with h5py.File(path,"r") as f:
            first=f["episodes"][sorted(f["episodes"])[0]]
            fields=sorted(first.keys())
            all_seeds=[int(scalar(group,"initial_seed")) for group in f["episodes"].values()]
        episodes=EpisodeDataset(policy,path,all_seeds,c["gamma"]).episodes;issues=[];short=[]
        obs_abs=action_abs=0.;reward_min=return_min=float("inf");reward_max=return_max=float("-inf")
        for e in episodes:
            if e.length<width:short.append({"seed":e.seed,"episode_id":e.episode_id,"length":e.length})
            arrays=(("observations",e.observations),("next_observations",e.next_observations),("actions",e.actions),("rewards",e.rewards),("returns",e.returns))
            for field,value in arrays:
                bad=np.argwhere(~np.isfinite(value))
                for index in bad:
                    index=tuple(map(int,index));issues.append({"policy":policy,"seed":e.seed,"episode_id":e.episode_id,"field":field,"timestep":index[0],"feature_index":index[1] if len(index)>1 else None,"value":str(value[index])})
            if np.isfinite(e.observations).all():obs_abs=max(obs_abs,float(np.abs(e.observations).max()))
            if np.isfinite(e.next_observations).all():obs_abs=max(obs_abs,float(np.abs(e.next_observations).max()))
            if np.isfinite(e.actions).all():action_abs=max(action_abs,float(np.abs(e.actions).max()))
            finite_reward=e.rewards[np.isfinite(e.rewards)];finite_return=e.returns[np.isfinite(e.returns)]
            if finite_reward.size:reward_min=min(reward_min,float(finite_reward.min()));reward_max=max(reward_max,float(finite_reward.max()))
            if finite_return.size:return_min=min(return_min,float(finite_return.min()));return_max=max(return_max,float(finite_return.max()))
        report["sources"][policy]={"path":str(path),"file_exists":path.is_file(),"episode_fields":fields,
            "train_episodes":len(train[policy].episodes),"validation_episodes":len(val[policy].episodes),
            "transitions":sum(e.length for e in episodes),"successful_episodes":sum(e.success for e in episodes),
            "terminated_endings":sum(bool(e.terminated[-1]) for e in episodes),"truncated_endings":sum(bool(e.truncated[-1]) for e in episodes),
            "full_prefix_learning_starts":sum(max(1,e.length-c["learning_sequence_length"]+1) for e in train[policy].episodes),
            "episodes_shorter_than_legacy_burn_plus_learning":len(short),"short_episodes":short,
            "episode_length_min":min(e.length for e in episodes),"episode_length_max":max(e.length for e in episodes),
            "obs_abs_max":obs_abs,"action_abs_max":action_abs,"reward_min":reward_min,"reward_max":reward_max,"return_min":return_min,"return_max":return_max,
            "non_finite_count":len(issues),
            "obs_dim":train[policy].obs_dim,"action_dim":train[policy].action_dim,
            "dones_equal_terminated_or_truncated":all(np.array_equal(e.dones,e.terminated|e.truncated) for e in episodes)}
        report["non_finite_issues"].extend(issues);report["non_finite_count"]+=len(issues)
    if report["non_finite_count"]:report["status"]="FAIL"
    rendered=json.dumps(report,indent=2);print(rendered)
    if a.output:Path(a.output).write_text(rendered+"\n")
    if report["status"]!="PASS":raise SystemExit(1)
if __name__=="__main__":main()
