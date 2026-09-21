#!/usr/bin/env python3
"""Read-only audit of the real Stage1 episode data required by Stage2.2."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import h5py
import numpy as np
from sequence_dataset import KEYS,POLICIES,load_splits

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default=str(Path(__file__).with_name("stage2_2_config.json")));p.add_argument("--dataset-root");a=p.parse_args()
    c=json.loads(Path(a.config).read_text());root=Path(a.dataset_root or c["dataset_root"])
    train,val=load_splits(root,range(c["train_seed_start"],c["train_seed_end"]+1),range(c["val_seed_start"],c["val_seed_end"]+1),c["gamma"])
    report={"status":"PASS","read_only":True,"canonical_observation_keys":list(KEYS),"sources":{}}
    width=c["burn_in_length"]+c["learning_sequence_length"]
    for policy in POLICIES:
        path=root/policy/"transitions.hdf5"
        with h5py.File(path,"r") as f:
            first=f["episodes"][sorted(f["episodes"])[0]]
            fields=sorted(first.keys())
        episodes=train[policy].episodes+val[policy].episodes
        report["sources"][policy]={"path":str(path),"file_exists":path.is_file(),"episode_fields":fields,
            "train_episodes":len(train[policy].episodes),"validation_episodes":len(val[policy].episodes),
            "transitions":sum(e.length for e in episodes),"successful_episodes":sum(e.success for e in episodes),
            "terminated_endings":sum(bool(e.terminated[-1]) for e in episodes),"truncated_endings":sum(bool(e.truncated[-1]) for e in episodes),
            "legal_sequence_starts":sum(max(0,e.length-width+1) for e in train[policy].episodes),
            "obs_dim":train[policy].obs_dim,"action_dim":train[policy].action_dim,
            "dones_equal_terminated_or_truncated":all(np.array_equal(e.dones,e.terminated|e.truncated) for e in episodes)}
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
