#!/usr/bin/env python3
"""Read-only real-artifact smoke check for a prepared progressive pair."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import h5py,torch
from stage3_new_agent import Stage3SAC,build_actor,progressive_schedule,state_hash,strict_stage2_load
from stage3_new_evaluation import KEYS
from stage3_new_handoff import FrozenRNNProposer
from train_stage3_new import device,file_hash

def read(path):
    with open(path,encoding="utf-8") as f:return json.load(f)
def main():
    p=argparse.ArgumentParser();p.add_argument("--pair-run-dir",required=True);p.add_argument("--device",default="cpu");a=p.parse_args();pair=Path(a.pair_run_dir).resolve();cfg=read(pair/"shared"/"config_resolved.json");sources=read(pair/"shared"/"stage2_source_manifest.json");d=device(a.device)
    actor_payload=torch.load(pair/"shared"/"actor_init.pth",map_location=d);actor=build_actor(cfg,d);actor.load_state_dict(actor_payload["actor_state_dict"],strict=True)
    if state_hash(actor)!=actor_payload["actor_hash"]:raise RuntimeError("Shared Actor hash mismatch")
    critic_hashes={}
    for group in ("rnn_q","multi_q"):
        critic,_=strict_stage2_load(sources[group]["checkpoint"],d,cfg);agent=Stage3SAC(build_actor(cfg,d),critic,cfg,d);critic_hashes[group]=state_hash(agent.critic)
        if state_hash(agent.target)!=critic_hashes[group]:raise RuntimeError(f"{group} target is not a hard step-0 copy")
    proposer=FrozenRNNProposer(cfg["bc_rnn_checkpoint"],d)
    with h5py.File(cfg["expert_dataset"],"r") as f:
        name=sorted(f["data"])[0];demo=f["data"][name];proposer.start_episode();proposal=proposer.action({key:demo["obs"][key][0] for key in KEYS})
    schedules={str(step):progressive_schedule(cfg,step) for step in (0,9999,10000,20000,30000)}
    print(json.dumps({"status":"PASS","pair_run_dir":str(pair),"actor_init_sha256":file_hash(pair/"shared"/"actor_init.pth"),"critic_hashes":critic_hashes,"bc_rnn_loaded":True,"bc_rnn_first_action_shape":list(proposal.shape),"schedules":schedules},indent=2,sort_keys=True))
if __name__=="__main__":main()
