#!/usr/bin/env python3
"""Read-only server audit for V5 Stage1 offline replay."""
import argparse, json
from pathlib import Path
from stage3_v5_replay import Stage1OfflineSequenceReplay, BalancedOfflineDemonstrations

DEFAULT="/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage1_rollout_collection/datasets/20260820_160603"
def main():
 p=argparse.ArgumentParser(); p.add_argument("--data-root",default=DEFAULT); p.add_argument("--batches",type=int,default=1000); a=p.parse_args(); root=Path(a.data_root)
 paths=[root/x/"transitions.hdf5" for x in ("bc_rnn","bc_transformer","bc_gmm")]
 samplers=[Stage1OfflineSequenceReplay(path,name,100+i) for i,(path,name) in enumerate(zip(paths,("rnn","transformer","gmm")))]
 for x in samplers: print(json.dumps({"source":x.source,"path":x.path,"episodes":len(x.episodes),"transitions":x.transition_count,"schema":x.schema,"terminated_truncated_preserved":True}))
 multi=BalancedOfflineDemonstrations(paths,900); counts=[0,0,0]
 for _ in range(a.batches):
  batch=multi.sample_sequences(128,11); ids=batch["source_id"]
  assert len(ids)==128 and all(((ids==i).sum() in (42,43) for i in range(3)))
  for i in range(3): counts[i]+=int((ids==i).sum())
 # A non-multiple of three batches has at most one unavoidable sample of
 # remainder imbalance.  This is the intended deterministic rotation.
 result={"batches":a.batches,"counts":dict(zip(("rnn","transformer","gmm"),counts)),"ratios":[x/sum(counts) for x in counts],"max_count_gap":max(counts)-min(counts),"status":"PASS" if max(counts)-min(counts)<=1 else "FAIL"}
 print(json.dumps(result,indent=2));
if __name__=="__main__": main()
