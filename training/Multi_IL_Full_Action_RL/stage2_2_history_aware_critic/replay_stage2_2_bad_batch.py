#!/usr/bin/env python3
"""Exactly replay a batch from the legacy sampler used by the failed runs."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from finite_diagnostics import numpy_batch_stats
from sequence_dataset import load_splits
from sequence_sampler import LegacyWindowSampler

def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=("rnn_q","multi_q"),required=True);p.add_argument("--step",type=int,required=True)
    p.add_argument("--config",default=str(Path(__file__).with_name("stage2_2_config.json")));p.add_argument("--dataset-root");p.add_argument("--dump-npz");p.add_argument("--output");a=p.parse_args()
    if a.step<1:raise ValueError("--step is one-based and must be positive")
    c=json.loads(Path(a.config).read_text());root=a.dataset_root or c["dataset_root"]
    train,_=load_splits(root,range(c["train_seed_start"],c["train_seed_end"]+1),range(c["val_seed_start"],c["val_seed_end"]+1),c["gamma"])
    sampler=LegacyWindowSampler(train,c["legacy_replay_burn_in_length"],c["learning_sequence_length"],c["horizon"],c["training_seed"],a.mode=="multi_q")
    batch=None
    for _ in range(a.step):batch=sampler.sample(c["sequence_batch_size"])
    metadata=[]
    for i in range(len(batch["start"])):metadata.append({key:(str(batch[key][i]) if key=="policy" else int(batch[key][i])) for key in ("policy","seed","episode_id","start","stop")})
    global_stats=numpy_batch_stats(batch);per_sequence=[]
    for i,item in enumerate(metadata):
        sliced={key:batch[key][i:i+1] for key in ("observations","previous_actions","progress","actions","returns")}
        per_sequence.append({"metadata":item,"tensor_statistics":numpy_batch_stats(sliced)})
    report={"status":"PASS" if all(v["isfinite"] for v in global_stats.values()) else "FAIL","sampler":"legacy_arbitrary_window_exact_replay","mode":a.mode,"step":a.step,"batch_metadata":metadata,"tensor_statistics":global_stats,"per_sequence":per_sequence}
    rendered=json.dumps(report,indent=2);print(rendered)
    if a.output:Path(a.output).write_text(rendered+"\n")
    if a.dump_npz:np.savez_compressed(a.dump_npz,**batch)
if __name__=="__main__":main()
