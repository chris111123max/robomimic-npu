#!/usr/bin/env python3
"""Compare local target variance in memoryless and learned history spaces."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from history_critic import load_checkpoint
from sequence_dataset import load_splits,previous_actions,POLICIES

def neighborhood_report(features,targets,progress,success,k):
    scale=np.maximum(features.std(0,keepdims=True),1e-6);x=(features-features.mean(0,keepdims=True))/scale
    values=[]; progress_spread=[]; success_mismatch=[]
    for i in range(len(x)):
        distance=np.sum((x-x[i])**2,axis=1);neighbors=np.argpartition(distance,min(k,len(x)-1))[:min(k+1,len(x))];neighbors=neighbors[neighbors!=i][:k]
        values.append(np.var(targets[neighbors]));progress_spread.append(np.mean(np.abs(progress[neighbors]-progress[i])));success_mismatch.append(np.mean(success[neighbors]!=success[i]))
    return {"target_variance":{"mean":float(np.mean(values)),"median":float(np.median(values)),"p95":float(np.percentile(values,95))},
            "mean_absolute_progress_gap":float(np.mean(progress_spread)),
            "success_label_mismatch_rate":float(np.mean(success_mismatch))}
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--checkpoint",required=True);p.add_argument("--output",required=True);p.add_argument("--device",default="cpu");p.add_argument("--max-samples",type=int,default=2048);p.add_argument("--neighbors",type=int,default=16);a=p.parse_args();c=json.loads(Path(a.config).read_text());device=torch.device(a.device)
    _,val=load_splits(c["dataset_root"],range(c["train_seed_start"],c["train_seed_end"]+1),range(c["val_seed_start"],c["val_seed_end"]+1),c["gamma"]);model,_=load_checkpoint(a.checkpoint,c,device);model.eval();rows=[]
    with torch.no_grad():
        for policy in POLICIES:
            for e in val[policy].episodes:
                o=torch.as_tensor(e.observations[None],device=device);pa=torch.as_tensor(previous_actions(e.actions)[None],device=device);pr=torch.arange(e.length,device=device).float()[None,:,None]/c["horizon"]
                z,_=model.q1.encode_history(o,pa,pr)
                for i in range(e.length):rows.append((e.observations[i],e.actions[i],z[0,i].cpu().numpy(),e.returns[i],i/c["horizon"],policy,e.success))
    rng=np.random.default_rng(c["training_seed"]);rows=[rows[i] for i in rng.choice(len(rows),min(len(rows),a.max_samples),replace=False)];obs=np.stack([r[0] for r in rows]);act=np.stack([r[1] for r in rows]);z=np.stack([r[2] for r in rows]);target=np.array([r[3] for r in rows])
    progress=np.array([r[4] for r in rows]);success=np.array([r[6] for r in rows],bool)
    report={"samples":len(rows),"neighbors":a.neighbors,
            "memoryless_neighborhood":neighborhood_report(np.c_[obs,act],target,progress,success,a.neighbors),
            "history_neighborhood":neighborhood_report(np.c_[z,act],target,progress,success,a.neighbors)}
    report["mean_target_variance_ratio_history_over_memoryless"]=report["history_neighborhood"]["target_variance"]["mean"]/max(report["memoryless_neighborhood"]["target_variance"]["mean"],1e-12)
    Path(a.output).write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
