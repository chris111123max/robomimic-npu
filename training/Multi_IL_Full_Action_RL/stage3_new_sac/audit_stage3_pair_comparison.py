#!/usr/bin/env python3
"""Compare compact RNN-Q and Multi-Q source-decomposition audit summaries."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def read(path):
    with open(path,encoding="utf-8") as f:return json.load(f)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n")
def extract(summary):
    expert,online=summary["expert"],summary["online"]
    return {"checkpoint_env_steps":summary["checkpoint_env_steps"],"alpha":summary["alpha"],"q_expert_data":expert["q_behavior_or_data"]["mean"],"q_expert_policy":expert["q_policy"]["mean"],"expert_policy_minus_data_q":expert["policy_minus_behavior_or_data_q"]["mean"],"expert_fraction_policy_gt_data":expert["fraction_q_policy_gt_q_data"],"expert_random_action_overestimation":expert["fraction_random_max_gt_q_data"],"expert_distance_advantage_pearson":expert["distance_advantage_pearson"],"expert_distance_advantage_spearman":expert["distance_advantage_spearman"],"q_online_behavior":online["q_behavior_or_data"]["mean"],"q_online_policy":online["q_policy"]["mean"],"online_policy_minus_behavior_q":online["policy_minus_behavior_or_data_q"]["mean"],"online_random_action_overestimation":online["fraction_random_max_gt_q_data"]}
def main():
    p=argparse.ArgumentParser();p.add_argument("--pair-run-dir",required=True);p.add_argument("--rnn-summary");p.add_argument("--multi-summary");p.add_argument("--output");a=p.parse_args();pair=Path(a.pair_run_dir).resolve();base=pair/"audits"/"q_source_decomposition";rpath=Path(a.rnn_summary) if a.rnn_summary else base/"rnn_q"/"source_q_summary.json";mpath=Path(a.multi_summary) if a.multi_summary else base/"multi_q"/"source_q_summary.json";rmanifest=read(rpath.parent/"sample_manifest.json");mmanifest=read(mpath.parent/"sample_manifest.json")
    if rmanifest["expert_indices"]!=mmanifest["expert_indices"] or rmanifest["seed"]!=mmanifest["seed"]:raise RuntimeError("RNN-Q and Multi-Q audits did not use identical expert samples")
    rnn,multi=extract(read(rpath)),extract(read(mpath));keys=sorted(set(rnn)&set(multi));result={"pair_run_dir":str(pair),"matched_expert_samples":True,"audit_seed":rmanifest["seed"],"sample_size":rmanifest["sample_size"],"rnn_q":rnn,"multi_q":multi,"multi_minus_rnn":{key:(multi[key]-rnn[key] if isinstance(rnn[key],(int,float)) and isinstance(multi[key],(int,float)) else None) for key in keys},"interpretation":"Read-only diagnostics; differences do not establish causality by themselves."};output=Path(a.output) if a.output else base/"pair_q_decomposition_summary.json";write(output,result);print(json.dumps(result,indent=2))
if __name__=="__main__":main()
