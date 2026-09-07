#!/usr/bin/env python3
"""Compare a completed CQL-lite pair with the immutable vanilla-SAC baseline."""
from __future__ import annotations
import argparse,json
from pathlib import Path

MILESTONES={1000,5000,10000,25000,50000,75000,100000,150000,200000,250000,300000}
METRICS=("qmin_mean","target_qmin_mean","td_target_mean","critic_loss","critic_loss_total","critic_td_loss","cql_loss_raw","cql_loss_weighted","anchor_loss_raw","anchor_loss_weighted","anchor_teacher_student_pearson_q1","anchor_teacher_student_pearson_q2","alpha","policy_minus_data_q_mean","random_max_minus_data_q_mean")
def read(path):
    with open(path,encoding="utf-8") as f:return json.load(f)
def metric_rows(group):
    result={}
    with open(group/"train_metrics.jsonl",encoding="utf-8") as f:
        for line in f:
            row=json.loads(line);step=int(row.get("env_steps",-1))
            if step in MILESTONES:result[step]={key:row.get(key) for key in METRICS}
    return result
def evaluations(group):
    result={}
    for path in (group/"evaluations").glob("step_*.json"):result[int(path.stem.split("_")[1])]=read(path).get("success_rate")
    return result
def source_final(pair,group):
    path=pair/group/"source_diagnostics.jsonl"
    if not path.is_file():return None
    last=None
    with open(path,encoding="utf-8") as f:
        for line in f:last=json.loads(line)
    return last
def main():
    p=argparse.ArgumentParser();p.add_argument("--cql-pair-run-dir",required=True);p.add_argument("--baseline-pair-run-dir",required=True);p.add_argument("--output");a=p.parse_args();cql=Path(a.cql_pair_run_dir).resolve();baseline=Path(a.baseline_pair_run_dir).resolve();groups={}
    for group in ("rnn_q","multi_q"):
        base_metrics,cql_metrics=metric_rows(baseline/group),metric_rows(cql/group);steps=sorted(set(base_metrics)|set(cql_metrics));groups[group]={"milestones":{str(step):{"baseline":base_metrics.get(step),"cql_lite":cql_metrics.get(step)} for step in steps},"evaluation_success_rate":{"baseline":evaluations(baseline/group),"cql_lite":evaluations(cql/group)},"cql_source_diagnostics_final":source_final(cql,group)}
    result={"baseline_pair_run_dir":str(baseline),"cql_pair_run_dir":str(cql),"groups":groups};output=Path(a.output) if a.output else cql/"pair_comparison"/"baseline_vs_cqllite.json";output.parent.mkdir(parents=True,exist_ok=True)
    with open(output,"w",encoding="utf-8") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps({"output":str(output.resolve()),"groups":list(groups)},indent=2))
if __name__=="__main__":main()
