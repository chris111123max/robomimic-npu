#!/usr/bin/env python3
"""Create the fixed three-group Stage4 comparison after all groups complete."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from stage4_core import atomic_json,read_json  # noqa

POINTS=(0,30000,50000,100000,200000,500000,1000000)
def rows(path):
    with Path(path).open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))
def number(value):return None if value in (None,"") else float(value)
def auc(points,end):
    selected=sorted((step,value) for step,value in points.items() if step<=end)
    if not selected or selected[0][0]!=0 or selected[-1][0]!=end:return None
    return float(np.trapz([value for _,value in selected],[step for step,_ in selected]))
def group_summary(directory,config):
    status=read_json(directory/"status.json");
    if status["status"]!="COMPLETE":raise RuntimeError(f"Incomplete Stage4 group: {directory}: {status}")
    evaluations={int(row["env_step"]):row for row in rows(directory/"evaluation_metrics.csv")};steps=sorted(evaluations);sr={step:float(evaluations[step]["success_rate"]) for step in steps};progress={step:float(evaluations[step]["mean_progress"]) for step in steps}
    initial=sr[0];first_200=[value for step,value in sr.items() if step<=200000];minimum=min(first_200);after=[step for step in steps if step>int(config["actor_freeze_steps"]) and sr[step]>=initial]
    thresholds={str(value):next((step for step in steps if sr[step]>=value),None) for value in (0.30,0.40,0.50,0.60,0.70)};best_step=max(steps,key=lambda step:(sr[step],progress[step],-step))
    training=rows(directory/"training_metrics.csv");weights=np.asarray([int(row["updates_this_episode"]) for row in training]);fractions=np.asarray([number(row["critic_fraction_clipped"]) or 0.0 for row in training])
    return {"evaluation_points":{str(point):sr.get(point) for point in POINTS},"initial_success_rate":initial,"minimum_success_rate_first_200k":minimum,"initial_degradation":initial-minimum,
        "recovery_step":None if not after else after[0],"threshold_crossings":thresholds,"auc_0_200k":auc(sr,200000),"auc_0_500k":auc(sr,500000),"auc_0_1m":auc(sr,1000000),
        "best_success_rate":sr[best_step],"best_success_step":best_step,"final_success_rate":sr[int(config["total_env_steps"])],"critic_fraction_clipped":None if weights.sum()==0 else float(np.average(fractions,weights=weights)),"nan_inf_failure":status["nan_inf_failure"],"success_curve":sr,"progress_curve":progress}
def main():
    p=argparse.ArgumentParser();p.add_argument("--run-dir",required=True);a=p.parse_args();run=Path(a.run_dir).resolve();active_groups=list(read_json(run/"pids.json"))
    if active_groups != ["rnn_only_critic","multi_il_critic"]:raise RuntimeError(f"Unexpected Stage4 active groups: {active_groups}")
    config=read_json(run/active_groups[0]/"config.json")
    audits={group:read_json(run/group/"initialization_audit.json") for group in active_groups};actor_hashes={value["actor_hash"] for value in audits.values()}
    if len(actor_hashes)!=1:raise RuntimeError(f"Stage4 Actor hashes differ: {audits}")
    critic_hashes={value["critic_hash"] for value in audits.values()}
    if len(critic_hashes)!=2:raise RuntimeError("Stage4 Critic initialization hashes are not distinct")
    summaries={group:group_summary(run/group,config) for group in active_groups};all_steps=sorted(set().union(*(set(value["success_curve"]) for value in summaries.values())))
    curve_path=run/"stage4_learning_curves.csv"
    with curve_path.open("w",newline="",encoding="utf-8") as handle:
        fields=["env_step","rnn_success_rate","multi_success_rate","rnn_mean_progress","multi_mean_progress"];writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for step in all_steps:writer.writerow({"env_step":step,"rnn_success_rate":summaries["rnn_only_critic"]["success_curve"].get(step),"multi_success_rate":summaries["multi_il_critic"]["success_curve"].get(step),"rnn_mean_progress":summaries["rnn_only_critic"]["progress_curve"].get(step),"multi_mean_progress":summaries["multi_il_critic"]["progress_curve"].get(step)})
    result={"stage":"Stage4","active_groups":active_groups,"random_critic_started":False,"run_directory":str(run),"pomdp_baselines_commit":config["pomdp_baselines_commit"],"actor_hash":next(iter(actor_hashes)),"actor_hashes_identical":True,"initialization_audits":audits,"config":config,"groups":summaries,"learning_curve_csv":str(curve_path),"stage5_started":False,"extra_seeds_started":False}
    atomic_json(run/"stage4_comparison.json",result)
    metrics=[("Step0 SR",lambda x:x["evaluation_points"]["0"]),("30k SR",lambda x:x["evaluation_points"]["30000"]),("50k SR",lambda x:x["evaluation_points"]["50000"]),("100k SR",lambda x:x["evaluation_points"]["100000"]),("200k SR",lambda x:x["evaluation_points"]["200000"]),("500k SR",lambda x:x["evaluation_points"]["500000"]),("1M SR",lambda x:x["evaluation_points"]["1000000"]),("Initial degradation",lambda x:x["initial_degradation"]),("Recovery step",lambda x:x["recovery_step"]),("First SR>=0.4",lambda x:x["threshold_crossings"]["0.4"]),("First SR>=0.5",lambda x:x["threshold_crossings"]["0.5"]),("First SR>=0.6",lambda x:x["threshold_crossings"]["0.6"]),("AUC 0-200k",lambda x:x["auc_0_200k"]),("AUC 0-500k",lambda x:x["auc_0_500k"]),("AUC 0-1M",lambda x:x["auc_0_1m"]),("Best SR",lambda x:x["best_success_rate"]),("Best step",lambda x:x["best_success_step"]),("Final SR",lambda x:x["final_success_rate"])]
    lines=["# Stage4 comparison","","Random Critic was not launched in this run.","","| Metric | RNN-only | Multi-IL |","|---|---:|---:|"]
    for label,getter in metrics:lines.append("| "+label+" | "+" | ".join(str(getter(summaries[group])) for group in active_groups)+" |")
    lines += ["","RANDOM CRITIC NOT STARTED","","NO STAGE5 STARTED","","NO EXTRA SEEDS STARTED",""];(run/"stage4_comparison.md").write_text("\n".join(lines),encoding="utf-8");print("Stage4 comparison:",run/"stage4_comparison.json")
if __name__=="__main__":main()
