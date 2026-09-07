#!/usr/bin/env python3
"""Compare matched Stage3-new probes and learning curves for the pair."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def stats(x):
    x=np.asarray(x,float);return {"mean":float(x.mean()),"std":float(x.std()),"median":float(np.median(x)),"p10":float(np.percentile(x,10)),"p90":float(np.percentile(x,90)),"p95":float(np.percentile(x,95))}
def ranks(x):
    x=np.asarray(x);order=np.argsort(x,kind="mergesort");r=np.empty(len(x),float);values=x[order];start=0
    for end in range(1,len(x)+1):
        if end==len(x) or values[end]!=values[start]:r[order[start:end]]=(start+end-1)/2;start=end
    return r
def corr(a,b):
    a,b=np.asarray(a).reshape(-1),np.asarray(b).reshape(-1);return None if len(a)<2 or np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])
def cosine(a,b):
    denom=np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12);return stats(np.sum(a*b,axis=1)/denom)
def geometry(a,b):
    a,b=np.asarray(a,float).reshape(-1),np.asarray(b,float).reshape(-1);ac,bc=a-a.mean(),b-b.mean();az=ac/max(a.std(),1e-12);bz=bc/max(b.std(),1e-12)
    return {"centered_pearson":corr(ac,bc),"centered_spearman":corr(ranks(ac),ranks(bc)),"zscore_pearson":corr(az,bz),"zscore_spearman":corr(ranks(az),ranks(bz))}
def compare_npz(left_path,right_path):
    result={}
    with np.load(left_path) as left,np.load(right_path) as right:
        probes=sorted({key.split("__",1)[0] for key in left.files})
        for probe in probes:
            row={}
            for head in ("q1","q2","qmin"):
                a,b=left[f"{probe}__{head}"].reshape(-1),right[f"{probe}__{head}"].reshape(-1);mean_a,mean_b=float(a.mean()),float(b.mean());absolute=abs(mean_a-mean_b);row[head]={"mean_abs_difference":float(np.mean(np.abs(a-b))),"pearson":corr(a,b),"spearman":corr(ranks(a),ranks(b)),"mean_q_rnn":mean_a,"mean_q_multi":mean_b,"absolute_mean_difference":absolute,"relative_mean_difference":float(absolute/max(abs(mean_a),abs(mean_b),1e-12)),"normalized_geometry":geometry(a,b)}
                if head=="qmin":row["qmin_absolute_scale"]={"mean_qmin_rnn":mean_a,"mean_qmin_multi":mean_b,"absolute_difference":absolute,"relative_difference":float(absolute/max(abs(mean_a),abs(mean_b),1e-12))}
            a,b=left[f"{probe}__mu"],right[f"{probe}__mu"];row["actor_action_l2_difference"]=stats(np.linalg.norm(a-b,axis=1));row["gradient_cosine_q1"]=cosine(left[f"{probe}__grad_q1"],right[f"{probe}__grad_q1"]);row["gradient_cosine_q2"]=cosine(left[f"{probe}__grad_q2"],right[f"{probe}__grad_q2"]);result[probe]=row
    return result
def load_evaluations(group):
    rows=[]
    for path in sorted((group/"evaluations").glob("step_*.json")):
        with open(path,encoding="utf-8") as f:row=json.load(f)
        row["env_steps"]=int(path.stem.split("_")[1]);rows.append(row)
    return rows
def load_source_diagnostics(group):
    path=group/"source_diagnostics.jsonl";last=None
    if path.exists():
        with open(path,encoding="utf-8") as f:
            for line in f:last=json.loads(line)
    return last
def load_jsonl_last(path):
    last=None
    if path.exists():
        with open(path,encoding="utf-8") as f:
            for line in f:last=json.loads(line)
    return last
def compare_pair(pair):
    pair=Path(pair);left,right=pair/"rnn_q",pair/"multi_q";left_steps={p.stem.split("_")[1]:p for p in (left/"probes").glob("step_*.npz")};right_steps={p.stem.split("_")[1]:p for p in (right/"probes").glob("step_*.npz")}
    probes={step:compare_npz(left_steps[step],right_steps[step]) for step in sorted(set(left_steps)&set(right_steps),key=int)};curves={"rnn_q":load_evaluations(left),"multi_q":load_evaluations(right)}
    output=pair/"pair_comparison";output.mkdir(exist_ok=True);write(output/"critic_washout.json",probes);write(output/"learning_curve_comparison.json",curves);summary={"pair_run_dir":str(pair),"matched_probe_steps":[int(x) for x in probes],"rnn_q_evaluations":len(curves["rnn_q"]),"multi_q_evaluations":len(curves["multi_q"]),"source_diagnostics_final":{"rnn_q":load_source_diagnostics(left),"multi_q":load_source_diagnostics(right)},"anchor_diagnostics_final":{"rnn_q":load_jsonl_last(left/"anchor_diagnostics.jsonl"),"multi_q":load_jsonl_last(right/"anchor_diagnostics.jsonl")},"final_probe_pair_geometry":probes.get("300000")};write(output/"stage3_new_summary.json",summary);return summary
def write(path,value):
    with open(path,"w",encoding="utf-8") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n")
def main():
    p=argparse.ArgumentParser();p.add_argument("--pair-run-dir",required=True);print(json.dumps(compare_pair(p.parse_args().pair_run_dir),indent=2))
if __name__=="__main__":main()
