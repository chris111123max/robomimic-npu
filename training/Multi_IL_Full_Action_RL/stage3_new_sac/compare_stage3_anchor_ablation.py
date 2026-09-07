#!/usr/bin/env python3
"""Three-stage ablation: vanilla vs CQL-lite vs CQL-lite + Stage2 anchor."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from compare_stage3_cql_baseline import evaluations,metric_rows,source_final

def last_jsonl(path):
    last=None
    if path.is_file():
        with open(path,encoding="utf-8") as f:
            for line in f:last=json.loads(line)
    return last
def main():
    p=argparse.ArgumentParser();p.add_argument("--vanilla-pair-run-dir",required=True);p.add_argument("--cql-pair-run-dir",required=True);p.add_argument("--anchor-pair-run-dir",required=True);p.add_argument("--output");a=p.parse_args();runs={"vanilla":Path(a.vanilla_pair_run_dir).resolve(),"cql_lite":Path(a.cql_pair_run_dir).resolve(),"cql_lite_anchor":Path(a.anchor_pair_run_dir).resolve()};groups={}
    for group in ("rnn_q","multi_q"):
        metrics={name:metric_rows(run/group) for name,run in runs.items()};steps=sorted(set().union(*(set(rows) for rows in metrics.values())));groups[group]={"milestones":{str(step):{name:rows.get(step) for name,rows in metrics.items()} for step in steps},"evaluation_success_rate":{name:evaluations(run/group) for name,run in runs.items()},"source_diagnostics_final":{name:source_final(run,group) for name,run in runs.items()},"anchor_diagnostics_final":last_jsonl(runs["cql_lite_anchor"]/group/"anchor_diagnostics.jsonl")}
    result={"runs":{k:str(v) for k,v in runs.items()},"groups":groups};output=Path(a.output) if a.output else runs["cql_lite_anchor"]/"pair_comparison"/"vanilla_vs_cql_vs_anchor.json";output.parent.mkdir(parents=True,exist_ok=True)
    with open(output,"w",encoding="utf-8") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps({"output":str(output.resolve()),"groups":list(groups)},indent=2))
if __name__=="__main__":main()
