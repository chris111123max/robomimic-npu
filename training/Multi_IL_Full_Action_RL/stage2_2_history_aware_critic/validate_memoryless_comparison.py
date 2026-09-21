#!/usr/bin/env python3
"""Compare Stage2.2 history-aware and in-directory matched baseline results."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--history-validation",required=True);p.add_argument("--memoryless-validation",required=True);p.add_argument("--output",required=True);a=p.parse_args();h=json.loads(Path(a.history_validation).read_text());m=json.loads(Path(a.memoryless_validation).read_text());policies=("bc_rnn","bc_transformer","bc_gmm");report={"protocol":"matched Stage2.2 control: same seeds, data, MC target, previous action, progress, optimizer, budget, and evaluation; only multi-step recurrence differs","policies":{p:{"history":h[p],"matched_memoryless":m[p],"history_minus_memoryless_twin_mean_mse":h[p]["twin_mean_mse"]-m[p]["twin_mean_mse"],"history_minus_memoryless_spearman":None if h[p]["spearman"] is None or m[p]["spearman"] is None else h[p]["spearman"]-m[p]["spearman"]} for p in policies}};Path(a.output).write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
