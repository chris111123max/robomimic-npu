#!/usr/bin/env python3
"""Emit explicit instructions/contract for matched Stage2.1 comparison."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--history-validation",required=True);p.add_argument("--memoryless-validation",required=True);p.add_argument("--output",required=True);a=p.parse_args();h=json.loads(Path(a.history_validation).read_text());m=json.loads(Path(a.memoryless_validation).read_text());policies=("bc_rnn","bc_transformer","bc_gmm");report={"protocol":"same seeds 10080-10099, same finite MC target, same policy holdouts","policies":{p:{"history":h[p],"memoryless":m[p],"twin_mean_mse_delta":h[p]["twin_mean_mse"]-m[p]["twin_mean_mse"]} for p in policies}};Path(a.output).write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
