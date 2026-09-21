#!/usr/bin/env python3
"""Summarize recorded Stage2.2 timing without launching formal training."""
import argparse,json
from pathlib import Path
import numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument("metrics");a=p.parse_args();rows=[json.loads(x) for x in Path(a.metrics).read_text().splitlines() if x.strip()];keys=("sample_ms","forward_ms","backward_ms","optimizer_ms");report={k:{"mean":float(np.mean([r[k] for r in rows])),"p95":float(np.percentile([r[k] for r in rows],95))} for k in keys};report["effective_timesteps_per_second"]=float(sum(r["effective_timesteps"] for r in rows)/(sum(sum(r[k] for k in keys) for r in rows)/1000));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
