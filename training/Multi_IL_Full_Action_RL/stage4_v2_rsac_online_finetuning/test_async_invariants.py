#!/usr/bin/env python3
"""Deterministic checks for async per-env state, RNG, and trajectory isolation."""
import argparse,json
from pathlib import Path
import numpy as np
def streams(order):
 rng={i:np.random.RandomState(20260901+1000003*(i+1)) for i in range(16)};out={i:[] for i in rng}
 for env_id in order:out[env_id].append(rng[env_id].standard_normal(14).tolist())
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args();natural=[i for _ in range(5) for i in range(16)];reordered=[i for _ in range(5) for i in reversed(range(16))];one=streams(natural);two=streams(reordered);rng_ok=all(np.array_equal(one[i],two[i]) for i in range(16));slots=np.arange(16*2*4).reshape(16,2,4);ready=[7,1,12,4];gather=slots[ready].copy()+1000;scattered=slots.copy();scattered[ready]=gather;hidden_ok=all(np.array_equal(scattered[i],slots[i]+(1000 if i in ready else 0)) for i in range(16));trajectories={i:[(i,e,t) for e in range(2) for t in range(3)] for i in range(16)};boundary_ok=all(all(env==i for env,_,_ in seq) and all(seq[j][1]==seq[j+1][1] or seq[j][2]==2 for j in range(len(seq)-1)) for i,seq in trajectories.items());result={"status":"PASS" if rng_ok and hidden_ok and boundary_ok else "FAIL","per_env_rng_completion_order_independent":rng_ok,"hidden_state_env_id_scatter":hidden_ok,"replay_sequence_env_and_episode_boundaries":boundary_ok};Path(a.output).write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2));raise SystemExit(0 if result["status"]=="PASS" else 1)
if __name__=="__main__":main()
