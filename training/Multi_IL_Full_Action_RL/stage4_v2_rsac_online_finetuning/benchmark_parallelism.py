#!/usr/bin/env python3
"""Rollout-only 8-vs-16 persistent-worker benchmark; no replay or learning."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent;V1=PROJECT/"stage4_rsac_online_finetuning"
for path in (V1,PROJECT/"stage1_rollout_collection",PROJECT/"stage3_actor_initialization"):
 if str(path) not in sys.path:sys.path.insert(0,str(path))
from stage4_core import initialize_models,read_json,select_device,seed_all
from stage4_parallel_env import Stage4ParallelEnvPool
from train_stage4_group import SHAPES,batched_actor_actions
def main():
 p=argparse.ArgumentParser();p.add_argument("--workers",type=int,choices=(8,16),required=True);p.add_argument("--transitions",type=int,default=4000);p.add_argument("--device",default="npu:0");p.add_argument("--config",required=True);p.add_argument("--affinity",required=True);p.add_argument("--output",required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.set_num_interop_threads(1);cfg=read_json(a.config);cfg.update(group="rnn_only_critic",device=a.device);device=select_device(a.device);seed_all(cfg["seed"]);actor,payload,_,_,_=initialize_models("rnn_only_critic",cfg,device);actor.eval();ids=read_json(a.affinity)["rnn_cpu_ids"][:a.workers]
 pool=Stage4ParallelEnvPool(payload["teacher_checkpoint"],payload["observation_keys"],SHAPES,a.workers,cfg["evaluation_horizon"],cfg["terminate_on_success"],cfg["seed"],cfg["parallel_start_method"],cfg["parallel_startup_timeout_seconds"],cfg["parallel_step_timeout_seconds"],ids);hidden=(torch.zeros(2,a.workers,400,device=device),torch.zeros(2,a.workers,400,device=device));counters=np.zeros(a.workers,np.int64);obs=pool.reset(range(a.workers),[cfg["seed"]+i for i in range(a.workers)]);done=0;infer=rollout=0.;cpu0=pool.worker_cpu_seconds();start=time.perf_counter()
 try:
  while done<a.transitions:
   workers=list(range(min(a.workers,a.transitions-done)));t=time.perf_counter();actions=batched_actor_actions(actor,obs,hidden,counters,workers,device);infer+=time.perf_counter()-t;t=time.perf_counter();results=pool.step(workers,actions);rollout+=time.perf_counter()-t
   for w in workers:
    obs[w]=results[w][0]
    if results[w][2]:obs.update(pool.reset([w],[cfg["seed"]+a.workers+done+w]));counters[w]=0;hidden[0][:,w].zero_();hidden[1][:,w].zero_()
   done+=len(workers)
 finally:
  elapsed=time.perf_counter()-start;worker_cpu=pool.worker_cpu_seconds()-cpu0;pool.close()
 result={"workers":a.workers,"transitions":done,"wall_seconds":elapsed,"transitions_per_second":done/elapsed,"rollout_seconds":rollout,"policy_inference_seconds":infer,"worker_cpu_percentage":100.0*worker_cpu/elapsed,"cpu_ids":ids,"learning_performed":False};Path(a.output).write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
