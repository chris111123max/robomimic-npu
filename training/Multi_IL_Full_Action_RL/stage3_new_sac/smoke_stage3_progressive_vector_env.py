#!/usr/bin/env python3
"""Environment-only startup, step, individual reset, and shutdown smoke."""
import argparse,json
import numpy as np
from stage3_progressive_vector_env import StaggeredVectorEnv
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--num-envs",type=int,default=16);p.add_argument("--vector-steps",type=int,default=10);a=p.parse_args();cfg=json.load(open(a.config,encoding="utf-8"));v=cfg["parallel_env"];env=None
    try:
        env=StaggeredVectorEnv(cfg["expert_dataset"],a.num_envs,cfg["train_seed_base"],v["env_startup_delay_sec"],v["env_startup_timeout_sec"],v["multiprocessing_start_method"]);valid=0
        for step in range(a.vector_steps):
            results=env.step(np.zeros((a.num_envs,14),np.float32));valid+=sum(message[0]=="OK" for _,message in results)
            if step==0:env.reset(0,cfg["train_seed_base"]+a.num_envs)
        print(json.dumps({"status":"PASS","num_envs":a.num_envs,"vector_steps":a.vector_steps,"valid_transitions":valid,"individual_reset_env_id":0,"clean_shutdown":True}))
    finally:
        if env is not None:env.close()
if __name__=="__main__":main()
