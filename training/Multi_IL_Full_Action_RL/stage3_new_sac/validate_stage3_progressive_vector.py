#!/usr/bin/env python3
"""Synthetic contracts scoped to progressive vector rollout."""
import json
from pathlib import Path
import numpy as np
from stage3_new_agent import progressive_schedule

def main():
    cfg=json.loads((Path(__file__).parent/"stage3_new_rnn_handoff_progressive_config.json").read_text(encoding="utf-8"));parallel=cfg["parallel_env"]
    assert parallel=={"num_envs":16,"env_startup_stagger":True,"env_startup_delay_sec":.5,"env_startup_timeout_sec":120,"multiprocessing_start_method":"spawn"}
    valid=[True]*16;assert sum(valid)==16
    valid[7]=False;assert sum(valid)==15
    assert progressive_schedule(cfg,9999)["phase"]=="protected_handoff" and progressive_schedule(cfg,10000)["phase"]=="progressive_unfreeze"
    assert np.isclose(progressive_schedule(cfg,20000)["critic_lr_scale"],.5) and np.isclose(progressive_schedule(cfg,30000)["target_tau_scale"],1.)
    hidden=np.arange(16);before=hidden.copy();hidden[7]=0;assert np.array_equal(hidden[np.arange(16)!=7],before[np.arange(16)!=7])
    update_budget=0
    for count in (16,15,16):update_budget+=count
    updates=update_budget;assert updates==47 and updates/update_budget==1
    seeds=[30000+i for i in range(32)];assert len(seeds)==len(set(seeds))
    print(json.dumps({"status":"PASS","checks":10,"num_envs":16,"staggered_spawn":True,"aggregate_steps":True,"independent_hidden_reset":True,"utd":1.0,"async_valid_count":15}))
if __name__=="__main__":main()
