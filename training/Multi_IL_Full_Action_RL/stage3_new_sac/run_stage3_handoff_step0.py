#!/usr/bin/env python3
"""No-update step-0 pure-RNN or target-Q arbitration evaluation."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import torch
from stage3_new_agent import Stage3SAC,build_actor,strict_stage2_load
from stage3_new_evaluation import build_env,close_env
from stage3_new_handoff import FrozenRNNProposer,evaluate_handoff
from train_stage3_new import device

def read(path):
    with open(path,encoding="utf-8") as f:return json.load(f)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp,"w",encoding="utf-8") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n")
    os.replace(tmp,path)
def update_comparison(out):
    names=("pure_rnn","rnn_q_selector","multi_q_selector");rows={name:read(out/f"{name}.json") for name in names if (out/f"{name}.json").is_file()}
    if rows:write(out/"comparison.json",{name:{key:row.get(key) for key in ("success_count","success_rate","mean_return","mean_length","rnn_selected_fraction","rl_selected_fraction","q_rnn_mean","q_rl_mean","q_margin_rl_minus_rnn_mean","q_margin_median","q_margin_p10","q_margin_p90","tie_count")} for name,row in rows.items()})
def main():
    p=argparse.ArgumentParser();p.add_argument("--pair-run-dir",required=True);p.add_argument("--mode",required=True,choices=("pure_rnn","rnn_q","multi_q"));p.add_argument("--device",required=True);a=p.parse_args();pair=Path(a.pair_run_dir).resolve();config=read(pair/"shared"/"config_resolved.json");sources=read(pair/"shared"/"stage2_source_manifest.json");d=device(a.device);env=build_env(config["expert_dataset"]);proposer=FrozenRNNProposer(config["bc_rnn_checkpoint"],d);seeds=config["step0_arbitration_seeds"];out=pair/"step0_arbitration"
    try:
        if a.mode=="pure_rnn":report=evaluate_handoff(None,None,proposer,env,seeds,config["horizon"],d,config["sim_error_handling"]["evaluation_retry_count"],True);name="pure_rnn"
        else:
            actor=build_actor(config,d);payload=torch.load(pair/"shared"/"actor_init.pth",map_location=d);actor.load_state_dict(payload["actor_state_dict"],strict=True);critic,_=strict_stage2_load(sources[a.mode]["checkpoint"],d,config);agent=Stage3SAC(actor,critic,config,d);report=evaluate_handoff(actor,agent.target,proposer,env,seeds,config["horizon"],d,config["sim_error_handling"]["evaluation_retry_count"]);name=f"{a.mode}_selector"
        report.update({"mode":name,"gradient_updates":0,"seeds":seeds,"actor_init_path":str((pair/"shared"/"actor_init.pth").resolve()),"bc_rnn_checkpoint":config["bc_rnn_checkpoint"],"critic_checkpoint":None if a.mode=="pure_rnn" else sources[a.mode]["checkpoint"]});write(out/f"{name}.json",report);update_comparison(out);print(json.dumps({"output":str((out/f'{name}.json').resolve()),"success_rate":report["success_rate"],"rnn_selected_fraction":report.get("rnn_selected_fraction"),"rl_selected_fraction":report.get("rl_selected_fraction")},indent=2))
    finally:close_env(env)
if __name__=="__main__":main()
