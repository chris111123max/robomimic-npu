#!/usr/bin/env python3
"""Evaluate one Stage3-R recurrent actor shard from exact Stage1 initial states."""
from __future__ import annotations
import argparse,copy,json,random,sys
from pathlib import Path
import h5py,numpy as np,torch
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent;S1=PROJECT/"stage1_rollout_collection";S3=PROJECT/"stage3_actor_initialization"
for p in (HERE,S1,S3):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from stage3_r_actor import load_actor,set_vendor_device  # noqa
from common import extract_canonical_observation  # noqa
from collect_multi_il_rollouts import progress_observation_schema  # noqa
from evaluate_stage3_actor import load_initial_states,env_success,extract_transport_progress,partial_progress_score,close_env  # noqa
import robomimic.utils.file_utils as FileUtils  # noqa
import robomimic.utils.obs_utils as ObsUtils  # noqa
def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if hasattr(torch,"npu") and torch.npu.is_available():torch.npu.manual_seed_all(s)
def device_of(name):
    if name.startswith("npu"):
        import torch_npu  # noqa
        torch.npu.set_device(name)
    return torch.device(name)
def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--seed-start",type=int,required=True);p.add_argument("--num-seeds",type=int,required=True);p.add_argument("--device",default="cpu");p.add_argument("--output",required=True);a=p.parse_args()
    device=device_of(a.device);set_vendor_device(device);actor,payload=load_actor(a.checkpoint,device);keys=payload["observation_keys"]
    shapes={"robot0_eef_pos":[3],"robot0_eef_quat":[4],"robot0_gripper_qpos":[2],"robot1_eef_pos":[3],"robot1_eef_quat":[4],"robot1_gripper_qpos":[2],"object":[41]}
    dataset=Path(payload["source_dataset"]);initial=dataset.parent.parent/"initial_states.hdf5";seeds=list(range(a.seed_start,a.seed_start+a.num_seeds));states=load_initial_states(initial,seeds)
    ck=FileUtils.maybe_dict_from_checkpoint(ckpt_path=payload["teacher_checkpoint"]);cfg,_=FileUtils.config_from_checkpoint(ckpt_dict=ck);ObsUtils.initialize_obs_utils_with_config(cfg);env,_=FileUtils.env_from_checkpoint(ckpt_dict=ck,render=False,render_offscreen=False,verbose=False)
    schema=progress_observation_schema(env,keys,shapes);rows=[];horizon=int(payload["evaluation_horizon"]);terminate=bool(payload["terminate_on_success"])
    try:
        for seed in seeds:
            seed_all(seed);obs=env.reset_to(copy.deepcopy(states[seed]));seed_all(seed);actor.reset();ret=0.;ever_t=ever_p=final_t=final_p=False;success=env_success(env);raw_done=False
            for step in range(horizon):
                canonical=extract_canonical_observation(obs,keys,shapes);flat=np.concatenate([canonical[k].reshape(-1) for k in keys]).astype(np.float32);x=torch.as_tensor(flat[None],device=device)
                with torch.no_grad():action=actor.act(x,deterministic=True)[0][0].cpu().numpy()
                obs,reward,raw_done,_=env.step(action);ret+=float(reward);nxt=extract_canonical_observation(obs,keys,shapes);progress=extract_transport_progress(nxt,schema);final_t=progress["trash_in_trash_bin"];final_p=progress["payload_in_target_bin"];ever_t|=final_t;ever_p|=final_p;success=env_success(env)
                if raw_done or (terminate and success):break
            length=step+1;rows.append({"initial_seed":seed,"success":int(success),"ever_trash_in_bin":bool(ever_t),"final_trash_in_bin":bool(final_t),"ever_payload_in_bin":bool(ever_p),"final_payload_in_bin":bool(final_p),"partial_progress_score":partial_progress_score(success,ever_t,ever_p),"episode_return":ret,"episode_length":length,"terminated":bool(raw_done),"truncated":bool(not raw_done)})
            print(f"seed={seed} success={int(success)} progress={rows[-1]['partial_progress_score']} length={length}",flush=True)
    finally:close_env(env)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,"w",encoding="utf-8") as f:json.dump({"checkpoint":a.checkpoint,"episodes":rows,"device":str(device),"horizon_reset_interval":10},f,indent=2)
if __name__=="__main__":main()
