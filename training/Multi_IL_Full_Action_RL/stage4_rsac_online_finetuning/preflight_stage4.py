#!/usr/bin/env python3
"""Non-environment Stage4 preflight for one group/device."""
import argparse,json,sys,tempfile
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from stage4_core import (OnlineSequenceReplay,Stage4SAC,assert_phase_contract,atomic_json,
                         initialize_models,model_audit,read_json,seed_all,select_device,state_hash)  # noqa
from train_stage4_group import build_environment,flatten  # noqa
from evaluate_stage3_actor import close_env,load_initial_states  # noqa
def main():
    p=argparse.ArgumentParser();p.add_argument("--group",required=True);p.add_argument("--device",required=True);p.add_argument("--config",default=str(HERE/"stage4_config.json"));p.add_argument("--output",required=True);a=p.parse_args()
    config=read_json(a.config);config.update(group=a.group,device=a.device);assert_phase_contract(config);device=select_device(a.device);seed_all(config["seed"])
    actor,payload,critic,target,source=initialize_models(a.group,config,device);audit=model_audit(a.group,actor,critic,target,source,config)
    if payload["architecture"]!={"state_dim":59,"action_dim":14,"lstm_hidden_size":400,"lstm_num_layers":2,"batch_first":True,"horizon":10,"sac_head":"pomdp-baselines TanhGaussianPolicy"}:raise RuntimeError("Stage3-R Actor architecture mismatch")
    zero=torch.zeros((1,59),device=device);actor.reset()
    with torch.no_grad():first=actor.act(zero,deterministic=True)[0]
    actor.reset()
    with torch.no_grad():second=actor.act(zero,deterministic=True)[0]
    if not torch.equal(first,second):raise RuntimeError("Actor hidden reset is not deterministic")
    initial=load_initial_states(config["initial_states"],[10080]);env=build_environment(payload)
    try:
        observation=env.reset_to(initial[10080]);state,_=flatten(observation,payload["observation_keys"]);actor.reset()
        with torch.no_grad():environment_action=actor.act(torch.as_tensor(state[None],device=device),deterministic=True)[0][0]
    finally:close_env(env)
    replay=OnlineSequenceReplay({**config,"replay_capacity":1000});obs=np.zeros((12,59),np.float32);act=np.zeros((12,14),np.float32);rew=np.zeros(12,np.float32);done=np.zeros(12,np.uint8);done[-1]=1
    replay.add_episode(obs,act,rew,done,obs,terminated=done,truncated=np.zeros_like(done));batch=replay.sample(2,device)
    if tuple(batch["obs"].shape)!=(10,2,59) or tuple(batch["act"].shape)!=(10,2,14):raise RuntimeError("Sequence replay batch shape mismatch")
    engine=Stage4SAC(actor,critic,target,config,device);before=state_hash(target)
    with tempfile.TemporaryDirectory() as temp:stats=engine.update(batch,60000,Path(temp))
    after=state_hash(target)
    if before==after:raise RuntimeError("Target Critic did not update")
    if not all(value is None or not isinstance(value,float) or np.isfinite(value) for value in stats.values()):raise RuntimeError("Non-finite preflight update")
    audit.update(status="PASS",deterministic_zero_action=first[0].cpu().numpy().tolist(),deterministic_seed10080_action=environment_action.cpu().numpy().tolist(),sequence_batch_shape=list(batch["obs"].shape),critic_update_finite=True,actor_update_finite=True,alpha_update_finite=True,target_update_finite=True,phase_contract=True)
    atomic_json(a.output,audit);print(json.dumps(audit,indent=2))
if __name__=="__main__":main()
