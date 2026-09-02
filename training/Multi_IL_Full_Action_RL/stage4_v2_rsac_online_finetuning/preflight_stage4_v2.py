#!/usr/bin/env python3
import argparse,json,math,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];V1=ROOT/"stage4_rsac_online_finetuning";sys.path.insert(0,str(V1))
import numpy as np
from preflight_stage4 import main as unused  # noqa
from stage4_core import OnlineSequenceReplay,Stage4SAC,assert_phase_contract,atomic_json,initialize_models,model_audit,phase_at,read_json,seed_all,select_device,state_hash
def main():
 p=argparse.ArgumentParser();p.add_argument("--group",required=True);p.add_argument("--device",required=True);p.add_argument("--config",required=True);p.add_argument("--output",required=True);a=p.parse_args();c=read_json(a.config);c.update(group=a.group,device=a.device);assert_phase_contract(c)
 assert c["total_env_steps"]==100000 and c["actor_freeze_steps"]==5000 and c["actor_warmup_end"]==20000
 assert c["parallel_envs"]==16 and c["collector_mode"]=="async_ready_queue"
 assert c["initial_entropy_alpha"]==0.001 and c["automatic_entropy_tuning"] is False
 assert c["automatic_entropy_tuning"] is False and c["initial_entropy_alpha"]==.001 and c["sequence_length"]==10
 before=phase_at(4999,c);boundary=phase_at(5000,c);middle=phase_at(10000,c);end=phase_at(20000,c)
 assert before["actor_updates"] is False and boundary["actor_updates"] is False and boundary["actor_lr"]==0.0
 assert middle["actor_updates"] is True and math.isclose(middle["actor_lr"],1e-4,rel_tol=0.0,abs_tol=1e-15)
 assert end["actor_updates"] is True and math.isclose(end["actor_lr"],float(c["actor_target_lr"]),rel_tol=0.0,abs_tol=1e-15)
 d=select_device(a.device);seed_all(c["seed"]);actor,payload,critic,target,source=initialize_models(a.group,c,d);audit=model_audit(a.group,actor,critic,target,source,c);replay=OnlineSequenceReplay({**c,"replay_capacity":1000})
 if replay.transitions or replay.episodes:raise RuntimeError("Replay is not empty at step0")
 obs=np.zeros((12,59),np.float32);act=np.zeros((12,14),np.float32);rew=np.zeros(12,np.float32);done=np.zeros(12,np.uint8);done[-1]=1;replay.add_episode(obs,act,rew,done,obs,done,np.zeros_like(done));engine=Stage4SAC(actor,critic,target,c,d)
 if engine.algo.automatic_entropy_tuning or hasattr(engine.algo,"alpha_entropy_optim"):raise RuntimeError("Stage4-v2 unexpectedly created alpha optimizer")
 before_alpha=engine.algo.alpha_entropy;before_actor=state_hash(actor);before_target=state_hash(target)
 with tempfile.TemporaryDirectory() as temp:stats=engine.update(replay.sample(2,d),6000,Path(temp))
 if engine.algo.alpha_entropy!=before_alpha or before_alpha!=.001:raise RuntimeError("Fixed alpha changed")
 if state_hash(actor)==before_actor or state_hash(target)==before_target:raise RuntimeError("Actor/target finite update did not execute")
 audit.update(status="PASS",replay_empty_at_step0=True,sequence_length=10,freeze_steps=5000,warmup="5000->20000",automatic_entropy_tuning=False,fixed_alpha=.001,alpha_optimizer_exists=False,actor_loss="alpha*log_pi-min(Q1,Q2)",critic_update_finite=bool(np.isfinite(stats["critic_loss"])),actor_update_finite=bool(np.isfinite(stats["actor_loss"])),target_update_finite=True)
 atomic_json(a.output,audit);print(json.dumps(audit,indent=2))
if __name__=="__main__":main()
