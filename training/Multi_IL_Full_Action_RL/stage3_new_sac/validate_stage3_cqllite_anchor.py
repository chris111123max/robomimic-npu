#!/usr/bin/env python3
"""Synthetic validation for CQL-lite + frozen Stage2 z-score anchor."""
from __future__ import annotations
import copy,json,sys,types
from pathlib import Path
import numpy as np,torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];S2=ROOT/"training"/"Multi_IL_Full_Action_RL"/"stage2_new_critic_pretraining"
if str(S2) not in sys.path:sys.path.insert(0,str(S2))
from critic_network import build_critic
from stage3_new_agent import Stage3SAC,build_actor
from stage3_new_evaluation import KEYS,evaluate

def config(cql=True,anchor=True):
    return {"hidden_dims":[256,256],"actor_lr":3e-4,"critic_lr":3e-4,"alpha_lr":3e-4,"critic_weight_decay":1e-4,"gamma":.99,"tau":.005,"target_entropy":-14.,"alpha_init":.01,"cql":{"enabled":cql,"lambda":.1,"num_random_actions":10,"num_policy_actions":1,"apply_to_expert":True,"apply_to_online":True,"detach_policy_actions":True},"anchor":{"enabled":anchor,"lambda":.1,"batch_size":256,"eps":1e-6}}
def replay_batch(n=16):
    rng=np.random.default_rng(7);return {"observations":rng.normal(size=(n,59)).astype(np.float32),"actions":rng.uniform(-1,1,size=(n,14)).astype(np.float32),"rewards":rng.integers(0,2,size=(n,1)).astype(np.float32),"next_observations":rng.normal(size=(n,59)).astype(np.float32),"terminals":rng.integers(0,2,size=(n,1)).astype(np.float32)}
def anchor_batch(n=256):
    rng=np.random.default_rng(9);return {"state":rng.normal(size=(n,59)).astype(np.float32),"action":rng.uniform(-1,1,size=(n,14)).astype(np.float32)}
def new_agent(cfg,teacher=True):
    actor=build_actor(cfg,"cpu");critic=build_critic(59,14,[256,256],"relu",True,"cpu");frozen=copy.deepcopy(critic) if teacher else None;return Stage3SAC(actor,critic,cfg,"cpu",teacher=frozen)

class FatalError(Exception):pass
class FakeActor:
    def __call__(self,state,deterministic=False):return (torch.zeros((len(state),14)),)
class FakeEnv:
    def __init__(self):self.failed=False
    def reset(self):return self.obs()
    def seed(self,seed):pass
    def obs(self):
        widths=(3,4,2,3,4,2,41);return {key:np.zeros(width,np.float32) for key,width in zip(KEYS,widths)}
    def is_success(self):return {"task":False}
    def step(self,action):
        if not self.failed:self.failed=True;raise FatalError("synthetic narrowphase")
        return self.obs(),0.0,True,{}

def main():
    teacher=torch.tensor([0.,1.,2.,3.]);tests={"offset":teacher+100,"positive_scale":5*teacher+20,"reversed":teacher.flip(0),"negative_scale":-teacher}
    losses={name:float(torch.nn.functional.mse_loss(Stage3SAC.zscore(value),Stage3SAC.zscore(teacher))) for name,value in tests.items()}
    assert losses["offset"]<1e-10 and losses["positive_scale"]<1e-10 and losses["reversed"]>1 and losses["negative_scale"]>1
    torch.manual_seed(1);cfg=config();agent=new_agent(cfg)
    with torch.no_grad():next(agent.critic.parameters()).add_(torch.randn_like(next(agent.critic.parameters()))*.01)
    values=agent.anchor_components(anchor_batch());agent.actor_optimizer.zero_grad(set_to_none=True);agent.critic_optimizer.zero_grad(set_to_none=True);values["loss"].backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in agent.critic.parameters());assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in agent.actor.parameters());assert all(p.grad is None for p in agent.teacher.parameters());optimizer_ids={id(p) for g in agent.critic_optimizer.param_groups for p in g["params"]};assert not any(id(p) in optimizer_ids for p in agent.teacher.parameters())
    metrics=agent.update(replay_batch(),anchor_batch());assert all(np.isfinite(metrics[k]) for k in ("critic_loss_total","cql_loss_weighted","anchor_loss_weighted","anchor_pearson_q1","anchor_pearson_q2"))
    torch.manual_seed(2);cql_only_cfg=config(True,False);cql_only=new_agent(cql_only_cfg,False);cql_metrics=cql_only.update(replay_batch());assert cql_metrics["anchor_loss_weighted"]==0
    torch.manual_seed(22);legacy_cfg=config(True,False);legacy_cfg.pop("anchor");legacy=new_agent(legacy_cfg,False);legacy_metrics=legacy.update(replay_batch());torch.manual_seed(22);disabled=new_agent(config(True,False),False);disabled_metrics=disabled.update(replay_batch());assert abs(legacy_metrics["critic_loss_total"]-disabled_metrics["critic_loss_total"])<1e-7
    torch.manual_seed(3);vanilla=new_agent(config(False,False),False);raw=replay_batch();state=torch.get_rng_state();tb={k:torch.as_tensor(v) for k,v in raw.items()};target=vanilla.target_components(tb)["td_target"];q1,q2=vanilla.critic(tb["observations"],tb["actions"]);expected=float(torch.nn.functional.mse_loss(q1,target)+torch.nn.functional.mse_loss(q2,target));torch.set_rng_state(state);vanilla_metrics=vanilla.update(raw);assert abs(vanilla_metrics["critic_loss_total"]-expected)<1e-6 and vanilla_metrics["cql_loss_weighted"]==0 and vanilla_metrics["anchor_loss_weighted"]==0
    previous=sys.modules.get("mujoco");sys.modules["mujoco"]=types.SimpleNamespace(FatalError=FatalError)
    try:report=evaluate(FakeActor(),FakeEnv(),[1],2,"cpu",1)
    finally:
        if previous is None:del sys.modules["mujoco"]
        else:sys.modules["mujoco"]=previous
    assert report["valid_episodes"]==1 and report["sim_error_episodes"]==0 and report["episodes"][0]["attempts"]==2
    print(json.dumps({"status":"PASS","checks":25,"zscore_losses":losses,"teacher_frozen":True,"actor_anchor_grad_zero":True,"combined_smoke":True,"anchor_disabled_regression":True,"cql_only_regression":True,"vanilla_regression":True,"synthetic_mujoco_retry":True}))
if __name__=="__main__":main()
