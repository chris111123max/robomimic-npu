#!/usr/bin/env python3
"""CPU synthetic validation for the optional Stage3-new CQL-lite path."""
from __future__ import annotations
import copy,json,sys
from pathlib import Path
import numpy as np,torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];S2=ROOT/"training"/"Multi_IL_Full_Action_RL"/"stage2_new_critic_pretraining"
if str(S2) not in sys.path:sys.path.insert(0,str(S2))
from critic_network import build_critic
from stage3_new_agent import Stage3SAC,build_actor

def config(enabled):
    return {"hidden_dims":[256,256],"actor_lr":3e-4,"critic_lr":3e-4,"alpha_lr":3e-4,"critic_weight_decay":1e-4,"gamma":.99,"tau":.005,"target_entropy":-14.,"alpha_init":.01,"cql":{"enabled":enabled,"lambda":.1,"num_random_actions":10,"num_policy_actions":1,"apply_to_expert":True,"apply_to_online":True,"detach_policy_actions":True}}
def batch(n=16):
    rng=np.random.default_rng(7);return {"observations":rng.normal(size=(n,59)).astype(np.float32),"actions":rng.uniform(-1,1,size=(n,14)).astype(np.float32),"rewards":rng.integers(0,2,size=(n,1)).astype(np.float32),"next_observations":rng.normal(size=(n,59)).astype(np.float32),"terminals":rng.integers(0,2,size=(n,1)).astype(np.float32)}
def main():
    high=Stage3SAC.cql_penalty(torch.tensor([[3.,2.]]),torch.tensor([[1.]])).item();low=Stage3SAC.cql_penalty(torch.tensor([[1.,0.]]),torch.tensor([[5.]])).item();assert high>0 and low<high
    torch.manual_seed(11);cfg=config(True);actor=build_actor(cfg,"cpu");critic=build_critic(59,14,[256,256],"relu",True,"cpu");agent=Stage3SAC(actor,critic,cfg,"cpu",-np.ones(14),np.ones(14));b={key:torch.as_tensor(value) for key,value in batch().items()};q1,q2=critic(b["observations"],b["actions"]);values=agent.cql_components(b["observations"],q1,q2);assert values["random_actions"].shape==(16,10,14) and values["q_random"].shape==(16,10);assert torch.all(values["random_actions"]>=-1) and torch.all(values["random_actions"]<=1);agent.actor_optimizer.zero_grad(set_to_none=True);agent.critic_optimizer.zero_grad(set_to_none=True);values["loss"].backward();assert all(parameter.grad is None or torch.count_nonzero(parameter.grad)==0 for parameter in actor.parameters());assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad)>0 for parameter in critic.parameters())
    torch.manual_seed(22);base_cfg=config(False);base_actor=build_actor(base_cfg,"cpu");base_critic=build_critic(59,14,[256,256],"relu",True,"cpu");baseline=Stage3SAC(base_actor,base_critic,base_cfg,"cpu");raw=batch();state=torch.get_rng_state();tb={key:torch.as_tensor(value) for key,value in raw.items()};target=baseline.target_components(tb)["td_target"];q1,q2=baseline.critic(tb["observations"],tb["actions"]);expected=(torch.nn.functional.mse_loss(q1,target)+torch.nn.functional.mse_loss(q2,target)).item();torch.set_rng_state(state);metrics=baseline.update(raw);assert abs(metrics["critic_loss"]-expected)<1e-6 and metrics["cql_loss_weighted"]==0
    torch.manual_seed(33);smoke=Stage3SAC(build_actor(cfg,"cpu"),build_critic(59,14,[256,256],"relu",True,"cpu"),cfg,"cpu")
    for _ in range(3):metrics=smoke.update(batch())
    required={"critic_td_loss","cql_loss_q1_raw","cql_loss_q2_raw","cql_loss_raw","cql_loss_weighted","q_data_mean","q_policy_cql_mean","q_random_mean","q_random_max_mean","policy_minus_data_q_mean"};assert required.issubset(metrics) and all(np.isfinite(value) for key,value in metrics.items() if key in required)
    print(json.dumps({"status":"PASS","checks":15,"cql_high_gap":high,"cql_low_gap":low,"actor_cql_grad_zero":True,"baseline_regression":True,"random_shape":list(values["random_actions"].shape)}))
if __name__=="__main__":main()
