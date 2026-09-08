#!/usr/bin/env python3
"""Synthetic sanity checks for protected handoff and progressive unfreezing."""
from __future__ import annotations
import copy,json
import numpy as np,torch
from stage3_new_agent import Stage3SAC,build_actor,progressive_schedule,state_hash
from critic_network import build_critic

def config():
    return {"hidden_dims":[256,256],"actor_lr":3e-4,"critic_lr":3e-4,"alpha_lr":3e-4,"critic_weight_decay":1e-4,"gamma":.99,"tau":.005,"target_entropy":-14.,"alpha_init":.01,"progressive_unfreeze":{"enabled":True,"protected_until_env_steps":10000,"unfreeze_end_env_steps":30000},"cql":{"enabled":True,"lambda":.1,"num_random_actions":10,"num_policy_actions":1,"apply_to_expert":True,"apply_to_online":True,"detach_policy_actions":True},"anchor":{"enabled":False},"handoff":{"enabled":True,"lambda_handoff":1.0}}
def batch(n=16):
    rng=np.random.default_rng(19)
    return {"observations":rng.normal(size=(n,59)).astype(np.float32),"actions":rng.uniform(-1,1,size=(n,14)).astype(np.float32),"rewards":rng.integers(0,2,size=(n,1)).astype(np.float32),"next_observations":rng.normal(size=(n,59)).astype(np.float32),"terminals":np.zeros((n,1),np.float32),"action_rl":rng.uniform(-1,1,size=(n,14)).astype(np.float32),"action_rnn":rng.uniform(-1,1,size=(n,14)).astype(np.float32),"rnn_next_actions":rng.uniform(-1,1,size=(n,14)).astype(np.float32),"selected_source":np.zeros((n,1),np.float32),"is_online":np.ones((n,1),np.float32)}
def main():
    cfg=config();s0=progressive_schedule(cfg,0);s9999=progressive_schedule(cfg,9999);s10=progressive_schedule(cfg,10000);s20=progressive_schedule(cfg,20000);s30=progressive_schedule(cfg,30000)
    assert s0["phase"]==s9999["phase"]=="protected_handoff" and s0["critic_lr_effective"]==s0["target_tau_effective"]==0
    assert s10["phase"]=="progressive_unfreeze" and s10["critic_lr_effective"]==s10["target_tau_effective"]==0
    assert np.isclose(s20["critic_lr_effective"],.5*cfg["critic_lr"]) and np.isclose(s20["target_tau_effective"],.5*cfg["tau"])
    assert s30["phase"]=="full_sac" and s30["critic_lr_effective"]==cfg["critic_lr"] and s30["target_tau_effective"]==cfg["tau"]
    torch.manual_seed(23);agent=Stage3SAC(build_actor(cfg,"cpu"),build_critic(59,14,[256,256],"relu",True,"cpu"),cfg,"cpu");critic0=state_hash(agent.critic);target0=state_hash(agent.target);actor0=state_hash(agent.actor);alpha0=float(agent.alpha.item())
    metrics=agent.update(batch(),env_steps=5000)
    assert metrics["phase"]=="protected_handoff" and metrics["actor_sac_loss"]==0 and metrics["cql_loss_raw"]==0
    assert state_hash(agent.critic)==critic0 and state_hash(agent.target)==target0 and float(agent.alpha.item())==alpha0
    assert state_hash(agent.actor)!=actor0 and any(p.grad is not None for p in agent.actor.parameters()) and all(p.grad is None for p in agent.critic.parameters())
    boundary=agent.update(batch(),env_steps=10000);assert boundary["critic_lr_effective"]==0 and boundary["target_tau_effective"]==0 and state_hash(agent.critic)==critic0 and state_hash(agent.target)==target0
    midpoint=agent.update(batch(),env_steps=20000);assert np.isclose(midpoint["critic_lr_effective"],1.5e-4) and np.isclose(midpoint["target_tau_effective"],.0025) and state_hash(agent.critic)!=critic0
    full=agent.update(batch(),env_steps=30000);assert np.isclose(full["critic_lr_effective"],3e-4) and np.isclose(full["target_tau_effective"],.005)
    print(json.dumps({"status":"PASS","checks":17,"phase_a_critic_frozen":True,"phase_a_target_frozen":True,"phase_a_actor_changed":True,"phase_a_alpha_fixed":True,"actor_gradient_isolated":True,"schedule_10k_20k_30k":True,"cql_disabled_in_phase_a":True}))
if __name__=="__main__":main()
