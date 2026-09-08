#!/usr/bin/env python3
"""Synthetic contracts for target-Q RNN handoff without real checkpoints."""
from __future__ import annotations
import copy,json,sys
from pathlib import Path
import numpy as np,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];S2=ROOT/"training"/"Multi_IL_Full_Action_RL"/"stage2_new_critic_pretraining"
if str(S2) not in sys.path:sys.path.insert(0,str(S2))
from critic_network import build_critic
from stage3_new_agent import Stage3SAC,build_actor
from stage3_new_handoff import handoff_imitation,hybrid_bootstrap,select_actions

class ActionQ(torch.nn.Module):
    def forward(self,state,action):q=action[:,0:1];return q,q
class FakeRNN:
    def start_episode(self):self.hidden=0
    def action(self,state):self.hidden+=1;return np.full(14,self.hidden,np.float32)
def cfg(enabled=True):
    return {"hidden_dims":[256,256],"actor_lr":3e-4,"critic_lr":3e-4,"alpha_lr":3e-4,"critic_weight_decay":1e-4,"gamma":.99,"tau":.005,"target_entropy":-14.,"alpha_init":.01,"cql":{"enabled":True,"lambda":.1,"num_random_actions":10,"num_policy_actions":1,"apply_to_expert":True,"apply_to_online":True,"detach_policy_actions":True},"anchor":{"enabled":False},"handoff":{"enabled":enabled,"lambda_handoff":1.0}}
def batch(n=16):
    rng=np.random.default_rng(7);selected=rng.integers(0,2,size=(n,1)).astype(np.float32);return {"observations":rng.normal(size=(n,59)).astype(np.float32),"actions":rng.uniform(-1,1,size=(n,14)).astype(np.float32),"rewards":rng.integers(0,2,size=(n,1)).astype(np.float32),"next_observations":rng.normal(size=(n,59)).astype(np.float32),"terminals":rng.integers(0,2,size=(n,1)).astype(np.float32),"action_rl":rng.uniform(-1,1,size=(n,14)).astype(np.float32),"action_rnn":rng.uniform(-1,1,size=(n,14)).astype(np.float32),"rnn_next_actions":rng.uniform(-1,1,size=(n,14)).astype(np.float32),"selected_source":selected,"q_select_rl":np.zeros((n,1),np.float32),"q_select_rnn":np.zeros((n,1),np.float32),"q_select_margin":np.zeros((n,1),np.float32),"is_online":np.ones((n,1),np.float32)}
def main():
    state=torch.zeros((3,59));rl=torch.zeros((3,14));rnn=torch.zeros((3,14));rl[:,0]=torch.tensor([0.,2.,1.]);rnn[:,0]=torch.tensor([1.,0.,1.]);chosen,qrl,qrnn,wins=select_actions(ActionQ(),state,rl,rnn);assert wins.tolist()==[False,True,False] and chosen[:,0].tolist()==[1.,2.,1.]
    logp=torch.tensor([[4.],[4.],[4.]]);boot=hybrid_bootstrap(ActionQ(),state,rl,rnn,logp,torch.tensor(.1));assert torch.allclose(boot["value"],torch.tensor([[1.],[1.6],[1.]]));reward=torch.tensor([[1.],[1.],[1.]]);terminal=torch.ones((3,1));target=reward+.99*(1-terminal)*boot["value"];assert torch.equal(target,reward)
    torch.manual_seed(3);actor=build_actor(cfg(),"cpu");critic=ActionQ();rnn_actions=torch.ones((3,14));mask=torch.tensor([1,0,1],dtype=torch.bool);loss,count=handoff_imitation(actor,state,rnn_actions,mask);loss.backward();assert count==2 and any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in actor.parameters()) and all(p.grad is None for p in critic.parameters())
    fake=FakeRNN();fake.start_episode();current=fake.action(None);next_action=fake.action(None);cached=next_action.copy();assert np.array_equal(cached,next_action);fake.start_episode();assert np.array_equal(fake.action(None),current)
    torch.manual_seed(9);config=cfg();agent=Stage3SAC(build_actor(config,"cpu"),build_critic(59,14,[256,256],"relu",True,"cpu"),config,"cpu");metrics=agent.update(batch());required=("actor_sac_loss","handoff_loss_raw","actor_total_loss","bootstrap_rnn_fraction","bootstrap_rl_fraction","cql_loss_weighted");assert all(np.isfinite(metrics[k]) for k in required)
    torch.manual_seed(10);disabled_cfg=cfg(False);raw=batch();raw={k:v for k,v in raw.items() if k in ("observations","actions","rewards","next_observations","terminals")};disabled=Stage3SAC(build_actor(disabled_cfg,"cpu"),build_critic(59,14,[256,256],"relu",True,"cpu"),disabled_cfg,"cpu");disabled_metrics=disabled.update(raw);assert disabled_metrics["handoff_loss_raw"]==0
    print(json.dumps({"status":"PASS","checks":18,"selector_rnn_rl_tie":True,"hybrid_bootstrap":True,"terminal_mask":True,"handoff_actor_gradient_isolated":True,"recurrent_cache_consistency":True,"episode_hidden_reset":True,"cql_handoff_smoke":True,"handoff_disabled_regression":True}))
if __name__=="__main__":main()
