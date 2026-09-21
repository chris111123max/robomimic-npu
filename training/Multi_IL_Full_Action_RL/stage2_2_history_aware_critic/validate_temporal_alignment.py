#!/usr/bin/env python3
"""CPU structural tests for temporal semantics and checkpoint contract."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from history_critic import build_critic,checkpoint_payload,load_checkpoint
from sequence_dataset import previous_actions,POLICIES
from sequence_sampler import SequenceSampler
HERE=Path(__file__).resolve().parent;CONFIG=json.loads((HERE/"stage2_2_config.json").read_text())

class TemporalTests(unittest.TestCase):
 def setUp(self): torch.manual_seed(3);self.m=build_critic(CONFIG);self.b,self.t=2,8;self.o=torch.randn(self.b,self.t,59);self.a=torch.randn(self.b,self.t,14);self.p=torch.arange(self.t).float()[None,:,None].repeat(self.b,1,1)/700
 def test_previous_action_and_episode_start(self):
  raw=np.arange(70,dtype=np.float32).reshape(5,14);p=previous_actions(raw);self.assertTrue(np.all(p[0]==0));np.testing.assert_array_equal(p[1:],raw[:-1])
 def test_windows_never_cross_episode_and_sources_balance(self):
  class E:
   def __init__(self,identity):
    self.episode_id=identity;self.length=60;self.observations=np.full((60,59),identity,np.float32);self.actions=np.full((60,14),identity,np.float32);self.returns=np.zeros(60,np.float32);self.success=False
  class D:
   def __init__(self,offset):self.episodes=[E(offset),E(offset+1)]
  datasets={p:D(i*10) for i,p in enumerate(POLICIES)};sampler=SequenceSampler(datasets,32,16,700,1,True);batch=sampler.sample(16)
  for row,episode_id in zip(batch["observations"],batch["episode_id"]):self.assertTrue(np.all(row==episode_id))
  for _ in range(2):sampler.sample(16)
  self.assertEqual(sorted(sampler.counts.values()),[16,16,16])
 def test_no_future_leakage(self):
  prev=torch.cat((torch.zeros(self.b,1,14),self.a[:,:-1]),1);z,_=self.m.q1.encode_history(self.o,prev,self.p);changed=self.o.clone();changed[:,5:]+=100;z2,_=self.m.q1.encode_history(changed,prev,self.p);torch.testing.assert_close(z[:,:5],z2[:,:5])
 def test_context_recurrence(self):
  prev=torch.cat((torch.zeros(self.b,1,14),self.a[:,:-1]),1);full,state=self.m.q1.encode_history(self.o[:,:5],prev[:,:5],self.p[:,:5]);zt,state2=self.m.q1.advance_context(self.o[:,5],self.a[:,4],self.p[:,5],state);torch.testing.assert_close(zt,self.m.q1.encode_history(self.o[:,:6],prev[:,:6],self.p[:,:6])[0][:,5],rtol=1e-5,atol=1e-6)
 def test_candidate_batch_and_twin_independence(self):
  prev=torch.cat((torch.zeros(self.b,1,14),self.a[:,:-1]),1);contexts,_=self.m.encode_history(self.o,prev,self.p);z=(contexts[0][:,-1],contexts[1][:,-1]);c=torch.randn(self.b,5,14);batched=self.m.q_from_context(z,c);loop=[torch.stack([self.m.q_from_context(z,c[:,k])[i] for k in range(5)],1) for i in range(2)];torch.testing.assert_close(batched[0],loop[0]);torch.testing.assert_close(batched[1],loop[1]);self.assertTrue(set(map(id,self.m.q1.parameters())).isdisjoint(set(map(id,self.m.q2.parameters()))))
 def test_gradients_and_checkpoint(self):
  prev=torch.cat((torch.zeros(self.b,1,14),self.a[:,:-1]),1);q1,q2=self.m.forward_sequence(self.o,prev,self.p,self.a);(q1.square().mean()+q2.square().mean()).backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.m.parameters()));opt=torch.optim.AdamW(self.m.parameters(),lr=3e-4,weight_decay=1e-4)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/"x.pth";torch.save(checkpoint_payload(self.m,opt,CONFIG,1,.1),path);restored,_=load_checkpoint(path,CONFIG);torch.testing.assert_close(next(self.m.parameters()),next(restored.parameters()));bad=copy.deepcopy(CONFIG);bad["horizon"]=699
   with self.assertRaises(RuntimeError):load_checkpoint(path,bad)
if __name__=="__main__":unittest.main()
