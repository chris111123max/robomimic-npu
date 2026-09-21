#!/usr/bin/env python3
"""Regression tests for Stage2.2 temporal, numerical, and metric contracts."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from evaluation import auc,corr,ranks
from finite_diagnostics import gradient_statistics,numpy_batch_stats
from history_critic import build_critic,checkpoint_payload,load_checkpoint
from sequence_dataset import previous_actions,POLICIES
from sequence_sampler import SequenceSampler,LegacyWindowSampler
HERE=Path(__file__).resolve().parent;CONFIG=json.loads((HERE/"stage2_2_config.json").read_text())

class E:
 def __init__(self,identity,length=60):
  self.policy=POLICIES[identity//10];self.seed=10000+identity;self.episode_id=identity;self.length=length;self.success=bool(identity%2)
  self.observations=np.full((length,59),identity,np.float32);self.next_observations=self.observations+.1;self.actions=np.arange(length*14,dtype=np.float32).reshape(length,14)/100
  self.returns=np.linspace(1,0,length,dtype=np.float32);self.rewards=np.zeros(length,np.float32)
class D:
 def __init__(self,offset,lengths=(60,60)):self.episodes=[E(offset+i,n) for i,n in enumerate(lengths)]
def datasets(lengths=(60,60)):return {p:D(i*10,lengths) for i,p in enumerate(POLICIES)}

class TemporalTests(unittest.TestCase):
 def setUp(self):torch.manual_seed(3);self.m=build_critic(CONFIG);self.b,self.t=2,8;self.o=torch.randn(self.b,self.t,59);self.a=torch.randn(self.b,self.t,14);self.p=torch.arange(self.t).float()[None,:,None].repeat(self.b,1,1)/700
 def test_01_previous_action_and_episode_start(self):
  raw=np.arange(70,dtype=np.float32).reshape(5,14);p=previous_actions(raw);self.assertTrue(np.all(p[0]==0));np.testing.assert_array_equal(p[1:],raw[:-1])
 def test_02_no_future_leakage(self):
  prev=torch.cat((torch.zeros(self.b,1,14),self.a[:,:-1]),1);z,_=self.m.q1.encode_history(self.o,prev,self.p);changed=self.o.clone();changed[:,5:]+=100;z2,_=self.m.q1.encode_history(changed,prev,self.p);torch.testing.assert_close(z[:,:5],z2[:,:5])
 def test_03_twin_independence_and_candidate_batch(self):
  prev=torch.cat((torch.zeros(self.b,1,14),self.a[:,:-1]),1);contexts,_=self.m.encode_history(self.o,prev,self.p);z=(contexts[0][:,-1],contexts[1][:,-1]);c=torch.randn(self.b,5,14);batched=self.m.q_from_context(z,c);loop=[torch.stack([self.m.q_from_context(z,c[:,k])[i] for k in range(5)],1) for i in range(2)];torch.testing.assert_close(batched[0],loop[0]);torch.testing.assert_close(batched[1],loop[1]);self.assertTrue(set(map(id,self.m.q1.parameters())).isdisjoint(set(map(id,self.m.q2.parameters()))))
 def test_04_full_vs_step_recurrence(self):
  prev=torch.cat((torch.zeros(self.b,1,14),self.a[:,:-1]),1);_,state=self.m.q1.encode_history(self.o[:,:5],prev[:,:5],self.p[:,:5]);zt,_=self.m.q1.advance_context(self.o[:,5],self.a[:,4],self.p[:,5],state);full=self.m.q1.encode_history(self.o[:,:6],prev[:,:6],self.p[:,:6])[0][:,5];torch.testing.assert_close(zt,full,rtol=1e-5,atol=1e-6)
 def test_05_early_short_and_no_cross_episode(self):
  s=SequenceSampler(datasets((8,8)),32,16,700,2,True);self.assertEqual(s.valid["bc_rnn"],[(0,0),(1,0)]);batch=s.sample(48)
  self.assertTrue(np.all(batch["start"]==0));self.assertTrue(np.all(batch["episode_length"]==8));self.assertTrue(np.all(batch["learning_mask"].sum((1,2))==8))
  for row,eid,valid in zip(batch["observations"],batch["episode_id"],batch["valid_mask"]):self.assertTrue(np.all(row[valid[:,0]]==eid))
 def test_06_source_balance(self):
  s=SequenceSampler(datasets(),32,16,700,1,True)
  for _ in range(3):s.sample(16)
  self.assertEqual(sorted(s.counts.values()),[16,16,16])
 def test_07_training_eval_history_identical(self):
  d=datasets();s=SequenceSampler(d,32,16,700,4,False);b=s.sample(1);e=next(e for e in d["bc_rnn"].episodes if e.episode_id==int(b["episode_id"][0]));stop=int(b["stop"][0]);prev=previous_actions(e.actions)
  with torch.no_grad():
   sampled=self.m.forward_sequence(torch.tensor(b["observations"]),torch.tensor(b["previous_actions"]),torch.tensor(b["progress"]),torch.tensor(b["actions"]))[0][0,:stop]
   full=self.m.forward_sequence(torch.tensor(e.observations[None,:stop]),torch.tensor(prev[None,:stop]),torch.arange(stop).float()[None,:,None]/700,torch.tensor(e.actions[None,:stop]))[0][0]
  torch.testing.assert_close(sampled,full)
 def test_08_gradients_and_nan_gradient_detection(self):
  prev=torch.cat((torch.zeros(self.b,1,14),self.a[:,:-1]),1);q1,q2=self.m.forward_sequence(self.o,prev,self.p,self.a);(q1.square().mean()+q2.square().mean()).backward();stats=gradient_statistics(self.m);self.assertFalse(stats["non_finite_gradient_parameter_names"]);next(self.m.q1.parameters()).grad.view(-1)[0]=float("nan");self.assertTrue(gradient_statistics(self.m)["non_finite_gradient_parameter_names"])
 def test_09_nan_input_detection(self):
  s=SequenceSampler(datasets(),32,16,700,1,False);b=s.sample(2);b["observations"][0,0,0]=np.nan;self.assertFalse(numpy_batch_stats(b)["observations"]["isfinite"])
 def test_10_checkpoint_contract(self):
  opt=torch.optim.AdamW(self.m.parameters(),lr=3e-4,weight_decay=1e-4)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/"x.pth";torch.save(checkpoint_payload(self.m,opt,CONFIG,1,.1,.1),path);restored,payload=load_checkpoint(path,CONFIG);torch.testing.assert_close(next(self.m.parameters()),next(restored.parameters()));self.assertEqual(payload["checkpoint_validation_metric"],.1)
   for key,value in (("horizon",699),("gamma",.5),("history_semantics","wrong")):
    bad=copy.deepcopy(CONFIG);bad[key]=value
    with self.assertRaises(RuntimeError):load_checkpoint(path,bad)
 def test_11_exact_legacy_replay(self):
  one=LegacyWindowSampler(datasets(),32,16,700,CONFIG["training_seed"],True);two=LegacyWindowSampler(datasets(),32,16,700,CONFIG["training_seed"],True)
  for _ in range(5):a=one.sample(16);b=two.sample(16)
  for key in ("policy","seed","episode_id","start","stop","observations"):np.testing.assert_array_equal(a[key],b[key])

class MetricTests(unittest.TestCase):
 def test_12_average_rank_ties(self):np.testing.assert_allclose(ranks(np.array([1,1,3,2])),[.5,.5,3,2])
 def test_13_all_equal_and_duplicate_spearman(self):
  self.assertIsNone(corr(ranks(np.ones(5)),ranks(np.arange(5))));self.assertAlmostEqual(corr(ranks(np.array([0,0,1,1])),ranks(np.array([0,0,1,1]))),1.)
 def test_14_auc_score_ties(self):
  self.assertAlmostEqual(auc(np.array([0.,0.,1.,1.]),np.array([0,1,0,1],bool)),.5);self.assertAlmostEqual(auc(np.array([0.,1.,2.,3.]),np.array([0,0,1,1],bool)),1.)

if __name__=="__main__":unittest.main()
