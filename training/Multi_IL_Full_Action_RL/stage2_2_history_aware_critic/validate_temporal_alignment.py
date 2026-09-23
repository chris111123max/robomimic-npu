#!/usr/bin/env python3
"""Regression tests for Stage2.2 temporal, numerical, and metric contracts."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from evaluation import auc,corr,ranks,evaluate,episode_windows
from finite_diagnostics import gradient_statistics,numpy_batch_stats
from history_critic import build_critic,checkpoint_payload,load_checkpoint
from sequence_dataset import previous_actions,POLICIES
from sequence_sampler import SequenceSampler,LegacyWindowSampler
HERE=Path(__file__).resolve().parent;CONFIG=json.loads((HERE/"stage2_2_config.json").read_text())

class E:
 def __init__(self,identity,length=60):
  self.policy=POLICIES[identity//10];self.seed=10000+identity;self.episode_id=identity;self.length=length;self.success=bool(identity%2)
  self.observations=np.full((length,59),identity,np.float32);self.next_observations=self.observations+.1
  self.actions=np.arange(length*14,dtype=np.float32).reshape(length,14)/100
  self.returns=np.linspace(1,0,length,dtype=np.float32);self.rewards=np.zeros(length,np.float32)
class D:
 def __init__(self,offset,lengths=(60,60)):self.episodes=[E(offset+i,n) for i,n in enumerate(lengths)]
def datasets(lengths=(60,60)):return {p:D(i*10,lengths) for i,p in enumerate(POLICIES)}

class TemporalTests(unittest.TestCase):
 def setUp(self):
  torch.manual_seed(3);self.m=build_critic(CONFIG);self.b,self.t=2,8
  self.o=torch.randn(self.b,self.t,59);self.a=torch.randn(self.b,self.t,14)
  self.p=torch.arange(self.t).float()[None,:,None].repeat(self.b,1,1)/700

 def test_01_previous_action_and_episode_start(self):
  raw=np.arange(70,dtype=np.float32).reshape(5,14);p=previous_actions(raw)
  self.assertTrue(np.all(p[0]==0));np.testing.assert_array_equal(p[1:],raw[:-1])

 def test_02_no_future_leakage(self):
  prev=torch.cat((torch.zeros(self.b,1,14),self.a[:,:-1]),1)
  z,_=self.m.q1.encode_history(self.o,prev,self.p)
  changed=self.o.clone();changed[:,5:]+=100;z2,_=self.m.q1.encode_history(changed,prev,self.p)
  torch.testing.assert_close(z[:,:5],z2[:,:5])

 def test_03_twin_independence_and_candidate_batch(self):
  prev=torch.cat((torch.zeros(self.b,1,14),self.a[:,:-1]),1)
  contexts,_=self.m.encode_history(self.o,prev,self.p);z=(contexts[0][:,-1],contexts[1][:,-1])
  c=torch.randn(self.b,5,14);batched=self.m.q_from_context(z,c)
  loop=[torch.stack([self.m.q_from_context(z,c[:,k])[i] for k in range(5)],1) for i in range(2)]
  torch.testing.assert_close(batched[0],loop[0]);torch.testing.assert_close(batched[1],loop[1])
  self.assertTrue(set(map(id,self.m.q1.parameters())).isdisjoint(set(map(id,self.m.q2.parameters()))))

 def test_04_context_is_capped_at_ten_and_first_previous_action_zero(self):
  s=SequenceSampler(datasets(),32,10,700,2,False)
  for _ in range(20):
   b=s.sample(32)
   self.assertEqual(b["observations"].shape[1],10)
   self.assertTrue(np.all(b["context_steps"]<=10))
   self.assertTrue(np.all(b["previous_actions"][:,0]==0))
   self.assertTrue(np.all(b["learning_mask"].sum((1,2))==1))

 def test_05_early_transition_uses_short_context_without_cross_episode(self):
  s=SequenceSampler(datasets((8,8)),32,10,700,2,True);batch=s.sample(48)
  self.assertTrue(np.all(batch["context_steps"]<=8))
  self.assertTrue(np.all(batch["start"]==0))
  self.assertTrue(np.all(batch["learning_mask"].sum((1,2))==1))
  for row,eid,valid in zip(batch["observations"],batch["episode_id"],batch["valid_mask"]):
   self.assertTrue(np.all(row[valid[:,0]]==eid))

 def test_06_source_balance(self):
  s=SequenceSampler(datasets(),32,10,700,1,True)
  for _ in range(3):s.sample(16)
  self.assertEqual(sorted(s.counts.values()),[16,16,16])

 def test_07_training_and_evaluation_window_semantics_identical(self):
  d=datasets();s=SequenceSampler(d,32,10,700,4,False);b=s.sample(1)
  e=next(e for e in d["bc_rnn"].episodes if e.episode_id==int(b["episode_id"][0]))
  target=int(b["target_step"][0]);o,pa,p,a,final=episode_windows(e,10,700)
  n=int(b["context_steps"][0])
  np.testing.assert_array_equal(b["observations"][0,:n],o[target,:n])
  np.testing.assert_array_equal(b["previous_actions"][0,:n],pa[target,:n])
  np.testing.assert_array_equal(b["actions"][0,:n],a[target,:n])
  self.assertEqual(n-1,int(final[target]))

 def test_08_old_history_before_window_cannot_change_input(self):
  e=E(0,30);o1,pa1,p1,a1,f1=episode_windows(e,10,700)
  e2=E(0,30);e2.observations[:10]+=999;e2.actions[:10]+=999
  o2,pa2,p2,a2,f2=episode_windows(e2,10,700)
  target=25
  np.testing.assert_array_equal(o1[target],o2[target])
  np.testing.assert_array_equal(pa1[target],pa2[target])
  np.testing.assert_array_equal(p1[target],p2[target])
  np.testing.assert_array_equal(f1,f2)

 def test_09_gradients_and_nan_gradient_detection(self):
  prev=torch.cat((torch.zeros(self.b,1,14),self.a[:,:-1]),1)
  q1,q2=self.m.forward_sequence(self.o,prev,self.p,self.a)
  (q1.square().mean()+q2.square().mean()).backward();stats=gradient_statistics(self.m)
  self.assertFalse(stats["non_finite_gradient_parameter_names"])
  next(self.m.q1.parameters()).grad.view(-1)[0]=float("nan")
  self.assertTrue(gradient_statistics(self.m)["non_finite_gradient_parameter_names"])

 def test_10_nan_input_detection(self):
  s=SequenceSampler(datasets(),32,10,700,1,False);b=s.sample(2);b["observations"][0,0,0]=np.nan
  self.assertFalse(numpy_batch_stats(b)["observations"]["isfinite"])

 def test_11_checkpoint_contract_rejects_old_history_semantics(self):
  opt=torch.optim.AdamW(self.m.parameters(),lr=3e-4,weight_decay=1e-4)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/"x.pth";torch.save(checkpoint_payload(self.m,opt,CONFIG,1,.1,.1),path)
   restored,payload=load_checkpoint(path,CONFIG);torch.testing.assert_close(next(self.m.parameters()),next(restored.parameters()))
   self.assertEqual(payload["checkpoint_validation_metric"],.1)
   for key,value in (("horizon",699),("gamma",.5),("history_semantics","full_episode_prefix_unroll_learning_mask"),("recurrent_context_length",11)):
    bad=copy.deepcopy(CONFIG);bad[key]=value
    with self.assertRaises(RuntimeError):load_checkpoint(path,bad)

 def test_12_exact_legacy_replay_unchanged(self):
  one=LegacyWindowSampler(datasets(),32,16,700,CONFIG["training_seed"],True)
  two=LegacyWindowSampler(datasets(),32,16,700,CONFIG["training_seed"],True)
  for _ in range(5):a=one.sample(16);b=two.sample(16)
  for key in ("policy","seed","episode_id","start","stop","observations"):np.testing.assert_array_equal(a[key],b[key])

 def test_13_evaluation_restores_training_mode(self):
  self.m.train();evaluate(self.m,datasets(),torch.device("cpu"),700,10);self.assertTrue(self.m.training)

class MetricTests(unittest.TestCase):
 def test_14_average_rank_ties(self):np.testing.assert_allclose(ranks(np.array([1,1,3,2])),[.5,.5,3,2])
 def test_15_auc_and_spearman_ties(self):
  self.assertIsNone(corr(ranks(np.ones(5)),ranks(np.arange(5))))
  self.assertAlmostEqual(corr(ranks(np.array([0,0,1,1])),ranks(np.array([0,0,1,1]))),1.)
  self.assertAlmostEqual(auc(np.array([0.,0.,1.,1.]),np.array([0,1,0,1],bool)),.5)
  self.assertAlmostEqual(auc(np.array([0.,1.,2.,3.]),np.array([0,0,1,1],bool)),1.)

if __name__=="__main__":unittest.main()
