from __future__ import annotations
import sys,tempfile,unittest
from pathlib import Path
import h5py,numpy as np,torch

HERE=Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from audit_stage3_q_source_decomposition import analyze_source
from audit_stage3_terminal_semantics import KEYS,audit_dataset
from stage3_new_agent import Stage3SAC

class FakeActor:
    def __call__(self,states,deterministic=False,reparameterize=False,return_log_prob=False):
        action=torch.full((len(states),14),.5,device=states.device);logp=torch.full((len(states),1),-2.,device=states.device)
        return action,None,None,logp,None,None,None,None
class FakeCritic:
    def __call__(self,states,actions):
        q=1+4*actions[:,:1];return q,q+.25
class FakeAgent:
    actor=FakeActor();critic=FakeCritic()
    def target_components(self,b):
        n=len(b["rewards"]);zero=torch.zeros((n,1),device=b["rewards"].device);return {"target_qmin":zero+4,"next_logp":zero-2,"entropy_bonus":zero+.2,"td_target":b["rewards"]+.99*(1-b["terminals"])*4.2}

class TargetHarness:
    actor=FakeActor();target=FakeCritic();alpha=torch.tensor(.1);config={"gamma":.99}

class AuditTests(unittest.TestCase):
    def make_hdf5(self,path):
        with h5py.File(path,"w") as f:
            data=f.create_group("data")
            for episode,(reward,done,terminated,truncated) in enumerate(((1,1,1,0),(0,0,0,1),(0,1,1,0))):
                demo=data.create_group(f"demo_{episode}");obs=demo.create_group("obs");nxt=demo.create_group("next_obs")
                for key_index,key in enumerate(KEYS):
                    values=np.full((2,1),episode*10+key_index,np.float32);next_values=values+1
                    if episode==0:next_values[-1]=10+key_index
                    obs.create_dataset(key,data=values);nxt.create_dataset(key,data=next_values)
                demo.create_dataset("actions",data=np.zeros((2,14),np.float32));demo.create_dataset("rewards",data=[0,reward]);demo.create_dataset("dones",data=[0,done]);demo.create_dataset("terminated",data=[0,terminated]);demo.create_dataset("truncated",data=[0,truncated])
    def test_terminal_bootstrap_and_cross_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"expert.hdf5";self.make_hdf5(path);_,summary,episodes,positives,masks=audit_dataset(path)
        self.assertEqual(summary["num_episodes"],3);self.assertEqual(summary["cross_episode_next_obs_match_count"],1);self.assertEqual(masks["positive_rewards"]["positive_reward_bootstrap_mask_0_count"],1);self.assertEqual(masks["episode_final_transitions"]["bootstrap_mask_1_count"],1);self.assertTrue(any("AUDIT WARNING" in warning for warning in masks["warnings"]));self.assertEqual(len(episodes),3);self.assertEqual(len(positives),1)
    def test_q_delta_and_random_reproducibility(self):
        n=8;data={"observations":np.zeros((n,59),np.float32),"actions":np.zeros((n,14),np.float32),"rewards":np.zeros((n,1),np.float32),"next_observations":np.zeros((n,59),np.float32),"terminals":np.zeros((n,1),np.float32)};indices=np.arange(n);ids=[("demo_0",i) for i in range(n)]
        first=analyze_source(FakeAgent(),data,indices,ids,"expert",10,20260906,"cpu");second=analyze_source(FakeAgent(),data,indices,ids,"expert",10,20260906,"cpu")
        self.assertAlmostEqual(first[0]["policy_minus_behavior_or_data_q"]["mean"],2.0);self.assertEqual([row["random_q_max"] for row in first[3]],[row["random_q_max"] for row in second[3]])
    def test_target_helper_exact_formula(self):
        b={"next_observations":torch.zeros((2,59)),"rewards":torch.tensor([[1.],[2.]]),"terminals":torch.tensor([[0.],[1.]])};result=Stage3SAC.target_components(TargetHarness(),b);expected=torch.tensor([[1+.99*3.2],[2.]])
        self.assertTrue(torch.allclose(result["td_target"],expected));self.assertTrue(torch.allclose(result["entropy_bonus"],torch.full((2,1),.2)))

if __name__=="__main__":unittest.main()
