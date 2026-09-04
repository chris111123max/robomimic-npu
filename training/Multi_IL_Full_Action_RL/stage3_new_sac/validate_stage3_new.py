#!/usr/bin/env python3
"""CPU synthetic smoke test for every Stage3-new algorithmic invariant."""
from __future__ import annotations
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
import h5py,numpy as np,torch
from compare_stage3_new_pair import compare_npz
from stage3_new_agent import Stage3SAC,build_actor,state_hash,strict_stage2_load
from stage3_new_dataset import ExpertDataset,KEYS
from stage3_new_evaluation import evaluate
from stage3_new_probe import record_probe
from stage3_new_replay import SymmetricSampler,TransitionBuffer
from train_stage3_new import load_resume,save

WIDTHS=(3,4,2,3,4,2,41)
def config():
    return {"hidden_dims":[256,256],"actor_lr":3e-4,"critic_lr":3e-4,"alpha_lr":3e-4,"critic_weight_decay":1e-4,"gamma":.99,"tau":.005,"target_entropy":-14.0,"alpha_init":.01,"training_seed":7}
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def expert_file(path):
    with h5py.File(path,"w") as f:
        data=f.create_group("data");data.attrs["env_args"]=json.dumps({"env_name":"TwoArmTransport"});demo=data.create_group("demo_0");obs=demo.create_group("obs");nxt=demo.create_group("next_obs");n=128
        for key,width in zip(KEYS,WIDTHS):obs.create_dataset(key,data=np.random.randn(n,width).astype(np.float32));nxt.create_dataset(key,data=np.random.randn(n,width).astype(np.float32))
        demo.create_dataset("actions",data=np.tanh(np.random.randn(n,14)).astype(np.float32));demo.create_dataset("rewards",data=np.random.randn(n).astype(np.float32));demo.create_dataset("dones",data=np.zeros(n,np.float32))
class FakeEnv:
    def __init__(self):self.step_index=0
    def obs(self):return {k:np.full(w,self.step_index/10,np.float32) for k,w in zip(KEYS,WIDTHS)}
    def seed(self,seed):self.local_seed=seed
    def reset(self):self.step_index=0;return self.obs()
    def step(self,action):assert np.asarray(action).shape==(14,);self.step_index+=1;return self.obs(),float(self.step_index==3),False,{}
    def is_success(self):return {"task":self.step_index>=3}
    def get_state(self):return {"states":np.asarray([self.step_index],np.float32)}
    def reset_to(self,state):self.step_index=int(state["states"][0]);return self.obs()
def main():
    torch.manual_seed(7);c=config()
    with tempfile.TemporaryDirectory(prefix="stage3_new_smoke_") as tmp:
        root=Path(tmp);expert_path=root/"expert.hdf5";expert_file(expert_path);offline=ExpertDataset(expert_path,1)
        from stage3_new_agent import build_critic
        original=build_critic(59,14,c["hidden_dims"],"relu",True,"cpu");s2=root/"stage2.pth";torch.save({"critic_state_dict":original.state_dict(),"model_config":{"obs_dim":59,"action_dim":14,"hidden_dims":[256,256],"activation":"relu","layer_norm":True},"gamma":.99},s2)
        critic,_=strict_stage2_load(s2,"cpu",c);target_copy={k:v.clone() for k,v in critic.state_dict().items()};actor0=build_actor(c,"cpu");shared={k:v.clone() for k,v in actor0.state_dict().items()};actor1=build_actor(c,"cpu");actor2=build_actor(c,"cpu");actor1.load_state_dict(shared);actor2.load_state_dict(shared)
        assert all(torch.equal(actor1.state_dict()[k],actor2.state_dict()[k]) for k in shared);state=np.random.randn(4,59).astype(np.float32);assert actor1(torch.as_tensor(state))[0].shape==(4,14);assert actor1(torch.as_tensor(state),deterministic=True)[0].shape==(4,14);assert torch.equal(actor1(torch.as_tensor(state),deterministic=True)[0],actor2(torch.as_tensor(state),deterministic=True)[0])
        agent=Stage3SAC(actor1,critic,c,"cpu");assert abs(agent.alpha.item()-.01)<1e-7;assert all(torch.equal(agent.target.state_dict()[k],target_copy[k]) for k in target_copy);names=[type(x).__name__ for x in agent.critic.q1.network];assert names==["Linear","LayerNorm","ReLU","Linear","LayerNorm","ReLU","Linear"] and agent.critic_optimizer.param_groups[0]["weight_decay"]==1e-4
        online=TransitionBuffer(2000,seed=2);sample=np.zeros(59,np.float32);action=np.zeros(14,np.float32)
        for _ in range(999):online.add(sample,action,0,sample,0)
        assert online.size<1000;online.add(sample,action,0,sample,0);assert online.size>=1000
        sampler=SymmetricSampler(offline,online,3)
        for _ in range(300):sampler.sample(257)
        fractions=sampler.fractions();assert abs(fractions["offline"]-.5)<1e-4
        before_alpha=agent.log_alpha.detach().clone();before_target={k:v.clone() for k,v in agent.target.state_dict().items()};metrics=agent.update(sampler.sample(256));assert agent.updates==1 and agent.log_alpha.item()!=before_alpha.item() and any(not torch.equal(before_target[k],agent.target.state_dict()[k]) for k in before_target)
        required_metrics={"reward_mean","reward_std","reward_min","reward_max","target_qmin_mean","target_qmin_std","entropy_bonus_mean","entropy_bonus_std","entropy_bonus_min","entropy_bonus_max","td_target_mean","td_target_std","td_target_min","td_target_max","q1_mean","q2_mean","qmin_mean","q1_std","q2_std","alpha","log_alpha","alpha_loss","target_entropy","policy_entropy"};assert required_metrics.issubset(metrics) and abs(metrics["alpha"]-.01)<1e-4
        for _ in range(10):online.add(sample,action,0,sample,0);agent.update(sampler.sample(256))
        assert agent.updates==11
        fake=FakeEnv();before=online.size;report=evaluate(agent.actor,fake,[1,2],5,"cpu");assert online.size==before and report["success_rate"]==1.0
        probes={"mixed":{"states":state,"actions":np.zeros((4,14),np.float32)}};left=root/"left.npz";right=root/"right.npz";summary=record_probe(agent.actor,agent.critic,probes,"cpu",left);record_probe(agent.actor,agent.critic,probes,"cpu",right);comparison=compare_npz(left,right);row=comparison["mixed"];assert summary["mixed"]["count"]==4 and {"q1","q2","qmin","actor_action_l2_norm","grad_q1_norm","grad_q2_norm"}.issubset(summary["mixed"]);assert row["actor_action_l2_difference"]["mean"]==0.0 and "p10" in row["gradient_cosine_q1"] and "qmin_absolute_scale" in row and "normalized_geometry" in row["qmin"]
        checkpoint=root/"checkpoint.pth";save(checkpoint,agent,online,sampler,offline,c,"rnn_q",str(s2),1010,2,None,fake);actor_new=build_actor(c,"cpu");critic_new,_=strict_stage2_load(s2,"cpu",c);agent_new=Stage3SAC(actor_new,critic_new,c,"cpu");offline_new=ExpertDataset(expert_path,1);payload,replay_new,sampler_new,context=load_resume(checkpoint,agent_new,offline_new,c,fake);assert payload["env_steps"]==1010 and replay_new.size==online.size and agent_new.updates==agent.updates and context is None
        incompatible=dict(c);incompatible["alpha_init"]=1.0
        try:load_resume(checkpoint,agent_new,offline_new,incompatible,fake);raise AssertionError("Cross-alpha resume was accepted")
        except RuntimeError as error:assert "incompatible" in str(error)
        old=root/"old_pair"/"shared";old.mkdir(parents=True);torch.save({"actor_state_dict":shared,"actor_hash":state_hash(actor0),"architecture":{"obs_dim":59,"action_dim":14,"hidden_dims":[256,256],"class":"rlkit TanhGaussianPolicy"},"training_seed":7},old/"actor_init.pth");(old/"seed_manifest.json").write_text(json.dumps({"training_seed":7,"train_seed_rule":"train_seed_base + episode_index","train_seed_base":30000,"evaluation_seeds":list(range(20000,20010))}),encoding="utf-8")
        new_root=root/"runs";subprocess.run([sys.executable,str(Path(__file__).with_name("prepare_stage3_new_pair.py")),"--output-root",str(new_root),"--run-id","alpha_test","--reference-pair-run-dir",str(old.parent),"--alpha-init","0.01","--expert-dataset",str(expert_path),"--stage2-run-dir",str(root),"--rnn-q-checkpoint",str(s2),"--multi-q-checkpoint",str(s2)],check=True,capture_output=True,text=True)
        new_shared=new_root/"alpha_test"/"shared";assert sha(old/"actor_init.pth")==sha(new_shared/"actor_init.pth") and sha(old/"seed_manifest.json")==sha(new_shared/"seed_manifest.json");resolved=json.loads((new_shared/"config_resolved.json").read_text());provenance=json.loads((new_shared/"reference_pair_manifest.json").read_text());stage2_sources=json.loads((new_shared/"stage2_source_manifest.json").read_text());assert resolved["alpha_init"]==.01 and provenance["new_alpha_init"]==.01 and provenance["reference_pair_run_dir"]==str(old.parent.resolve());assert all(Path(source["checkpoint"])==s2.resolve() for source in stage2_sources.values());assert not any((new_root/"alpha_test"/"rnn_q").iterdir()) and not any((new_root/"alpha_test"/"multi_q").iterdir())
        print(json.dumps({"status":"PASS","offline_online_fractions":fractions,"checks":40,"configured_alpha_init":c["alpha_init"],"resumed_alpha":agent_new.alpha.item(),"reference_actor_exact":True,"reference_seeds_exact":True}))
if __name__=="__main__":main()
