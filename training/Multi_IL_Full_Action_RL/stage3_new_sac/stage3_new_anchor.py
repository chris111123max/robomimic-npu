"""Frozen Stage2 value-geometry anchor and reproducible IL-data routing."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[3];S2=ROOT/"training"/"Multi_IL_Full_Action_RL"/"stage2_new_critic_pretraining"
if str(S2) not in sys.path:sys.path.insert(0,str(S2))
from stage2_new_dataset import POLICIES,load_split_datasets  # noqa:E402
from stage2_new_sampler import MultiPolicyBalancedSampler,RNNTransitionSampler  # noqa:E402

def read(path):
    with open(path,encoding="utf-8") as f:return json.load(f)

class AnchorData:
    """Reuse the audited Stage2 split and samplers without preprocessing drift."""
    def __init__(self,stage2_run_dir,group,seed):
        config=read(Path(stage2_run_dir)/"config_resolved.json")
        train_seeds=range(int(config["train_seed_start"]),int(config["train_seed_end"])+1)
        val_seeds=range(int(config["val_seed_start"]),int(config["val_seed_end"])+1)
        self.train,self.val=load_split_datasets(config["dataset_root"],train_seeds,val_seeds,float(config["gamma"]))
        self.group,self.seed=group,int(seed)
        self.sampler=RNNTransitionSampler(self.train["bc_rnn"],self.seed) if group=="rnn_q" else MultiPolicyBalancedSampler(self.train,self.seed)
        self.config={"dataset_root":config["dataset_root"],"train_seed_start":int(config["train_seed_start"]),"train_seed_end":int(config["train_seed_end"]),"val_seed_start":int(config["val_seed_start"]),"val_seed_end":int(config["val_seed_end"]),"sampling":"rnn_transition_uniform" if group=="rnn_q" else config["multi_policy_sampling"]}
    def sample(self,size):return self.sampler.sample(int(size))
    def fixed(self,split,size,seed):
        datasets=self.train if split=="train" else self.val;rng=np.random.default_rng(int(seed))
        if self.group=="rnn_q":
            d=datasets["bc_rnn"];idx=rng.choice(d.transition_count,min(int(size),d.transition_count),replace=False);return d.batch(idx)
        base,remainder=divmod(int(size),len(POLICIES));parts=[]
        for i,policy in enumerate(POLICIES):
            d=datasets[policy];count=base+int(i<remainder);parts.append(d.batch(rng.choice(d.transition_count,min(count,d.transition_count),replace=False)))
        return {key:np.concatenate([part[key] for part in parts]) for key in parts[0]}
    def state_dict(self):
        result={"rng_state":self.sampler.rng.bit_generator.state}
        for name in ("batch_index","counts"):
            if hasattr(self.sampler,name):result[name]=getattr(self.sampler,name)
        return result
    def load_state_dict(self,state):
        self.sampler.rng.bit_generator.state=state["rng_state"]
        for name in ("batch_index","counts"):
            if name in state:setattr(self.sampler,name,state[name])
    def proportions(self):return self.sampler.proportions() if hasattr(self.sampler,"proportions") else {"bc_rnn":1.0}

def ranks(values):
    x=np.asarray(values).reshape(-1);order=np.argsort(x,kind="mergesort");rank=np.empty(len(x),float);ordered=x[order];start=0
    for end in range(1,len(x)+1):
        if end==len(x) or ordered[end]!=ordered[start]:rank[order[start:end]]=(start+end-1)/2;start=end
    return rank

def correlation(a,b):
    a,b=np.asarray(a).reshape(-1),np.asarray(b).reshape(-1)
    return None if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12 else float(np.corrcoef(a,b)[0,1])

def geometry_diagnostics(agent,batch):
    values=agent.anchor_components(batch);result={key:float(value.detach().item()) for key,value in values.items() if torch.is_tensor(value) and value.numel()==1}
    for head in ("q1","q2"):
        student=values[f"student_{head}"].detach().cpu().numpy();teacher=values[f"teacher_{head}"].detach().cpu().numpy()
        result[f"spearman_{head}"]=correlation(ranks(student),ranks(teacher));result[f"max_abs_diff_{head}"]=float(np.max(np.abs(student-teacher)))
    return result

def probe_geometry(critic,probes,device):
    means={}
    with torch.no_grad():
        for name,data in probes.items():
            state=torch.as_tensor(data["states"],dtype=torch.float32,device=device);action=torch.as_tensor(data["actions"],dtype=torch.float32,device=device);q1,q2=critic(state,action);means[name]=float(torch.minimum(q1,q2).mean().item())
    success=means.get("rnn_success");failure=means.get("other_failure")
    return {"probe_qmin_means":means,"qmin_success_mean":success,"qmin_failure_mean":failure,"success_failure_q_gap":None if success is None or failure is None else success-failure}
