"""Standard feed-forward SAC with a Stage2-new-compatible Twin Critic."""
from __future__ import annotations
import copy, hashlib, sys
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[3]; S2=ROOT/"training"/"Multi_IL_Full_Action_RL"/"stage2_new_critic_pretraining"; RLKIT=ROOT/"rlkit"
for path in (str(S2),str(RLKIT)):
    if path not in sys.path: sys.path.insert(0,path)
from critic_network import build_critic  # noqa:E402
from rlkit.torch.sac.policies import TanhGaussianPolicy  # noqa:E402

def build_actor(config,device=None):
    actor=TanhGaussianPolicy(hidden_sizes=list(config["hidden_dims"]),obs_dim=59,action_dim=14,std=None)
    return actor if device is None else actor.to(device)

def state_hash(module):
    digest=hashlib.sha256()
    for key,value in module.state_dict().items(): digest.update(key.encode());digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
def strict_stage2_load(path,device,config):
    payload=torch.load(path,map_location=device); mc=payload.get("model_config",{})
    expected={"obs_dim":59,"action_dim":14,"hidden_dims":list(config["hidden_dims"]),"activation":"relu","layer_norm":True}
    for key,value in expected.items():
        if mc.get(key)!=value: raise RuntimeError(f"Stage2 checkpoint {key}={mc.get(key)!r}, expected {value!r}")
    if float(payload.get("gamma",-1))!=0.99: raise RuntimeError("Stage2 checkpoint gamma must be 0.99")
    critic=build_critic(59,14,config["hidden_dims"],"relu",True,device); critic.load_state_dict(payload["critic_state_dict"],strict=True); return critic,payload

class Stage3SAC:
    def __init__(self,actor,critic,config,device):
        self.actor,self.critic,self.target,self.config,self.device=actor,critic,copy.deepcopy(critic).to(device),config,device;self.target.requires_grad_(False)
        for key,value in critic.state_dict().items():
            if not torch.equal(value,self.target.state_dict()[key]): raise AssertionError("Target Critic is not an exact step-0 copy")
        self.actor_optimizer=torch.optim.Adam(actor.parameters(),lr=float(config["actor_lr"]),weight_decay=0.0)
        self.critic_optimizer=torch.optim.AdamW(critic.parameters(),lr=float(config["critic_lr"]),weight_decay=float(config["critic_weight_decay"]))
        self.log_alpha=torch.tensor(np.log(float(config["initial_alpha"])),dtype=torch.float32,device=device,requires_grad=True);self.alpha_optimizer=torch.optim.Adam([self.log_alpha],lr=float(config["alpha_lr"]),weight_decay=0.0);self.updates=0
    @property
    def alpha(self): return self.log_alpha.exp()
    def action(self,state,deterministic=False):
        with torch.no_grad(): return self.actor(torch.as_tensor(state,dtype=torch.float32,device=self.device).reshape(-1,59),deterministic=deterministic)[0].cpu().numpy()
    def update(self,batch):
        b={k:torch.as_tensor(v,dtype=torch.float32,device=self.device) for k,v in batch.items()}; gamma=float(self.config["gamma"])
        next_action,_,_,next_logp,*_=self.actor(b["next_observations"],reparameterize=True,return_log_prob=True)
        with torch.no_grad(): tq1,tq2=self.target(b["next_observations"],next_action); target=b["rewards"]+gamma*(1-b["terminals"])*(torch.minimum(tq1,tq2)-self.alpha.detach()*next_logp)
        q1,q2=self.critic(b["observations"],b["actions"]);l1=torch.nn.functional.mse_loss(q1,target);l2=torch.nn.functional.mse_loss(q2,target);critic_loss=l1+l2
        self.critic_optimizer.zero_grad(set_to_none=True);critic_loss.backward();self.critic_optimizer.step()
        action,_,_,logp,*_=self.actor(b["observations"],reparameterize=True,return_log_prob=True)
        for p in self.critic.parameters(): p.requires_grad_(False)
        aq1,aq2=self.critic(b["observations"],action); actor_loss=(self.alpha.detach()*logp-torch.minimum(aq1,aq2)).mean();self.actor_optimizer.zero_grad(set_to_none=True);actor_loss.backward();self.actor_optimizer.step()
        for p in self.critic.parameters(): p.requires_grad_(True)
        alpha_loss=-(self.log_alpha*(logp.detach()+float(self.config["target_entropy"]))).mean();self.alpha_optimizer.zero_grad(set_to_none=True);alpha_loss.backward();self.alpha_optimizer.step()
        tau=float(self.config["tau"])
        with torch.no_grad():
            for source,dest in zip(self.critic.parameters(),self.target.parameters()): dest.mul_(1-tau).add_(source,alpha=tau)
        self.updates+=1; td=torch.cat((q1-target,q2-target))
        sampled=action.detach()
        return {"critic_loss":float(critic_loss.item()),"q1_loss":float(l1.item()),"q2_loss":float(l2.item()),"actor_loss":float(actor_loss.item()),"alpha":float(self.alpha.item()),"alpha_loss":float(alpha_loss.item()),"entropy":float((-logp).mean().item()),"mean_q1":float(q1.mean().item()),"mean_q2":float(q2.mean().item()),"target_q":float(target.mean().item()),"td_error":float(td.abs().mean().item()),"action_mean":sampled.mean(dim=0).cpu().tolist(),"action_std":sampled.std(dim=0,unbiased=False).cpu().tolist(),"action_saturation_rate":(sampled.abs()>0.99).float().mean(dim=0).cpu().tolist()}
