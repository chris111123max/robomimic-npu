"""Standard feed-forward SAC with a Stage2-new-compatible Twin Critic."""
from __future__ import annotations
import copy, hashlib, math, sys
from pathlib import Path
import numpy as np, torch

ROOT=Path(__file__).resolve().parents[3]; S2=ROOT/"training"/"Multi_IL_Full_Action_RL"/"stage2_new_critic_pretraining"; RLKIT=ROOT/"rlkit"
for path in (str(S2),str(RLKIT)):
    if path not in sys.path: sys.path.insert(0,path)
from critic_network import build_critic  # noqa:E402
from rlkit.torch.sac.policies import TanhGaussianPolicy  # noqa:E402
from stage3_new_handoff import handoff_imitation,hybrid_bootstrap  # noqa:E402

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
    def __init__(self,actor,critic,config,device,action_low=None,action_high=None,teacher=None):
        self.actor,self.critic,self.target,self.config,self.device=actor,critic,copy.deepcopy(critic).to(device),config,device;self.target.requires_grad_(False)
        for key,value in critic.state_dict().items():
            if not torch.equal(value,self.target.state_dict()[key]): raise AssertionError("Target Critic is not an exact step-0 copy")
        self.actor_optimizer=torch.optim.Adam(actor.parameters(),lr=float(config["actor_lr"]),weight_decay=0.0)
        self.critic_optimizer=torch.optim.AdamW(critic.parameters(),lr=float(config["critic_lr"]),weight_decay=float(config["critic_weight_decay"]))
        alpha_init=float(config["alpha_init"])
        if not math.isfinite(alpha_init) or alpha_init<=0: raise ValueError(f"alpha_init must be finite and positive, got {alpha_init}")
        self.log_alpha=torch.tensor(math.log(alpha_init),dtype=torch.float32,device=device,requires_grad=True);self.alpha_optimizer=torch.optim.Adam([self.log_alpha],lr=float(config["alpha_lr"]),weight_decay=0.0);self.updates=0
        low=-np.ones(14,np.float32) if action_low is None else np.asarray(action_low,np.float32);high=np.ones(14,np.float32) if action_high is None else np.asarray(action_high,np.float32)
        if low.shape!=(14,) or high.shape!=(14,) or not np.isfinite(low).all() or not np.isfinite(high).all() or not np.all(low<high):raise ValueError("Invalid environment action bounds")
        self.action_low=torch.as_tensor(low,dtype=torch.float32,device=device);self.action_high=torch.as_tensor(high,dtype=torch.float32,device=device)
        self.teacher=teacher
        if self.teacher is not None:
            self.teacher.eval();self.teacher.requires_grad_(False)
            optimizer_ids={id(p) for group in self.critic_optimizer.param_groups for p in group["params"]}
            if any(id(p) in optimizer_ids for p in self.teacher.parameters()):raise RuntimeError("Frozen teacher entered Critic optimizer")
    @property
    def alpha(self): return self.log_alpha.exp()
    def action(self,state,deterministic=False):
        with torch.no_grad(): return self.actor(torch.as_tensor(state,dtype=torch.float32,device=self.device).reshape(-1,59),deterministic=deterministic)[0].cpu().numpy()
    def target_components(self,b):
        """Return the exact target terms used by training without updating state."""
        next_action,_,_,next_logp,*_=self.actor(b["next_observations"],reparameterize=True,return_log_prob=True)
        handoff=bool(self.config.get("handoff",{}).get("enabled",False))
        if handoff:
            boot=hybrid_bootstrap(self.target,b["next_observations"],next_action,b["rnn_next_actions"],next_logp,self.alpha);target_qmin=torch.where(boot["rl_wins"],boot["q_rl"],boot["q_rnn"] )[:,None];entropy_bonus=torch.where(boot["rl_wins"][:,None],-self.alpha.detach()*next_logp,torch.zeros_like(next_logp));td_target=b["rewards"]+float(self.config["gamma"])*(1-b["terminals"])*boot["value"]
            return {"target_qmin":target_qmin,"next_logp":next_logp.detach(),"entropy_bonus":entropy_bonus,"td_target":td_target,"bootstrap":boot}
        with torch.no_grad():
            tq1,tq2=self.target(b["next_observations"],next_action);target_qmin=torch.minimum(tq1,tq2);entropy_bonus=-self.alpha.detach()*next_logp;td_target=b["rewards"]+float(self.config["gamma"])*(1-b["terminals"])*(target_qmin+entropy_bonus)
        return {"target_qmin":target_qmin,"next_logp":next_logp.detach(),"entropy_bonus":entropy_bonus,"td_target":td_target}
    def cql_components(self,states,q1_data,q2_data):
        cql=self.config.get("cql",{});batch=len(states);k_random=int(cql.get("num_random_actions",10));k_policy=int(cql.get("num_policy_actions",1))
        if k_random!=10 or k_policy!=1:raise RuntimeError("CQL-lite contract requires K_random=10 and K_policy=1")
        with torch.no_grad():policy_action=self.actor(states,reparameterize=True,return_log_prob=False)[0].detach()
        random_action=torch.rand((batch,k_random,14),dtype=states.dtype,device=states.device);random_action=self.action_low+(self.action_high-self.action_low)*random_action
        candidates=torch.cat((policy_action[:,None,:],random_action),dim=1);expanded=states[:,None,:].expand(-1,k_policy+k_random,-1).reshape(-1,59);flat=candidates.reshape(-1,14);ood_q1,ood_q2=self.critic(expanded,flat);ood_q1=ood_q1.reshape(batch,-1);ood_q2=ood_q2.reshape(batch,-1)
        raw_q1=self.cql_penalty(ood_q1,q1_data);raw_q2=self.cql_penalty(ood_q2,q2_data);raw=raw_q1+raw_q2;q_data=torch.minimum(q1_data,q2_data).reshape(-1);q_policy=torch.minimum(ood_q1[:,0],ood_q2[:,0]);q_random=torch.minimum(ood_q1[:,1:],ood_q2[:,1:]);random_max=q_random.max(dim=1).values
        return {"loss_q1":raw_q1,"loss_q2":raw_q2,"loss":raw,"policy_actions":policy_action,"random_actions":random_action,"q_data":q_data,"q_policy":q_policy,"q_random":q_random,"q_random_max":random_max}
    @staticmethod
    def cql_penalty(q_ood,q_data):return torch.logsumexp(q_ood,dim=1).mean()-q_data.mean()
    @staticmethod
    def zscore(q,eps=1e-6):
        q=q.reshape(-1);return (q-q.mean())/(q.std(unbiased=False)+float(eps))
    @staticmethod
    def pearson(a,b,eps=1e-12):
        a,b=a.reshape(-1),b.reshape(-1);ac,bc=a-a.mean(),b-b.mean();return (ac*bc).mean()/(ac.std(unbiased=False)*bc.std(unbiased=False)+float(eps))
    def anchor_components(self,batch):
        if self.teacher is None:raise RuntimeError("Anchor enabled without frozen Stage2 teacher")
        cfg=self.config.get("anchor",{});eps=float(cfg.get("eps",1e-6));states=torch.as_tensor(batch["state"],dtype=torch.float32,device=self.device);actions=torch.as_tensor(batch["action"],dtype=torch.float32,device=self.device)
        sq1,sq2=self.critic(states,actions)
        with torch.no_grad():tq1,tq2=self.teacher(states,actions)
        sq1,sq2,tq1,tq2=(x.reshape(-1) for x in (sq1,sq2,tq1,tq2));l1=torch.nn.functional.mse_loss(self.zscore(sq1,eps),self.zscore(tq1,eps));l2=torch.nn.functional.mse_loss(self.zscore(sq2,eps),self.zscore(tq2,eps));raw=.5*(l1+l2)
        return {"loss_q1":l1,"loss_q2":l2,"loss":raw,"student_q1":sq1,"student_q2":sq2,"teacher_q1":tq1,"teacher_q2":tq2,"student_q1_mean":sq1.mean(),"student_q1_std":sq1.std(unbiased=False),"student_q2_mean":sq2.mean(),"student_q2_std":sq2.std(unbiased=False),"teacher_q1_mean":tq1.mean(),"teacher_q1_std":tq1.std(unbiased=False),"teacher_q2_mean":tq2.mean(),"teacher_q2_std":tq2.std(unbiased=False),"pearson_q1":self.pearson(sq1,tq1),"pearson_q2":self.pearson(sq2,tq2),"teacher_std_near_zero":((tq1.std(unbiased=False)<1e-8)|(tq2.std(unbiased=False)<1e-8)).float()}
    @staticmethod
    def ensure_finite(values):
        bad=[name for name,value in values.items() if torch.is_tensor(value) and not torch.isfinite(value).all()]
        if bad:raise FloatingPointError(f"Non-finite Stage3 SAC/CQL tensors: {bad}")
    def source_diagnostics(self,batch):
        b={k:torch.as_tensor(v,dtype=torch.float32,device=self.device) for k,v in batch.items()}
        with torch.no_grad():
            q1,q2=self.critic(b["observations"],b["actions"]);cql=self.cql_components(b["observations"],q1,q2);target=self.target_components(b);gap=cql["q_policy"]-cql["q_data"];random_gap=cql["q_random_max"]-cql["q_data"]
        values={"q_data_mean":cql["q_data"].mean(),"q_policy_mean":cql["q_policy"].mean(),"q_random_max_mean":cql["q_random_max"].mean(),"policy_minus_data_q_mean":gap.mean(),"policy_gt_data_fraction":(gap>0).float().mean(),"random_max_minus_data_q_mean":random_gap.mean(),"random_max_gt_data_fraction":(random_gap>0).float().mean(),"target_qmin_mean":target["target_qmin"].mean(),"entropy_bonus_mean":target["entropy_bonus"].mean(),"td_target_mean":target["td_target"].mean(),"reward_mean":b["rewards"].mean()}
        self.ensure_finite(values);return {key:float(value.item()) for key,value in values.items()}
    def update(self,batch,anchor_batch=None):
        b={k:torch.as_tensor(v,dtype=torch.float32,device=self.device) for k,v in batch.items()};components=self.target_components(b);target_qmin=components["target_qmin"];entropy_bonus=components["entropy_bonus"];target=components["td_target"]
        q1,q2=self.critic(b["observations"],b["actions"]);l1=torch.nn.functional.mse_loss(q1,target);l2=torch.nn.functional.mse_loss(q2,target);critic_td_loss=l1+l2;cql_cfg=self.config.get("cql",{});cql_enabled=bool(cql_cfg.get("enabled",False))
        cql_values=self.cql_components(b["observations"],q1,q2) if cql_enabled else None;cql_raw=cql_values["loss"] if cql_enabled else critic_td_loss.new_zeros(());cql_weighted=float(cql_cfg.get("lambda",.1))*cql_raw
        anchor_cfg=self.config.get("anchor",{});anchor_enabled=bool(anchor_cfg.get("enabled",False))
        if anchor_enabled and anchor_batch is None:raise RuntimeError("Every anchored Critic update requires an independent anchor batch")
        anchor_values=self.anchor_components(anchor_batch) if anchor_enabled else None;anchor_raw=anchor_values["loss"] if anchor_enabled else critic_td_loss.new_zeros(());anchor_weighted=float(anchor_cfg.get("lambda",.1))*anchor_raw;critic_loss=critic_td_loss+cql_weighted+anchor_weighted
        self.ensure_finite({"critic_td_loss":critic_td_loss,"cql_loss":cql_raw,"anchor_loss":anchor_raw,"critic_loss":critic_loss,"q1":q1,"q2":q2,"target_qmin":target_qmin,"td_target":target,"alpha":self.alpha})
        self.critic_optimizer.zero_grad(set_to_none=True);critic_loss.backward();self.critic_optimizer.step()
        self.ensure_finite({f"critic_parameter_{index}":parameter for index,parameter in enumerate(self.critic.parameters())})
        action,_,_,logp,*_=self.actor(b["observations"],reparameterize=True,return_log_prob=True)
        for p in self.critic.parameters(): p.requires_grad_(False)
        aq1,aq2=self.critic(b["observations"],action); actor_sac_loss=(self.alpha.detach()*logp-torch.minimum(aq1,aq2)).mean();handoff_cfg=self.config.get("handoff",{});handoff_enabled=bool(handoff_cfg.get("enabled",False));handoff_loss,mask_count=handoff_imitation(self.actor,b["observations"],b["action_rnn"],(b["is_online"]>.5)&(b["selected_source"]<.5)) if handoff_enabled else (actor_sac_loss.new_zeros(()),0);handoff_weighted=float(handoff_cfg.get("lambda_handoff",1.0))*handoff_loss;actor_loss=actor_sac_loss+handoff_weighted;self.actor_optimizer.zero_grad(set_to_none=True);actor_loss.backward();self.actor_optimizer.step()
        for p in self.critic.parameters(): p.requires_grad_(True)
        alpha_loss=-(self.log_alpha*(logp.detach()+float(self.config["target_entropy"]))).mean();self.alpha_optimizer.zero_grad(set_to_none=True);alpha_loss.backward();self.alpha_optimizer.step()
        self.ensure_finite({"actor_loss":actor_loss,"alpha_loss":alpha_loss,"alpha":self.alpha,**{f"actor_parameter_{index}":parameter for index,parameter in enumerate(self.actor.parameters())}})
        tau=float(self.config["tau"])
        with torch.no_grad():
            for source,dest in zip(self.critic.parameters(),self.target.parameters()): dest.mul_(1-tau).add_(source,alpha=tau)
        self.updates+=1; td=torch.cat((q1-target,q2-target));qmin=torch.minimum(q1,q2)
        sampled=action.detach()
        def tensor_stats(name,value,minimum_maximum=False):
            result={f"{name}_mean":float(value.mean().item()),f"{name}_std":float(value.std(unbiased=False).item())}
            if minimum_maximum:result.update({f"{name}_min":float(value.min().item()),f"{name}_max":float(value.max().item())})
            return result
        policy_entropy=float((-logp).mean().item())
        metrics={"critic_loss":float(critic_loss.item()),"critic_loss_total":float(critic_loss.item()),"critic_td_loss":float(critic_td_loss.item()),"q1_loss":float(l1.item()),"q2_loss":float(l2.item()),"cql_loss_q1_raw":float(cql_values["loss_q1"].item()) if cql_enabled else 0.0,"cql_loss_q2_raw":float(cql_values["loss_q2"].item()) if cql_enabled else 0.0,"cql_loss_raw":float(cql_raw.item()),"cql_loss_weighted":float(cql_weighted.item()),"anchor_loss_q1_raw":float(anchor_values["loss_q1"].item()) if anchor_enabled else 0.0,"anchor_loss_q2_raw":float(anchor_values["loss_q2"].item()) if anchor_enabled else 0.0,"anchor_loss_raw":float(anchor_raw.item()),"anchor_loss_weighted":float(anchor_weighted.item()),"actor_loss":float(actor_loss.item()),"actor_sac_loss":float(actor_sac_loss.item()),"handoff_loss_raw":float(handoff_loss.item()),"handoff_loss_weighted":float(handoff_weighted.item()),"handoff_mask_count":mask_count,"handoff_mask_fraction":float(mask_count/len(b["observations"])),"actor_total_loss":float(actor_loss.item()),"alpha":float(self.alpha.item()),"log_alpha":float(self.log_alpha.item()),"alpha_loss":float(alpha_loss.item()),"target_entropy":float(self.config["target_entropy"]),"policy_entropy":policy_entropy,"entropy":policy_entropy,"mean_q1":float(q1.mean().item()),"mean_q2":float(q2.mean().item()),"target_q":float(target.mean().item()),"td_error":float(td.abs().mean().item()),"action_mean":sampled.mean(dim=0).cpu().tolist(),"action_std":sampled.std(dim=0,unbiased=False).cpu().tolist(),"action_saturation_rate":(sampled.abs()>0.99).float().mean(dim=0).cpu().tolist()}
        if handoff_enabled:
            boot=components["bootstrap"];metrics.update({"bootstrap_rnn_fraction":float((~boot["rl_wins"]).float().mean().item()),"bootstrap_rl_fraction":float(boot["rl_wins"].float().mean().item()),"q_boot_rnn_mean":float(boot["q_rnn"].mean().item()),"q_boot_rl_mean":float(boot["q_rl"].mean().item()),"boot_value_mean":float(boot["value"].mean().item())})
        if anchor_enabled:
            for name in ("teacher_q1_mean","teacher_q1_std","teacher_q2_mean","teacher_q2_std","student_q1_mean","student_q1_std","student_q2_mean","student_q2_std","pearson_q1","pearson_q2","teacher_std_near_zero"):metrics[f"anchor_{name}"]=float(anchor_values[name].item())
            metrics["anchor_teacher_student_pearson_q1"]=metrics["anchor_pearson_q1"];metrics["anchor_teacher_student_pearson_q2"]=metrics["anchor_pearson_q2"]
        if cql_enabled:
            gap=cql_values["q_policy"]-cql_values["q_data"];random_gap=cql_values["q_random_max"]-cql_values["q_data"];metrics.update({"q_data_mean":float(cql_values["q_data"].mean().item()),"q_policy_cql_mean":float(cql_values["q_policy"].mean().item()),"q_random_mean":float(cql_values["q_random"].mean().item()),"q_random_max_mean":float(cql_values["q_random_max"].mean().item()),"policy_minus_data_q_mean":float(gap.mean().item()),"policy_minus_data_q_median":float(gap.median().item()),"random_max_minus_data_q_mean":float(random_gap.mean().item()),"fraction_policy_gt_data":float((gap>0).float().mean().item()),"fraction_random_max_gt_data":float((random_gap>0).float().mean().item())})
        metrics.update(tensor_stats("reward",b["rewards"],True));metrics.update(tensor_stats("target_qmin",target_qmin));metrics.update(tensor_stats("entropy_bonus",entropy_bonus,True));metrics.update(tensor_stats("td_target",target,True));metrics.update(tensor_stats("q1",q1));metrics.update(tensor_stats("q2",q2));metrics["qmin_mean"]=float(qmin.mean().item())
        return metrics
