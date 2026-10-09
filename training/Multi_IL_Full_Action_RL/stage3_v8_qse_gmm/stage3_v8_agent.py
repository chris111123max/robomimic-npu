"""V7 Q update retained; success NLL gradient added before original clipping."""
import copy
import math
import torch
import stage3_v8_paths
from stage3_v6_agent import RecurrentGMMTD3V6, strict_stage2_load
from stage3_v8_good_replay import GoodReplay, bind_pool
from stage3_v8_gmm_loss import good_loss, validate_windows

def parameter_group(name):
    for group,token in (("mean_head","decoder.nets.mean"),("logits_head","decoder.nets.logits"),
                        ("std_head","decoder.nets.scale"),("shared_rnn","rnn.nets")):
        if token in name: return group
    return "other"

def gradient_summary(named, q_grads, good_grads, coefficient):
    result={}
    for group in ("total","shared_rnn","mean_head","logits_head","std_head","other"):
        ids=[i for i,(n,p) in enumerate(named) if group=="total" or parameter_group(n)==group]
        qs=[];gs=[];dots=[];cs=[]
        for i in ids:
            p=named[i][1]
            q=torch.zeros_like(p) if q_grads[i] is None else q_grads[i].detach()
            g=torch.zeros_like(p) if good_grads[i] is None else good_grads[i].detach()
            qs.append(q.square().sum());gs.append(g.square().sum());dots.append((q*g).sum())
            cs.append((q+coefficient*g).square().sum())
        def summed(xs): return float(torch.stack(xs).sum()) if xs else 0.
        qn=math.sqrt(summed(qs));gn=math.sqrt(summed(gs));dot=summed(dots)
        result[group]=dict(q_norm=qn,good_norm=gn,combined_norm=math.sqrt(summed(cs)),
            weighted_good_ratio=coefficient*gn/qn if qn else None,
            cosine=dot/(qn*gn) if qn*gn else None)
    return result

class RecurrentGMMTD3(RecurrentGMMTD3V6):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.v8_loss=copy.deepcopy(self.config["v8_loss"])
        coefficient=self.v8_loss.get("lambda_good")
        if coefficient is None or not math.isfinite(float(coefficient)) or coefficient<0:
            raise RuntimeError("V8 needs finite nonnegative calibrated lambda")
        if coefficient and (self.v8_loss["std_mode"]!="fixed" or
                             self.v8_loss.get("supervision_std") is None):
            raise RuntimeError("Production V8 uses calibrated fixed supervision std only")
        self.good_replay=GoodReplay(self.v8_loss,int(self.config["training_seed"])+
                                   int(self.v8_loss["good_rng_seed_offset"]))
        if self.config.get("v8_load_offline_good",False):
            self.good_replay.load_offline(self.config["offline_sources"])
        bind_pool(self.good_replay)
        self.last_v8_metrics={}

    def actor_update(self,sequences,env_steps,collect_metrics=True,good_batch=None):
        coefficient=float(self.v8_loss["lambda_good"])
        # No sampling, graph, hook, RNG or optimizer changes in lambda=0 branch.
        if coefficient==0:
            return super().actor_update(sequences,env_steps,collect_metrics=collect_metrics)
        if not self.actor_gate_open or self.critic_updates % int(self.config["policy_delay"]):
            raise RuntimeError("V8 Actor gate/delay violation")
        validate_windows(sequences)
        if good_batch is None:
            good_batch=self.good_replay.sample(int(self.v8_loss["good_batch_size"]))
        self.actor.train()
        loss=good_loss(self.actor,good_batch,self.action_scale,self.action_offset,self.v8_loss)
        named=list(self.actor.named_parameters())
        grads=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
        if not all(g is None or torch.isfinite(g).all() for g in grads):
            raise FloatingPointError("Nonfinite good gradient")
        # Fixed-std NLL has no scale-head path. Every other supervised parameter
        # must receive a Q gradient, verified before the original optimizer step.
        handles=[]; observed=set(); q_grads=[None]*len(named)
        def add_good(i):
            def hook(q):
                observed.add(i)
                if collect_metrics: q_grads[i]=q.detach().clone()
                return q+coefficient*grads[i].detach()
            return hook
        for i,(_,p) in enumerate(named):
            if grads[i] is not None:
                handles.append(p.register_hook(add_good(i)))
        def verify(opt,args,kwargs):
            missing=[named[i][0] for i,g in enumerate(grads) if g is not None and i not in observed]
            if missing: raise RuntimeError("Good gradient unsupported by inherited Q graph: "+str(missing))
        pre=self.actor_optimizer.register_step_pre_hook(verify)
        try:
            metrics=super().actor_update(sequences,env_steps,collect_metrics=collect_metrics)
        finally:
            for h in handles: h.remove()
            pre.remove()
        if collect_metrics:
            summary=gradient_summary(named,q_grads,grads,coefficient)
            self.last_v8_metrics=dict(good_loss=float(loss.detach()),lambda_good=coefficient,
                                     gradient_groups=summary,**self.good_replay.metrics())
            metrics.update(actor_good_loss=float(loss.detach()),lambda_good=coefficient,
                actor_total_loss=metrics["actor_rl_loss"]+coefficient*float(loss.detach()),
                good_gradient_norm=summary["total"]["good_norm"],
                q_gradient_norm=summary["total"]["q_norm"],
                good_q_gradient_ratio=summary["total"]["weighted_good_ratio"],
                good_q_gradient_cosine=summary["total"]["cosine"],**self.good_replay.metrics())
        return metrics
