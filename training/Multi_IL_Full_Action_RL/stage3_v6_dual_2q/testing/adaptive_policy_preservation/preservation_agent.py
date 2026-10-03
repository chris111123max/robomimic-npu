"""Testing-only loss extension; all other production agent methods inherited."""
import copy,inspect,json,os,sys,textwrap
from pathlib import Path
import numpy as np
import torch
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'actor_temporal_supervision'))
import run_temporal as T
import stage3_v5_agent as V5
import stage3_v6_agent as V6
REG=json.loads((H/'preregistration.json').read_text())
Original=V6.RecurrentGMMTD3

def norm(gs):
 vs=[g.detach().float().square().sum() for g in gs if g is not None]
 return torch.stack(vs).sum().sqrt() if vs else torch.tensor(0.)

def scalar(x):return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)

def log(path,row):
 with Path(path).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

class PreservationAgent(Original):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.branch=os.environ['PRESERVATION_BRANCH'];self.test_out=Path(os.environ['PRESERVATION_GROUP_OUT'])
  self.test_out.mkdir(parents=True,exist_ok=True)
  self.preserve_reference=copy.deepcopy(self.actor).eval().requires_grad_(False)
  p=torch.load(REG['source_checkpoint'],map_location='cpu',weights_only=False)
  assert p['env_steps']==140000 and p['actor_updates']==0 and not p['actor_optimizer']['state']
  self.preserve_reference.load_state_dict(p['actor'],strict=True)
  self.reference_hash=T.module_hash(self.preserve_reference)
  self._hcur=None;self._href=None
  self.actor.nets['rnn'].nets.register_forward_hook(lambda m,i,o:setattr(self,'_hcur',o[0].detach()))
  self.preserve_reference.nets['rnn'].nets.register_forward_hook(lambda m,i,o:setattr(self,'_href',o[0].detach()))
  assert not any(id(p)==id(q) for p in self.preserve_reference.parameters() for pg in self.actor_optimizer.param_groups for q in pg['params'])

 def _begin_preservation(self,dist,b,rl,env_steps,contexts):
  with torch.no_grad():rd,_=T.forward(self.preserve_reference,b['observations'])
  rm=rd.component_distribution.base_dist.loc.detach();rp=rd.mixture_distribution.probs.detach()
  m=dist.component_distribution.base_dist.loc;p=dist.mixture_distribution.probs
  mu=(rp*(m-rm).square().sum(-1)).sum(-1).mean()
  kl=(rp*(rd.mixture_distribution.logits.detach()-dist.mixture_distribution.logits)).sum(-1).mean()
  distance=mu+REG['beta']*kl
  lam=REG['lambda_max']*float(np.clip((scalar(distance)-REG['D_safe'])/(REG['D_hard']-REG['D_safe']),0,1)) if self.branch=='ADAPTIVE' else 0.
  pars=list(self.actor.parameters())
  rg=torch.autograd.grad(rl,pars,retain_graph=True,allow_unused=True)
  ag=torch.autograd.grad(distance,pars,retain_graph=True,allow_unused=True)
  rnorm=scalar(norm(rg));anorm=scalar(norm(ag))
  gn=scalar(norm([r+lam*a if r is not None and a is not None else r if a is None else lam*a for r,a in zip(rg,ag)]))
  std_rg=[g for (n,p),g in zip(self.actor.named_parameters(),rg) if T.group(n)=='std']
  self._preservation_record={'testing_only':True,'branch':self.branch,'env_steps':int(env_steps),'actor_updates':self.actor_updates+1,'lambda':lam,'lambda_max_fraction':lam/REG['lambda_max'],'D_mu':scalar(mu),'D_p':scalar(kl),'D':scalar(distance),'anchor_loss':scalar(distance),'actor_RL_loss':scalar(rl),'total_loss':scalar(rl)+lam*scalar(distance),'RL_grad_norm':rnorm,'unit_anchor_grad_norm':anorm,'anchor_grad_norm':lam*anorm,'anchor_RL_grad_ratio':lam*anorm/rnorm if rnorm else 0.,'actor_total_grad_norm_preclip':gn,'std_RL_grad_norm':scalar(norm(std_rg)),'actor_lr':self.actor_optimizer.param_groups[0]['lr']}
  return distance,lam,rd

 def _finish_preservation(self,b,rl,rd,contexts):
  with torch.no_grad():
   post,_=T.forward(self.actor,b['observations']);hidden_current=self._hcur.clone();hidden_ref=self._href.clone()
   q,_,_,_,_=T.component_mean_q(self.critic,contexts,T.final_distribution(post),self.action_scale,self.action_offset,twin_min=False)
   gain=scalar(q.mean()+rl.detach())
  row=self._preservation_record;row['same_critic_optimizer_step_RL_gain']=gain;row['Q_before']=-scalar(rl);row['Q_after']=scalar(q.mean())
  if self.actor_updates%25==0 or self.critic_updates%int(self.config["train_metrics_interval_updates"])==0:
   with torch.no_grad():
    m=post.component_distribution.base_dist.loc;rm=rd.component_distribution.base_dist.loc
    p=post.mixture_distribution.probs;rp=rd.mixture_distribution.probs
    wm=(p[...,None]*m).sum(-2);wr=(rp[...,None]*rm).sum(-2)
    diag={'execution_matched_action_drift':None,'component_mean_RMS':scalar((m-rm).square().mean().sqrt()),'weighted_mean_L2':scalar((wm-wr).norm(dim=-1).mean()),'hidden_L2':scalar((hidden_current-hidden_ref).norm(dim=-1).mean()),'categorical_KL':scalar((rp*(rd.mixture_distribution.logits-post.mixture_distribution.logits)).sum(-1).mean()),'gmm_entropy':scalar(-(p*p.clamp_min(1e-12).log()).sum(-1).mean()),'top1_probability':scalar(p.max(-1).values.mean()),'top1_mode_histogram':torch.bincount(p.argmax(-1).reshape(-1).cpu(),minlength=5).cpu().tolist(),'mode_rank_change_fraction':scalar((p.argsort(-1)!=rp.argsort(-1)).any(-1).float().mean()),'component_pairwise_distance':scalar(torch.cdist(m.reshape(-1,5,14),m.reshape(-1,5,14))[...,~torch.eye(5,dtype=torch.bool,device=m.device)].mean())}
    mode=self.actor.training
    try:
     cur=T.sampled_stream(self.actor,b['observations']);ref=T.sampled_stream(self.preserve_reference,b['observations'])
     diag['execution_matched_action_drift']=scalar((cur-ref).norm(dim=-1).mean())
    finally:self.actor.train(mode);self.preserve_reference.eval()
    accum={}
    for (n,p),(rn,rp) in zip(self.actor.named_parameters(),self.preserve_reference.named_parameters()):
     assert n==rn;k=T.group(n);ss=scalar((p-rp).square().sum());old,count=accum.get(k,(0.,0));accum[k]=(old+ss,count+p.numel())
    diag['parameter_RMS']={k:(ss/count)**.5 for k,(ss,count) in accum.items()}
    diag['parameter_RMS']['total']=(sum(x[0] for x in accum.values())/sum(x[1] for x in accum.values()))**.5
   assert T.module_hash(self.preserve_reference)==self.reference_hash
   assert not any(p.grad is not None for p in self.preserve_reference.parameters())
   row['policy_diagnostics']=diag;row['reference_hash_unchanged']=True
  log(self.test_out/'actor_update_diagnostics.jsonl',row)

# Derive solely the Actor function from the exact current production source.
# Keep all gates, final Q1 objective, optimizer, clip, counter and production diagnostics.
src=textwrap.dedent(inspect.getsource(V5.RecurrentGMMTD3.actor_update))
assert src.count('actor_rl = -expected.mean()')==1 and src.count('actor_rl.backward()')==1 and src.count('self.actor_updates += 1')==1
src=src.replace('actor_rl = -expected.mean()','actor_rl = -expected.mean()\n        preservation_loss, preservation_lambda, preservation_ref_distribution = self._begin_preservation(sequence_distribution, b, actor_rl, env_steps, final_contexts)',1)
src=src.replace('actor_rl.backward()','(actor_rl + preservation_lambda * preservation_loss).backward() if self.branch == "ADAPTIVE" else actor_rl.backward()',1)
src=src.replace('self.actor_updates += 1','self.actor_updates += 1\n    self._finish_preservation(b, actor_rl, preservation_ref_distribution, final_contexts)',1)
src=src.replace('"actor_total_loss": actor_rl,', '"actor_total_loss": actor_rl + preservation_lambda * preservation_loss,',1)
namespace=dict(vars(V5));exec(compile(src,str(H/'derived_actor_update'), 'exec'),namespace)
PreservationAgent.actor_update=namespace['actor_update']
