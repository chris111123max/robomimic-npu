#!/usr/bin/env python3
"""Testing-only, fixed-critic, gradient-calibrated Actor-to-140K anchor."""
import argparse,copy,json,sys,hashlib,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'actor_temporal_supervision'
sys.path.insert(0,str(PREV))
import run_temporal as T
MILESTONES=(0,1,10,25,50,100,250,500,750,1000)
BRANCHES=('BASELINE','WEAK','MEDIUM','STRONG')

def save(path,value):
 T.dump(path,value)

def gradients(actor,loss,retain=True):
 named=list(actor.named_parameters())
 gs=torch.autograd.grad(loss,[p for n,p in named],allow_unused=True,retain_graph=retain)
 return [(T.group(n),np.zeros(p.numel(),np.float32) if g is None else g.detach().cpu().numpy().reshape(-1).copy()) for (n,p),g in zip(named,gs)]

def summarize_grad(rl,mu,prob):
 result={}
 for key in ('rnn','mean','logits','std','encoder','other','total'):
  rows=[i for i,(g,a) in enumerate(rl) if key=='total' or g==key]
  result[key]={}
  for tag,values in (('rl',rl),('mu',mu),('p',prob)):
   result[key][tag]=float(np.sqrt(sum(np.dot(values[i][1].astype(np.float64),values[i][1].astype(np.float64)) for i in rows)))
  result[key]['mu_p_dot']=float(sum(np.dot(mu[i][1].astype(np.float64),prob[i][1].astype(np.float64)) for i in rows))
 return result

def losses(actor,ref,critic,obs,contexts,scale,offset):
 d,_=T.forward(actor,obs)
 with torch.no_grad():
  rd,_=T.forward(ref,obs)
  rm=rd.component_distribution.base_dist.loc.detach()
  rp=rd.mixture_distribution.probs.detach()
  rlog=rd.mixture_distribution.logits.detach()
 q,_,_,_,_=T.component_mean_q(critic,tuple(c[:,-1] for c in contexts),T.final_distribution(d),scale,offset,twin_min=False)
 m=d.component_distribution.base_dist.loc
 lp=d.mixture_distribution.logits
 mu=(rp*(m-rm).square().sum(-1)).sum(-1).mean()
 prob=(rp*(rlog-lp)).sum(-1).mean()
 return -q.mean(),mu,prob

def anchor_norm(g,beta):
 return max(0,g['mu']**2+beta**2*g['p']**2+2*beta*g['mu_p_dot'])**.5

def extra_metrics(current,baseline,actor,initial):
 m=T.metrics(current,baseline,actor,initial)
 d=current['means'].astype(np.float64)-baseline['means'].astype(np.float64)
 p=current['probs'].astype(np.float64); rp=baseline['probs'].astype(np.float64)
 kl=(rp*(np.log(np.maximum(rp,1e-30))-np.log(np.maximum(p,1e-30)))).sum(-1)
 dist=np.linalg.norm(current['means'][..., :,None,:]-baseline['means'][...,None,:,:],axis=-1)
 nearest=dist.argmin(-1)
 g=m['gmm']
 g.update(component_rms=float(np.sqrt(np.mean(d*d))),component_rms_per_mode=np.sqrt(np.mean(d*d,axis=(0,1,3))).tolist(),component_rms_per_action_dimension=np.sqrt(np.mean(d*d,axis=(0,1,2))).tolist(),categorical_kl_ref_current=float(kl.mean()),kl_by_timestep=kl.mean(0).tolist(),top1_probability=float(p.max(-1).mean()),top1_mode_histogram=np.bincount(p.argmax(-1).reshape(-1),minlength=5).tolist(),nearest_reference_mode_mapping_counts=[[int((nearest[...,i]==j).sum()) for j in range(5)] for i in range(5)],nearest_mode_nonidentity_fraction=float(np.mean(nearest!=np.arange(5))))
 m['execution_drift']=float(np.mean(m['sampled_action_drift']['l2_mean_by_timestep']))
 m['RL_improvement']=float(m['frozen_Q1_gain_by_timestep'][-1])
 m['RL_loss']=-float(m['frozen_Q1_mean_by_timestep'][-1])
 return m

def setup(run,previous,device):
 cfg=json.loads((run/'shared/config_resolved.json').read_text())
 data=dict(np.load(previous/'fixed_temporal_contexts.npz'))
 actor,rollout,metadata=T.load_exact_actor(run/'shared/bc_rnn_gmm_source.pth',device)
 ck=torch.load(run/'mean2q/multi_q/checkpoints/critic_ready.pth',map_location='cpu',weights_only=False)
 assert ck['env_steps']==140000 and ck['actor_updates']==0 and len(ck['actor_optimizer']['state'])==0
 actor.load_state_dict(ck['actor'],strict=True)
 manifest=json.loads((run/'shared/stage2_source_manifest.json').read_text())
 critic,_=T.strict_stage2_load(manifest['multi_q']['checkpoint'],device)
 critic.load_state_dict(ck['q1_q2'],strict=True);critic.eval().requires_grad_(False)
 scale=torch.as_tensor(rollout.action_normalization_stats['actions']['scale'],dtype=torch.float32,device=device).reshape(1,1,1,14)
 offset=torch.as_tensor(rollout.action_normalization_stats['actions']['offset'],dtype=torch.float32,device=device).reshape(1,1,1,14)
 obs=torch.as_tensor(data['observations'][:,9:],device=device)
 with torch.no_grad():
  windows=[]
  for t in range(10):
   c=T.encode_replay_contexts(critic,torch.as_tensor(data['observations'][:,t:t+10],device=device),torch.as_tensor(data['actions'][:,t:t+10],device=device),torch.as_tensor(data['episode_steps'][:,t:t+10],device=device),700)
   windows.append(tuple(x[:,-1] for x in c))
  aligned=tuple(torch.stack([c[j] for c in windows],1).detach() for j in range(2))
 return cfg,data,actor,critic,scale,offset,obs,aligned,ck,metadata

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='npu:0');args=ap.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'calibration.json').exists() or (out/'offline_results.json').exists():raise FileExistsError(out)
 device=T.resolve_device(args.device)
 torch.manual_seed(T.SEED);torch.npu.manual_seed(T.SEED)
 cfg,data,actor,critic,scale,offset,obs,contexts,ck,contract=setup(args.run,args.previous,device)
 actor.eval();ref=copy.deepcopy(actor).eval().requires_grad_(False)
 initial={n:p.detach().cpu().clone() for n,p in actor.named_parameters()}
 refhash=T.module_hash(ref);chash=T.module_hash(critic)
 refnorm=float(np.sqrt(sum(float(p.double().square().sum()) for p in initial.values())))
 baseline=T.evaluate(ref,obs,critic,contexts,scale,offset)
 old=dict(np.load(args.previous/'baseline_outputs.npz'))
 assert all(np.array_equal(baseline[k],old[k]) for k in old),'previous baseline differs'
 np.savez_compressed(out/'reference_outputs.npz',**baseline)
 schedule=json.loads((args.previous/'schedule.json').read_text())
 lrs=schedule['actor_lrs'];assert len(lrs)==1000
 assert all(abs(lr-2e-6*min(1,max(0,e-140000)/140000))<1e-18 for lr,e in zip(lrs,schedule['env_steps']))
 schedule['kind']='APPROXIMATE_PRODUCTION_LR_REPLAY';save(out/'schedule.json',schedule)
 batches=[np.concatenate((np.arange(32)+(j%2)*32,np.arange(16)+64+(j%4)*16,np.arange(16)+128+(j%4)*16)) for j in range(4)]
 save(out/'experiment_contract.json',{'actor':contract,'checkpoint':str(args.run/'mean2q/multi_q/checkpoints/critic_ready.pth'),'env_steps':ck['env_steps'],'actor_updates':ck['actor_updates'],'optimizer_initial_state_entries':len(ck['actor_optimizer']['state']),'policy_delay':cfg['policy_delay'],'actor_max_grad_norm':cfg['actor_max_grad_norm'],'batch_size':64,'production_batch_size':cfg['batch_size'],'batch_limitation':'Testing-only deterministic 64, identical to previous CURRENT; not exact production stochastic replay batches','data':str(args.previous/'fixed_temporal_contexts.npz'),'data_sha256':hashlib.sha256((args.previous/'fixed_temporal_contexts.npz').read_bytes()).hexdigest(),'batches':[x.tolist() for x in batches],'reference_hash':refhash,'reference_norm':refnorm,'RL':'negative probability-weighted Q1 at final token; full 10-token BPTT','anchor':'mean_batch_time sum_k p_ref[k] ||mu_current[k]-mu_ref[k]||^2 + beta KL(p_ref||p_current); normalized actions; no std; same RL batch zero-init','selection_predeclared':'smallest weak/medium lambda with >=30% execution drift reduction and >=10% RL improvement retention; strong upper bound; no success-conditioned retuning'})
 def new_optimizer(clone):
  opt=torch.optim.Adam(clone.parameters(),lr=0.,weight_decay=0.)
  opt.load_state_dict(copy.deepcopy(ck['actor_optimizer']))
  assert len(opt.state)==0
  return opt
 probe=copy.deepcopy(actor).train();opt=new_optimizer(probe);calrows=[]
 for u in range(1,26):
  ids=batches[(u-1)%4];opt.zero_grad(set_to_none=True)
  for pg in opt.param_groups:pg['lr']=lrs[u-1]
  rl,mu,prob=losses(probe,ref,critic,obs[ids],tuple(c[ids] for c in contexts),scale,offset)
  rl.backward();torch.nn.utils.clip_grad_norm_(probe.parameters(),cfg['actor_max_grad_norm']);opt.step()
  rl,mu,prob=losses(probe,ref,critic,obs[ids],tuple(c[ids] for c in contexts),scale,offset)
  r=gradients(probe,rl);a=gradients(probe,mu);p=gradients(probe,prob,False)
  row={'update':u,'lr':lrs[u-1],'RL_loss':float(rl.detach().cpu()),'L_mu':float(mu.detach().cpu()),'L_p':float(prob.detach().cpu()),'gradients':summarize_grad(r,a,p)}
  calrows.append(row)
  del r,a,p,rl,mu,prob
 pe=T.evaluate(probe,obs,critic,contexts,scale,offset)
 pm=extra_metrics(pe,baseline,probe,initial)
 # Mean displacement of each frequently executed component must stay tiny.
 component_drift=np.sqrt(((pe['means']-baseline['means'])**2).sum(-1))
 max_probe_drift=float(component_drift.max())
 assert max_probe_drift<=1e-3,max_probe_drift
 beta=float(np.median([row['gradients']['total']['mu']/row['gradients']['total']['p'] for row in calrows]))
 ratios=np.array([anchor_norm(row['gradients']['total'],beta)/row['gradients']['total']['rl'] for row in calrows])
 unit_ratio=float(np.median(ratios));lambdas={'BASELINE':0.,'WEAK':.1/unit_ratio,'MEDIUM':.3/unit_ratio,'STRONG':1./unit_ratio}
 calibration={'method':'25 deterministic production-RL virtual probe updates at unmodified LR; median ratios after each small update; all main branches reset to original 140K empty Adam','max_probe_normalized_component_L2':max_probe_drift,'beta_method':'median ||g_mu||/||g_KL|| balances unscaled full-Actor gradient norms','beta':beta,'lambda_method':'target / median(||g_mu + beta*g_KL|| / ||g_RL||)','lambdas':lambdas,'probe_metrics':pm,'samples':calrows,'ratios_by_branch':{b:{k:float(np.median([v*anchor_norm(row['gradients'][k],beta)/row['gradients'][k]['rl'] for row in calrows])) if any(row['gradients'][k]['rl']>0 for row in calrows) else None for k in ('total','rnn','mean','logits')} for b,v in lambdas.items()}}
 save(out/'calibration.json',calibration)
 print('CALIBRATION '+json.dumps({k:calibration[k] for k in ('max_probe_normalized_component_L2','beta','lambdas','ratios_by_branch')}),flush=True)
 result={'run':str(args.run),'device':str(device),'updates':1000,'schedule_kind':schedule['kind'],'calibration':str(out/'calibration.json'),'branches':{},'reference_integrity':[],'critic_frozen_hash':chash}
 def integrity(branch,u):
  norm=float(np.sqrt(sum(float(p.detach().cpu().double().square().sum()) for p in ref.parameters())))
  with torch.no_grad():d,_=T.forward(ref,obs[:64]);m=d.component_distribution.base_dist.loc.cpu().numpy();p=d.mixture_distribution.probs.cpu().numpy()
  check={'branch':branch,'update':u,'hash_unchanged':T.module_hash(ref)==refhash,'norm':norm,'norm_unchanged':norm==refnorm,'means_max_difference':float(np.max(abs(m-baseline['means'][:64]))),'probabilities_max_difference':float(np.max(abs(p-baseline['probs'][:64]))),'reference_grad_present':any(p.grad is not None for p in ref.parameters())}
  assert check['hash_unchanged'] and check['norm_unchanged'] and check['means_max_difference']==0 and check['probabilities_max_difference']==0 and not check['reference_grad_present']
  result['reference_integrity'].append(check)
 for branch in BRANCHES:
  clone=copy.deepcopy(actor);opt=new_optimizer(clone);lam=lambdas[branch]
  assert T.module_hash(clone)==refhash
  milestones={'0':extra_metrics(baseline,baseline,clone,initial)};gradrows={};start=time.monotonic()
  for u in range(1,1001):
   clone.train();ids=batches[(u-1)%4]
   for pg in opt.param_groups:pg['lr']=lrs[u-1]
   rl,mu,prob=losses(clone,ref,critic,obs[ids],tuple(c[ids] for c in contexts),scale,offset)
   loss=rl+lam*(mu+beta*prob)
   if u in MILESTONES:
    rg=gradients(clone,rl);mg=gradients(clone,mu);pg=gradients(clone,prob)
    g=summarize_grad(rg,mg,pg)
    gradrows[str(u)]={'RL_loss':float(rl.detach().cpu()),'L_mu':float(mu.detach().cpu()),'L_p':float(prob.detach().cpu()),'total_loss':float(loss.detach().cpu()),'gradients':g,'anchor_RL_ratio':{k:lam*anchor_norm(v,beta)/v['rl'] if v['rl']>0 else None for k,v in g.items()}}
    del rg,mg,pg
   opt.zero_grad(set_to_none=True);loss.backward()
   if not bool(torch.isfinite(loss)) or any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in clone.parameters()):raise FloatingPointError((branch,u))
   torch.nn.utils.clip_grad_norm_(clone.parameters(),cfg['actor_max_grad_norm']);opt.step()
   if u in MILESTONES:
    e=T.evaluate(clone,obs,critic,contexts,scale,offset)
    if any(not np.isfinite(v).all() for v in e.values()):raise FloatingPointError((branch,u,'outputs'))
    np.savez_compressed(out/f'{branch}_outputs_{u:04d}.npz',**e)
    m=extra_metrics(e,baseline,clone,initial);milestones[str(u)]=m;integrity(branch,u)
    print(json.dumps({'branch':branch,'update':u,'elapsed_s':time.monotonic()-start,'execution_drift':m['execution_drift'],'early':m['early'],'final':m['final'],'RL_improvement':m['RL_improvement']}),flush=True)
  base=result['branches'].get('BASELINE',{}).get('milestones',{}).get('1000',m)
  m['drift_reduction_vs_baseline']=1-m['execution_drift']/base['execution_drift']
  m['RL_retention_vs_baseline']=m['RL_improvement']/base['RL_improvement']
  result['branches'][branch]={'lambda':lam,'beta':beta,'milestones':milestones,'gradient_checks':gradrows,'finite':True}
  # New testing-only checkpoint, never a training checkpoint.
  path=out/f'{branch}_testing_actor_1000.pth'
  if path.exists():raise FileExistsError(path)
  torch.save({'testing_only':True,'actor':{k:v.detach().cpu() for k,v in clone.state_dict().items()},'actor_virtual_updates':1000,'source_env_steps':140000,'branch':branch,'lambda':lam,'beta':beta},path)
 assert T.module_hash(critic)==chash
 result['critic_hash_unchanged']=True
 previous=json.loads((args.previous/'offline_results.json').read_text())['branches']['CURRENT']['milestones']['1000']
 result['baseline_previous_CURRENT_differences']={k:result['branches']['BASELINE']['milestones']['1000'][k]-previous[k] for k in ('early','mid','late','final')}
 assert all(abs(v)<1e-5 for v in result['baseline_previous_CURRENT_differences'].values()),result['baseline_previous_CURRENT_differences']
 eligible=[b for b in ('WEAK','MEDIUM') if result['branches'][b]['milestones']['1000']['drift_reduction_vs_baseline']>=.3 and result['branches'][b]['milestones']['1000']['RL_retention_vs_baseline']>=.1]
 selected=eligible[0] if eligible else 'WEAK'
 result['closed_loop_selection']={'selected':selected,'eligible':eligible,'reason':'predeclared smallest lambda, >=30% drift reduction and >=10% RL retention' if eligible else 'no weak/medium qualified; WEAK diagnostic and STRONG upper bound, not an effective-anchor claim','branches':['ORIGINAL_140K','BASELINE',selected,'STRONG'],'success_conditioned_retuning':False}
 save(out/'offline_results.json',result)
 print('OFFLINE_COMPLETE '+json.dumps(result['closed_loop_selection']),flush=True)
if __name__=='__main__':main()
