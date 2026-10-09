"""Frozen conditional sensitivity and episode-disjoint diagnostics. No simulation."""
import os,sys,json,copy,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;TEST=HERE.parent;ROOT=HERE.parents[4];RL=ROOT/'training/Multi_IL_Full_Action_RL';OUT=HERE/'round1'
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for d in ['stage3_v5_rgmm_td3','stage3_v6_dual_2q','stage2_2_history_aware_critic']:sys.path.insert(0,str(RL/d))
sys.path.insert(0,str(TEST/'mean_multi_collapse_diagnosis'))
import torch
from core import setup,RUN,module_hash,strict_stage2_load
from stage3_v5_actor import load_exact_actor
from stage3_v5_agent import target_final_distribution_vectorized
from stage3_v5_history_critic import encode_replay_contexts,previous_actions
from sequence_dataset import load_splits
from sequence_sampler import SequenceSampler
from scipy.stats import spearmanr,rankdata
from scipy.spatial.distance import cdist

def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def corr(x,y):return float(spearmanr(x,y).statistic) if len(x)>2 and np.std(x)>1e-12 and np.std(y)>1e-12 else None
def stats(x):
 x=np.asarray(x,float);return dict(mean=float(x.mean()),std=float(x.std()),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max()))
def fit_predict(x,y,z):
 m=x.mean(0);sd=np.maximum(x.std(0),1e-4);x=(x-m)/sd;z=(z-m)/sd;x=np.c_[np.ones(len(x)),x];z=np.c_[np.ones(len(z)),z];pen=np.eye(x.shape[1]);pen[0,0]=0
 return z@np.linalg.solve(x.T@x+10*pen,x.T@y)
def main():
 started=time.time();OUT.mkdir(exist_ok=False);os.chdir(OUT);torch.set_num_threads(1)
 device,ready,ref,td,scale,offset=setup();assert str(device)=='npu:0';ref.eval().requires_grad_(False)
 manifest=json.loads((RUN/'shared/stage2_source_manifest.json').read_text());mc,payload=strict_stage2_load(manifest['multi_q']['checkpoint'],device);mc.eval().requires_grad_(False);critics={'ready':td,'MC':mc};actors={'readiness':ref}
 bc,_,_=load_exact_actor(RUN/'shared/bc_rnn_gmm_source.pth',device);bc.eval().requires_grad_(False);actors['BC']=bc
 for name,p in [('post625',TEST/'mean_multi_collapse_diagnosis/round1/actor_625.pth'),('post1250',TEST/'mean_multi_collapse_diagnosis/round1/actor_1250.pth'),('proposal_old',TEST/'competence_backtracking_test/accepted_block_10.pth')]:
  a=copy.deepcopy(ref);a.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['actor']);a.eval().requires_grad_(False);actors[name]=a
 oldstate=actors['proposal_old'].state_dict();full=torch.load(TEST/'competence_backtracking_test/proposal_block_17_full.pth',map_location='cpu',weights_only=False)['actor'];a=copy.deepcopy(actors['proposal_old'])
 with torch.no_grad():
  for n,p in a.named_parameters():p.copy_(oldstate[n]+(full[n].to(device)-oldstate[n])/512)
 a.eval().requires_grad_(False);actors['proposal_bad']=a;hashes={k:module_hash(v) for k,v in {**critics,**actors}.items()}
 cfg=json.loads((RL/'stage2_2_history_aware_critic/stage2_2_config.json').read_text());train,val=load_splits(cfg['dataset_root'],range(10000,10080),range(10080,10100),.99)
 checks={'next_observation_shift_max':0.,'return_recursion_max':0.,'first_previous_action_zero':True,'candidate_is_action_t':True,'next_history_shift':True}
 for data in [train,val]:
  for src,ds in data.items():
   for e in ds.episodes:
    checks['next_observation_shift_max']=max(checks['next_observation_shift_max'],float(np.max(np.abs(e.next_observations[:-1]-e.observations[1:]))))
    checks['return_recursion_max']=max(checks['return_recursion_max'],float(np.max(np.abs(e.returns[:-1]-(e.rewards[:-1]+.99*e.returns[1:])))))
    assert np.array_equal(e.dones,e.terminated|e.truncated) and not e.dones[:-1].any() and e.dones[-1]
 sampler=SequenceSampler(train,0,10,700,20261008,True);sample=sampler.sample(192)
 for i in range(len(sample['seed'])):
  e=next(e for e in train[sample['policy'][i]].episodes if e.seed==sample['seed'][i]);s,t=int(sample['start'][i]),int(sample['target_step'][i]);n=t-s+1
  assert np.all(sample['previous_actions'][i,0]==0);assert np.array_equal(sample['actions'][i,n-1],e.actions[t]);assert np.array_equal(sample['previous_actions'][i,1:n],e.actions[s:t]);assert sample['learning_mask'][i].sum()==1 and sample['learning_mask'][i,n-1,0];assert abs(float(sample['returns'][i,n-1,0])-float(e.returns[t]))<1e-7
 assert checks['next_observation_shift_max']<1e-6 and checks['return_recursion_max']<1e-6
 class Recording:
  def encode_history(self,o,pa,p):self.saved=(o,pa,p);return (o,o),(None,None)
 rc=Recording();o=torch.arange(2*10*59,dtype=torch.float32,device=device).reshape(2,10,59);a=torch.arange(2*10*14,dtype=torch.float32,device=device).reshape(2,10,14);st=torch.arange(20,30,device=device).repeat(2,1)
 encode_replay_contexts(rc,o,a,st,700,next_observations=o+1);assert torch.equal(rc.saved[1][:,1:],a[:,1:]) and bool((rc.saved[1][:,0]==0).all());assert torch.allclose(rc.saved[2],(st+1).float().unsqueeze(-1)/700)
 rows=[]
 for split,data in [('train',train),('val',val)]:
  for sid,(src,ds) in enumerate(data.items()):
   for e in ds.episodes:
    for t in np.unique(np.linspace(9,e.length-1,12,dtype=int)):
     rows.append(dict(region='offline_'+split,split=split,source=sid,seed=e.seed,episode=src+'_'+str(e.seed),step=int(t),o=e.observations[t-9:t+1],a=e.actions[t-9:t+1],steps=np.arange(t-9,t+1),mc=float(e.returns[t]),success=e.success))
 tracefiles={'BC_success':TEST/'mean_multi_collapse_diagnosis/round1/READY_trajectories.jsonl','early_degradation':TEST/'critic_gradient_rootcause/round1/evaluations/FIXED_STATES/OLD_ACCEPTED_BLOCK_10_trajectories.jsonl','proposal_degradation':TEST/'critic_gradient_rootcause/round1/evaluations/FIXED_STATES/BLOCK_17_trajectories.jsonl','collapsed':TEST/'mean_multi_collapse_diagnosis/round1/FROZEN_1250_trajectories.jsonl'}
 for region,p in tracefiles.items():
  raw=[json.loads(l) for l in p.read_text().splitlines()]
  for seed in sorted({x['seed'] for x in raw}):
   rr=sorted([r for r in raw if r['seed']==seed],key=lambda r:r['timestep'])
   for t in sorted(set(np.linspace(9,len(rr)-1,24,dtype=int).tolist()+([282] if len(rr)>282 else []))):
    h=rr[t-9:t+1];rows.append(dict(region=region,split='discovery',source=-1,seed=seed,episode=region+'_'+str(seed),step=t,o=np.asarray([r['observation_flat'] for r in h],np.float32),a=np.asarray([r['action'] for r in h],np.float32),steps=np.asarray([r['timestep'] for r in h]),mc=rr[t]['finite_mc_return'],success=rr[-1]['success']))
 arrays={k:np.stack([r[k] for r in rows]) for k in ['o','a','steps']};arrays.update({k:np.asarray([r[k] for r in rows]) for k in ['region','split','source','seed','episode','step','mc','success']});np.savez_compressed(OUT/'contexts.npz',**arrays)
 qs={};controls={};probe={};gradient={};actor_values={};candidates={}
 def ten(x):return torch.as_tensor(x,device=device,dtype=torch.float32)
 for cname,c in critics.items():
  qparts=[];ctparts={};pparts=[];gparts=[];eparts={};aparts=[]
  for start in range(0,len(rows),128):
   end=start+128;o=ten(arrays['o'][start:end]);a=ten(arrays['a'][start:end]);st=torch.as_tensor(arrays['steps'][start:end],device=device);pa=previous_actions(a,st);progress=st.float().unsqueeze(-1)/700
   with torch.no_grad():
    enc=encode_replay_contexts(c,o,a,st,700);ctx=tuple(z[:,-1] for z in enc);q1,q2=c.q_from_context(ctx,a[:,-1]);qparts.append(torch.cat([q1,q2],-1).cpu().numpy())
    values={'zero_action':c.q_from_context(ctx,torch.zeros_like(a[:,-1])),'fixed_action':c.q_from_context(ctx,ten(arrays['a'][0,-1]).expand(len(o),14)),'shuffled_action':c.q_from_context(ctx,a.flip(0)[:,-1])}
    for name,pvalue in [('progress_zero',torch.zeros_like(progress)),('progress_mid',torch.full_like(progress,.5)),('progress_shift',torch.clamp(progress+.1,0,1))]:
     z,_=c.encode_history(o,pa,pvalue);values[name]=c.q_from_context(tuple(x[:,-1] for x in z),a[:,-1])
    z,_=c.encode_history(o[:,-1:].expand_as(o),pa,progress);values['repeat_current_obs']=c.q_from_context(tuple(x[:,-1] for x in z),a[:,-1])
    z,_=c.encode_history(o,torch.zeros_like(pa),progress);values['zero_history_actions']=c.q_from_context(tuple(x[:,-1] for x in z),a[:,-1])
    for name,v in values.items():ctparts.setdefault(name,[]).append(torch.cat(v,-1).cpu().numpy())
   if not np.any(arrays['split'][start:end]!='train'):continue
   cand=[a[:,-1].clamp(-1,1)];names=['replay'];expect={}
   with torch.no_grad():
    for name,actor in actors.items():
     torch.manual_seed(20261008+start);torch.npu.manual_seed(20261008+start)
     dist,_=target_final_distribution_vectorized(actor,o,episode_steps=st,horizon=10);mu=dist.component_distribution.base_dist.loc*scale+offset;p=dist.mixture_distribution.probs
     cand.append((dist.sample()*scale+offset).clamp(-1,1));names.append(name)
     u,v=c.q_from_context(ctx,mu);expect[name]=torch.stack([(p*u.squeeze(-1)).sum(-1),(p*v.squeeze(-1)).sum(-1)],-1).cpu().numpy()
    b=cand[names.index('BC')];bad=cand[names.index('post1250')]
    for alpha in [.25,.5,.75]:cand.append((b+alpha*(bad-b)).clamp(-1,1));names.append('interpolation_'+str(alpha))
    stack=torch.stack(cand,1);u,v=c.q_from_context(ctx,stack);pparts.append(np.stack([u.squeeze(-1).cpu().numpy(),v.squeeze(-1).cpu().numpy()],-1));aparts.append(stack.cpu().numpy())
    for name,v in expect.items():eparts.setdefault(name,[]).append(v)
   aa=a[:,-1].detach().clone().requires_grad_(True);q1,q2=c.q_from_context(tuple(z.detach() for z in ctx),aa);g1=torch.autograd.grad(q1.sum(),aa,retain_graph=True)[0];g2=torch.autograd.grad(q2.sum(),aa)[0];gparts.append(torch.stack([g1,g2],1).detach().cpu().numpy())
  qs[cname]=np.concatenate(qparts);controls[cname]={k:np.concatenate(v) for k,v in ctparts.items()};probe[cname]=np.concatenate(pparts);gradient[cname]=np.concatenate(gparts);actor_values[cname]={k:np.concatenate(v) for k,v in eparts.items()};candidates[cname]=np.concatenate(aparts);print('CRITIC_DONE',cname,len(rows),flush=True)
 probeids=np.concatenate([np.arange(s,min(s+128,len(rows))) for s in range(0,len(rows),128) if np.any(arrays['split'][s:s+128]!='train')])
 np.savez_compressed(OUT/'probes.npz',probe_ids=probeids,candidate_names=np.asarray(names),**{f'{n}_q':v for n,v in qs.items()},**{f'{n}_candidate_q':v for n,v in probe.items()},**{f'{n}_candidate_actions':v for n,v in candidates.items()},**{f'{n}_gradient':v for n,v in gradient.items()},**{f'{n}_expected_{k}':v for n,vs in actor_values.items() for k,v in vs.items()},**{f'{n}_{k}':v for n,vs in controls.items() for k,v in vs.items()})
 result={'device':str(device),'no_simulator':True,'actor_updates':0,'critic_updates':0,'alignment_checks':checks,'episode_split':{'train':list(range(10000,10080)),'val':list(range(10080,10100)),'discovery_only_existing_traces':True,'seed_rule':'fixed linspace12 per offline episode,24 per trace plus existing fork282; no outcome-based selection'},'critics':{},'candidate_names':names,'candidate_probe_rows':len(probeids),'source_MC_step':payload.get('step')}
 progress=arrays['step']/700;src=np.eye(4)[np.maximum(arrays['source'],0)];knots=np.arange(.1,1,.1);px=np.c_[progress,progress**2,progress**3,np.maximum(progress[:,None]-knots,0),src];hx=np.c_[arrays['o'].reshape(len(rows),-1),arrays['a'][:,:-1].reshape(len(rows),-1),px];ax=np.c_[hx,arrays['a'][:,-1]];tr=arrays['split']=='train';va=arrays['split']=='val';y=arrays['mc'];baseline={}
 for name,x in [('progress_source',px),('history_only',hx),('history_action',ax)]:
  pred=fit_predict(x[tr],y[tr],x[va]);baseline[name]={'heldout_mse':float(np.mean((pred-y[va])**2)),'heldout_spearman':corr(pred,y[va]),'fit_rows':int(tr.sum()),'validation_rows':int(va.sum()),'method':'fixed ridge10; exploratory baseline, not causal action contribution'}
 result['diagnostic_baselines']=baseline
 for cn in critics:
  z=qs[cn].mean(-1);r={}
  for region in sorted(set(arrays['region'])):
   m=arrays['region']==region;r[region]={'n':int(m.sum()),'q_mc_spearman':corr(z[m],y[m]),'q_progress_spearman':corr(z[m],progress[m]),'mc_progress_spearman':corr(y[m],progress[m]),'mse':float(np.mean((z[m]-y[m])**2)),'q_std':float(z[m].std()),'control_mean_abs_change':{k:float(np.mean(abs(v[m].mean(-1)-z[m]))) for k,v in controls[cn].items()},'fixed_action_q_mc_spearman':corr(controls[cn]['fixed_action'][m].mean(-1),y[m]),'zero_action_q_mc_spearman':corr(controls[cn]['zero_action'][m].mean(-1),y[m])}
   pp=(arrays['region'][probeids]==region);q=probe[cn][pp];g=gradient[cn][pp]
   if pp.any():
    r[region]['candidate_range']=stats(q.mean(-1).max(-1)-q.mean(-1).min(-1));r[region]['gradient_q1_norm']=stats(np.linalg.norm(g[:,0]*scale.cpu().numpy(),axis=-1));r[region]['gradient_twin_cos']=stats(np.sum(g[:,0]*g[:,1],-1)/(np.linalg.norm(g[:,0],axis=-1)*np.linalg.norm(g[:,1],axis=-1)+1e-12));r[region]['post1250_minus_BC']=stats(q[:,names.index('post1250')].mean(-1)-q[:,names.index('BC')].mean(-1))
  X=np.c_[np.ones(va.sum()),px[va]];rankz=rankdata(z[va]);ranky=rankdata(y[va]);rz=rankz-X@np.linalg.lstsq(X,rankz,rcond=None)[0];ry=ranky-X@np.linalg.lstsq(X,ranky,rcond=None)[0]
  per_ep={e:corr(z[arrays['episode']==e],y[arrays['episode']==e]) for e in sorted(set(arrays['episode'][va]))};phase={str(i):corr(z[va&(progress>=i/5)&(progress<(i+1)/5)],y[va&(progress>=i/5)&(progress<(i+1)/5)]) for i in range(5)}
  f=hx[va];sd=np.maximum(hx[tr].std(0),.01);dist=cdist(f/sd,f/sd,'sqeuclidean')/f.shape[1];ep=arrays['episode'][va];dist[ep[:,None]==ep[None,:]]=np.inf;nn=np.argsort(dist,axis=1)[:,:12];local=[corr(z[va][ix],y[va][ix]) for ix in nn];nonnull=[v for v in local if v is not None]
  r['validation_controlled']={'partial_rank_progress_source':float(np.corrcoef(rz,ry)[0,1]),'phase_spearman':phase,'within_episode_spearman':per_ep,'neighbor_spearman_mean':float(np.mean(nonnull)) if nonnull else None,'neighbor_valid_count':len(nonnull),'neighbor_distance':stats(np.take_along_axis(dist,nn,1).mean(1)),'caveat':'nearest cross-episode histories approximate only; labels belong to executed neighbor actions'};result['critics'][cn]=r
 assert all(module_hash(v)==hashes[k] for k,v in {**critics,**actors}.items());result['model_hashes_unchanged']=True;result['wall_seconds']=time.time()-started;dump(OUT/'result.json',result)
 print(json.dumps({'baselines':baseline,'ready_val':result['critics']['ready']['offline_val'],'MC_val':result['critics']['MC']['offline_val'],'ready_BC':result['critics']['ready']['BC_success'],'MC_BC':result['critics']['MC']['BC_success']}),flush=True)
if __name__=='__main__':main()
