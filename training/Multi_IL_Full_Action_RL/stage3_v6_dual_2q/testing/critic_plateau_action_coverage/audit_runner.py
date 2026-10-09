import sys,os,json,time,hashlib,collections
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,h5py
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
D=Path(__file__).resolve().parent; R=D.parents[2]; T=D.parent
RUN=Path('/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_readiness_v2_multi_mean_random_formal_20260929_130945'); G=RUN/'mean2q/multi_q'; OLD=T/'actor_critic_improvement_consistency'
os.chdir(D)
def convert(x):
 if isinstance(x,np.ndarray): return x.tolist()
 if isinstance(x,np.generic): return x.item()
 if isinstance(x,Path): return str(x)
 raise TypeError(type(x))
def dump(name,x): (D/name).write_text(json.dumps(x,indent=2,default=convert,allow_nan=False)+'\n')
def event(s): print(time.strftime('%H:%M:%S'),s,flush=True)
def loadrows(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
cfg=json.loads((RUN/'shared/config_resolved.json').read_text()); manifest=json.loads((RUN/'shared/stage2_source_manifest.json').read_text())
protected=[p for p in R.rglob('*') if p.is_file() and 'testing' not in p.parts and p.suffix in ['.py','.json','.sh']]
protected+=list(G.glob('*.json*'))+[G/'checkpoints/critic_ready.pth',G/'checkpoints/critic_ready.sequences.npy',Path(manifest['multi_q']['checkpoint']),RUN/'shared/bc_rnn_gmm_source.pth']+[Path(p) for p in cfg['offline_sources'].values()]
protected+=list(OLD.glob('*.npz'))+[OLD/'context_metadata.json']
before={str(p):sha(p) for p in sorted(set(protected))}; dump('safety_before.json',before); event('safety baseline '+str(len(before)))
rows=loadrows(G/'train_metrics.jsonl'); readyrows=loadrows(G/'readiness_metrics.jsonl'); rows0=[r for r in rows if r['env_steps']<=140000 and r.get('actor_updates',0)==0]; rows0=sorted({r['env_steps']:r for r in rows0}.values(),key=lambda r:r['env_steps']); x=np.array([r['env_steps'] for r in rows0]);
with (D/'critic_only_metrics.jsonl').open('w') as f:
 for r in rows0: f.write(json.dumps(r)+'\n')
plateau={}
for key in ['critic_loss_q1','critic_loss_q2']:
 y=np.array([r[key] for r in rows0]); stats=[]
 for width in [30000,15000,45000]:
  m=x>140000-width; xx=x[m]; yy=y[m]; edges=np.linspace(140000-width,140000,4); blocks=[yy[(xx>edges[i])&(xx<=edges[i+1])] for i in range(3)]; means=[float(b.mean()) for b in blocks]; rel=(means[-1]-means[0])/max(means[0],1e-15); slope=np.polyfit((xx-140000)/width,yy,1)[0]/yy.mean(); ac=float(np.corrcoef(yy[:-1],yy[1:])[0,1]);
  stats.append({'window':width,'records':len(yy),'block_means':means,'relative_block_change':rel,'relative_linear_change':float(slope),'std_over_mean':float(yy.std()/yy.mean()),'lag1_autocorrelation':ac,'effective_n_AR1_diagnostic':float(len(yy)*(1-ac)/(1+ac)),'pass':bool(abs(rel)<=.1 and abs(slope)<=.1),'continued_decrease':bool(rel<-.1 and slope<-.1)})
 status='PASS' if all(s['pass'] for s in stats) else ('FAIL' if all(s['continued_decrease'] for s in stats) else 'INCONCLUSIVE')
 plateau[key]={'status':status,'classification':{'PASS':'PLATEAU_CONFIRMED','FAIL':'PLATEAU_NOT_REACHED','INCONCLUSIVE':'PLATEAU_INCONCLUSIVE'}[status],'windows':stats}
plateau.update({'first_observed_critic_only_env_step':int(x[0]),'first_overall_ready_v2':next(r['env_steps'] for r in readyrows if r['overall_ready_v2']),'first_critic_ready':next(r['env_steps'] for r in readyrows if r['critic_ready']),'first_actor_logged_update':next({k:r[k] for k in ['env_steps','actor_updates','updates','gate_open_step']} for r in rows if r.get('actor_updates',0)>0),'readiness_rows':readyrows,'critic_lr_logged':'NOT_AVAILABLE','critic_only_lr_reconstructed':cfg['critic_lr'],'td_mae_full_training_curve':'NOT_AVAILABLE','td_mae_readiness_curve':[{k:r[k] for k in ['env_steps','td_e1_mae','td_e2_mae','td_target_mean','td_target_std']} for r in readyrows],'heldout_validation_status':'INCONCLUSIVE','independent_validation_curve':'NOT_AVAILABLE','limitations':['Moving TD target, continuously growing replay, averages of training batches, no independent convergence curve. Plateau is operational loss stability, not value correctness.','Stage2 validation trajectories enter Stage3 offline training verbatim.','Cannot quantify benefit of longer training without observed independent improvement; no retraining performed.']})
dump('critic_loss_plateau_audit.json',plateau)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,2,figsize=(12,7))
for i,key in enumerate(['critic_loss_q1','critic_loss_q2']):
 yy=np.array([r[key] for r in rows0]); ax[0,i].plot(x,yy,alpha=.25,lw=.6); rolling=np.convolve(yy,np.ones(25)/25,mode='valid'); ax[0,i].plot(x[24:],rolling,label='25 logged records mean'); ax[0,i].axvline(140000,c='r');ax[0,i].set_title(key);ax[0,i].set_yscale('log');ax[0,i].legend()
 for k in ['q'+str(i+1)+'_mean','q'+str(i+1)+'_std','td_target_mean','critic_grad_norm']:
  ax[1,i].plot(x,[r.get(k,np.nan) for r in rows0],label=k,alpha=.75)
 ax[1,i].legend();ax[1,i].set_xlabel('environment steps')
fig.tight_layout();fig.savefig(D/'critic_loss_curves.png',dpi=160);plt.close(fig); event('Round1 loss complete '+str({k:plateau[k]['status'] for k in ['critic_loss_q1','critic_loss_q2']}))
keys=['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','robot1_eef_pos','robot1_eef_quat','robot1_gripper_qpos','object']; episodes=[]
def mc(r):
 out=np.zeros(len(r),np.float32); v=0.
 for i in range(len(r)-1,-1,-1): v=float(r[i])+.99*v;out[i]=v
 return out
def scalar(g,k): return np.asarray(g.attrs[k] if k in g.attrs else g[k][()]).reshape(-1)[0].item()
for source,path in cfg['offline_sources'].items():
 with h5py.File(path,'r') as f:
  for name,g in sorted(f['episodes'].items()):
   a=np.asarray(g['actions'],np.float32);n=len(a);o=np.concatenate([np.asarray(g['obs'][k]).reshape(n,-1) for k in keys],1).astype(np.float32);r=np.asarray(g['rewards']).reshape(-1);seed=int(scalar(g,'initial_seed'))
   episodes.append({'id':len(episodes),'source':source,'seed':seed,'success':bool(scalar(g,'episode_success')),'o':o,'a':a,'r':r,'mc':mc(r),'complete':True,'stage2_train':seed<=10079})
replay=np.load(G/'checkpoints/critic_ready.sequences.npy',allow_pickle=True).item();el=[r for r in loadrows(G/'episode_metrics.jsonl') if r['env_steps']<=140000]; assert len(el)==len(replay['episodes'])
online_match=[]
for i,e in enumerate(replay['episodes']):
 log=el[i]; assert len(e['actions'])==log['length'] and bool(e['success'])==bool(log['success']); online_match.append({'index':i,'episode_id':log['episode_id'],'seed':log['seed'],'length':log['length'],'success':log['success']});r=np.asarray(e['rewards']).reshape(-1)
 episodes.append({'id':len(episodes),'source':'online_bc_rnn','seed':int(log['seed']),'success':bool(e['success']),'o':np.asarray(e['observations'],np.float32),'a':np.asarray(e['actions'],np.float32),'r':r,'mc':mc(r),'complete':True,'stage2_train':False})
# Active prefixes have no terminal return and are not independent completed episodes.
for env,e in sorted(replay['current'].items()):
 if len(e['actions'])==0: continue
 o=np.asarray(e['observations'],np.float32);a=np.asarray(e['actions'],np.float32);r=np.asarray(e['rewards']).reshape(-1)
 prev=[z for z in el if z['env_id']==int(env)];seed=int(prev[-1]['seed']+cfg['parallel_env']['num_envs']) if prev else int(cfg['train_seed_base']+int(env))
 episodes.append({'id':len(episodes),'source':'online_bc_rnn_prefix','seed':seed,'success':False,'o':o,'a':a,'r':r,'mc':np.full(len(a),np.nan,np.float32),'complete':False,'stage2_train':False})
dump('online_episode_log_alignment.json',{'method':'chronological completed-episode order, every length and success agrees; prefix seed reconstructed from verified new_context generation scheme','completed':online_match})
def epstats(es):
 n=[len(e['a']) for e in es];return {'episodes':len(es),'independent_seeds':len(set(e['seed'] for e in es)),'transitions':sum(n),'full_history_windows_10':sum(max(0,z-9) for z in n),'critic_successor_windows_11':sum(max(0,z-10) for z in n),'aligned_windows':sum(z//10 for z in n),'success_episodes':sum(e['success'] for e in es),'complete_episodes':sum(e['complete'] for e in es),'episode_lengths_quantiles':np.quantile(n,[0,.25,.5,.75,1])}
source_stats={s:epstats([e for e in episodes if e['source']==s]) for s in sorted(set(e['source'] for e in episodes))}
data={'by_source':source_stats,'stage2_train':epstats([e for e in episodes if e['stage2_train']]),'stage2_validation':epstats([e for e in episodes if e['source'].startswith('bc_') and not e['stage2_train']]),'stage3_all_offline':epstats(episodes[:300]),'ready_online_complete':epstats(episodes[300:554]),'ready_all':epstats(episodes),'replay_transition_counter':replay['transitions'],'stage3_sample_fractions':{'offline':.5,'online':.5,'offline_each_source':1/6},'distinct_episode_usage_counter':'NOT_AVAILABLE; available pool and cumulative draws do not establish every episode was sampled','validation_leakage':{'stage2_train_val_shared_seeds':0,'stage2_val_episodes_in_stage3_offline_training':60,'stage2_val_seeds_in_stage3_offline_training':20,'identical_whole_episodes_and_prefixes':True,'stage3_fixed_readiness_diagnostics':'drawn from training replay; not independent validation'},'sampling':{'online':'categorical GMM plus Gaussian component sample','normalized_eval_std':.0001,'external_noise':0,'pre_env_clipping':False,'realized_component_ID_frequencies':'NOT_AVAILABLE','raw_command':'same action sent to env and stored in replay; critic uses last raw actions field','controller_executed_torque_or_scaled_delta':'NOT_AVAILABLE in saved replay; controller transformations not interchangeable with raw command'},'task_phase':'object flags payload_in_target_bin index45 and trash_in_trash_bin46 (18+27,18+28); only coarse observable phase, no latent manipulation phase/goal field saved'}
dump('data_source_audit.json',data)
def actionstats(a):
 sd=a.std(0);cov=np.cov(a,rowvar=False);v=np.linalg.eigvalsh(cov).clip(0);rank=float(v.sum()**2/max((v*v).sum(),1e-20));uniq=len(np.unique(np.round(a,5),axis=0));return {'transitions':len(a),'mean':a.mean(0),'std':sd,'quantiles':np.quantile(a,[0,.01,.05,.5,.95,.99,1],axis=0),'correlation':np.corrcoef(a,rowvar=False),'covariance_effective_rank':rank,'unique_actions_rounded_1e_5':uniq,'repeat_fraction_rounded_1e_5':1-uniq/len(a),'raw_outside_nominal_bounds_fraction':float(np.mean(np.any(abs(a)>1,axis=1))),'actual_pre_env_clipping_fraction':0,'controller_internal_clipping_fraction':'NOT_AVAILABLE'}
action={s:actionstats(np.concatenate([e['a'] for e in episodes if e['source']==s])) for s in source_stats};action['stage2_train']=actionstats(np.concatenate([e['a'] for e in episodes if e['stage2_train']])); action['stage3_all']=actionstats(np.concatenate([e['a'] for e in episodes]));dump('action_coverage_audit.json',{'groups':action,'global_coverage_classification':'INCONCLUSIVE','interpretation':'Nonzero variance and many unique commands establish global variation; no universal sufficiency criterion for 14D candidate coverage. Conditional results reported separately.','component_labels':'NOT_AVAILABLE; no guessed component labels'})
fig,ax=plt.subplots(figsize=(10,4))
for s in source_stats: ax.plot(range(14),action[s]['std'],marker='.',label=s)
ax.set_xlabel('raw action dimension');ax.set_ylabel('global standard deviation');ax.legend();fig.tight_layout();fig.savefig(D/'global_action_variation.png',dpi=160);plt.close(fig);event('Round0/global complete')
# Geometry carries the full current 10-frame Critic history, not embedding-only distance.
H=[];A=[];E=[];S=[];P=[];C=[];V=[];U=[];W=[];src=[]
for e in episodes:
 for t in range(9,len(e['a']),10):
  o=e['o'][t-9:t+1];a=e['a'][t-9:t+1];H.append(np.r_[o.ravel(),a[:9].ravel(),t/700].astype(np.float32));A.append(a[-1]);E.append(e['id']);S.append(e['seed']);P.append(min(t//70,9));C.append(int(o[-1,45]>.5)+2*int(o[-1,46]>.5));V.append(e['mc'][t]);U.append(e['success']);W.append(len(e['a'])-1-t);src.append(e['source'])
H=np.stack(H);A=np.stack(A);E=np.array(E);S=np.array(S);P=np.array(P);C=np.array(C);V=np.array(V);U=np.array(U);W=np.array(W);src=np.array(src)
# Save every sampled reference ID; neighbouring windows never count as independent episodes.
np.savez_compressed(D/'reference_windows.npz',episode=E,seed=S,progress_bin=P,phase=C,action=A,return_label=V,remaining=W,source=src)
ref=np.flatnonzero(S%5!=0); calpool=np.flatnonzero(S%5==0);cal=calpool[np.linspace(0,len(calpool)-1,512,dtype=int)];center=H[ref].mean(0);scale=np.maximum(H[ref].std(0),.01);scale[-1]=.1;HS=(H-center)/scale;ascal=np.maximum(A[ref].std(0),.01)
rng=np.random.default_rng(20261009);proj=(rng.normal(size=(H.shape[1],48))/np.sqrt(48)).astype(np.float32);HP=HS@proj
class Neighbours:
 def __init__(self,indices):
  self.trees={}
  for p in range(10):
   for c in range(4):
    ids=indices[(P[indices]==p)&(C[indices]==c)]
    if len(ids): self.trees[p,c]=(ids,cKDTree(HP[ids]))
 def get(self,h,p,c,seed=None):
  if (p,c) not in self.trees: return np.array([],int),np.array([])
  ids,tree=self.trees[p,c];z=((h-center)/scale).astype(np.float32);_,ix=tree.query(z@proj,k=min(512,len(ids)));cand=ids[np.atleast_1d(ix)]
  if seed is not None: cand=cand[S[cand]!=seed]
  d=np.sqrt(np.mean((HS[cand]-z)**2,axis=1));order=np.argsort(d); seen=set();selected=[];dist=[]
  for j in order:
   ep=int(E[cand[j]])
   if ep in seen: continue
   seen.add(ep);selected.append(cand[j]);dist.append(d[j])
   if len(selected)==32: break
  return np.array(selected,int),np.array(dist)
geom=Neighbours(ref);old=[]
for i in cal:
 ids,dist=geom.get(H[i],int(P[i]),int(C[i]),int(S[i]));old.append((i,ids,dist))
r5=np.array([d[4] for _,ids,d in old if len(ids)>=5]);assert len(r5)>=30
radii={str(q):float(np.quantile(r5,q)) for q in [.95,.99]};local_cal={};actioncal=[];diversity=[]
for q in [.95,.99]:
 rad=radii[str(q)];values=[]
 for i,ids,dist in old:
  ns=ids[dist<=rad]
  if len(ns)>=5: values.append(float(np.min(np.sqrt(np.mean(((A[ns]-A[i])/ascal)**2,axis=1)))))
 local_cal[str(q)]={'eligible_cases':len(values),'action_radius':float(np.quantile(values,q)) if len(values)>=30 else None}
for i,ids,dist in old:
 ns=ids[dist<=radii['0.95']]
 if len(ns)<5:continue
 a=A[ns];cov=np.cov(a,rowvar=False);eig=np.linalg.eigvalsh(cov).clip(0);pairs=np.sqrt(np.mean(((a[:,None]-a[None,:])/ascal)**2,axis=-1));up=pairs[np.triu_indices(len(a),1)];vals=V[ns];valid=np.isfinite(vals)
 diversity.append({'query_reference_index':int(i),'independent_episodes':len(ns),'independent_seeds':len(set(S[ns])),'sources':dict(collections.Counter(src[ns])),'action_variance_per_dim':a.var(0),'covariance_effective_rank':float(eig.sum()**2/max(np.sum(eig*eig),1e-20)),'distinct_actions_1e_5':len(np.unique(np.round(a,5),axis=0)),'action_pair_rms_quantiles':np.quantile(up,[0,.1,.5,.9,1]),'return_range':float(np.ptp(vals[valid])) if valid.any() else None,'return_std':float(np.std(vals[valid])) if valid.any() else None,'all_labels_nearly_equal':bool(np.ptp(vals[valid])<1e-5) if valid.any() else None,'mixed_success_labels':bool(len(set(U[ns][valid]))>1),'remaining_steps_range':int(np.ptp(W[ns])),'history_neighbour_rms_quantiles':np.quantile(dist[dist<=radii['0.95']],[0,.5,1])})
conditional={'reference_windows':len(H),'reference_episodes':len(set(E)),'calibration_query_count':len(cal),'reference_split_windows':len(ref),'exclude_same_seed_across_sources':True,'calibration_is_not_Critic_heldout_validation':True,'history_radii':radii,'conditional_action_calibration':local_cal,'valid_calibration_histories':len(diversity),'insufficient_conditional_histories':len(cal)-len(diversity),'local_neighbour_summary':{'median_independent_episodes':float(np.median([z['independent_episodes'] for z in diversity])),'median_action_covariance_effective_rank':float(np.median([z['covariance_effective_rank'] for z in diversity])),'median_action_difference_rms':float(np.median([z['action_pair_rms_quantiles'][2] for z in diversity]))},'details':diversity,'classification':'CONDITIONAL_ACTION_COVERAGE_LIMITED' if len(diversity)<len(cal)*.95 else 'INCONCLUSIVE','limitations':['Aligned 1/10 sampling of available replay; local covariance cannot establish coverage of all candidate actions.','Projected shortlist of 512, then exact full standardized history distance; approximate nearest search.','Normal-neighbour radii establish relative geometry, not simulator state equality or causal identification.','Task phases use two observable success-progress flags; latent task phase and goal annotations unavailable.','Geometry split is independent of episodes/seeds, but all its data was available to Critic training.']}
dump('conditional_action_diversity.json',conditional)
valid=np.isfinite(V);formula=np.where(U,.99**W,0);residual=np.abs(V[valid]-formula[valid]);ret={'classification':'RETURN_DIFFERENCE_UNIDENTIFIABLE','finite_labels':int(valid.sum()),'formula_success_times_gamma_remaining_max_abs_error':float(residual.max()),'fraction_formula_error_below_1e_5':float(np.mean(residual<1e-5)),'conditional_neighbourhoods':len(diversity),'fraction_nearly_identical_labels':float(np.mean([z['all_labels_nearly_equal'] for z in diversity])),'fraction_mixed_success_labels':float(np.mean([z['mixed_success_labels'] for z in diversity])),'median_conditional_return_range':float(np.median([z['return_range'] for z in diversity])),'actual_historical_Bellman_targets_per_replay_transition':'NOT_AVAILABLE; moving target logs only aggregated','interpretation':'Labels contain observational success/timing signal. Differences are not paired same-state, same-continuation action advantages; source policy, state and horizon confound action effects.','stage2_source_policies':['bc_rnn','bc_transformer','bc_gmm'],'stage3_online_labels':'archived BC rollout MC used only as diagnostic; production training uses TD targets'}
dump('return_identifiability_audit.json',ret);event('Round2 geometry calibrated '+str(local_cal))
s2idx=np.flatnonzero((S<=10079)&np.char.startswith(src,'bc_'));s2ref=s2idx[S[s2idx]%5!=0];s2pool=s2idx[S[s2idx]%5==0];s2cal=s2pool[np.linspace(0,len(s2pool)-1,256,dtype=int)];s2geom=Neighbours(s2ref);s2old=[]
for i in s2cal:
 ns,ds=s2geom.get(H[i],int(P[i]),int(C[i]),int(S[i]));s2old.append((i,ns,ds))
s2r=float(np.quantile([ds[4] for _,ns,ds in s2old if len(ns)>=5],.95));s2details=[]
for i,ns,ds in s2old:
 ns=ns[ds<=s2r]
 if len(ns)<5:continue
 vv=np.linalg.eigvalsh(np.cov(A[ns],rowvar=False)).clip(0);s2details.append({'query_index':int(i),'independent_episodes':len(ns),'sources':dict(collections.Counter(src[ns])),'variance_per_dim':A[ns].var(0),'covariance_effective_rank':float(vv.sum()**2/max(np.sum(vv*vv),1e-20)),'return_range':float(np.ptp(V[ns]))})
conditional['stage2_train_only']={'windows':len(s2idx),'calibration_cases':256,'history_radius95':s2r,'eligible_histories':len(s2details),'details':s2details};dump('conditional_action_diversity.json',conditional)
# Existing Actor outputs and existing rollout commands: no Actor inference or new sampling.
bank=np.load(OLD/'fixed_replay_histories.npz'); cache=np.load(OLD/'frozen_numeric_evidence.npz');meta=json.loads((OLD/'context_metadata.json').read_text());qids=list(np.linspace(0,4095,512,dtype=int));qh=[];qseed=[];qactual=[]
for i in qids:
 o=bank['observations'][i];a=bank['actions'][i];st=int(bank['episode_steps'][i,-1]);qh.append(np.r_[o.ravel(),a[:9].ravel(),st/700]);qseed.append(None);qactual.append(a[-1])
traces={}
for owner,file in [(0,'READY'),(625,'FROZEN_625'),(1250,'FROZEN_1250')]:
 rs=loadrows(T/'mean_multi_collapse_diagnosis/round1'/ (file+'_trajectories.jsonl'))
 for seed in sorted(set(r['seed'] for r in rs)):traces[owner,seed]=sorted([r for r in rs if r['seed']==seed],key=lambda r:r['timestep'])
for i,m in enumerate(meta):
 if i<4096 or m['step']%10!=9:continue
 rs=traces[m['owner'],m['seed']];t=m['step'];o=np.array([r['observation_flat'] for r in rs[t-9:t+1]],np.float32);a=np.array([r['action'] for r in rs[t-9:t+1]],np.float32);qids.append(i);qh.append(np.r_[o.ravel(),a[:9].ravel(),t/700]);qseed.append(m['seed']);qactual.append(a[-1])
qh=np.stack(qh).astype(np.float32);qids=np.array(qids);qactual=np.stack(qactual);np.savez_compressed(D/'query_selection.npz',cache_indices=qids,histories=qh,recorded_raw_actions=qactual)
# Query the full archived training reference, excluding any exact self trajectory.
allgeom=Neighbours(np.arange(len(H)));neighbours=[];dists=[]
for j,h in enumerate(qh):
 p=min(int(round(h[-1]*700))//70,9);c=int(h[9*59+45]>.5)+2*int(h[9*59+46]>.5);ns,ds=allgeom.get(h,p,c,qseed[j])
 # Fixed bank omits episode identity. Zero-distance match identifies its entire source seed for exclusion.
 if qseed[j] is None and len(ds) and ds[0]<1e-6:ns,ds=allgeom.get(h,p,c,int(S[ns[0]]))
 neighbours.append(ns);dists.append(ds)
status={};distances={};support={}
for q in [.95,.99]:
 qr=str(q);rad=radii[qr];ar=local_cal[qr]['action_radius'];states=np.zeros((3,len(qids),5),np.int8);ad=np.full((3,len(qids),5),np.nan,np.float32);nc=[];nr=[]
 for j,(ns,ds) in enumerate(zip(neighbours,dists)):
  ns=ns[ds<=rad];nc.append(len(ns));nr.append(len(set(src[ns])))
  if len(ns)<5:continue
  for si,u in enumerate([0,625,1250]):
   candidate=cache[str(u)+'_means'][qids[j]];aa=np.sqrt(np.mean(((candidate[:,None,:]-A[ns][None,:,:])/ascal)**2,axis=-1));dd=aa.min(1);ad[si,j]=dd;states[si,j]=np.where(dd<=ar,2,1) if ar is not None else 3
 status[qr]=states;distances[qr]=ad
 result={}
 for si,u in enumerate([0,625,1250]):
  pp=cache[str(u)+'_probs'][qids];result[str(u)]={}
  groups={'fixed_replay':np.arange(len(qids))<512,'own_rollout':np.array([i>=4096 and meta[i].get('owner')==u for i in qids]),'all_contexts':np.ones(len(qids),bool)}
  for group,mask in groups.items():
   p=pp[mask];st=states[si,mask];res={'histories':int(mask.sum()),'history_unsupported_fraction':float(np.mean(st[:,0]==0)),'history_supported_action_unsupported_probability':float(np.mean(np.sum(p*(st==1),axis=1))),'history_and_action_supported_probability':float(np.mean(np.sum(p*(st==2),axis=1))),'inconclusive_probability':float(np.mean(np.sum(p*(st==3),axis=1))),'mean_eligible_independent_neighbours':float(np.mean(np.array(nc)[mask])),'mean_independent_sources':float(np.mean(np.array(nr)[mask])),'mean_mode_weights':p.mean(0),'mode_action_unsupported_fraction':np.mean(st==1,axis=0),'mode_history_and_action_supported_fraction':np.mean(st==2,axis=0)}
   result[str(u)][group]=res
 support[qr]=result
np.savez_compressed(D/'conditional_support_records.npz',cache_indices=qids,primary_status=status['0.95'],sensitivity_status=status['0.99'],primary_action_distance=distances['0.95'],sensitivity_action_distance=distances['0.99'])
actor_audit={'calibration':{'history_radii':radii,'action_radii':local_cal},'status_codes':{'0':'HISTORY_INSUFFICIENT_CONDITIONAL_SUPPORT','1':'HISTORY_SUPPORTED_ACTION_UNSUPPORTED','2':'HISTORY_AND_ACTION_RELATIVELY_SUPPORTED','3':'INCONCLUSIVE'},'results':support,'saved_actor_outputs_reused':True,'new_actor_inference':False,'sampling_std_normalized':{str(u):np.unique(cache[str(u)+'_std'][qids]).tolist() for u in [0,625,1250]},'action_shifts_fixed':{str(u):{'mean_weighted_absolute_per_dimension':np.mean(np.sum(cache[str(u)+'_probs'][qids[:512],:,None]*abs(cache[str(u)+'_means'][qids[:512]]-cache['0_means'][qids[:512]]),axis=1),axis=0),'weighted_mode_rms_shift':float(np.mean(np.sum(cache[str(u)+'_probs'][qids[:512]]*np.sqrt(np.mean((cache[str(u)+'_means'][qids[:512]]-cache['0_means'][qids[:512]])**2,axis=-1)),axis=1)))} for u in [625,1250]},'interpretation':'Mode centers and actual Gaussian sampling parameters reused; real mode IDs unavailable. Support probabilities are mixture weights on mode centers, not fabricated realized mode frequencies. Tiny Gaussian residual not assumed exactly zero. Geometry is observational relative coverage, not causal value error.'}
dump('actor_candidate_support_audit.json',actor_audit);event('Round3 support complete')
