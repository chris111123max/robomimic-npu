"""Read-only frozen-network audit. No training or simulator entry points."""
import os,sys,json,hashlib,time,subprocess
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; TEST=HERE.parent; RL=TEST.parents[1]; ROOT=RL.parents[1]
os.chdir(HERE)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
def dump(name,d):
 p=HERE/name
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
start=time.monotonic()
prior=TEST/'mean_multi_collapse_diagnosis/round1'
preregister={'checkpoint_updates':[0,625,1250],'fixed_training_bank_batches':64,'histories_per_batch':64,'gaussian_independent_antithetic_pairs':32,'samples_per_component':64,'categorical':'integrate all5 modes exactly (Rao-Blackwell), equivalent expectation to mode sampling','offline_rng_seed':20261009,'batch_size':64,'occupancy':'all existing READY/FROZEN_625/FROZEN_1250 complete trajectories; equal seed weighting; discount gamma .99','support_reference':'first32 reconstructed bank batches; calibration last32; q99 threshold fixed before results','support_groups':['10step_observations','previous_actions','current_executed_actions','progress','critic_embedding','combined'],'support_scope':'aligned full10step endpoints only to avoid reset/phase/length confounding','simulator_default':'NO; only all5 user trigger conditions permit one validation','thresholds':'direction using paired Gaussian-only95% t MC interval, no repeated sampling; NN outside fixed calibrated99% is potential support shift, not OOD truth'}
dump('preregistration.json',preregister)
oldbaseline=read(TEST/'critic_continuation_semantics/safety_before.json')['files']
paths=[Path(p) for p in oldbaseline if Path(p).is_file()]
paths+=list(prior.glob('*'))
paths=[p for p in paths if p.is_file()]
before={str(p.resolve()):sha(p) for p in paths}
dump('safety_before.json',{'files':before,'fusion_exists':(ROOT/'fusion_result.json').exists()})
sys.path.insert(0,str(TEST/'mean_multi_collapse_diagnosis'))
from core import setup,build_batch_bank,module_hash,load_exact_actor,RUN
from stage3_v5_actor import flat_to_obs
from stage3_v5_agent import target_final_distribution_vectorized
from stage3_v5_history_critic import encode_replay_contexts
import torch,numpy as np
from scipy.spatial import cKDTree
print('BASELINE_READY',len(before),flush=True)
device,ready,bc,critic,scale,offset=setup();assert str(device)=='npu:0'
critic.eval().requires_grad_(False); bc.eval().requires_grad_(False)
actors={0:bc}; checkpoints=[]
for u in [0,625,1250]:
 p=RUN/'mean2q/multi_q/checkpoints/critic_ready.pth' if u==0 else prior/f'actor_{u}.pth'
 if u:
  actor,_,_=load_exact_actor(RUN/'shared/bc_rnn_gmm_source.pth',device)
  payload=torch.load(p,map_location='cpu',weights_only=False); assert payload['actor_virtual_updates']==u
  actor.load_state_dict(payload['actor'],strict=True);actor.eval().requires_grad_(False);actors[u]=actor
 checkpoints.append({'updates':u,'path':str(p),'sha256':sha(p),'state_key':'actor'})
for p in [RUN/'mean2q/multi_q/checkpoints/critic_ready.pth',RUN/'shared/bc_rnn_gmm_source.pth',Path(ready['online_sequence_replay'])]:
 before.setdefault(str(p),sha(p))
model_before={str(u):module_hash(a) for u,a in actors.items()}; model_before['critic']=module_hash(critic)
bank,_,replaypath=build_batch_bank(ready['config'],ready,64)
fixed={k:np.concatenate([b[k] for b in bank]) for k in ['observations','actions','episode_steps']}
assert fixed['observations'].shape==(4096,10,59)
np.savez_compressed(HERE/'fixed_replay_histories.npz',**fixed)
trace_paths={0:prior/'READY_trajectories.jsonl',625:prior/'FROZEN_625_trajectories.jsonl',1250:prior/'FROZEN_1250_trajectories.jsonl'}
traces={}; entries=[]; episode_summaries={}
for u,p in trace_paths.items():
 records={}
 for line in p.read_text().splitlines():
  x=json.loads(line);records.setdefault(x['seed'],[]).append(x)
 assert len(records)==4
 ep=[]
 for seed,rs in records.items():
  rs.sort(key=lambda x:x['timestep']);assert [x['timestep'] for x in rs]==list(range(len(rs)))
  obs=np.asarray([x['observation_flat'] for x in rs],np.float32)
  raw=np.asarray([x['action'] for x in rs],np.float32)
  actions=raw.copy() # exact env.step/Replay command; production disables pre-step clipping
  for t,r in enumerate(rs):
   g=sum(.99**(j-t)*rs[j]['reward'] for j in range(t,len(rs)))
   assert abs(g-r['finite_mc_return'])<1e-10
   s=max(0,t-9)
   entries.append({'owner':u,'seed':seed,'step':t,'o':obs[s:t+1],'a':actions[s:t+1],'raw_action':raw[t],'executed_action':actions[t],'steps':np.arange(s,t+1),'next_observation':obs[t+1] if t+1<len(rs) else None,'mc':g,'success':bool(rs[-1]['success'])})
  ep.append({'seed':seed,'length':len(rs),'success':bool(rs[-1]['success']),'start_MC':rs[0]['finite_mc_return'],'raw_outside_bounds_fraction':float(np.mean(np.any(np.abs(raw)>1,axis=1)))})
 traces[u]=records;episode_summaries[str(u)]=ep
assert len(set(tuple(r.keys()) for r in traces.values()))==1
print('INPUTS_READY',len(entries),'trace contexts',flush=True)
assert ready['config']['exploration']['external_action_noise_std']==0
assert not ready['config']['exploration']['clip_to_env_bounds']
# Archive exact source evidence. Internal controller transforms are the plant;
# Q action coordinate is the14dim denormalized command passed to env.step/replay.
source_specs=[('stage3_v5_rgmm_td3/stage3_v5_agent.py',304,374),('stage3_v5_rgmm_td3/stage3_v5_history_critic.py',27,84),('stage3_v5_rgmm_td3/stage3_v5_actor.py',67,112),('stage3_v5_rgmm_td3/stage3_v5_actor.py',240,322),('stage3_v5_rgmm_td3/stage3_v5_rollout.py',1,71),('stage3_v6_dual_2q/stage3_v6_agent.py',105,160),('stage3_v6_dual_2q/train_stage3_v6_vector.py',475,500),('stage3_v6_dual_2q/train_stage3_v6_vector.py',664,756),('stage3_v5_rgmm_td3/stage3_v5_replay.py',65,105)]
source=[]
for rel,a,b in source_specs:
 p=RL/rel;lines=p.read_text().splitlines();source.append({'path':str(p),'sha256':sha(p),'start':a,'end':b,'text':'\n'.join(f'{i+1}: {s}' for i,s in enumerate(lines) if a<=i+1<=b)})
controller=Path('/data/home/3220251075/lerobot_workspace/miniconda3/envs/robosuite_npu/lib/python3.10/site-packages/robosuite/controllers/parts/controller.py')
lines=controller.read_text().splitlines();source.append({'path':str(controller),'sha256':sha(controller),'start':149,'end':168,'text':'\n'.join(f'{i+1}: {s}' for i,s in enumerate(lines) if 149<=i+1<=168)})
dump('source_evidence.json',source)
md=['# Actual optimization/execution source audit','Production/checkpoints read-only. No structural bug assumed.','Actor gradient: negative mean category-weighted Q1 at means; Q2 diagnostic only. Probabilities and means have gradient; Gaussian std excluded from this RL objective. Critic frozen during Actor gradient, history independent of candidate; no detach of candidate. No Q/reward normalization in this path.','Actor windows start0,10,..., initialized zero, BPTT10, objective final token only. Critic zero-state sliding10 includes observation, previous raw env.step command and t/700; first predecessor0. These coincide at block-final t%10=9. Away from that phase Actor uses latest reset suffix, Critic sliding10; these are different representations, not automatically bugs or full physical state.','Execution: categorical sample then eval Gaussian std1e-4, normalization scale/offset; formal and saved eval external noise0, pre-env clipping disabled. Environment internal controller saturates/scales commands as part of transition. Q input remains raw14dim env.step command, not torques. Actual internal actuator trace NOT_MEASURABLE.','Formal BoundarySnapshotExecutor uses per-block parameter snapshot with bounded lag32; static frozen Actor audit has no version changes. Version-lag causality in formal collapse NOT_MEASURABLE here.']
for s in source:md+=['\n## '+s['path']+':'+str(s['start']),'SHA256 '+s['sha256'],'```python',s['text'],'```']
(HERE/'source_audit.md').write_text('\n\n'.join(md)+'\n')
# Array layout: all fixed aligned windows then all old closed-loop steps.
N=4096+len(entries); metrics={}; distributions={}; critic_embeddings=np.zeros((N,192),np.float32)
context_meta=[{'dataset':'fixed_replay','index':i,'step':int(fixed['episode_steps'][i,-1])} for i in range(4096)]+[{k:e[k] for k in ['owner','seed','step','mc','success']} for e in entries]
dump('context_metadata.json',context_meta)
for u in actors:
 metrics[u]={k:np.zeros((N,2),np.float64) for k in ['proxy','exec','exec_clipped_diagnostic']}
 metrics[u]['replicates']=np.zeros((N,32,2),np.float64)
 metrics[u]['action_samples']=np.zeros((N,5,64,14),np.float32) if False else None
 distributions[u]={'probs':np.zeros((N,5),np.float32),'means':np.zeros((N,5,14),np.float32),'std':np.zeros((N,5,14),np.float32),'bound_exceed':np.zeros(N,np.float64)}
train_eval_parity=0.
batches=[]
for s in range(0,4096,64):batches.append((np.arange(s,min(s+64,4096)),fixed['observations'][s:s+64],fixed['actions'][s:s+64],fixed['episode_steps'][s:s+64]))
for length in range(1,11):
 inds=[i for i,e in enumerate(entries) if len(e['o'])==length]
 for k in range(0,len(inds),64):
  sel=inds[k:k+64];batches.append((np.array(sel)+4096,np.stack([entries[i]['o'] for i in sel]),np.stack([entries[i]['a'] for i in sel]),np.stack([entries[i]['steps'] for i in sel])))
# Fixed Gaussian random numbers independent across histories, shared across Actors.
rng=np.random.default_rng(20261009)
with torch.no_grad():
 for number,(ids,on,an,sn) in enumerate(batches):
  obs=torch.as_tensor(on,device=device,dtype=torch.float32); acts=torch.as_tensor(an,device=device,dtype=torch.float32); steps=torch.as_tensor(sn,device=device,dtype=torch.long)
  enc=encode_replay_contexts(critic,obs,acts,steps,700);ctx=tuple(x[:,-1] for x in enc)
  critic_embeddings[ids]=torch.cat(ctx,-1).cpu().numpy()
  noise=rng.standard_normal((len(ids),5,32,14)).astype(np.float32);noise=np.concatenate([noise,-noise],axis=2)
  eps=torch.as_tensor(noise,device=device)
  baseline_samples=None
  for u,actor in actors.items():
   d,_=target_final_distribution_vectorized(actor,obs,steps,10)
   loc=d.component_distribution.base_dist.loc;std=d.component_distribution.base_dist.scale;p=d.mixture_distribution.probs
   if number==0:
    actor.train();td,_=target_final_distribution_vectorized(actor,obs,steps,10);actor.eval()
    train_eval_parity=max(train_eval_parity,float((td.component_distribution.base_dist.loc-loc).abs().max().cpu()),float((td.mixture_distribution.probs-p).abs().max().cpu()))
   means=loc*scale+offset
   q=critic.q_from_context(ctx,means);metrics[u]['proxy'][ids]=torch.stack([(p*x.squeeze(-1)).sum(-1) for x in q],-1).cpu().numpy()
   samples=(loc[:,:,None,:]+std[:,:,None,:]*eps)*scale+offset
   values=critic.q_from_context(ctx,samples.flatten(1,2))
   rep=torch.stack([(p[:,:,None]*x.squeeze(-1).reshape(len(ids),5,64)).sum(1) for x in values],-1)
   antithetic=(rep[:,:32]+rep[:,32:])/2
   metrics[u]['replicates'][ids]=antithetic.cpu().numpy();metrics[u]['exec'][ids]=antithetic.mean(1).cpu().numpy()
   cq=critic.q_from_context(ctx,samples.clamp(-1,1).flatten(1,2))
   metrics[u]['exec_clipped_diagnostic'][ids]=torch.stack([(p*x.squeeze(-1).reshape(len(ids),5,64).mean(-1)).sum(-1) for x in cq],-1).cpu().numpy()
   distributions[u]['probs'][ids]=p.cpu().numpy();distributions[u]['means'][ids]=means.cpu().numpy();distributions[u]['std'][ids]=std.cpu().numpy()
   distributions[u]['bound_exceed'][ids]=(p*(samples.abs()>1).any(-1).float().mean(-1)).sum(-1).cpu().numpy()
  if number%25==0:print('OFFLINE_BATCH',number,'/',len(batches),flush=True)
print('ROUND1_2_NETWORK_PASS_DONE',flush=True)
# Persist finite raw numerical evidence, not just summary averages.
raw={'critic_embeddings':critic_embeddings}
for u in actors:
 for k,v in metrics[u].items():
  if v is not None:raw[f'{u}_{k}']=v
 for k,v in distributions[u].items():raw[f'{u}_{k}']=v
np.savez_compressed(HERE/'frozen_numeric_evidence.npz',**raw)
def interval(replicates,weights):
 weights=np.asarray(weights,np.float64);weights=weights/weights.sum()
 x=np.einsum('n,nsk->sk',weights,replicates)
 mean=x.mean(0);se=x.std(0,ddof=1)/np.sqrt(32)
 return {'mean':mean.tolist(),'gaussian_only_MC_95_interval':np.stack([mean-2.039513*se,mean+2.039513*se],-1).tolist(),'se':se.tolist(),'scope':'32 independent antithetic Gaussian pairs; categorical exactly integrated; finite fixed histories. Not environment MC or trajectory/population confidence interval.'}
def aggregate(u,ids,w=None):
 if w is None:w=np.ones(len(ids))
 w=np.asarray(w);w=w/w.sum()
 m=metrics[u];base=metrics[0]
 def avg(x):return np.einsum('n,nk->k',w,x[ids]).tolist()
 result={'count':len(ids),'Jproxy':avg(m['proxy']),'Jexec':interval(m['replicates'][ids],w),'proxy_exec_gap':avg(m['proxy']-m['exec']),'delta_proxy_vs_BC':avg(m['proxy']-base['proxy']),'delta_exec_vs_BC':interval(m['replicates'][ids]-base['replicates'][ids],w),'clamped_command_Q_diagnostic_delta':avg(m['exec_clipped_diagnostic']-m['exec']),'probability_mean':np.einsum('n,nk->k',w,distributions[u]['probs'][ids]).tolist(),'probability_L1_change_mean':float(np.dot(w,np.abs(distributions[u]['probs'][ids]-distributions[0]['probs'][ids]).sum(-1))),'probability_top1_change_fraction':float(np.dot(w,np.argmax(distributions[u]['probs'][ids],-1)!=np.argmax(distributions[0]['probs'][ids],-1))),'component_means_RMS_change':float(np.sqrt(np.einsum('n,n->',w,((distributions[u]['means'][ids]-distributions[0]['means'][ids])**2).mean((1,2))))),'eval_std_min':float(distributions[u]['std'][ids].min()),'eval_std_max':float(distributions[u]['std'][ids].max()),'pre_env_clipping_rate':0.,'nominal_bound_exceed_probability':float(np.dot(w,distributions[u]['bound_exceed'][ids]))}
 # Coupled categorical quantiles, shared equal Gaussian noise cancels exactly.
 assert np.array_equal(distributions[u]['std'][ids],distributions[0]['std'][ids])
 quantiles=(np.arange(64)+.5)/64
 chosen=[]
 for v in [0,u]:
  cumulative=distributions[v]['probs'][ids].cumsum(-1)
  mode=(quantiles[None,:,None]>cumulative[:,None,:]).sum(-1).clip(0,4)
  chosen.append(distributions[v]['means'][ids][np.arange(len(ids))[:,None],mode])
 result['paired_categorical_quantile_env_command_drift_L2']=float(np.dot(w,np.linalg.norm(chosen[1]-chosen[0],axis=-1).mean(-1)))
 return result
fixed_ids=np.arange(4096)
proxy={str(u):aggregate(u,fixed_ids) for u in actors}
proxy_status='PASS' if all(proxy[str(u)]['delta_proxy_vs_BC'][0]>0 and proxy[str(u)]['delta_exec_vs_BC']['gaussian_only_MC_95_interval'][0][0]>0 for u in [625,1250]) else 'INCONCLUSIVE'
if any(proxy[str(u)]['delta_proxy_vs_BC'][0]>0 and proxy[str(u)]['delta_exec_vs_BC']['gaussian_only_MC_95_interval'][0][1]<=0 for u in [625,1250]):proxy_status='FAIL'
dump('proxy_execution_audit.json',{'status':proxy_status,'checkpoints':checkpoints,'critic':str(RUN/'mean2q/multi_q/checkpoints/critic_ready.pth'),'fixed_histories':4096,'teacher_forced':True,'same_critic_and_order':True,'same_actor_independent_RNN_rebuild':True,'actual_rollout_external_noise':0,'actual_rollout_pre_env_clip':False,'category_exact_gaussian_32_antithetic_pairs':True,'train_eval_mean_probability_parity_first64_max_abs':train_eval_parity,'results':proxy,'missing_early_1_250_actor_weights':'NOT_MEASURABLE full execution expectation; previous paired_q outputs only, no checkpoint and no retraining','history_data':'exact reconstructed64-batch training bank using saved sampler states; online plus offline sources, not pure BC occupancy','internal_actuator_actions':'NOT_MEASURABLE in these traces; Q uses same raw14dim command as production','classification_scope':'PASS means observed old-history proxy gains survive actual command distribution; not true return or universal execution equivalence'})
# Calibrate NN distances against held-out old data; aligned t%10=9 only.
obs10=np.zeros((N,10,59),np.float32); prior10=np.zeros((N,10,14),np.float32); actions1=np.zeros((N,14),np.float32); progress=np.zeros((N,1),np.float32)
for ids,on,an,sn in batches:
 L=on.shape[1];obs10[ids,-L:]=on;actions1[ids]=an[:,-1];progress[ids,0]=sn[:,-1]/700
 if L>1:prior10[ids,-L+1:]=an[:,:-1]
features={'10step_observations':obs10.reshape(N,-1),'previous_actions':prior10.reshape(N,-1),'current_executed_actions':actions1,'progress':progress,'critic_embedding':critic_embeddings}
features['combined']=np.concatenate([features[k]/np.sqrt(features[k].shape[1]) for k in ['10step_observations','previous_actions','current_executed_actions','progress']],-1)
reference_ids=np.arange(2048); calibration_ids=np.arange(2048,4096)
support={}; individual_distance={}
aligned=np.array([i for i,e in enumerate(entries) if e['step']%10==9])+4096
for name,x in features.items():
 mean=x[reference_ids].mean(0);sd=np.maximum(x[reference_ids].std(0),1e-3)
 z=(x-mean)/sd/np.sqrt(x.shape[1]);tree=cKDTree(z[reference_ids])
 cal=tree.query(z[calibration_ids],k=1,workers=1)[0];threshold=float(np.quantile(cal,.99))
 dist=tree.query(z[aligned],k=1,workers=1)[0];individual_distance[name]=dict(zip(aligned.tolist(),dist.tolist()))
 group={}
 for u in actors:
  mask=np.array([context_meta[i]['owner']==u for i in aligned]);ds=dist[mask]
  group[str(u)]={'count':int(mask.sum()),'median':float(np.median(ds)),'p99':float(np.quantile(ds,.99)),'outside_calibrated_q99_fraction':float(np.mean(ds>max(threshold,1e-12))),'potential_support_shift':bool(np.mean(ds>max(threshold,1e-12))>.01)}
 support[name]={'old_old_calibration_count':2048,'reference_count':2048,'old_old_median':float(np.median(cal)),'old_old_q99':threshold,'groups':group,'limits':'Empirical NN metric only; overlapping episodes/windows or duplicates may make calibration optimistic; not proof of true support or causal OOD.'}
 print('SUPPORT_COMPLETE',name,flush=True)
occupancy={}; phase_results={}; local=[]
for owner in actors:
 ids=np.array([i+4096 for i,e in enumerate(entries) if e['owner']==owner]);w=np.array([.99**entries[i-4096]['step'] for i in ids])
 occupancy[str(owner)]={'count':len(ids),'equal_seed_count':4,'discount_weighting':'gamma^t per recorded step; equal seeds before normalized finite-occupancy means','cross_actor_results':{str(u):aggregate(u,ids,w) for u in actors},'known_closed_loop':episode_summaries[str(owner)]}
 for u in actors:
  occupancy[str(owner)]['cross_actor_results'][str(u)]['predicted_finite_performance_difference_sum']=np.einsum('n,nk->k',w/4,(metrics[u]['exec']-metrics[0]['exec'])[ids]).tolist()
  occupancy[str(owner)]['cross_actor_results'][str(u)]['warning']='Approximate network advantages on compressed histories. This finite weighted sum is NOT an exact performance identity.'
 for phase in range(10):
  si=np.array([i for i in ids if context_meta[i]['step']%10==phase]);sw=np.array([.99**context_meta[i]['step'] for i in si])
  phase_results[f'{owner}_phase{phase}']=aggregate(owner,si,sw)
 for lo,hi in [(0,100),(100,200),(200,300),(300,400),(400,500),(500,700)]:
  si=np.array([i for i in ids if lo<=context_meta[i]['step']<hi]);sw=np.array([.99**context_meta[i]['step'] for i in si])
  if len(si):phase_results[f'{owner}_steps{lo}_{hi}']=aggregate(owner,si,sw)
 # Common same-seed time interval controls long failed-tail composition.
 common=np.array([i for i in ids if context_meta[i]['step']<len(traces[0][context_meta[i]['seed']])]);cw=np.array([.99**context_meta[i]['step'] for i in common])
 occupancy[str(owner)]['common_BC_time_range_result']=aggregate(owner,common,cw)
 for i in ids:
  reps=metrics[owner]['replicates'][i]-metrics[0]['replicates'][i];avg=reps.mean(0);se=reps.std(0,ddof=1)/np.sqrt(32)
  if owner and avg[0]+2.039513*se[0]<-1e-5:
   local.append({'updates':owner,'context_index':int(i),'seed':context_meta[i]['seed'],'step':context_meta[i]['step'],'delta_Q1':float(avg[0]),'delta_Q2':float(avg[1]),'gaussian_only_Q1_interval':[float(avg[0]-2.039513*se[0]),float(avg[0]+2.039513*se[0])],'proxy_Q':metrics[owner]['proxy'][i].tolist(),'executed_Q':metrics[owner]['exec'][i].tolist(),'physical_state_available':False,'phase':context_meta[i]['step']%10})
local.sort(key=lambda x:(x['updates'],x['step'],x['seed']))
# Distribution and neighboring changes from full saved traces, not parameter L2.
shifts={}
for u in actors:
 ids=np.array([i+4096 for i,e in enumerate(entries) if e['owner']==u]);commonpairs=[];motions=[]
 for seed,rs in traces[u].items():
  ref=traces[0][seed]
  for t,r in enumerate(rs):
   if t<len(ref):commonpairs.append(np.linalg.norm(np.array(r['observation_flat'])-np.array(ref[t]['observation_flat'])))
   if t+1<len(rs):motions.append(np.linalg.norm(np.array(rs[t+1]['observation_flat'])-np.array(r['observation_flat'])))
 shifts[str(u)]={'observation_common_seed_time_L2_median':float(np.median(commonpairs)),'neighbor_observation_change_L2_median':float(np.median(motions)),'progress_mean':float(progress[ids].mean()),'critic_embedding_mean':critic_embeddings[ids].mean(0).tolist(),'critic_embedding_coordinate_std_mean':float(critic_embeddings[ids].std(0).mean()),'raw_command_mean':actions1[ids].mean(0).tolist(),'same_initial_observation':all(np.array_equal(traces[u][s][0]['observation_flat'],traces[0][s][0]['observation_flat']) for s in traces[u])}
transport='PASS' if all(occupancy[str(u)]['cross_actor_results'][str(u)]['delta_exec_vs_BC']['gaussian_only_MC_95_interval'][0][0]>0 for u in [625,1250]) else 'INCONCLUSIVE'
if any(occupancy[str(u)]['cross_actor_results'][str(u)]['delta_exec_vs_BC']['gaussian_only_MC_95_interval'][0][1]<0 for u in [625,1250]):transport='FAIL'
dump('occupancy_shift_audit.json',{'prediction_transport_status':transport,'replay_to_onpolicy_causal_status':'INCONCLUSIVE','full_trajectories':12,'sampled_existing_transition_contexts':len(entries),'occupancy_results':occupancy,'support_calibration':support,'distribution_shift':shifts,'phase_and_timebin_results':phase_results,'local_negative_predictions_first20':local[:20],'limits':['Teacher-forced BC and new Actors independently rebuild suffix from same history; no use of another Actors hidden state','Ten-step history is not full physical state or proven sufficient Markov state','Support distances are calibrated diagnostics, not proof of OOD or cause','Different average absolute Q across progress/duration is not used as quality ordering','Source trace actions are raw env.step commands, not hidden controller actuator states','Gaussian-only intervals do not capture trajectory or network value error']})
first=local[0] if local else None
onpolicy_positive=transport=='PASS'
location='Critic/strategy-improvement link on new occupancy (positive estimated advantage with lower realized success)' if onpolicy_positive and proxy_status=='PASS' else 'Replay-conditioned gain transfer on visited histories' if transport=='FAIL' else 'No reliable early link uniquely localized'
earliest={'earliest_observable_checkpoint_updates':625,'updates_interval':'(0,625]; checkpoints1/250 missing; exact first update NOT_MEASURABLE','checkpoint':str(prior/'actor_625.pth'),'observed_success_counts':[4,3,0],'mechanism_location':location,'fixed_history_proxy_execution':proxy['625'],'new_history_conditional_advantage':occupancy['625']['cross_actor_results']['625'],'first_descriptive_negative_Q_context':first,'temporal_precedes_performance_decline':'INCONCLUSIVE: no earlier complete Actor/rollout checkpoint;625 already3/4. First adverse predicted context within saved episode is not onset of training failure.','support_shift_at_625':{k:v['groups']['625'] for k,v in support.items()},'evidence_level':'CONFIRMED measured network/trajectory quantities; SUPPORTED predictive/correlational link; INCONCLUSIVE unique causality','strict_physical_snapshot_available_for_first_conflict':'NOT_MEASURABLE: saved trajectory fields omit full simulation/controller state','Round4_trigger_checks':{'1_clear_earliest_conflict':False,'2_existing_trajectory_causal_gap':True,'3_strict_reproducible_physical_snapshot':False,'4_specific_discriminating_hypothesis':'INCONCLUSIVE','5_one4env_can_discriminate_two_candidates':False},'simulator_invocations':0}
dump('earliest_mismatch.json',earliest)
print('ROUND2_SUMMARIES_DONE','proxy',proxy_status,'prediction_transport',transport,flush=True)
true_start={str(u):float(np.mean([e['start_MC'] for e in episode_summaries[str(u)]])) for u in actors}
true_delta={str(u):true_start[str(u)]-true_start['0'] for u in actors}
theory={'finite_formula':'Jnew-JBC = E_new[sum_t gamma^t A_t^BC(s_t,a_t)] for matching full state/policy memory/time and terminal absorbing semantics; A=QBC-VBC','assumptions_not_proven':['Critic sliding history10 sufficient for hidden physics/controller/task state and Actor memory','Ready network accurate QBC and coherent VBC across full occupancy','Match finite700 horizon and success terminal vs TD truncation/bootstrapping semantics','Replay final-token objective guarantees pointwise new-occupancy improvement','Formal snapshot lag captured by static checkpoint comparison'],'weighted_estimate':{str(u):occupancy[str(u)]['cross_actor_results'][str(u)]['predicted_finite_performance_difference_sum'] for u in actors},'realized_start_return_mean':true_start,'realized_start_return_delta':true_delta,'statement':'Finite network sums are estimates, NOT true MC advantages or numerical theorem verification; intervals only Gaussian integration uncertainty.'}
dump('policy_improvement_theory_audit.json',theory)
remaining=['Unreliable matched conditional advantages on new histories, including approximation or finite-history aliasing','Replay final-token shared-parameter/continuation/visitation mismatch, without a unique demonstrated code error']
summary={'status':'COMPLETE_STOPPED','proxy_to_execution':proxy_status,'replay_to_onpolicy':'INCONCLUSIVE','replay_gain_prediction_transport':transport,'earliest_mismatch':location,'root_cause':'INCONCLUSIVE','checkpoints':checkpoints,'critic_checkpoint':str(RUN/'mean2q/multi_q/checkpoints/critic_ready.pth'),'fixed_replay_histories':4096,'fixed_unique_histories':int(np.unique(fixed['observations'].reshape(4096,-1),axis=0).shape[0]),'trajectory_contexts':len(entries),'completed_rollouts_reused':12,'proxy_results':proxy,'own_occupancy_results':{str(u):occupancy[str(u)]['cross_actor_results'][str(u)] for u in actors},'known_success_counts':{str(u):sum(e['success'] for e in episode_summaries[str(u)]) for u in actors},'realized_start_returns':true_start,'finite_weighted_advantage_prediction':theory['weighted_estimate'],'earliest_observable_updates':625,'exact_earliest_update':'NOT_MEASURABLE','specific_code_error':'NOT_CONFIRMED','remaining_alternatives':remaining,'simulator':{'invocations':0,'env_steps':0,'initialized':0,'used':0,'closed':0,'reason':'All five trigger conditions not met; no simulator created. Four-env contract applies only if invoked.'},'new_Actor_updates':0,'new_Critic_updates':0,'production_changed':'NO','formal_checkpoints_changed':'NO','formal_training':'STOPPED','next_step_proposed_only':'Separately authorized return-identifiable conditional advantage evaluation at earliest observed625-update histories with fixed continuation, distinguishing value/history error from replay/shared-policy mismatch. No next task started.','elapsed_seconds':time.monotonic()-start}
model_after={str(u):module_hash(a) for u,a in actors.items()};model_after['critic']=module_hash(critic)
assert model_before==model_after
changes=[p for p,h in before.items() if not Path(p).is_file() or sha(p)!=h];assert not changes,changes
assert (ROOT/'fusion_result.json').exists()==read(HERE/'safety_before.json')['fusion_exists']
active=[]
for line in subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines():
 if 'python' in line and any(k in line for k in ['train_stage','acceptance_runner','spawn_main']):active.append(line)
assert not active,active
safety={'protected_files':len(before),'files_unchanged':True,'model_hashes_unchanged':True,'model_hashes_before':model_before,'model_hashes_after':model_after,'fusion_existence_unchanged':True,'residual_simulator_or_formal_training':active,'new_files_confined_to':str(HERE)}
dump('safety_after.json',safety);summary['safety']=safety
proxy_rows='\n'.join('| {} | {:.9f} | {:.9f} | {:.9f} | {:.9f} | {:.9g} | {:.9g} |'.format(u,*proxy[str(u)]['delta_proxy_vs_BC'],*proxy[str(u)]['delta_exec_vs_BC']['mean'],*proxy[str(u)]['proxy_exec_gap']) for u in actors)
occupancy_rows='\n'.join('| {} | {} | {:.9f} | {:.9f} | {} | {:.9f} |'.format(u,occupancy[str(u)]['count'],*occupancy[str(u)]['cross_actor_results'][str(u)]['delta_exec_vs_BC']['mean'],sum(e['success'] for e in episode_summaries[str(u)]),true_start[str(u)]) for u in actors)
support_rows='\n'.join('| {} | {:.6g} | {:.3%} | {:.3%} | {:.3%} |'.format(k,v['old_old_q99'],*[v['groups'][str(u)]['outside_calibrated_q99_fraction'] for u in actors]) for k,v in support.items())
report=f'''# Actor-Critic Policy Improvement Consistency Audit

Earliest observable break: **{location}**.
Proxy -> Execution: **{proxy_status}**, limited to predicted-Q gain transport.
Replay -> On-policy causal conclusion: INCONCLUSIVE; measured conditional-Q
prediction transport: {transport}. Unique full-collapse root cause: INCONCLUSIVE.
Earliest deteriorated checkpoint625 (3/4), then1250 (0/4).
Exact onset NOT_MEASURABLE; observable interval(0,625], not a precise first update.

## Round0: actual chain

source_audit.md records absolute file/function paths, lines and source excerpts.
Objective is category-weighted Q1 at component means; Q2 diagnostic only.
mean2q combines TD targets, not Actor-gradient heads. Means/logits get gradient;
Gaussian std is excluded from this RL mean objective. Critic context is zero-state
sliding10 observation/prior env command/t700; candidate current action is separate.
Actor zero reset0,10,...; BPTT10 and block-final objective only. Other phases use
Actor reset suffix versus Critic sliding history. This is representation semantics,
not proof of a bug or full physical-state equality.

Rollout samples category then eval Gaussian std1e-4, inverts source normalization.
Production external noise0, pre-env clipping FALSE, confirmed in readiness config
and saved frozen runner. Artificially clamped commands would audit another policy.
Controller saturation/scaling belongs to the plant; Q input remains raw14dim
command, not torques. Actuator/internal physical states NOT_MEASURABLE from traces.
Clamped-Q diagnostics are separate, not production execution. Boundary snapshots
hold parameters per reset block with lag bound32; frozen static audit does not
establish lag causality in formal training.

## Round1: proxy versus actual command-distribution Q

Checkpoints:0,625,1250 with paths/hashes.1/250 have previous paired_q outputs only;
complete execution audit NOT_MEASURABLE without weights. No training reconstruction.
Same frozen Ready Critic,4096 exact reconstructed64-bank histories, same order
and normalization; mixed offline/online, not pure BC occupancy. Independently
rebuild each Actor RNN suffix; teacher forcing is not new-policy closed-loop history.

| Updates | Delta proxyQ1 | proxyQ2 | Delta execQ1 | execQ2 | gapQ1 | gapQ2 |
|---|---:|---:|---:|---:|---:|---:|
{proxy_rows}

Category integrated exactly over5 modes;32 independent antithetic Gaussian pairs
(64 samples/component), paired across Actors. No expansion/tuning. JSON includes
Gaussian-only95% t integration intervals; not environment-MC or Critic-error CIs.
Probabilities, means, std, nominal-bound exceedance, paired categorical-quantile
command drift and clamped-Q diagnostic are recorded. Train/eval mean/prob parity
first64 max error {train_eval_parity:.9g}. All model hashes unchanged, no optimizer.
PASS means actual sampling preserves predicted-Q gain on these histories, not
true policy improvement. Do not substitute learned training std for eval1e-4 std.
'''
report+=f'''
## Round2: old replay to visited distributions

All12 saved full trajectories reused, same four seeds at all stages. Complete
observation/action/reward/progress/success/termination sequences; finite returns
recomputed. Critic history and each Actor suffix reconstructed independently.
Missing final successor/opaque physical/actuator fields NOT_MEASURABLE.
Only Ready TD is used; MC source is not silently called pure BC Q.
Compare BC/new command distributions within the SAME history using same Critic;
never rank policies by absolute Q across different progress/remaining length.

| Owner updates | Histories | Discount-weighted DeltaQ1 | DeltaQ2 | Success/4 | Actual mean startG |
|---|---:|---:|---:|---:|---:|
{occupancy_rows}

JSON includes cross-Actor/occupancy values, all reset phases, preregistered time
bins, common seed/time ranges excluding extra failed tails, neighboring state
changes, previous/current actions, embeddings and GMM outputs; raw arrays saved.

Reference first32 old batches, calibration last32; support queries full aligned
t%10=9 windows only. Standardized coordinate RMS NN distance; fixed old-old99%
calibration, no threshold tuning:

| Feature | Old-old q99 | BC outside |625 outside |1250 outside |
|---|---:|---:|---:|---:|
{support_rows}

POTENTIAL_SUPPORT_SHIFT is only a geometric diagnostic. Overlapping windows or
high-dimensional embeddings can make calibration optimistic; it is not proof of
real support exclusion or causal OOD error. Progress/long failure tails can change
occupancy as an effect of failure. No proof shift preceded degradation:625 already
has3/4, earlier complete checkpoint/rollout pairs missing.

## Earliest mismatch and evidence strength

Earliest available partially degraded checkpoint625; collapse1250. earliest_mismatch
records checkpoint, old proxy/exec, new conditional advantages, local negative
predictions if present, raw context indices/seed/time and support diagnostics.
The first negative predicted context is descriptive, not the first training update
or proven collapse cause. A same10step input is not same complete simulator state.
Measured network and recorded trajectory facts CONFIRMED; the association between
predicted improvement and lower success SUPPORTED; unique mechanism INCONCLUSIVE.
No strict causal rescue intervention was performed in this task.

## Round3: policy improvement conditions

For matched full states/policy memory/time/terminal semantics, finite performance
difference uses E_new[sum_t gamma^t A_t^BC]. Infinite normalized discounted-occupancy
form requires the appropriate absorbing-tail conditions. Our estimates use gamma.99,
equal four seeds, all saved t, success stopfirst reward, failure stop700.
These approximate network advantages are NOT real MC advantages and do not verify
that identity numerically. History sufficiency, accurate/coherent old-policy value,
time-limit bootstrap matching and pointwise improvement on new occupancy are not
established. Formal snapshot/version behavior adds another unmeasured distinction.

If old proxy and execution improve and new-history estimated advantage remains
positive while actual return falls, occupancy shift alone is not the identified
cause. Remaining alternatives: unreliable matched advantages/history aliasing,
and replay-final-token/shared-policy/continuation/visitation improvement mismatch.
If a subset has negative estimated advantages, this is a candidate location and
correlation, not proof the network estimates those true advantages correctly.
No concrete code-level error proven to explain complete collapse. No LR, anchor,
step-size, UTD, weaker training or new gate direction pursued.

## Optional simulation and stop

Zero simulator calls, zero new env steps. Five trigger conditions not all met:
no reliably resolved early update and no saved strict physical/controller snapshot
for that earliest conflict; one discriminating intervention not established.
Thus no4-env pool was initialized;0/0/0 is reported honestly rather than fake4/4/4.
Prior frozen experiment worker contract is reused evidence, not a new simulation.
Production/checkpoints/history preserved; all protected hashes unchanged and no
residual simulator/formal training. New artifacts only in {HERE}.

ACTOR_CRITIC_IMPROVEMENT_CONSISTENCY_DIAGNOSIS_COMPLETE

FORMAL_TRAINING_REMAINS_STOPPED
'''
if proxy_status=='FAIL':summary['next_step_proposed_only']='Targeted verification of the identified proxy/execution mechanism with a minimal correction, only after separate authorization.'
elif transport=='FAIL':summary['next_step_proposed_only']='Targeted mechanism validation of replay-to-visited-history conditional gain failure, only after separate authorization.'
report+='\nUnique next step (proposed only): '+summary['next_step_proposed_only']+'\n'
(HERE/'FINAL_REPORT.md').write_text(report)
dump('final_summary.json',summary)
print(json.dumps({'DONE':True,'proxy':proxy_status,'prediction_transport':transport,'success_counts':summary['known_success_counts'],'fixed_histories':4096,'trajectory_contexts':len(entries),'protected_files':len(before),'elapsed_seconds':time.monotonic()-start}),flush=True)
