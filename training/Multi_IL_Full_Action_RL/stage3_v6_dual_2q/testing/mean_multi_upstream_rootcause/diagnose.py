"""Mean-only upstream causal diagnostics. No production mutation."""
import sys,json,copy,hashlib,subprocess,os
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
TEST=HERE.parent
PREV=TEST/'mean_multi_collapse_diagnosis'
sys.path.insert(0,str(PREV))
from core import *
HERE=Path(__file__).resolve().parent
OUT=HERE
from stage3_v5_history_critic import encode_replay_contexts
from run_optimizer_step_probe import sampled_execution_stream
from scipy.spatial.distance import cdist
ALPHAS=(0.,.25,.5,.75,1.,1.25)

def report(p,body):
 p=Path(p);assert not p.exists();p.write_text(body)
def loadj(p):return json.loads(Path(p).read_text())
def safety():
 old=loadj(PREV/'safety_before.json')
 return {k:hashlib.sha256(Path(k).read_bytes()).hexdigest() for k in old if Path(k).is_file()}
def round0():
 out=OUT/'round0';out.mkdir(parents=True,exist_ok=True)
 evidence={}
 for p in (PREV/'round0_existing').glob('*.json'):evidence[str(p)]=loadj(p)
 for p in [PREV/'round1/result.json',PREV/'round2/result.json',PREV/'final_summary.json']:
  evidence[str(p)]=loadj(p)
 history={}
 for directory in ['actor_objective_alignment','actor_temporal_supervision','actor_init_anchor','adaptive_policy_preservation','actor_optimizer_diagnostics']:
  history[directory]={str(p):p.read_text() for p in (TEST/directory).glob('*REPORT*.md')}
 evidence['historical_reports']=history
 evidence['causal_graph']='frozen ready Critic ascent -> both Q increase -> actions drift -> success collapses; execution retraction rescues'
 evidence['already_causal']=['frozen ready learned-Q ascent sufficient','executed-action displacement dose response']
 evidence['unknown']='Why does the readiness Critic assign increasing value to Actor movement that eventually damages real closed-loop success?'
 dump(out/'existing_evidence.json',evidence)
 dump(OUT/'safety_before.json',safety())
 report(out/'ROUND_REPORT.md',"""# Round 0
Test: read formal checkpoint/log metadata and all previous evidence, including mean-only frozen experiment and retraction.
Original collapse: formal last online success at153973; frozen1250 updates gives0/4. Before collapse: both Q already favor drift at625.
Causal: frozen optimization and action rollback; off-support and bootstrap provenance UNKNOWN.
Next: on actual successful reference trajectory contexts, scan the REAL drift direction; do not rerun existing full-policy interpolation rollouts.
DECISION: CONTINUE_TO_ROUND_1
""")
 print('ROUND0_DONE',flush=True)

def prepare():
 device,ready,reference,critic,scale,offset=setup()
 bad=copy.deepcopy(reference)
 bad.load_state_dict(torch.load(PREV/'round1/actor_1250.pth',map_location='cpu',weights_only=False)['actor'])
 bad.eval();reference.eval()
 records=[json.loads(l) for l in (PREV/'round1/READY_trajectories.jsonl').read_text().splitlines()]
 contexts={'observations':[],'actions':[],'episode_steps':[],'mc':[],'seed':[]}
 for seed in SEEDS:
  rows=[r for r in records if r['seed']==seed]
  assert rows[-1]['success']
  for t in range(9,len(rows),10):
   window=rows[t-9:t+1]
   contexts['observations'].append([r['observation_flat'] for r in window])
   contexts['actions'].append([r['action'] for r in window])
   contexts['episode_steps'].append([r['timestep'] for r in window])
   contexts['mc'].append(rows[t]['finite_mc_return'])
   contexts['seed'].append(seed)
 return device,ready,reference,bad,critic,scale,offset,{k:np.asarray(v) for k,v in contexts.items()}

def round1():
 out=OUT/'round1';out.mkdir(exist_ok=True)
 device,ready,ref,bad,critic,scale,offset,data=prepare()
 obs=torch.as_tensor(data['observations'],device=device,dtype=torch.float32)
 refout=actor_outputs(ref,obs,device);badout=actor_outputs(bad,obs,device)
 # Same-RNG sampled actions at aligned horizon end, same reference-visited observations.
 sample0=sampled_execution_stream(ref,obs,device)[:,-1]*scale+offset
 sample1=sampled_execution_stream(bad,obs,device)[:,-1]*scale+offset
 actions={'sampled':(sample0,sample1),
          'weighted':(refout['weighted'][:,-1]*scale+offset,badout['weighted'][:,-1]*scale+offset)}
 scores={};raw={}
 with torch.no_grad():
  for kind,(a,b) in actions.items():
   scores[kind]=[];raw[kind]={}
   for alpha in ALPHAS:
    q1s=[];q2s=[]
    for lo in range(0,len(obs),32):
     ctxobs=obs[lo:lo+32]
     ca=torch.as_tensor(data['actions'][lo:lo+32],device=device,dtype=torch.float32)
     st=torch.as_tensor(data['episode_steps'][lo:lo+32],device=device,dtype=torch.long)
     enc=encode_replay_contexts(critic,ctxobs,ca,st,700)
     q1,q2=critic.q_from_context((enc[0][:,-1],enc[1][:,-1]),a[lo:lo+32]+alpha*(b[lo:lo+32]-a[lo:lo+32]))
     q1s.extend(q1.cpu().numpy().reshape(-1));q2s.extend(q2.cpu().numpy().reshape(-1))
    x=np.array(q1s);y=np.array(q2s);raw[kind][str(alpha)]={'q1':x.tolist(),'q2':y.tolist()}
    baseline=raw[kind]['0.0']
    d1=x-np.array(baseline['q1']);d2=y-np.array(baseline['q2'])
    scores[kind].append({'alpha':alpha,'q1':float(x.mean()),'q2':float(y.mean()),'qmean':float(((x+y)/2).mean()),'delta_q1':float(d1.mean()),'delta_q2':float(d2.mean()),'both_increase_fraction':float(np.mean((d1>0)&(d2>0))),'twin_median':float(np.median(abs(x-y))),'twin_p95':float(np.percentile(abs(x-y),95)),'normalized_reference_distance':float(torch.linalg.vector_norm(alpha*(b-a)/scale,dim=-1).mean())})
   data[kind+'_ref']=a.cpu().numpy();data[kind+'_bad']=b.cpu().numpy()
 data['ref_component_means']=refout['means'][:,-1].cpu().numpy()
 data['bad_component_means']=badout['means'][:,-1].cpu().numpy()
 np.savez_compressed(out/'successful_contexts.npz',**data)
 behavior=loadj(PREV/'round2/result.json')['behavior']
 dump(out/'result.json',{'contexts':len(obs),'all_successful_reference_trajectories':True,'scores':scores,'raw':raw,'behavior_reused':behavior,'behavior_endpoint_reused':loadj(PREV/'round1/result.json')['behavior'],'scope':'fresh successful trajectory landscape; old valid 4-env behavioral interpolation reused, not duplicate rollout','mc_caveat':'recorded MC belongs ONLY to executed reference actions, NOT candidate interpolated actions','component_mean_displacement_rms':float(torch.sqrt(torch.mean((badout['means']-refout['means'])**2))),'classification':'OFF_BEHAVIOR_CRITIC_OVEROPTIMISM_SUPPORTED' if scores['sampled'][-2]['delta_q1']>0 and scores['sampled'][-2]['delta_q2']>0 else 'STATE_DEPENDENT_OR_INCONCLUSIVE'})
 print(json.dumps(scores),flush=True)

def support_neighbors(query,steps,bank,k):
 features=np.concatenate([b['observations'][:,-1] for b in bank])
 actions=np.concatenate([b['actions'][:,-1] for b in bank])
 timesteps=np.concatenate([b['episode_steps'][:,-1] for b in bank])
 sd=np.maximum(features.std(0),.01)
 # No exact transition duplication inflating local density.
 unique=np.unique(np.round(np.c_[features,actions,timesteps],6),axis=0,return_index=True)[1]
 features=features[unique];actions=actions[unique];timesteps=timesteps[unique]
 d=cdist(query/sd,features/sd,'sqeuclidean')/features.shape[1]+((steps[:,None]-timesteps[None,:])/700)**2
 idx=np.argpartition(d,k-1,axis=1)[:,:k]
 return actions[idx],{'unique_transitions':len(unique),'neighbors':k,'obs_std_floor':.01,'state_metric':'mean standardized observation squared distance plus normalized episode-time squared distance'}
def support_stats(a,neighbors,scale):
 n=neighbors/scale
 nn=np.linalg.norm(a[:,None]/scale-n,axis=-1).min(1)
 pair=np.linalg.norm(n[:,:,None]-n[:,None,:],axis=-1)
 pair[:,np.arange(n.shape[1]),np.arange(n.shape[1])]=np.inf
 thresholds=np.percentile(pair.min(2),95,axis=1)
 center=n.mean(1);rad=np.percentile(np.linalg.norm(n-center[:,None],axis=-1),95,axis=1)
 return {'nearest_action_distance_mean':float(nn.mean()),'median':float(np.median(nn)),'p95':float(np.percentile(nn,95)),'outside_leave_one_neighbor_out_p95_fraction':float(np.mean(nn>thresholds)),'outside_local_ball_p95_fraction':float(np.mean(np.linalg.norm(a/scale-center,axis=-1)>rad)),'radius_mean':float(rad.mean())},nn,thresholds

def round2():
 out=OUT/'round2';out.mkdir(exist_ok=True)
 device,ready,ref,bad,critic,scale,offset,_=prepare()
 bank,probe,replay=build_batch_bank(ready['config'],ready,64)
 data=dict(np.load(OUT/'round1/successful_contexts.npz'))
 rows={};arrays={}
 for k in (8,32,64):
  near,contract=support_neighbors(data['observations'][:,-1],data['episode_steps'][:,-1],bank,k)
  rows[str(k)]={}
  for kind in ('sampled','weighted'):
   rows[str(k)][kind]=[]
   for a in ALPHAS:
    act=data[kind+'_ref']+a*(data[kind+'_bad']-data[kind+'_ref'])
    st,nn,threshold=support_stats(act,near,scale.cpu().numpy())
    rows[str(k)][kind].append({'alpha':a,**st})
    if k==32:arrays[f'{kind}_{a}_nn']=nn;arrays[f'{kind}_{a}_threshold']=threshold
 np.savez_compressed(out/'support_raw.npz',**arrays)
 dump(out/'result.json',{'sources':{'replay':str(replay),'offline':ready['config']['offline_sources']},'support_contract':contract,'metrics':rows,'caveats':['conditional nearest-state action neighborhood is density PROXY, not exact manifold','64 fixed production batches; no claim of exhaustive replay support','MC for non-executed candidate actions unknown'],'intervention_precommit':{'kind':'remove outward radial mean-output gradient outside local replay action p95 ball','k':32,'radius_percentile':95,'updates':1250,'control':'same 64 batches and original production gradient','anti_freeze_check':'report actual action drift and Q gain retention; no tuning if fails'}})
 print(json.dumps(rows['32']),flush=True)

def round3():
 out=OUT/'round3';out.mkdir(exist_ok=True)
 device,ready,ref,bad,critic,scale,offset,_=prepare()
 cfg=ready['config'];ch=module_hash(critic)
 bank,probe,replay=build_batch_bank(cfg,ready,64)
 query=np.concatenate([b['observations'][:,-1] for b in bank])
 steps=np.concatenate([b['episode_steps'][:,-1] for b in bank])
 # For training queries omit exact own state/action duplicates before nearest neighbors.
 # support_neighbors uses state matching; ball includes own replay action, intended support.
 near,contract=support_neighbors(query,steps,bank,32)
 normalized=(near-offset.cpu().numpy())/scale.cpu().numpy()
 neighbor_tensor=torch.as_tensor(normalized,device=device,dtype=torch.float32)
 pairs=np.linalg.norm(normalized[:,:,None]-normalized[:,None,:],axis=-1)
 pairs[:,np.arange(32),np.arange(32)]=np.inf
 radii=torch.as_tensor(np.percentile(pairs.min(2),95,axis=1),device=device,dtype=torch.float32)
 envs,lrs=schedule(cfg,1250)
 probeobs=torch.as_tensor(probe['observations'],device=device,dtype=torch.float32)
 baseout=actor_outputs(ref,probeobs,device)
 contexts=dict(np.load(TEST/'actor_collapse_diagnosis/results/fixed_contexts.npz'))
 results={};actors=[]
 for branch in ('BASELINE','SUPPORT_RADIAL_REJECTION'):
  actor=copy.deepcopy(ref)
  optim=torch.optim.Adam(actor.parameters(),lr=0.)
  optim.load_state_dict(copy.deepcopy(ready['actor_optimizer']))
  totals={'outward_removed':0,'output_vectors':0,'relative_gradient_removed_sum':0.}
  metrics={}
  with (out/f'{branch}_trace.jsonl').open('x') as f:
   for u in range(1,1251):
    actor.train();j=(u-1)%64
    for pg in optim.param_groups:pg['lr']=lrs[u-1]
    optim.zero_grad(set_to_none=True)
    loss,extra=production_actor_loss(actor,critic,bank[j],scale,offset,cfg,device)
    local={}
    if branch!='BASELINE':
     means=extra['means_normalized']
     neighbors=neighbor_tensor[j*64:(j+1)*64]
     dist=(means.detach()[:,:,None]-neighbors[:,None]).square().sum(-1)
     ix=dist.argmin(-1)
     center=neighbors.gather(1,ix[...,None].expand(-1,-1,14))
     radius=radii[j*64:(j+1)*64,None]
     d=means.detach()-center
     outside=torch.linalg.vector_norm(d,dim=-1)>radius
     def hook(g):
      dot=(g*d).sum(-1)
      reject=outside & (dot<0)
      removed=torch.where(reject[...,None],dot[...,None]*d/d.square().sum(-1,keepdim=True).clamp_min(1e-12),torch.zeros_like(g))
      local.update(rejected=int(reject.sum()),vectors=reject.numel(),fraction=float(torch.linalg.vector_norm(removed)/torch.linalg.vector_norm(g).clamp_min(1e-12)))
      return g-removed
     means.register_hook(hook)
    loss.backward()
    assert bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
    torch.nn.utils.clip_grad_norm_(actor.parameters(),cfg['actor_max_grad_norm'])
    optim.step()
    if local:
     totals['outward_removed']+=local['rejected'];totals['output_vectors']+=local['vectors'];totals['relative_gradient_removed_sum']+=local['fraction']
    f.write(json.dumps({'update':u,'loss':float(loss),'lr':lrs[u-1],**local},allow_nan=False)+'\n');f.flush()
    if u%100==0:print(json.dumps({'branch':branch,'update':u,'loss':float(loss),**local}),flush=True)
    if u in (625,1250):
     actor.eval();q,raw=qprobe(actor,ref,critic,contexts,device,scale,offset)
     params={n:p.detach() for n,p in ref.named_parameters()}
     drift=float(torch.sqrt(sum((p-params[n]).square().sum() for n,p in actor.named_parameters())))
     metrics[str(u)]={'q':q,'policy_drift':policy_drift(actor_outputs(actor,probeobs,device),baseout),'parameter_drift_unique_l2':drift}
   assert module_hash(critic)==ch
  results[branch]={'actor_updates':1250,'critic_updates':0,'metrics':metrics,'projection':totals}
  actors.append((branch,[copy.deepcopy(actor)],1))
 dump(out/'offline.json',results)
 os.chdir(out)
 behavior,contractenv=evaluate_four(actors,scale,offset,out)
 dump(out/'result.json',{'branches':results,'behavior':behavior,'env_contract':contractenv,'support_contract':contract,'critic_hash_unchanged':module_hash(critic)==ch,'actor_objective':'production unchanged; intervention removes only off-support outward mean gradient','caveat':'output-gradient projection does not strictly constrain Adam parameter-step geometry or prevent logits movement; frozen Critic, no claims of TD learning','formal_training_stopped':True})
 print('ROUND3_DONE',flush=True)

if __name__=='__main__':
 {'round0':round0,'round1':round1,'round2':round2,'round3':round3}[sys.argv[1]]()
