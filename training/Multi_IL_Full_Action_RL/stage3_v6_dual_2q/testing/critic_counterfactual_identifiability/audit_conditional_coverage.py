"""Complete window scan, episode-disjoint continuous-neighborhood calibration."""
import os,json,time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np,h5py,torch
from scipy.spatial.distance import cdist
HERE=Path(__file__).resolve().parent
RUN=Path('/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_readiness_v2_multi_mean_random_formal_20260929_130945')
KEYS=('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','robot1_eef_pos','robot1_eef_quat','robot1_gripper_qpos','object')
def main():
 start=time.time();ready=torch.load(RUN/'mean2q/multi_q/checkpoints/critic_ready.pth',map_location='cpu',weights_only=False);bc=torch.load(RUN/'shared/bc_rnn_gmm_source.pth',map_location='cpu',weights_only=False);scale=np.asarray(bc['action_normalization_stats']['actions']['scale']).reshape(14);replay=np.load(ready['online_sequence_replay'],allow_pickle=True).item();episodes=[]
 for label,collection in [('online_complete',replay['episodes']),('online_incomplete',list(replay['current'].values()))]:
  for j,e in enumerate(collection):
   if len(e['actions'])>=10:episodes.append({'label':label,'o':np.asarray(e['observations'],np.float32),'a':np.asarray(e['actions'],np.float32),'steps':np.asarray(e['episode_steps']).reshape(-1),'mc_train':False,'rw':np.asarray(e['rewards']).reshape(-1),'complete':label=='online_complete'})
 for label,path in ready['config']['offline_sources'].items():
  with h5py.File(path,'r') as f:
   for name,g in f['episodes'].items():
    seed=int(np.asarray(g.attrs['initial_seed'] if 'initial_seed' in g.attrs else g['initial_seed'][()]).reshape(-1)[0]);a=np.asarray(g['actions'],np.float32);n=len(a);o=np.concatenate([np.asarray(g['obs'][k],np.float32).reshape(n,-1) for k in KEYS],1)
    episodes.append({'label':label,'o':o,'a':a,'steps':np.arange(n),'mc_train':seed<=10079,'seed':seed,'rw':np.asarray(g['rewards']).reshape(-1),'complete':True})
 trainobs=np.concatenate([e['o'] for e in episodes if e['mc_train']]);sd=np.maximum(trainobs.std(0),.01);del trainobs
 reg=json.loads((HERE/'round2/preregistration.json').read_text());d=np.load(HERE/'round1/contexts.npz');p=np.load(HERE/'round1/probes.npz');names=p['candidate_names'].tolist();queries=[];cal=[]
 for s in reg['contexts']:
  i=s['context_index'];queries.append({'o':d['o'][i],'a':d['a'][i],'steps':d['steps'][i]})
 for e in episodes:
  if e.get('seed') in [10080,10081]:
   for s in reg['contexts']:
    t=min(s['step'],len(e['a'])-1);cal.append({'o':e['o'][t-9:t+1],'a':e['a'][t-9:t+1],'steps':e['steps'][t-9:t+1]})
 def feature(o,a,steps):
  return np.concatenate((o[:,-1]/sd/np.sqrt(59),(o/sd).reshape(len(o),-1)/np.sqrt(590),(a[:,:-1]/scale).reshape(len(o),-1)/np.sqrt(126),steps/700/np.sqrt(10)),1)
 q=feature(np.stack([x['o'] for x in queries+cal]),np.stack([x['a'] for x in queries+cal]),np.stack([x['steps'] for x in queries+cal]));cal_min=np.full(len(cal),np.inf);parts=[];actions=[];eps=[];labels=[];mctrain=[];windowcount=0
 for ei,e in enumerate(episodes):
  o=e['o'];a=e['a'];n=len(a);oh=np.lib.stride_tricks.sliding_window_view(o,10,axis=0).transpose(0,2,1);ah=np.lib.stride_tricks.sliding_window_view(a,10,axis=0).transpose(0,2,1);steps=np.lib.stride_tricks.sliding_window_view(e['steps'],10);f=feature(oh,ah,steps);dist=cdist(q,f,'euclidean');parts.append(dist[:4]);actions.append(a[9:]);eps.append(np.full(n-9,ei));labels.append(np.repeat(e['label'],n-9));mctrain.append(np.full(n-9,e['mc_train']));windowcount+=n-9
  if e['mc_train']:cal_min=np.minimum(cal_min,dist[4:].min(1))
  if ei%100==0:print('COVERAGE_EPISODE',ei,windowcount,flush=True)
 distances=np.concatenate(parts,1);acts=np.concatenate(actions);epids=np.concatenate(eps);labels=np.concatenate(labels);is_mc=np.concatenate(mctrain);radius=float(np.percentile(cal_min,95));result={'windows':windowcount,'episodes':len(episodes),'original_MC_training_windows':int(is_mc.sum()),'calibration':{'rule':'first2validationseeds/source at4lockedforktimes; full nearest training window;episode disjoint;radius nearest-distance p95;no labels used','count':len(cal),'nearest_distance':cal_min.tolist(),'radius_p95':radius},'queries':[],'limits':['Approximate history metric, not physical identity.','Unique episodes, not overlapping windows, count local variation.','Observed labels belong to original actions AND continuation; no counterfactual ground truth.','Incomplete online prefix return is censored and unused.','Calibrated radius presence/absence alone cannot establish slope error.']}
 for i,s in enumerate(reg['contexts']):
  a0=p['ready_candidate_actions'][s['probe_index'],names.index('BC')];bad=p['ready_candidate_actions'][s['probe_index'],names.index('post1250')];delta=(bad-a0)/scale;direction=delta/(np.linalg.norm(delta)+1e-12);entry={'context':s,'candidate_displacement_scaled':float(np.linalg.norm(delta)),'sets':{}}
  for label,mask in [('MC_train',is_mc),('ready_replay',np.ones(len(epids),bool))]:
   eligible=np.where(mask)[0];order=eligible[np.argsort(distances[i,eligible])];unique=[];seen=set()
   for j in order:
    if int(epids[j]) not in seen:unique.append(j);seen.add(int(epids[j]))
    if len(unique)==64:break
   near=np.asarray(unique);local=near[distances[i,near]<=radius];tables={}
   for k in [8,32,64]:
    ids=near[:k];aa=(acts[ids]-a0)/scale;projection=aa@direction;orthogonal=np.linalg.norm(aa-projection[:,None]*direction,axis=1)
    tables[str(k)]={'episodes':len(ids),'distance_minmax':[float(distances[i,ids].min()),float(distances[i,ids].max())],'BC_nearest_action_distance':float(np.linalg.norm((acts[ids]-a0)/scale,axis=1).min()),'bad_nearest_action_distance':float(np.linalg.norm((acts[ids]-bad)/scale,axis=1).min()),'direction_projection_minmax':[float(projection.min()),float(projection.max())],'direction_projection_std':float(projection.std()),'orthogonal_distance_median':float(np.median(orthogonal)),'source_counts':{str(v):int((labels[ids]==v).sum()) for v in set(labels[ids])}}
   entry['sets'][label]={'nearest_history_distance':float(distances[i,near[0]]),'within_calibrated_radius_unique_episodes_up_to64':len(local),'topK':tables,'conditional_history_support':'within calibration' if len(local) else 'outside calibration; cannot isolate action extrapolation'}
  result['queries'].append(entry)
 result['wall_seconds']=time.time()-start
 with (HERE/'round3/conditional_coverage.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
 print(json.dumps({'windows':windowcount,'radius':radius,'queries':[{'seed':e['context']['seed'],'step':e['context']['step'],'MC':e['sets']['MC_train']['conditional_history_support'],'ready':e['sets']['ready_replay']['conditional_history_support'],'MC_nearest':e['sets']['MC_train']['nearest_history_distance'],'ready_nearest':e['sets']['ready_replay']['nearest_history_distance']} for e in result['queries']],'seconds':result['wall_seconds']}),flush=True)
if __name__=='__main__':main()
