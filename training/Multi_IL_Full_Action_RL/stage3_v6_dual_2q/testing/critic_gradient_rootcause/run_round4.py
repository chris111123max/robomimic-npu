"""Exhaustive observed-data directional support audit; no simulation or optimization."""
import json,time
from pathlib import Path
import numpy as np,h5py,torch
from run_round1 import HERE,read,dump
OUT=HERE/'round4'
RUN=Path('/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_readiness_v2_multi_mean_random_formal_20260929_130945')
KEYS=('robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','robot1_eef_pos','robot1_eef_quat','robot1_gripper_qpos','object')
def run():
 started=time.time();reg=read(OUT/'preregistration.json');contexts=read(HERE/'round2/contexts.json');r2=read(HERE/'round2/result.json');bc=torch.load(RUN/'shared/bc_rnn_gmm_source.pth',map_location='cpu',weights_only=False);scale=np.asarray(bc['action_normalization_stats']['actions']['scale']).reshape(14);assert np.all(scale>0)
 ready=torch.load(RUN/'mean2q/multi_q/checkpoints/critic_ready.pth',map_location='cpu',weights_only=False);replay=np.load(ready['online_sequence_replay'],allow_pickle=True).item();episodes=[];sources=[]
 for label,data in [('online_complete',replay['episodes']),('online_current',list(replay['current'].values()))]:
  n=0
  for j,e in enumerate(data):
   o=np.asarray(e['observations'],dtype=np.float32);a=np.asarray(e['actions'],dtype=np.float32);rw=np.asarray(e['rewards'],dtype=np.float64).reshape(-1);steps=np.asarray(e['episode_steps']).reshape(-1)
   if len(o)<10:continue
   mc=np.zeros(len(rw));ret=0.
   for t in range(len(rw)-1,-1,-1):ret=rw[t]+.99*ret;mc[t]=ret
   episodes.append({'label':label,'episode':str(j),'obs':o,'actions':a,'steps':steps,'mc':mc,'mc_complete':label=='online_complete'});n+=len(o)
  sources.append({'source':label,'transitions_with_history_eligible_episode':n,'episodes':len(data)})
 for policy,path in sorted(ready['config']['offline_sources'].items()):
  n=0;ne=0
  with h5py.File(path,'r') as f:
   for name in sorted(f['episodes']):
    e=f['episodes'][name];o=np.concatenate([np.asarray(e['obs'][k],dtype=np.float32).reshape(len(e['actions']),-1) for k in KEYS],axis=1);a=np.asarray(e['actions'],dtype=np.float32);rw=np.asarray(e['rewards']).reshape(-1);steps=np.asarray(e['timestep']).reshape(-1);mc=np.zeros(len(rw));ret=0.
    for t in range(len(rw)-1,-1,-1):ret=float(rw[t])+.99*ret;mc[t]=ret
    assert o.shape[1]==59 and a.shape[1]==14
    episodes.append({'label':policy,'episode':name,'obs':o,'actions':a,'steps':steps,'mc':mc,'mc_complete':True});n+=len(o);ne+=1
  sources.append({'source':policy,'path':path,'transitions':n,'episodes':ne})
 allobs=np.concatenate([e['obs'] for e in episodes]);sd=np.maximum(allobs.std(0),reg['observation_std_floor']);del allobs
 arrays=[{k:[] for k in ('obs_rms','history_rms','previous_action_rms','dt','projection','perpendicular','action_distance','source','episode','timestep','mc','mc_complete')} for _ in contexts]
 for e in episodes:
  o=e['obs'];ac=e['actions'];n=len(o)
  if n<10:continue
  oh=np.lib.stride_tricks.sliding_window_view(o,10,axis=0).transpose(0,2,1);ah=np.lib.stride_tricks.sliding_window_view(ac,10,axis=0).transpose(0,2,1);ix=np.arange(9,n)
  for i,c in enumerate(contexts):
   z=arrays[i];qo=np.asarray(c['observations']);qa=np.asarray(c['actions']);a0=np.asarray(r2['branches']['0']['probes'][i]['action']);shift=(np.asarray(r2['branches'][str(1/128)]['probes'][i]['action'])-a0)/scale;radius=float(np.linalg.norm(shift));assert radius>1e-9;direction=shift/radius
   delta=(ac[ix]-a0)/scale;projection=delta@direction;distance=np.linalg.norm(delta,axis=1);perp=np.sqrt(np.maximum(0,distance**2-projection**2))
   values={'obs_rms':np.sqrt(np.mean(((o[ix]-qo[-1])/sd)**2,axis=1)),'history_rms':np.sqrt(np.mean(((oh-qo[None])/sd)**2,axis=(1,2))),'previous_action_rms':np.sqrt(np.mean(((ah[:,:-1]-qa[None,:-1])/scale)**2,axis=(1,2))),'dt':np.abs(e['steps'][ix]-c['timestep']),'projection':projection,'perpendicular':perp,'action_distance':distance,'source':np.repeat(e['label'],len(ix)),'episode':np.repeat(e['episode'],len(ix)),'timestep':e['steps'][ix],'mc':e['mc'][ix],'mc_complete':np.repeat(e['mc_complete'],len(ix))}
   for k,v in values.items():z[k].append(v)
 print('FULL_DATA_SCAN_DONE',sources,flush=True)
 table=[]
 for i,c in enumerate(contexts):
  z={k:np.concatenate(v) for k,v in arrays[i].items()};a0=np.asarray(r2['branches']['0']['probes'][i]['action']);shift=(np.asarray(r2['branches'][str(1/128)]['probes'][i]['action'])-a0)/scale;radius=float(np.linalg.norm(shift));rows=[]
  metric=z['obs_rms']**2+z['history_rms']**2+z['previous_action_rms']**2+(z['dt']/700)**2
  nearest=np.argsort(metric)[:32]
  for cr in reg['state_context_radii']:
   mask=(z['obs_rms']<=cr)&(z['history_rms']<=cr)&(z['previous_action_rms']<=cr)&(z['dt']<=10)
   for factor in (1,8,64):
    tube=mask&(z['perpendicular']<=factor*radius)&(np.abs(z['projection'])<=4*radius)
    plus=tube&(z['projection']>radius/2);minus=tube&(z['projection']<-radius/2)
    rows.append({'context_radius':cr,'orthogonal_radius_factor':factor,'context_count':int(mask.sum()),'tube_count':int(tube.sum()),'plus_count':int(plus.sum()),'minus_count':int(minus.sum()),'two_sided':bool(plus.any() and minus.any()),'context_projection_std':float(z['projection'][mask].std()) if mask.any() else None,'context_projection_minmax':list(map(float,[z['projection'][mask].min(),z['projection'][mask].max()])) if mask.any() else None,'closest_action_distance_in_context':float(z['action_distance'][mask].min()) if mask.any() else None})
  near=[{'source':str(z['source'][j]),'episode':str(z['episode'][j]),'timestep':int(z['timestep'][j]),'obs_rms':float(z['obs_rms'][j]),'history_rms':float(z['history_rms'][j]),'previous_action_rms':float(z['previous_action_rms'][j]),'dt':int(z['dt'][j]),'projection':float(z['projection'][j]),'perpendicular':float(z['perpendicular'][j]),'action_distance':float(z['action_distance'][j]),'recorded_mc':float(z['mc'][j]) if z['mc_complete'][j] else None} for j in nearest]
  np.savez_compressed(OUT/('seed'+str(c['seed'])+'_directional_scan.npz'),**z)
  table.append({'seed':c['seed'],'timestep':c['timestep'],'full_scan_history_windows':len(metric),'executed_delta_action_l2':radius,'rows':rows,'nearest32_contexts':near,'nearest_action_distance_in_nearest32':float(z['action_distance'][nearest].min())})
 tight=[next(x for x in c['rows'] if x['context_radius']==.1 and x['orthogonal_radius_factor']==1) for c in table]
 twosided=sum(x['two_sided'] for x in tight)
 one_sided=sum((x['plus_count']==0 and x['minus_count']>0) for x in tight)
 label='ONE_SIDED_ACTION_SUPPORT_EXTRAPOLATION_SUPPORTED_NOT_CAUSAL' if one_sided else 'NO_TWO_SIDED_LOCAL_DIRECTIONAL_SUPERVISION_FOUND' if twosided==0 else 'LOCAL_SUPPORT_HETEROGENEOUS'
 result={'classification':label,'upstream_root_cause':'INCONCLUSIVE','evidence_type':'CORRELATIONAL / NEGATIVE','top_two_discriminated':False,'two_sided_contexts_tight':twosided,'one_sided_minus_only_contexts_tight':one_sided,'sources':sources,'online_snapshot_recorded_transitions':replay['transitions'],'table':table,'wall_seconds':time.time()-started,'env_steps':0,'optimizer_steps':0,'decision':'STOP_MAX_4_ROUNDS','limits':['Exhaustive snapshot/offline histories scanned; approximate context metric is not exact simulator state','No simultaneous randomized returns for neighboring replay actions; no causal derivative identified from neighbor labels','Lack of two-sided support supports underconstraint hypothesis, not proof of wrong slope or causal provenance','Round2 sparse MC and clipping prohibit universal smooth gradient sign claim','No bootstrap-slope provenance experiment; bootstrap bias remains unproven']}
 dump(OUT/'result.json',result)
 text='# Round4: exhaustive directional support audit\n\nPrevious strongest conclusion: cross-seed policy-level harm; isolated tiny directions have sparse/nonmonotone MC; H32 persistence not sufficient.\nCurrent question: directional replay supervision deficit versus state/continuation-dependent composition.\nWhy: cannot attribute wrong derivative merely from high Q and low success; check whether observed training data brackets the queried actions at matched history.\nExisting data sufficient? YES for exhaustive observed-support audit; NO for randomized same-state return derivative or target-slope provenance.\nAnalysis: all actual ready snapshot episodes/current plus all three offline rollout sources; full10 observations and previous9 actions conditioned, directional tube and both sides counted.\nEvidence: CORRELATIONAL / NEGATIVE, not upstream causal proof.\n\nClassification: '+label+'\nUpstream root cause: INCONCLUSIVE\n\n|seed|history windows|local context count|plus / minus|action displacement|nearest action among32context neighbors|\n|---|---|---|---|---|---|\n'
 for c,x in zip(table,tight):text+=f"|{c['seed']}|{c['full_scan_history_windows']}|{x['context_count']}|{x['plus_count']}/{x['minus_count']}|{c['executed_delta_action_l2']:.9g}|{c['nearest_action_distance_in_nearest32']:.9g}|\n"
 text+='\nWhat was ruled out: unconditional claim of demonstrated double-sided local supervision if zero; not all replay coverage or function approximation hypotheses. Current top hypothesis: history-conditioned action derivative underconstraint combined with state/continuation sensitivity; still unproven source. Decision: STOP_MAX_4_ROUNDS. No production fix proposed; only remaining targeted same-state action-return identification should be a future task.\n'
 with (OUT/'ROUND_REPORT.md').open('x') as f:f.write(text)
 print(json.dumps({'classification':label,'sources':sources,'tight':tight,'table_brief':[{'seed':c['seed'],'windows':c['full_scan_history_windows'],'action_delta':c['executed_delta_action_l2'],'nearest32_min_action':c['nearest_action_distance_in_nearest32']} for c in table]},indent=2),flush=True)
if __name__=='__main__':run()
