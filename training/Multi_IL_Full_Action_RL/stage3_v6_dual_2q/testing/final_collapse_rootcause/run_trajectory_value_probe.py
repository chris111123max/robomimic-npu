"""Actual reset-phase-correct policy-Q changes on full existing trajectories."""
import os,sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from helpers import HERE,OLD,read,dump
from run_value_audit import setup_actors
import numpy as np

def stats(x):
 x=np.asarray(x,dtype=float)
 return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'minimum':float(x.min()),'maximum':float(x.max()),'positive_fraction_gt1e7':float(np.mean(x>1e-7))}

def run():
 import torch
 from scipy.stats import spearmanr,pearsonr
 from stage3_v5_history_critic import encode_replay_contexts
 from stage3_v5_agent import target_final_distribution_vectorized
 out=HERE/'round5';os.chdir(out);start=time.time();core,device,ready,actors,critic,scale,offset=setup_actors()
 hashes={n:core.module_hash(a) for n,a in actors.items()};ch=core.module_hash(critic)
 files={'BC':OLD/'round1/evaluations/SELECT_04/READINESS_trajectories.jsonl','OLD':OLD/'round1/evaluations/FIXED_STATES/OLD_ACCEPTED_BLOCK_10_trajectories.jsonl','BAD':OLD/'round1/evaluations/FIXED_STATES/BLOCK_17_trajectories.jsonl'}
 results={};records=[]
 with torch.no_grad():
  for region,path in files.items():
   raw=[json.loads(l) for l in path.read_text().splitlines()];windows=[]
   for seed in sorted(set(r['seed'] for r in raw)):
    rows=sorted([r for r in raw if r['seed']==seed],key=lambda r:r['timestep'])
    for t in range(9,len(rows)):
     h=rows[t-9:t+1];windows.append({'region':region,'seed':seed,'timestep':t,'phase':t%10,'mc':rows[t]['finite_mc_return'],'o':[x['observation_flat'] for x in h],'a':[x['action'] for x in h],'steps':[x['timestep'] for x in h]})
   local=[]
   for startidx in range(0,len(windows),64):
    batch=windows[startidx:startidx+64];o=torch.as_tensor(np.asarray([x['o'] for x in batch],np.float32),device=device);a=torch.as_tensor(np.asarray([x['a'] for x in batch],np.float32),device=device);steps=torch.as_tensor(np.asarray([x['steps'] for x in batch],np.int64),device=device)
    enc=encode_replay_contexts(critic,o,a,steps,700);contexts=tuple(c[:,-1] for c in enc);qb1,qb2=critic.q_from_context(contexts,a[:,-1].clamp(-1,1));values={}
    for name,actor in actors.items():
     dist,_=target_final_distribution_vectorized(actor,o,episode_steps=steps,horizon=10);mu=(dist.component_distribution.base_dist.loc*scale+offset).clamp(-1,1);q1,q2=critic.q_from_context(contexts,mu);p=dist.mixture_distribution.probs
     values[name]=np.stack([(p*q1.squeeze(-1)).sum(-1).cpu().numpy(),(p*q2.squeeze(-1)).sum(-1).cpu().numpy()],axis=1)
    for i,x in enumerate(batch):
     z={k:x[k] for k in ['region','seed','timestep','phase','mc']};z['behavior_q1']=float(qb1[i].item());z['behavior_q2']=float(qb2[i].item());z['expected']={n:v[i].tolist() for n,v in values.items()};z['bad_minus_old']=(values['BAD'][i]-values['OLD'][i]).tolist();z['bad_minus_ref']=(values['BAD'][i]-values['REF'][i]).tolist();local.append(z)
   def summarize(rows):
    d={}
    for pair in ['bad_minus_old','bad_minus_ref']:
     v=np.asarray([x[pair] for x in rows]);d[pair]={'q1':stats(v[:,0]),'q2':stats(v[:,1]),'both_positive_gt1e7_fraction':float(np.mean(np.all(v>1e-7,axis=1)))}
    mc=np.asarray([x['mc'] for x in rows]);q=.5*np.asarray([x['behavior_q1']+x['behavior_q2'] for x in rows]);d['behavior']={'mc_mean':float(mc.mean()),'qmean_mean':float(q.mean()),'mean_signed_q_error':float((q-mc).mean()),'spearman':float(spearmanr(q,mc).statistic) if np.std(mc)>0 and np.std(q)>0 else None,'pearson':float(pearsonr(q,mc).statistic) if np.std(mc)>0 and np.std(q)>0 else None};return d
   results[region]={'all':summarize(local),'by_seed':{str(seed):summarize([x for x in local if x['seed']==seed]) for seed in sorted(set(x['seed'] for x in local))},'by_reset_phase':{str(phase):summarize([x for x in local if x['phase']==phase]) for phase in range(10)},'before_fork_seed28':summarize([x for x in local if x['seed']==20028 and x['timestep']<=282])};records.extend(local);print('TRAJECTORY_Q_DONE',region,len(local),flush=True)
 assert core.module_hash(critic)==ch and all(core.module_hash(actors[n])==h for n,h in hashes.items())
 dump(out/'result.json',{'regions':results,'records':records,'history':'sliding10; Actor correct suffix since most recent episode-step multiple10, NOT a new final-token supervision experiment','no_optimizer_steps':True,'model_hashes_unchanged':True,'cost':{'offline_probe':True,'simulator_calls':0,'episodes':0,'env_steps':0,'actor_updates':0,'critic_updates':0,'wall_seconds':time.time()-start}})
 print('TRAJECTORY_VALUE_DONE',flush=True)
if __name__=='__main__':run()
