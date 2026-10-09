"""Preregistered action/continuation factorial; no model updates."""
import os,sys,json,copy,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from helpers import HERE,OLD,read,dump,load_tools,snapshot_hash
import numpy as np
OUT=HERE/(sys.argv[1] if len(sys.argv)>1 else 'round4')
def progress(o):
 f=np.asarray(o);return {'payload_z':float(f[20]),'trash_z':float(f[27]),'lid_z':float(f[34]),'payload_target_distance':float(np.linalg.norm(f[18:21]-f[39:42])),'trash_bin_distance':float(np.linalg.norm(f[25:28]-f[42:45])),'payload_in_target':bool(f[45]>.5),'trash_in_bin':bool(f[46]>.5)}
def run():
 import torch
 bt,core=load_tools()
 from stage3_v5_history_critic import encode_replay_contexts
 import stage3_v5_vector_env as vector
 from snapshot_worker import snapshot_worker
 vector._worker=snapshot_worker
 device,ready,bc,critic,scale,offset=core.setup();critic_hash=core.module_hash(critic)
 reg=read(OUT/'preregistration.json');seeds=reg['seeds'];fork=reg['fork'];spec=next(s for s in read(OLD/'round1/preregistration.json')['states'] if s['block']==17)
 state=torch.load(spec['old_checkpoint'],map_location='cpu',weights_only=False)['actor'];full=torch.load(spec['candidate_checkpoint'],map_location='cpu',weights_only=False)['actor']
 old=copy.deepcopy(bc);old.load_state_dict(state);bad=copy.deepcopy(old)
 with torch.no_grad():
  for n,p in bad.named_parameters():p.copy_(state[n].to(device)+spec['fraction']*(full[n].to(device)-state[n].to(device)))
 actors={'REF':bc,'OLD':old,'BAD':bad};actor_hashes={n:core.module_hash(a) for n,a in actors.items()}
 for a in actors.values():a.eval().requires_grad_(False)
 source=None
 vec=None;results={};baseline_snapshots={};baseline_context=None;contract={};started=time.time();snapshot_bank={};context_bank={};os.chdir(OUT)
 try:
  vec=core.StaggeredVectorEnv(core.DATASET,4,20007,delay=.5,timeout=120,startup_parallelism=4,shared_memory=False)
  contract.update(parallel_envs_initialized=len(vec.initial_observations),worker_pids=[p.pid for p in vec.processes]);assert vec.alive_worker_count()==4
  for branch in reg['branches']:
   label=branch['label'];prefix=branch['prefix'];baseline_snapshots=snapshot_bank.get(prefix,{});baseline_context=context_bank.get(prefix);raw=[json.loads(l) for l in Path(branch['prefix_trace']).read_text().splitlines()];source={seed:sorted([r for r in raw if r['seed']==seed],key=lambda r:r['timestep']) for seed in seeds};ex={n:core.BatchedGMMExecutor(a,scale,offset,4,horizon=10) for n,a in actors.items()};obs=vec.reset_many(dict(enumerate(seeds)));torch.manual_seed(reg['continuation_rng_seed']);torch.npu.manual_seed_all(reg['continuation_rng_seed'])
   hist=[[] for _ in seeds];active=set(range(4));won=[False]*4;probe=None
   for t in range(700):
    before=bt.acc.rng();samples={};after=None
    names=['REF'] if t<(fork//10)*10 else list(ex) if t<=fork else [branch['continuation']]
    for n in names:
     e=ex[n]
     bt.acc.restore_rng(before);samples[n]=e.actions_for(list(range(4)),[obs[i] for i in range(4)],0.,None,None)
     if after is None:after=bt.acc.rng()
    bt.acc.restore_rng(after);actions=np.asarray(samples[branch['continuation']] if t>=fork else samples['REF'],dtype=np.float32).copy()
    if t<fork:
     for i in sorted(active):
      assert np.max(np.abs(core.obs_to_flat(obs[i])-np.asarray(source[seeds[i]][t]['observation_flat'])))<1e-6
      actions[i]=np.asarray(source[seeds[i]][t]['action'],dtype=np.float32)
    if t==fork:
     assert len(active)==4
     for i in range(4):vec.connections[i].send(('snapshot',None))
     snapshots=[]
     for i in range(4):
      msg=vec._recv(i,120,'diagnostic_snapshot');assert msg[0]=='SNAPSHOT',msg;snapshot=msg[1];snapshots.append(snapshot)
      if not baseline_snapshots:pass
      elif snapshot_hash(snapshot)!=snapshot_hash(baseline_snapshots[i]):
       dump(OUT/(label+'_STATE_MISMATCH_'+str(i)+'.json'),{'reference':baseline_snapshots[i],'actual':snapshot});raise AssertionError('Strict state identity failed; no factorial result accepted')
     if not baseline_snapshots:baseline_snapshots={i:x for i,x in enumerate(snapshots)};snapshot_bank[prefix]=baseline_snapshots
     co=np.asarray([[h['observation_flat'] for h in hist[i][-9:]]+[core.obs_to_flat(obs[i]).tolist()] for i in range(4)],dtype=np.float32);ca=np.asarray([[h['action_raw'] for h in hist[i][-9:]]+[np.asarray(samples['OLD'][i]).tolist()] for i in range(4)],dtype=np.float32);cs=np.asarray([[h['timestep'] for h in hist[i][-9:]]+[t] for i in range(4)],dtype=np.int64)
     if baseline_context is None:baseline_context=(co.copy(),ca.copy());context_bank[prefix]=baseline_context
     else:assert np.array_equal(co,baseline_context[0]) and np.array_equal(ca,baseline_context[1])
     with torch.no_grad():
      enc=encode_replay_contexts(critic,torch.as_tensor(co,device=device),torch.as_tensor(ca,device=device),torch.as_tensor(cs,device=device),700);features=tuple(e[:,-1] for e in enc);action_values={}
      for n,sampled in samples.items():
       raw=np.asarray(sampled,dtype=np.float32);executed=np.clip(raw,vec.action_low,vec.action_high).astype(np.float32);assert np.all(executed>=vec.action_low) and np.all(executed<=vec.action_high)
       q1,q2=critic.q_from_context(features,torch.as_tensor(executed,device=device));action_values[n]={'raw':raw.tolist(),'executed':executed.tolist(),'clipping_l2':np.linalg.norm(raw-executed,axis=1).tolist(),'q1':q1.cpu().numpy().reshape(-1).tolist(),'q2':q2.cpu().numpy().reshape(-1).tolist(),'qmean':((q1+q2)/2).cpu().numpy().reshape(-1).tolist()}
     fixed=np.asarray(branch['fixed_fork_action'],dtype=np.float32);assert fixed.shape==(14,) and np.all(fixed>=vec.action_low) and np.all(fixed<=vec.action_high)
     fixed=np.repeat(fixed[None],4,axis=0)
     with torch.no_grad():
      fq1,fq2=critic.q_from_context(features,torch.as_tensor(fixed,device=device))
     action_values['FIXED']={'raw':fixed.tolist(),'executed':fixed.tolist(),'clipping_l2':[0.]*4,'q1':fq1.cpu().numpy().reshape(-1).tolist(),'q2':fq2.cpu().numpy().reshape(-1).tolist(),'qmean':((fq1+fq2)/2).cpu().numpy().reshape(-1).tolist()}
     actions=np.asarray(action_values[branch['action']]['executed'],dtype=np.float32);probe={'actions':action_values,'observations':co.tolist(),'past_actions_production_raw':ca.tolist(),'episode_steps':cs.tolist(),'state_hashes':[snapshot_hash(x) for x in snapshots],'same_physical_state':True,'task_progress':[progress(core.obs_to_flat(obs[i])) for i in range(4)]}
     dump(OUT/(label+'_fork.json'),{'probe':probe,'snapshots':snapshots})
    ids=sorted(active);current={i:core.obs_to_flat(obs[i]).tolist() for i in ids}
    for i,msg in vec.step([actions[i] for i in ids],ids):
     assert msg[0]=='OK',msg
     _,no,reward,done,win,info=msg;nf=core.obs_to_flat(no).tolist();raw=actions[i];executed=np.clip(raw,vec.action_low,vec.action_high)
     hist[i].append({'seed':seeds[i],'replicate':i,'timestep':t,'observation_flat':current[i],'action_raw':raw.tolist(),'action_executed_feasible':executed.tolist(),'next_observation_flat':nf,'reward':float(reward),'success':bool(win),'terminated':bool(done or win),'truncated':bool(t==699 and not(done or win)),'task_progress':progress(nf)});obs[i]=no;won[i]=bool(win)
     if done or win or t==699:active.remove(i)
    if t%100==0:print('FACTORIAL_ROLLOUT',label,t,len(active),flush=True)
    if not active:break
   assert probe is not None;contract['parallel_envs_used']=4;episodes=[];trace=OUT/(label+'_trace.jsonl')
   with trace.open('x') as f:
    for i,h in enumerate(hist):
     ret=0.
     for row in reversed(h):ret=row['reward']+.99*ret;row['finite_mc_return']=ret
     episodes.append({'seed':seeds[i],'replicate':i,'success':won[i],'length':len(h),'mc_from_fork':h[fork]['finite_mc_return'],'reward_sum':sum(x['reward'] for x in h),'payload_min_z_postfork':min(x['task_progress']['payload_z'] for x in h[fork:]),'payload_goal_first_postfork':next((x['timestep'] for x in h[fork:] if x['task_progress']['payload_in_target']),None),'trash_goal_first_postfork':next((x['timestep'] for x in h[fork:] if x['task_progress']['trash_in_bin']),None),'final_progress':h[-1]['task_progress'],'sim_error':False})
     for row in h:f.write(json.dumps(row,allow_nan=False)+'\n')
   results[label]={'action':branch['action'],'continuation':branch['continuation'],'episodes':episodes,'probe':probe,'trace':str(trace)};dump(OUT/(label+'_result.json'),results[label]);print('FACTORIAL_DONE',label,episodes,flush=True)
 finally:
  if vec is not None:
   vec.close();contract['parallel_envs_closed']=sum(not p.is_alive() and p.exitcode==0 for p in vec.processes);contract['worker_exitcodes']=[p.exitcode for p in vec.processes]
 contract['test_valid']=all(contract[k]==4 for k in ('parallel_envs_initialized','parallel_envs_used','parallel_envs_closed'));assert contract['test_valid'];assert core.module_hash(critic)==critic_hash;assert all(core.module_hash(actors[n])==h for n,h in actor_hashes.items())
 dump(OUT/'result.json',{'branches':results,'env_contract':contract,'env_steps':sum(e['length'] for b in results.values() for e in b['episodes']),'episodes':len(results)*4,'wall_seconds':time.time()-started,'actor_updates':0,'critic_updates':0,'model_hashes_unchanged':True,'strict_state_hash_all_branches_match':True,'state_match_scope':'within each fixed-prefix group across continuations; different prefixes intentionally have different states','readiness_env_steps':ready['env_steps'],'readiness_actor_updates':ready['actor_updates'],'source_specs':spec,'actor_hashes':actor_hashes,'formal_training_stopped':True})
 print('ROUND1_FACTORIAL_DONE',flush=True)
if __name__=='__main__':run()
