"""Fixed executed action displacement persistence; diagnostic, not mitigation."""
import os,sys,json,copy,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_round1 import read,dump,load_tools,HERE
OUT=HERE/'round3'
def run():
 import torch,numpy as np
 bt,core=load_tools()
 from stage3_v5_history_critic import encode_replay_contexts
 device,ready,bc,critic,scale,offset=core.setup();ch=core.module_hash(critic)
 reg=read(OUT/'preregistration.json');ctx=read(HERE/'round2/contexts.json');r2=read(HERE/'round2/result.json');seeds=reg['seeds'];forks=[x['timestep'] for x in ctx]
 zero=r2['branches']['0'];plus=r2['branches'][str(1/128)];shifts=[np.asarray(p['action'],dtype=np.float32)-np.asarray(z['action'],dtype=np.float32) for p,z in zip(plus['probes'],zero['probes'])]
 st=next(x for x in read(HERE/'round1/preregistration.json')['states'] if x['block']==17)
 old=copy.deepcopy(bc);old.load_state_dict(torch.load(st['old_checkpoint'],map_location='cpu',weights_only=False)['actor'])
 for a in (bc,old):a.eval().requires_grad_(False)
 raw=[json.loads(l) for l in (HERE/'round1/evaluations/FIXED_STATES/OLD_ACCEPTED_BLOCK_10_trajectories.jsonl').read_text().splitlines()]
 source={s:sorted([r for r in raw if r['seed']==s],key=lambda r:r['timestep']) for s in seeds}
 results={'1':{'episodes':plus['episodes'],'probes':plus['probes'],'reused':True,'trace':plus['trace']}};contract={};vec=None;started=time.time();os.chdir(OUT)
 try:
  vec=core.StaggeredVectorEnv(core.DATASET,4,20007,delay=.5,timeout=120,startup_parallelism=4,shared_memory=False)
  contract.update(parallel_envs_initialized=len(vec.initial_observations),worker_pids=[p.pid for p in vec.processes]);assert vec.alive_worker_count()==4
  for horizon in reg['new_horizons']:
   ex=[core.BatchedGMMExecutor(a,scale,offset,4,horizon=10) for a in (bc,old)]
   obs=vec.reset_many(dict(enumerate(seeds)));torch.manual_seed(20007);torch.npu.manual_seed_all(20007)
   hist=[[] for _ in range(4)];active=set(range(4));won=[False]*4;probes=[[] for _ in range(4)]
   for t in range(700):
    before=bt.acc.rng();samples=[];after=None
    for j,e in enumerate(ex):
     bt.acc.restore_rng(before);samples.append(e.actions_for(list(range(4)),[obs[i] for i in range(4)],0.,None,None))
     if j==0:after=bt.acc.rng()
    bt.acc.restore_rng(after);actions=np.asarray(samples[0],dtype=np.float32).copy()
    for i in sorted(active):
     if t<forks[i]:
      assert np.max(np.abs(core.obs_to_flat(obs[i])-np.asarray(source[seeds[i]][t]['observation_flat'])))<1e-6
      actions[i]=np.asarray(source[seeds[i]][t]['action'],dtype=np.float32)
     if forks[i]<=t<forks[i]+horizon:
      a0=np.asarray(samples[1][i] if t==forks[i] else samples[0][i],dtype=np.float32);raw=a0+shifts[i];query=np.clip(raw,vec.action_low,vec.action_high).astype(np.float32)
      co=np.asarray([h['observation_flat'] for h in hist[i][-9:]]+[core.obs_to_flat(obs[i]).tolist()],dtype=np.float32);ca=np.asarray([h['action'] for h in hist[i][-9:]]+[a0.tolist()],dtype=np.float32);cs=np.asarray([h['timestep'] for h in hist[i][-9:]]+[t],dtype=np.int64)
      if t==forks[i]:assert np.max(np.abs(co-np.asarray(ctx[i]['observations'])))<1e-6
      with torch.no_grad():
       enc=encode_replay_contexts(critic,torch.as_tensor(co[None],device=device),torch.as_tensor(ca[None],device=device),torch.as_tensor(cs[None],device=device),700);features=tuple(e[:,-1] for e in enc)
       q0=critic.q_from_context(features,torch.as_tensor(a0[None],device=device));q=critic.q_from_context(features,torch.as_tensor(query[None],device=device))
      probes[i].append({'timestep':t,'q1_delta':float((q[0]-q0[0]).item()),'q2_delta':float((q[1]-q0[1]).item()),'clipped':bool(np.any(raw<vec.action_low) or np.any(raw>vec.action_high))});actions[i]=query
    ids=sorted(active);before={i:core.obs_to_flat(obs[i]).tolist() for i in ids}
    for i,msg in vec.step([actions[i] for i in ids],ids):
     assert msg[0]=='OK',msg
     _,no,r,done,win,info=msg
     hist[i].append({'seed':seeds[i],'timestep':t,'observation_flat':before[i],'action':actions[i].tolist(),'next_observation_flat':core.obs_to_flat(no).tolist(),'reward':float(r),'success':bool(win),'terminated':bool(done or win),'truncated':bool(t==699 and not(done or win))});obs[i]=no;won[i]=bool(win)
     if done or win or t==699:active.remove(i)
    if t%100==0:print('HORIZON_ROLLOUT',horizon,t,len(active),flush=True)
    if not active:break
   contract['parallel_envs_used']=4;episodes=[];trace=OUT/('H'+str(horizon)+'_trace.jsonl')
   with trace.open('x') as f:
    for i,h in enumerate(hist):
     ret=0.
     for row in reversed(h):ret=row['reward']+.99*ret;row['finite_mc_return']=ret
     assert len(probes[i])==horizon
     episodes.append({'seed':seeds[i],'success':won[i],'length':len(h),'mc_from_fork':h[forks[i]]['finite_mc_return'],'reward_sum':sum(x['reward'] for x in h),'horizon_returns':{str(k):sum(.99**j*x['reward'] for j,x in enumerate(h[forks[i]:forks[i]+k])) for k in (1,4,8,16,32)},'sim_error':False})
     for row in h:f.write(json.dumps(row,allow_nan=False)+'\n')
   results[str(horizon)]={'episodes':episodes,'probes':probes,'reused':False,'trace':str(trace)};dump(OUT/('H'+str(horizon)+'_result.json'),results[str(horizon)]);print('HORIZON_DONE',horizon,episodes,flush=True)
 finally:
  if vec is not None:
   vec.close();contract['parallel_envs_closed']=sum(not p.is_alive() and p.exitcode==0 for p in vec.processes);contract['worker_exitcodes']=[p.exitcode for p in vec.processes]
 contract['test_valid']=all(contract[k]==4 for k in ('parallel_envs_initialized','parallel_envs_used','parallel_envs_closed'));assert contract['test_valid'];assert core.module_hash(critic)==ch
 dump(OUT/'result.json',{'branches':results,'zero_reused':zero,'env_contract':contract,'new_env_steps':sum(e['length'] for b in results.values() if not b['reused'] for e in b['episodes']),'wall_seconds':time.time()-started,'critic_hash_unchanged':True,'actor_updates':0,'critic_updates':0,'formal_training_stopped':True})
 print('ROUND3_DONE',flush=True)
if __name__=='__main__':run()
