"""Fixed-context action finite differences; testing only, no optimizer."""
import os,sys,json,copy,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_round1 import read,dump,load_tools,HERE
OUT=HERE/'round2'
def prep():
 a=read(HERE/'round1/analysis.json');assert a['decision']=='CONTINUE_TO_ROUND_2'
 seeds=a['independent_seeds'];rows=[json.loads(l) for l in (HERE/'round1/evaluations/FIXED_STATES/OLD_ACCEPTED_BLOCK_10_trajectories.jsonl').read_text().splitlines()]
 lengths=[sum(r['seed']==s for r in rows) for s in seeds]
 dump(OUT/'preregistration.json',{'seeds':seeds,'fork_times':[lengths[0]//4,199,199,lengths[3]//2],'proposal_blocks':[17,17,29,29],'branches':[0,1/512,-1/512,1/256,-1/256,1/128,-1/128],'direction':'actual common-RNG subfraction proposal action change; normalized to full proposal displacement in physical action-scale units','prefix':'exact recorded theta_old actions; validate observation identity','continuation':'readiness BC after one action injection','selection':'quarter successful old seed0, t199 failed old seeds1/2, midpoint successful old seed3; selected before Q/J outcomes','device':'npu:0','parallel_envs':4,'optimizer_updates':0,'q_floor':1e-7,'mc_floor':1e-9})
def run():
 import torch,numpy as np
 bt,core=load_tools()
 from stage3_v5_history_critic import encode_replay_contexts
 device,ready,bc,critic,scale,offset=core.setup();ch=core.module_hash(critic)
 reg=read(OUT/'preregistration.json');seeds=reg['seeds'];forks=reg['fork_times']
 specs={s['block']:s for s in read(HERE/'round1/preregistration.json')['states']}
 old=copy.deepcopy(bc);old.load_state_dict(torch.load(specs[17]['old_checkpoint'],map_location='cpu',weights_only=False)['actor'])
 actors=[bc,old]
 for b in (17,29):
  full=torch.load(specs[b]['candidate_checkpoint'],map_location='cpu',weights_only=False)['actor'];base=old.state_dict()
  sub=copy.deepcopy(old);af=copy.deepcopy(old);af.load_state_dict(full)
  with torch.no_grad():
   for n,p in sub.named_parameters():p.copy_(base[n]+specs[b]['fraction']*(full[n].to(device)-base[n]))
  actors.extend([sub,af])
 for a in actors:a.eval().requires_grad_(False)
 rows=[json.loads(l) for l in (HERE/'round1/evaluations/FIXED_STATES/OLD_ACCEPTED_BLOCK_10_trajectories.jsonl').read_text().splitlines()]
 source={s:sorted([r for r in rows if r['seed']==s],key=lambda r:r['timestep']) for s in seeds}
 vec=None;results={};contexts={};contract={};started=time.time();os.chdir(OUT)
 try:
  vec=core.StaggeredVectorEnv(core.DATASET,4,20007,delay=.5,timeout=120,startup_parallelism=4,shared_memory=False)
  contract.update(parallel_envs_initialized=len(vec.initial_observations),worker_pids=[p.pid for p in vec.processes]);assert vec.alive_worker_count()==4
  for epsilon in reg['branches']:
   label='A'+str(epsilon).replace('.','p').replace('-','m')
   executors=[core.BatchedGMMExecutor(a,scale,offset,4,horizon=10) for a in actors]
   obs=vec.reset_many(dict(enumerate(seeds)));torch.manual_seed(20007);torch.npu.manual_seed_all(20007)
   hist=[[],[],[],[]];probes={};active=set(range(4));won=[False]*4
   for t in range(700):
    before=bt.acc.rng();samples=[];after=None
    for j,ex in enumerate(executors):
     bt.acc.restore_rng(before)
     samples.append(ex.actions_for(list(range(4)),[obs[i] for i in range(4)],0.,None,None))
     if j==0:after=bt.acc.rng()
    bt.acc.restore_rng(after);actions=np.asarray(samples[0],dtype=np.float32).copy()
    for i in sorted(active):
     if t<forks[i]:
      assert np.max(np.abs(core.obs_to_flat(obs[i])-np.asarray(source[seeds[i]][t]['observation_flat'])))<1e-6
      actions[i]=np.asarray(source[seeds[i]][t]['action'],dtype=np.float32)
     if t==forks[i]:
      a0=np.asarray(samples[1][i],dtype=np.float32);j=2 if reg['proposal_blocks'][i]==17 else 4
      tiny=np.asarray(samples[j][i])-a0;full=np.asarray(samples[j+1][i])-a0
      tn=float(np.linalg.norm(tiny/scale.detach().cpu().numpy()));fn=float(np.linalg.norm(full/scale.detach().cpu().numpy()));assert tn>1e-8 and fn>1e-8
      d=tiny/tn*fn
      co=np.asarray([h['observation_flat'] for h in hist[i][-9:]]+[core.obs_to_flat(obs[i]).tolist()],dtype=np.float32)
      ca=np.asarray([h['action'] for h in hist[i][-9:]]+[a0.tolist()],dtype=np.float32)
      cs=np.asarray([h['timestep'] for h in hist[i][-9:]]+[t],dtype=np.int64)
      ctx={'seed':seeds[i],'timestep':t,'proposal_block':reg['proposal_blocks'][i],'observations':co.tolist(),'actions':ca.tolist(),'steps':cs.tolist(),'a0':a0.tolist(),'direction':d.tolist(),'tiny_scaled_norm':tn,'full_scaled_norm':fn}
      if epsilon==0:contexts[i]=ctx
      else:
       assert np.max(np.abs(co-np.asarray(contexts[i]['observations'])))<1e-6
       assert np.max(np.abs(d-np.asarray(contexts[i]['direction'])))<1e-6
      raw=a0+epsilon*d;action=np.clip(raw,vec.action_low,vec.action_high).astype(np.float32)
      enc=encode_replay_contexts(critic,torch.as_tensor(co[None],device=device),torch.as_tensor(ca[None],device=device),torch.as_tensor(cs[None],device=device),700)
      features=tuple(e[:,-1].detach() for e in enc)
      qa=torch.as_tensor(action[None],device=device);q1,q2=critic.q_from_context(features,qa)
      probe={'q1':float(q1.item()),'q2':float(q2.item()),'action':action.tolist(),'clipped':bool(np.any(raw<vec.action_low) or np.any(raw>vec.action_high)),'epsilon':epsilon}
      if epsilon==0:
       qa=qa.detach().requires_grad_(True);q1,q2=critic.q_from_context(features,qa)
       probe['autograd_directional']=[float((torch.autograd.grad(q.sum(),qa,retain_graph=True)[0]*torch.as_tensor(d[None],device=device)).sum().item()) for q in (q1,q2)]
      probes[i]=probe;actions[i]=action
    ids=sorted(active);before={i:core.obs_to_flat(obs[i]).tolist() for i in ids}
    for i,msg in vec.step([actions[i] for i in ids],ids):
     assert msg[0]=='OK',msg
     _,no,r,done,win,info=msg
     hist[i].append({'seed':seeds[i],'timestep':t,'observation_flat':before[i],'action':actions[i].tolist(),'next_observation_flat':core.obs_to_flat(no).tolist(),'reward':float(r),'success':bool(win),'terminated':bool(done or win),'truncated':bool(t==699 and not(done or win))})
     obs[i]=no;won[i]=bool(win)
     if done or win or t==699:active.remove(i)
    if t%100==0:print('ACTION_ROLLOUT',epsilon,t,len(active),flush=True)
    if not active:break
   assert len(probes)==4;contract['parallel_envs_used']=4;episodes=[]
   with (OUT/(label+'_trace.jsonl')).open('x') as f:
    for i,h in enumerate(hist):
     ret=0.
     for row in reversed(h):ret=row['reward']+.99*ret;row['finite_mc_return']=ret
     tail=h[forks[i]:]
     episodes.append({'seed':seeds[i],'success':won[i],'length':len(h),'mc_from_fork':tail[0]['finite_mc_return'],'reward_sum':sum(x['reward'] for x in h),'horizon_returns':{str(k):sum(.99**j*x['reward'] for j,x in enumerate(tail[:k])) for k in (1,4,8,16,32)},'sim_error':False})
     for row in h:f.write(json.dumps(row,allow_nan=False)+'\n')
   results[str(epsilon)]={'episodes':episodes,'probes':[probes[i] for i in range(4)],'trace':str(OUT/(label+'_trace.jsonl'))}
   dump(OUT/(label+'_result.json'),results[str(epsilon)]);print('ACTION_BRANCH_DONE',epsilon,episodes,flush=True)
 finally:
  if vec is not None:
   vec.close();contract['parallel_envs_closed']=sum(not p.is_alive() and p.exitcode==0 for p in vec.processes);contract['worker_exitcodes']=[p.exitcode for p in vec.processes]
 contract['test_valid']=all(contract[k]==4 for k in ('parallel_envs_initialized','parallel_envs_used','parallel_envs_closed'));assert contract['test_valid']
 assert core.module_hash(critic)==ch
 dump(OUT/'contexts.json',[contexts[i] for i in range(4)])
 dump(OUT/'result.json',{'branches':results,'env_contract':contract,'env_steps':sum(e['length'] for b in results.values() for e in b['episodes']),'wall_seconds':time.time()-started,'critic_hash_unchanged':True,'actor_updates':0,'critic_updates':0,'formal_training_stopped':True})
 print('ROUND2_DONE',flush=True)
if __name__=='__main__':
 if sys.argv[1]=='prepare':prep()
 elif sys.argv[1]=='run':run()
