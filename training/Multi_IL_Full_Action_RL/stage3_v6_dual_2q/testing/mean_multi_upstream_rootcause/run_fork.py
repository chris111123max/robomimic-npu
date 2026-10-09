"""Bounded same-state single-action fork; four simultaneous workers only."""
import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from diagnose import *
FORK=199

def main():
 out=OUT/'round4';out.mkdir(exist_ok=True)
 prior=loadj(OUT/'round3/result.json')
 assert prior['env_contract']['test_valid']
 device,ready,ref,bad,critic,scale,offset,_=prepare()
 ref.eval();bad.eval()
 vec=None;results={};prefixes={};contract={'device':'npu:0','seeds':list(SEEDS),'parallel_envs_initialized':0,'parallel_envs_used':0,'parallel_envs_closed':0}
 os.chdir(out)
 try:
  vec=StaggeredVectorEnv(DATASET,4,20000,delay=.5,timeout=120,startup_parallelism=4,shared_memory=False)
  contract.update(parallel_envs_initialized=len(vec.initial_observations),worker_pids=[p.pid for p in vec.processes])
  assert vec.alive_worker_count()==4
  for alpha in (0.,.25,1.):
   name='FORK_'+str(alpha).replace('.','p')
   executors=[BatchedGMMExecutor(a,scale,offset,4,horizon=10) for a in (ref,bad)]
   initial=vec.reset_many(dict(enumerate(SEEDS)));obs=[initial[i] for i in range(4)]
   torch.manual_seed(20007);torch.npu.manual_seed_all(20007)
   active=[True]*4;hist=[[] for _ in range(4)];steps=[0]*4;success=[False]*4
   probe=None
   for t in range(700):
    rng=torch.get_rng_state();nrng=torch.npu.get_rng_state();samples=[]
    for ex in executors:
     torch.set_rng_state(rng);torch.npu.set_rng_state(nrng)
     samples.append(ex.actions_for(list(range(4)),obs,0.,None,None))
    refactions=samples[0];badactions=samples[1]
    act=[np.asarray(a) for a in refactions]
    if t==FORK:
     assert all(active)
     cobs=np.asarray([[r['obs'] for r in h[-9:]]+[obs_to_flat(obs[i]).tolist()] for i,h in enumerate(hist)])
     chact=np.asarray([[r['action'] for r in h[-9:]]+[np.asarray(refactions[i]).tolist()] for i,h in enumerate(hist)])
     csteps=np.asarray([[r['t'] for r in h[-9:]]+[t] for h in hist])
     if alpha==0:prefixes['obs']=cobs
     else:assert np.allclose(cobs,prefixes['obs'],atol=1e-6,rtol=0),float(np.max(abs(cobs-prefixes['obs'])))
     with torch.no_grad():
      enc=encode_replay_contexts(critic,torch.as_tensor(cobs,device=device,dtype=torch.float32),torch.as_tensor(chact,device=device,dtype=torch.float32),torch.as_tensor(csteps,device=device,dtype=torch.long),700)
      a0=torch.as_tensor(np.asarray(refactions),device=device,dtype=torch.float32)
      a1=torch.as_tensor(np.asarray(badactions),device=device,dtype=torch.float32)
      qa=critic.q_from_context((enc[0][:,-1],enc[1][:,-1]),a0)
      a=a0+alpha*(a1-a0)
      qb=critic.q_from_context((enc[0][:,-1],enc[1][:,-1]),a)
      probe={'q1_reference':qa[0].cpu().numpy().reshape(-1).tolist(),'q2_reference':qa[1].cpu().numpy().reshape(-1).tolist(),'q1_candidate':qb[0].cpu().numpy().reshape(-1).tolist(),'q2_candidate':qb[1].cpu().numpy().reshape(-1).tolist(),'action_drift_normalized':torch.linalg.vector_norm((a-a0)/scale,dim=-1).cpu().tolist(),'prefix_max_obs_diff':float(np.max(abs(cobs-prefixes['obs'])))}
      act=list(a.cpu().numpy())
    ids=[i for i in range(4) if active[i]]
    before={i:obs_to_flat(obs[i]).tolist() for i in ids}
    for i,m in vec.step([act[i] for i in ids],ids):
     assert m[0]=='OK',m
     _,no,r,done,win,info=m
     hist[i].append({'t':t,'obs':before[i],'action':np.asarray(act[i]).tolist(),'reward':float(r),'success':bool(win),'terminated':bool(done or win),'truncated':bool(t==699 and not(done or win))})
     steps[i]+=1;success[i]=bool(win);obs[i]=no;active[i]=not(done or win or t==699)
    if t%100==0:print(json.dumps({'event':'fork_rollout','alpha':alpha,'t':t,'active':sum(active)}),flush=True)
    if not any(active):break
   assert probe is not None
   contract['parallel_envs_used']=4
   episodes=[]
   for i,h in enumerate(hist):
    ret=0.
    for row in reversed(h):ret=row['reward']+.99*ret;row['mc']=ret
    episodes.append({'seed':SEEDS[i],'success':success[i],'length':steps[i],'mc_from_fork':h[FORK]['mc'],'return':sum(r['reward'] for r in h),'sim_error':False})
   with (out/f'{name}_trace.jsonl').open('x') as f:
    for i,h in enumerate(hist):
     for row in h:f.write(json.dumps({'seed':SEEDS[i],**row},allow_nan=False)+'\n')
   results[str(alpha)]={'alpha':alpha,'episodes':episodes,'success_count':sum(success),'probe':probe}
   print(json.dumps({'event':'fork_done','alpha':alpha,'success':sum(success),'episodes':episodes,'probe':probe}),flush=True)
 finally:
  if vec is not None:
   vec.close();contract['parallel_envs_closed']=sum(not p.is_alive() and p.exitcode==0 for p in vec.processes);contract['worker_exitcodes']=[p.exitcode for p in vec.processes]
 contract['test_valid']=all(contract[k]==4 for k in ('parallel_envs_initialized','parallel_envs_used','parallel_envs_closed'))
 assert contract['test_valid']
 for alpha in ('0.25','1.0'):
  r=results[alpha];b=results['0.0'];dmc=[];dq=[];dqs=[]
  for i,e in enumerate(r['episodes']):
   dmc.append(e['mc_from_fork']-b['episodes'][i]['mc_from_fork'])
   d1=r['probe']['q1_candidate'][i]-r['probe']['q1_reference'][i]
   d2=r['probe']['q2_candidate'][i]-r['probe']['q2_reference'][i]
   dq.append((d1+d2)/2);dqs.append([d1,d2])
  r['paired_delta_mc']=dmc;r['paired_delta_qmean']=dq;r['paired_delta_q1_q2']=dqs
 dump(out/'result.json',{'branches':results,'env_contract':contract,'intervention':'one action at fixed t199 along same collapsed drift; all other actions from unchanged BC on own visited observations','fork_t':FORK,'actor_updates':0,'critic_updates':0,'limitations':['one predetermined context per each of four known-good seeds, not population','BC continuation MC evaluates single-action advantage, not sustained collapsed policy','same observed history verified; deterministic simulator reset/prefix used rather than serialized hidden simulator state'],'formal_training_stopped':True})
 print('ROUND4_DONE',flush=True)

if __name__=='__main__':main()
