"""Strict single-action acceptance, frozen networks and four simulator workers."""
import sys,os,json,copy,time,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;TEST=HERE.parent;RL=HERE.parents[2]
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for name in ['stage3_v5_rgmm_td3','stage3_v6_dual_2q','stage2_2_history_aware_critic']:sys.path.insert(0,str(RL/name))
sys.path.insert(0,str(TEST/'mean_multi_collapse_diagnosis'))
from core import setup,RUN,module_hash,strict_stage2_load,StaggeredVectorEnv,DATASET
from stage3_v5_actor import BatchedGMMExecutor,obs_to_flat,flat_to_obs,load_exact_actor
from stage3_v5_history_critic import encode_replay_contexts
from stage3_v5_agent import target_final_distribution_vectorized
import torch,numpy as np
sys.path.insert(0,str(TEST/'final_collapse_rootcause'))
from snapshot_worker import snapshot_worker
from helpers import snapshot_hash
import stage3_v5_vector_env as vector
vector._worker=snapshot_worker

def dump(path,x):
 with Path(path).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def hh(ex,i):
 state=ex.hidden[i];items=[] if state is None else (state if isinstance(state,tuple) else [state])
 return hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in items)+str(ex.counters[i]).encode()).hexdigest()
def module_hash_state(state):
 d=hashlib.sha256()
 for name,value in state.items():d.update(name.encode());d.update(value.cpu().numpy().tobytes())
 return d.hexdigest()
class MeanExecutor(BatchedGMMExecutor):
 def actions_with_uniforms(self,obs,uniform):
  reset=torch.tensor([x%10==0 for x in self.counters],device=self.device,dtype=torch.bool)
  state=self._pack_hidden(self.hidden,reset)
  flat=torch.as_tensor(np.stack([obs_to_flat(o) for o in obs]),device=self.device)
  with torch.no_grad():
   dist,state=self.actor.forward_train_step(flat_to_obs(flat),rnn_state=state)
   probs=dist.mixture_distribution.probs.detach().cpu().numpy();loc=dist.component_distribution.base_dist.loc
   modes=np.minimum((uniform[:,None]>probs.cumsum(-1)).sum(-1),probs.shape[-1]-1)
   ids=torch.as_tensor(modes,device=self.device);actions=loc[torch.arange(4,device=self.device),ids]*self.scale+self.offset
   assert bool((actions.abs()<=1.000001).all())
   raw=actions.cpu().numpy();self.hidden=self._unpack_hidden(state,4);self.counters=[x+1 for x in self.counters]
  return raw,probs,modes

def main():
 stage=sys.argv[1];out=HERE/stage;os.chdir(out);reg=json.loads((out/'preregistration.json').read_text());specs=reg['contexts'];torch.set_num_threads(1)
 device,ready,ref,td,scale,offset=setup();assert str(device)=='npu:0'
 bc,_,_=load_exact_actor(RUN/'shared/bc_rnn_gmm_source.pth',device)
 assert module_hash(bc)==module_hash(ref);assert module_hash(ref)==module_hash_state(ready['target_actor'])
 actors={'BC':bc}
 for name,updates in [('POST1250',1250)]:
  a=copy.deepcopy(ref);a.load_state_dict(torch.load(TEST/f'mean_multi_collapse_diagnosis/round1/actor_{updates}.pth',map_location='cpu',weights_only=False)['actor']);actors[name]=a
 for a in actors.values():a.eval().requires_grad_(False)
 manifest=json.loads((RUN/'shared/stage2_source_manifest.json').read_text());mc,_=strict_stage2_load(manifest['multi_q']['checkpoint'],device);mc.eval().requires_grad_(False);critics={'ready':td,'MC':mc};models={**actors,**critics};hashes={k:module_hash(v) for k,v in models.items()}
 sources={'BC':TEST/'mean_multi_collapse_diagnosis/round1/READY_trajectories.jsonl','OLD':TEST/'critic_gradient_rootcause/round1/evaluations/FIXED_STATES/OLD_ACCEPTED_BLOCK_10_trajectories.jsonl'}
 prefixes=[]
 for s in specs:
  if s['prefix']=='GENERATE_BC':prefixes.append(None)
  else:prefixes.append(sorted([json.loads(l) for l in sources[s['prefix']].read_text().splitlines() if json.loads(l)['seed']==s['seed']],key=lambda x:x['timestep']))
 vec=None;contract={'initialized':0,'used':0,'closed':0};results=[];baseline_pair={};candidates={};forks={};started=time.time()
 if stage=='round2_extension':
  old=json.loads((HERE/'round1/result.json').read_text())
  for i,sp in enumerate(specs):
   e=next(e for e in old['episodes'] if e['seed']==sp['seed']);baseline_pair[i]=e['pair'];candidates[i]={k:(np.asarray(v['raw'],np.float32),np.asarray(v['executed'],np.float32)) for k,v in e['all_candidates'].items()}
 try:
  vec=StaggeredVectorEnv(DATASET,4,20007,delay=.5,timeout=120,startup_parallelism=4,shared_memory=False);assert vec.alive_worker_count()==4;contract['initialized']=4;contract['worker_pids']=[x.pid for x in vec.processes]
  for repetition,future in enumerate(reg['future_rng_seeds']):
   uniforms=np.random.RandomState(future).uniform(size=(700,4))
   for branch in reg['unique_candidate_branches']:
    initial=vec.reset_many({i:s['seed'] for i,s in enumerate(specs)});obs=[initial[i] for i in range(4)];executors={k:BatchedGMMExecutor(a,scale,offset,4,horizon=10) for k,a in actors.items()};cont=MeanExecutor(ref,scale,offset,4,horizon=10)
    torch.manual_seed(reg['candidate_common_rng']);torch.npu.manual_seed(reg['candidate_common_rng']);active=[True]*4;traces=[[] for _ in specs];metadata={}
    for t in range(700):
     rng=torch.get_rng_state();nrng=torch.npu.get_rng_state();samples={}
     need_candidates=any(t<=forks.get(i,sp['fork']) for i,sp in enumerate(specs))
     for k,ex in (executors.items() if need_candidates else []):
      torch.set_rng_state(rng);torch.npu.set_rng_state(nrng);samples[k]=ex.actions_for(list(range(4)),obs,0.,None,None)
      if k=='BC':after=(torch.get_rng_state(),torch.npu.get_rng_state())
     if not need_candidates:samples={k:np.zeros((4,14),np.float32) for k in actors};after=(rng,nrng)
     samples['READINESS']=samples['BC']
     torch.set_rng_state(after[0]);torch.npu.set_rng_state(after[1]);means,probs,modes=cont.actions_with_uniforms(obs,uniforms[t]);actions=[]
     for i,s in enumerate(specs):
      fork=forks.get(i,s['fork'])
      if stage=='round3' and i not in forks and t>=120:
       if (traces[i] and any(np.sign(samples['BC'][i][j])!=np.sign(traces[i][-1]['action_executed'][j]) for j in [6,13])) or t==350:forks[i]=t;fork=t
      if stage=='round3' and i not in forks:fork=701
      action=np.asarray(samples['BC'][i],np.float32)
      if not active[i]:actions.append(action);continue
      if t<fork and prefixes[i] is not None:
       err=float(np.max(np.abs(obs_to_flat(obs[i])-prefixes[i][t]['observation_flat'])));assert err<1e-6,(i,t,err)
       action=np.asarray(prefixes[i][t]['action'],np.float32)
      elif t>fork:action=means[i]
      elif t==fork:
       vec.connections[i].send(('snapshot',None));msg=vec._recv(i,120,'fork_snapshot');assert msg[0]=='SNAPSHOT';sh=snapshot_hash(msg[1])
       rows=traces[i][-9:];o=np.asarray([x['observation_flat'] for x in rows]+[obs_to_flat(obs[i]).tolist()],np.float32);a=np.asarray([x['action_executed'] for x in rows]+[np.zeros(14).tolist()],np.float32);steps=np.arange(t-9,t+1)
       pair={'physical':sh,'history':hashlib.sha256(o.tobytes()+a[:-1].tobytes()+steps.tobytes()).hexdigest(),'continuation_hidden':hh(cont,i),'parent_rng':hashlib.sha256(after[0].numpy().tobytes()+after[1].cpu().numpy().tobytes()).hexdigest(),'matched':True}
       if i not in baseline_pair:baseline_pair[i]=pair
       assert pair==baseline_pair[i],(repetition,branch,i,'fork mismatch')
       cand={'A_BC':np.asarray(samples['BC'][i],np.float32),'B_READINESS':np.asarray(samples['READINESS'][i],np.float32),'C_POST1250':np.asarray(samples['POST1250'][i],np.float32)}
       assert np.array_equal(cand['A_BC'],cand['B_READINESS'])
       cand['D_INTERPOLATION_0.5']=(cand['A_BC']+cand['C_POST1250'])/2
       if i not in candidates:candidates[i]={k:(v.copy(),v.clip(-1,1)) for k,v in cand.items()}
       for k,v in cand.items():assert np.array_equal(v,candidates[i][k][0]),(i,k,'candidate changed')
       raw,action=candidates[i][branch];q={}
       with torch.no_grad():
        ot=torch.as_tensor(o[None],device=device);at=torch.as_tensor(a[None],device=device);st=torch.as_tensor(steps[None],device=device);act=torch.as_tensor(action[None],device=device)
        for k,c in critics.items():
         z=encode_replay_contexts(c,ot,at,st,700);u,v=c.q_from_context(tuple(x[:,-1] for x in z),act);q[k]={'Q1':float(u.item()),'Q2':float(v.item()),'Qmean':float((u+v).item()/2)}
       metadata[i]={'fork':t,'q':q,'raw_candidate':raw.tolist(),'executed_candidate':action.tolist(),'all_candidates':{k:{'raw':v[0].tolist(),'executed':v[1].tolist()} for k,v in candidates[i].items()},'pair':pair,'progress':t/700,'history_observations':o.tolist(),'history_actions':a.tolist(),'episode_steps':steps.tolist(),'fork_state':msg[1]}
      actions.append(action)
     ids=[i for i in range(4) if active[i]];before={i:obs_to_flat(obs[i]).tolist() for i in ids}
     messages=vec.step([np.asarray(actions[i]).clip(-1,1) for i in ids],ids)
     for i,msg in messages:
      assert msg[0]=='OK',msg;_,no,r,done,won,info=msg
      row={'seed':specs[i]['seed'],'timestep':t,'observation_flat':before[i],'next_observation_flat':obs_to_flat(no).tolist(),'action_raw':np.asarray(actions[i]).tolist(),'action_executed':np.asarray(actions[i]).clip(-1,1).tolist(),'reward':float(r),'success':bool(won),'terminated':bool(done or won),'truncated':bool(t==699 and not(done or won)),'continuation_mode':int(modes[i]) if t>metadata.get(i,{}).get('fork',700) else None}
      traces[i].append(row);obs[i]=no;active[i]=not(done or won or t==699)
     if t%100==0:print('PROGRESS',stage,repetition,branch,t,sum(active),flush=True)
     if not any(active):break
    assert len(metadata)==4,('missing preregistered fork',len(metadata));contract['used']=4;episodes=[]
    for i,rs in enumerate(traces):
     ret=0.
     for row in reversed(rs):ret=row['reward']+.99*ret;row['finite_mc_return']=ret
     m=metadata[i];e={'seed':specs[i]['seed'],'group':specs[i]['group'],'branch':branch,'future_rng':future,'length':len(rs),'success':rs[-1]['success'],'mc_from_fork':rs[m['fork']]['finite_mc_return'],'reward_values':sorted(set(r['reward'] for r in rs)),**m}
     assert all(r['reward'] in [0,1] for r in rs);episodes.append(e)
    stem=f'rep{repetition}_{branch}'
    with (out/(stem+'_trace.jsonl')).open('x') as f:
     for rs in traces:
      for row in rs:f.write(json.dumps(row,allow_nan=False)+'\n')
    dump(out/(stem+'_result.json'),{'episodes':episodes});results.extend(episodes)
    print('BRANCH_DONE',stem,[(e['seed'],e['success'],e['length']) for e in episodes],flush=True)
 finally:
  if vec is not None:
   vec.close();contract['closed']=sum(not p.is_alive() and p.exitcode==0 for p in vec.processes);contract['exitcodes']=[p.exitcode for p in vec.processes]
 assert all(contract[k]==4 for k in ['initialized','used','closed']);assert all(module_hash(v)==hashes[k] for k,v in models.items())
 dump(out/'result.json',{'episodes':results,'env_contract':contract,'actor_updates':0,'critic_updates':0,'model_hashes_unchanged':True,'model_hashes':hashes,'wall_seconds':time.time()-started,'semantics':{'ready':'MATCHED_TARGET_CATEGORICAL_COMPONENT_MEAN_CONTINUATION','MC':'SEMANTICS_MISMATCH_WITH_MIXED_COLLECTION_CONTINUATION'},'candidate_B_alias':'A_BC','device':str(device)})
 print('DONE',stage,'initialized=4 used=4 closed=4',flush=True)
if __name__=='__main__':main()
