"""Four concurrent workers, strict paired candidate actions followed by BC."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_round1 import *
# Reuse read-only snapshot implementation, never its output-writing helpers.
sys.path.insert(0,str(TEST/'final_collapse_rootcause'))
from snapshot_worker import snapshot_worker
from helpers import snapshot_hash
import stage3_v5_vector_env as vector
from stage3_v5_actor import BatchedGMMExecutor,obs_to_flat
vector._worker=snapshot_worker

def hidden_hash(ex,i):
 import hashlib
 st=ex.hidden[i];parts=[]
 if st is not None:
  for x in st if isinstance(st,tuple) else [st]:parts.append(x.detach().cpu().numpy().tobytes())
 return hashlib.sha256(b''.join(parts)+str(ex.counters[i]).encode()).hexdigest()
def main():
 out=HERE/'round2';os.chdir(out);reg=json.loads((out/'preregistration.json').read_text());data=np.load(HERE/'round1/contexts.npz');pr=np.load(HERE/'round1/probes.npz');names=pr['candidate_names'].tolist();specs=reg['contexts'];seeds=[s['seed'] for s in specs]
 device,ready,bc,td,scale,offset=setup();bc.eval().requires_grad_(False);manifest=json.loads((RUN/'shared/stage2_source_manifest.json').read_text());mc,_=strict_stage2_load(manifest['multi_q']['checkpoint'],device);mc.eval().requires_grad_(False);critics={'ready':td,'MC':mc};hashes={k:module_hash(v) for k,v in {**critics,'BC':bc}.items()}
 sources={'BC_success':TEST/'mean_multi_collapse_diagnosis/round1/READY_trajectories.jsonl','early_degradation':TEST/'critic_gradient_rootcause/round1/evaluations/FIXED_STATES/OLD_ACCEPTED_BLOCK_10_trajectories.jsonl'};prefixes=[]
 for s in specs:
  raw=[json.loads(l) for l in sources[s['region']].read_text().splitlines()];prefixes.append(sorted([r for r in raw if r['seed']==s['seed']],key=lambda r:r['timestep']))
 vec=None;branches={};snapshots={};hist_hashes={};hidden_hashes={};rng_hashes={};contract={'initialized':0,'used':0,'closed':0};started=time.time()
 try:
  from core import StaggeredVectorEnv,DATASET
  vec=StaggeredVectorEnv(DATASET,4,20007,delay=.5,timeout=120,startup_parallelism=4,shared_memory=False);assert vec.alive_worker_count()==4;contract['initialized']=len(vec.initial_observations);contract['worker_pids']=[p.pid for p in vec.processes]
  for branch in reg['branches']:
   initial=vec.reset_many(dict(enumerate(seeds)));obs=[initial[i] for i in range(4)];ex=BatchedGMMExecutor(bc,scale,offset,4,horizon=10);torch.manual_seed(20007);torch.npu.manual_seed_all(20007);active=[True]*4;traces=[[] for _ in range(4)];scores={};pair_checks={};raw_rewards=[]
   for t in range(700):
    actions=ex.actions_for(list(range(4)),obs,0.,None,None)
    for i,s in enumerate(specs):
     if not active[i]:continue
     if t<=s['step']:
      diff=float(np.max(np.abs(obs_to_flat(obs[i])-np.asarray(prefixes[i][t]['observation_flat']))));assert diff<1e-6,(i,t,diff)
     if t<s['step']:actions[i]=np.asarray(prefixes[i][t]['action'],np.float32)
     elif t==s['step']:
      vec.connections[i].send(('snapshot',None));msg=vec._recv(i,120,'paired_snapshot');assert msg[0]=='SNAPSHOT';sh=snapshot_hash(msg[1]);hh=hidden_hash(ex,i)
      context_index=s['context_index'];o=data['o'][context_index];aa=data['a'][context_index];steps=data['steps'][context_index];assert np.allclose(o[-1],obs_to_flat(obs[i]),atol=1e-6,rtol=0)
      rh=__import__('hashlib').sha256(torch.get_rng_state().numpy().tobytes()+torch.npu.get_rng_state().cpu().numpy().tobytes()).hexdigest();ih=__import__('hashlib').sha256(o.tobytes()+aa[:-1].tobytes()+steps.tobytes()).hexdigest()
      if branch=='BC':snapshots[i]=sh;hidden_hashes[i]=hh;rng_hashes[i]=rh;hist_hashes[i]=ih
      assert snapshots[i]==sh and hidden_hashes[i]==hh and rng_hashes[i]==rh and hist_hashes[i]==ih,(branch,i,'pair mismatch')
      action=pr['ready_candidate_actions'][s['probe_index'],names.index(branch)];actions[i]=action.copy();assert np.all(np.abs(action)<=1)
      with torch.no_grad():
       ot=torch.as_tensor(o[None],device=device);at=torch.as_tensor(aa[None],device=device);st=torch.as_tensor(steps[None],device=device);a=torch.as_tensor(action[None],device=device);q={}
       for name,c in critics.items():
        z=encode_replay_contexts(c,ot,at,st,700);u,v=c.q_from_context(tuple(x[:,-1] for x in z),a);q[name]=[float(u.item()),float(v.item())]
      scores[i]={'q':q,'raw_candidate':action.tolist(),'executed_candidate':action.tolist(),'history_observations':o.tolist(),'history_actions':aa.tolist(),'episode_steps':steps.tolist()};pair_checks[i]={'physical_hash':sh,'BC_hidden_hash':hh,'parent_rng_hash':rh,'history_hash':ih,'matched':True}
    ids=[i for i in range(4) if active[i]];before={i:obs_to_flat(obs[i]).copy() for i in ids};messages=vec.step([actions[i] for i in ids],ids)
    for i,m in messages:
     assert m[0]=='OK',m;_,no,r,done,won,info=m;traces[i].append({'seed':seeds[i],'timestep':t,'observation_flat':before[i].tolist(),'next_observation_flat':obs_to_flat(no).tolist(),'action_raw':np.asarray(actions[i]).tolist(),'action_executed':np.asarray(actions[i]).clip(-1,1).tolist(),'reward':float(r),'success':bool(won),'terminated':bool(done or won),'truncated':bool(t==699 and not(done or won))});obs[i]=no;active[i]=not(done or won or t==699)
    if t%100==0:print('SIM_PROGRESS',branch,t,sum(active),flush=True)
    if not any(active):break
   assert len(scores)==4;contract['used']=4;episodes=[]
   for i,rows in enumerate(traces):
    ret=0.
    for r in reversed(rows):ret=r['reward']+.99*ret;r['finite_mc_return']=ret
    e={'seed':seeds[i],'region':specs[i]['region'],'fork':specs[i]['step'],'length':len(rows),'success':rows[-1]['success'],'mc_from_fork':rows[specs[i]['step']]['finite_mc_return'],'reward_sum':sum(r['reward'] for r in rows),'reward_values':sorted(set(r['reward'] for r in rows)),**scores[i],'pair_check':pair_checks[i]};episodes.append(e)
   with (out/(branch+'_strict_trace.jsonl')).open('x') as f:
    for rs in traces:
     for r in rs:f.write(json.dumps(r,allow_nan=False)+'\n')
   branches[branch]={'episodes':episodes,'success_count':sum(e['success'] for e in episodes)};dump(out/(branch+'_strict_result.json'),branches[branch]);print('SIM_BRANCH_DONE',branch,branches[branch]['success_count'],flush=True)
 finally:
  if vec is not None:
   vec.close();contract['closed']=sum(not p.is_alive() and p.exitcode==0 for p in vec.processes);contract['exitcodes']=[p.exitcode for p in vec.processes]
 assert all(contract[k]==4 for k in ['initialized','used','closed']),contract
 assert all(module_hash(v)==hashes[k] for k,v in {**critics,'BC':bc}.items());dump(out/'strict_result.json',{'branches':branches,'env_contract':contract,'pairing':'strict exposed simulator/controller/RNG+history+BC_hidden hashes','caveat':'deterministic prefix replay, not serialization of every opaque simulator internal; one common future RNG realization per context','actor_updates':0,'critic_updates':0,'wall_seconds':time.time()-started,'model_hashes_unchanged':True,'reward_components_available':'scalar sparse reward only; no decomposed reward components exposed'});print('ROUND2_DONE initialized / used / closed = 4 / 4 / 4',flush=True)
if __name__=='__main__':main()
