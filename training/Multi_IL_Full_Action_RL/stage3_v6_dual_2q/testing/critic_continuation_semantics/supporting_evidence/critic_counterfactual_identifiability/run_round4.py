"""Independent same-action, continuation-labelled value evaluation. No repair."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_round2 import hidden_hash,snapshot_hash,snapshot_worker
from run_round1 import *
import stage3_v5_vector_env as vector
from stage3_v5_actor import BatchedGMMExecutor,obs_to_flat
vector._worker=snapshot_worker

def main():
 out=HERE/'round4';os.chdir(out);reg=json.loads((out/'preregistration.json').read_text());seeds=reg['seeds'];fork=reg['fork'];device,ready,bc,td,scale,offset=setup();bc.eval().requires_grad_(False);current=copy.deepcopy(bc);current.load_state_dict(torch.load(TEST/'mean_multi_collapse_diagnosis/round1/actor_1250.pth',map_location='cpu',weights_only=False)['actor']);current.eval().requires_grad_(False);manifest=json.loads((RUN/'shared/stage2_source_manifest.json').read_text());mc,_=strict_stage2_load(manifest['multi_q']['checkpoint'],device);mc.eval().requires_grad_(False);critics={'ready':td,'MC':mc};hashes={k:module_hash(v) for k,v in {**critics,'BC':bc,'current':current}.items()};baseline_pairs={};vec=None;branches={};contract={'initialized':0,'used':0,'closed':0};started=time.time()
 try:
  from core import StaggeredVectorEnv,DATASET
  vec=StaggeredVectorEnv(DATASET,4,20007,delay=.5,timeout=120,startup_parallelism=4,shared_memory=False);assert vec.alive_worker_count()==4;contract['initialized']=len(vec.initial_observations);contract['worker_pids']=[p.pid for p in vec.processes]
  for branch in reg['branches']:
   initial=vec.reset_many(dict(enumerate(seeds)));obs=[initial[i] for i in range(4)];refex=BatchedGMMExecutor(bc,scale,offset,4,horizon=10);curex=BatchedGMMExecutor(current,scale,offset,4,horizon=10);torch.manual_seed(reg['RNG']);torch.npu.manual_seed_all(reg['RNG']);traces=[[] for _ in range(4)];hist=[[] for _ in range(4)];active=[True]*4;scores={};pairchecks={}
   for t in range(700):
    rcpu=torch.get_rng_state();rnpu=torch.npu.get_rng_state();reference=refex.actions_for(list(range(4)),obs,0.,None,None);aftercpu=torch.get_rng_state();afternpu=torch.npu.get_rng_state();torch.set_rng_state(rcpu);torch.npu.set_rng_state(rnpu);candidate=curex.actions_for(list(range(4)),obs,0.,None,None);torch.set_rng_state(aftercpu);torch.npu.set_rng_state(afternpu)
    actions=np.asarray(candidate if branch=='CURRENT' and t>fork else reference,np.float32).clip(-1,1)
    if t==fork:
     assert all(active),'No reselection if seed completes before locked fork'
     for i in range(4):vec.connections[i].send(('snapshot',None))
     for i in range(4):
      msg=vec._recv(i,120,'heldout_snapshot');assert msg[0]=='SNAPSHOT';o=np.asarray([r['o'] for r in hist[i][-9:]]+[obs_to_flat(obs[i]).tolist()],np.float32);a=np.asarray([r['a'] for r in hist[i][-9:]]+[actions[i].tolist()],np.float32);st=np.arange(t-9,t+1)
      pair={'physical_hash':snapshot_hash(msg[1]),'history_hash':__import__('hashlib').sha256(o.tobytes()+a[:-1].tobytes()+st.tobytes()).hexdigest(),'BC_hidden':hidden_hash(refex,i),'CURRENT_hidden':hidden_hash(curex,i),'parent_rng':__import__('hashlib').sha256(aftercpu.numpy().tobytes()+afternpu.cpu().numpy().tobytes()).hexdigest(),'candidate':actions[i].tolist()}
      if branch=='BC':baseline_pairs[i]=pair
      assert baseline_pairs[i]==pair,(branch,i,'pair mismatch');pairchecks[i]=pair
      with torch.no_grad():
       ot=torch.as_tensor(o[None],device=device);at=torch.as_tensor(a[None],device=device);steps=torch.as_tensor(st[None],device=device);act=torch.as_tensor(actions[i][None],device=device);values={}
       for name,c in critics.items():
        z=encode_replay_contexts(c,ot,at,steps,700);u,v=c.q_from_context(tuple(x[:,-1] for x in z),act);values[name]=[float(u.item()),float(v.item())]
      scores[i]={'q':values,'history_observations':o.tolist(),'history_actions':a.tolist(),'episode_steps':st.tolist(),'candidate':actions[i].tolist()}
    ids=[i for i in range(4) if active[i]];before={i:obs_to_flat(obs[i]).copy() for i in ids};messages=vec.step([actions[i] for i in ids],ids)
    for i,m in messages:
     assert m[0]=='OK',m;_,no,r,done,won,info=m;hist[i].append({'o':before[i].tolist(),'a':actions[i].tolist()});traces[i].append({'seed':seeds[i],'timestep':t,'observation_flat':before[i].tolist(),'next_observation_flat':obs_to_flat(no).tolist(),'action':actions[i].tolist(),'reward':float(r),'success':bool(won),'terminated':bool(done or won),'truncated':bool(t==699 and not(done or won))});obs[i]=no;active[i]=not(done or won or t==699)
    if t%100==0:print('HELDOUT_PROGRESS',branch,t,sum(active),flush=True)
    if not any(active):break
   assert len(scores)==4;contract['used']=4;episodes=[]
   for i,rows in enumerate(traces):
    ret=0.
    for r in reversed(rows):ret=r['reward']+.99*ret;r['finite_mc_return']=ret
    episodes.append({'seed':seeds[i],'fork':fork,'length':len(rows),'success':rows[-1]['success'],'mc_from_fork':rows[fork]['finite_mc_return'],'reward_sum':sum(r['reward'] for r in rows),'reward_values':sorted(set(r['reward'] for r in rows)),**scores[i],'pair_check':pairchecks[i]})
   with (out/(branch+'_heldout_trace.jsonl')).open('x') as f:
    for rs in traces:
     for r in rs:f.write(json.dumps(r,allow_nan=False)+'\n')
   branches[branch]={'episodes':episodes,'success_count':sum(e['success'] for e in episodes)};dump(out/(branch+'_heldout_result.json'),branches[branch]);print('HELDOUT_BRANCH_DONE',branch,branches[branch]['success_count'],flush=True)
 finally:
  if vec is not None:
   vec.close();contract['closed']=sum(not p.is_alive() and p.exitcode==0 for p in vec.processes);contract['exitcodes']=[p.exitcode for p in vec.processes]
 assert all(contract[k]==4 for k in ['initialized','used','closed']),contract;assert all(module_hash(v)==hashes[k] for k,v in {**critics,'BC':bc,'current':current}.items());rows=[]
 for e,b in zip(branches['CURRENT']['episodes'],branches['BC']['episodes']):rows.append({'seed':e['seed'],'BC_success':b['success'],'current_success':e['success'],'BC_length':b['length'],'current_length':e['length'],'BC_mc':b['mc_from_fork'],'current_mc':e['mc_from_fork'],'delta_mc':e['mc_from_fork']-b['mc_from_fork'],'strict_pair':e['pair_check']==b['pair_check'],'same_Q':e['q']==b['q'],'q':e['q']})
 errors={}
 for name in critics:
  errors[name]={branch:float(np.mean([abs(np.mean(e['q'][name])-e['mc_from_fork']) for e in b['episodes']])) for branch,b in branches.items()}
 dump(out/'result.json',{'branches':branches,'pairs':rows,'env_contract':contract,'semantic_evaluation_mean_abs_realized_return_error':errors,'model_hashes_unchanged':True,'actor_updates':0,'critic_updates':0,'wall_seconds':time.time()-started,'limits':['One common future RNG per context,not conditional mean or confidence interval.','Continuation-labelled empirical values are an evaluation audit,not a fitted Critic or causal training repair.','Reference success is not assumed;unscreened seeds retained even if BC fails.'],'classification':'CONTINUATION_VALUE_GAP_HELDOUT_VALIDATION'});print('ROUND4_DONE initialized / used / closed = 4 / 4 / 4',json.dumps({'pairs':rows,'errors':errors}),flush=True)
if __name__=='__main__':main()
