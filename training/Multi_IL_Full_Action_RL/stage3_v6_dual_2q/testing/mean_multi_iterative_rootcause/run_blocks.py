"""Adaptive mean-only diagnosis: Round0 and first action-block test only."""
import sys,json,hashlib,subprocess,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
TEST=HERE.parent
PREV=TEST/'mean_multi_upstream_rootcause'
sys.path.insert(0,str(PREV))
import diagnose as prior
from core import setup,SEEDS,DATASET,RUN,dump,StaggeredVectorEnv,BatchedGMMExecutor,obs_to_flat
from stage3_v5_history_critic import encode_replay_contexts
import numpy as np
import torch
START=199
def read(p):return json.loads(Path(p).read_text())
def report(p,s):p=Path(p);assert not p.exists();p.write_text(s)
def sha_snapshot():
 old=read(PREV/'safety_before.json')
 return {k:hashlib.sha256(Path(k).read_bytes()).hexdigest() for k in old}

def round0():
 out=HERE/'round0';out.mkdir(exist_ok=True)
 prev=read(PREV/'final_summary.json')
 reports={str(p):p.read_text() for p in [PREV/'FINAL_REPORT.md',PREV/'round4/ROUND_REPORT.md']}
 meta=read(TEST/'mean_multi_collapse_diagnosis/round0_existing/round0_existing_evidence.json')
 dump(out/'result.json',{'previous':prev,'reports':reports,'formal_metadata_existing':meta,'unknown':'When does tolerable local action substitution become cumulative harmful execution, and why does repeated Q ascent fail to stop it?','causal_known':['frozen-ready Critic/replay sufficient','failed-weight execution retraction dose response'],'decision':'CONTINUE_TO_ROUND_1'})
 dump(HERE/'safety_before.json',sha_snapshot())
 report(out/'ROUND_REPORT.md',"""# ROUND 0
Previous conclusion: repeated Q ascent causes cumulative action drift; upstream exact cause unresolved.
Question: isolated failed-actor action is tolerable; when do short blocks overwhelm recovery?
Existing data: previous same-state single action4/4; whole failed policy0/4; support-gradient rejection0/4. Full JSON/reports read and integrated.
Method: existing-data only; source/formal checksum snapshot.
Simulator required: NO.
Evidence: previous causal facts retained, upstream INCONCLUSIVE.
DOES THIS EXPLAIN FORMAL MEAN2Q COLLAPSE? PARTLY.
Decision: CONTINUE_TO_ROUND_1.
""")
 print('ROUND0_DONE',flush=True)

def round1():
 # Must explicitly read previous report and result before design execution.
 previous_report=(HERE/'round0/ROUND_REPORT.md').read_text()
 previous=read(HERE/'round0/result.json');assert previous['decision']=='CONTINUE_TO_ROUND_1'
 out=HERE/'round1';out.mkdir(exist_ok=True)
 dump(out/'design.json',{'previous_round_conclusion':previous['unknown'],'why_next':'single-action tolerability vs cumulative execution requires block intervention, not another training run','lengths':[8,32],'reused':[0,1],'fork_start':START,'device':'npu:0','seeds':list(SEEDS),'all_future_rounds_undecided':True})
 device,ready,ref,bad,critic,scale,offset,_=prior.prepare()
 ref.eval();bad.eval()
 rows=[json.loads(l) for l in (PREV/'round4/FORK_0p0_trace.jsonl').read_text().splitlines()]
 baseline={seed:[r for r in rows if r['seed']==seed] for seed in SEEDS}
 sd=np.maximum(np.asarray([r['obs'] for r in rows]).std(0),.01)
 vec=None;results={};contract={'device':'npu:0','seeds':list(SEEDS),'parallel_envs_initialized':0,'parallel_envs_used':0,'parallel_envs_closed':0,'sim_error_count':0}
 os.chdir(out)
 try:
  vec=StaggeredVectorEnv(DATASET,4,20000,delay=.5,timeout=120,startup_parallelism=4,shared_memory=False)
  contract.update(parallel_envs_initialized=len(vec.initial_observations),worker_pids=[p.pid for p in vec.processes])
  assert vec.alive_worker_count()==4
  for length in (8,32):
   exes=[BatchedGMMExecutor(a,scale,offset,4,horizon=10) for a in (ref,bad)]
   initial=vec.reset_many(dict(enumerate(SEEDS)));obs=[initial[i] for i in range(4)]
   torch.manual_seed(20007);torch.npu.manual_seed_all(20007)
   active=[True]*4;hist=[[] for _ in range(4)];won=[False]*4
   for t in range(700):
    samples=[];rng=torch.get_rng_state();nrng=torch.npu.get_rng_state()
    for ex in exes:
     torch.set_rng_state(rng);torch.npu.set_rng_state(nrng)
     samples.append(ex.actions_for(list(range(4)),obs,0.,None,None))
    a0=np.asarray(samples[0]);a1=np.asarray(samples[1])
    injected=START<=t<START+length
    actions=a1 if injected else a0
    if t==START:
     for i,seed in enumerate(SEEDS):
      assert np.allclose(obs_to_flat(obs[i]),baseline[seed][t]['obs'],atol=1e-6,rtol=0)
    ids=[i for i in range(4) if active[i]]
    before={i:obs_to_flat(obs[i]) for i in ids}
    qinfo={}
    if injected:
     cobs=np.asarray([[r['obs'] for r in hist[i][-9:]]+[before[i].tolist()] for i in ids])
     cacts=np.asarray([[r['action'] for r in hist[i][-9:]]+[a0[i].tolist()] for i in ids])
     st=np.asarray([[r['t'] for r in hist[i][-9:]]+[t] for i in ids])
     with torch.no_grad():
      enc=encode_replay_contexts(critic,torch.as_tensor(cobs,device=device,dtype=torch.float32),torch.as_tensor(cacts,device=device,dtype=torch.float32),torch.as_tensor(st,device=device,dtype=torch.long),700)
      ctx=(enc[0][:,-1],enc[1][:,-1])
      qr=critic.q_from_context(ctx,torch.as_tensor(a0[ids],device=device,dtype=torch.float32))
      qb=critic.q_from_context(ctx,torch.as_tensor(a1[ids],device=device,dtype=torch.float32))
      for j,i in enumerate(ids):qinfo[i]={'q1_ref':float(qr[0][j]),'q2_ref':float(qr[1][j]),'q1_bad':float(qb[0][j]),'q2_bad':float(qb[1][j])}
    for i,m in vec.step([actions[i] for i in ids],ids):
     assert m[0]=='OK',m
     _,no,r,done,win,info=m
     row={'t':t,'seed':SEEDS[i],'obs':before[i].tolist(),'action':actions[i].tolist(),'reference_action':a0[i].tolist(),'failed_action':a1[i].tolist(),'injected':injected,'reward':float(r),'success':bool(win),'terminated':bool(done or win),'truncated':bool(t==699 and not(done or win)),**qinfo.get(i,{})}
     if t<len(baseline[SEEDS[i]]):
      d=before[i]-np.asarray(baseline[SEEDS[i]][t]['obs'])
      row['state_deviation_standardized']=float(np.linalg.norm(d/sd)/np.sqrt(59))
      row['eef_deviation_m']=float(np.linalg.norm(d[[0,1,2,9,10,11]]))
     row['action_vs_current_bc']=float(np.linalg.norm((actions[i]-a0[i])/scale.cpu().numpy()))
     hist[i].append(row);won[i]=bool(win);obs[i]=no;active[i]=not(done or win or t==699)
    if t%100==0:print(json.dumps({'event':'block_rollout','length':length,'t':t,'active':sum(active)}),flush=True)
    if not any(active):break
   contract['parallel_envs_used']=4;episodes=[]
   for i,h in enumerate(hist):
    mc=0.
    for r in reversed(h):mc=r['reward']+.99*mc;r['mc']=mc
    post=[r for r in h if START+length<=r['t']<START+length+50 and 'state_deviation_standardized' in r]
    inj=[r for r in h if r['injected']]
    base_return=baseline[SEEDS[i]][START]['mc']
    episodes.append({'seed':SEEDS[i],'success':won[i],'length':len(h),'mc_from_start':h[START]['mc'],'delta_mc':h[START]['mc']-base_return,'sim_error':False,'post_recovery_state_deviation_mean':float(np.mean([r['state_deviation_standardized'] for r in post])) if post else None,'post_recovery_eef_deviation_mean_m':float(np.mean([r['eef_deviation_m'] for r in post])) if post else None,'block_max_state_deviation':float(max(r.get('state_deviation_standardized',0) for r in inj)),'block_max_eef_deviation_m':float(max(r.get('eef_deviation_m',0) for r in inj)),'mean_injected_action_drift':float(np.mean([r['action_vs_current_bc'] for r in inj]))})
   with (out/f'BLOCK_{length}_trace.jsonl').open('x') as f:
    for h in hist:
     for r in h:f.write(json.dumps(r,allow_nan=False)+'\n')
   results[str(length)]={'success_count':sum(won),'episodes':episodes}
   print(json.dumps({'event':'block_done','block':length,'result':results[str(length)]}),flush=True)
 finally:
  if vec is not None:
   vec.close();contract['parallel_envs_closed']=sum(not p.is_alive() and p.exitcode==0 for p in vec.processes);contract['worker_exitcodes']=[p.exitcode for p in vec.processes]
 contract['test_valid']=all(contract[k]==4 for k in ['parallel_envs_initialized','parallel_envs_used','parallel_envs_closed'])
 assert contract['test_valid'],contract
 dump(out/'result.json',{'branches':results,'env_contract':contract,'reused_single_action':read(PREV/'round4/result.json'),'fork_start':START,'actor_updates':0,'critic_updates':0,'formal_training_stopped':True})
 print('ROUND1_DONE',flush=True)

if __name__=='__main__':{'round0':round0,'round1':round1}[sys.argv[1]]()
