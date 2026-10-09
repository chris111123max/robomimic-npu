"""Frozen Actor proposals with independent competence acceptance; testing only."""
import sys,os,json,copy,time,hashlib,random
from pathlib import Path
HERE=Path(__file__).resolve().parent
TEST=HERE.parent
PREV=TEST/'q_gain_matched_path_test'
sys.path.insert(0,str(PREV))
import run_test as qg
import core
import numpy as np
import torch
qg.HERE=HERE
REPORTING=(20008,20002,20005,20007)
MAX_UPDATES=10000
BLOCK=250
INITIAL_SEED=20261004
SELECTION_GROUPS=[(20003,20004,20010,20011)]+[tuple(range(20012+4*i,20016+4*i)) for i in range(19)]
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):return core.dump(p,x)
def rng():
 return {'cpu':torch.get_rng_state().clone(),'npu':torch.npu.get_rng_state().clone(),'numpy':copy.deepcopy(np.random.get_state()),'python':random.getstate()}
def restore_rng(x):
 torch.set_rng_state(x['cpu']);torch.npu.set_rng_state(x['npu']);np.random.set_state(x['numpy']);random.setstate(x['python'])
def rnghash(x):
 h=hashlib.sha256()
 h.update(x['cpu'].numpy().tobytes());h.update(x['npu'].numpy().tobytes());h.update(repr(x['numpy']).encode());h.update(repr(x['python']).encode())
 return h.hexdigest()
def torch_equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(torch_equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(torch_equal(x,y) for x,y in zip(a,b))
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 return a==b
def actor_save(actor,optimizer,count,name,rstate):
 p=HERE/(name+'.pth');assert not p.exists()
 torch.save({'testing_only':True,'actor':actor.state_dict(),'actor_optimizer':optimizer.state_dict(),'accepted_actor_updates':count,'rng':rstate,'source_env_steps':140000},p)
 return str(p)
def metrics(actor,ref,critic,contexts,device,scale,offset,probeobs,base,name,count):
 return qg.measure(actor,ref,critic,contexts,device,scale,offset,probeobs,base,name,count)
def evaluate(actor,seeds,name,scale,offset):
 # Reuse the exact earlier four-worker evaluator; each invocation closes its OWN pool.
 assert len(seeds)==4
 prior_rng=rng();prior_cwd=Path.cwd();oldseeds=core.SEEDS
 out=HERE/'evaluations'/name;out.mkdir(parents=True,exist_ok=False)
 start=time.time()
 try:
  core.SEEDS=tuple(seeds);os.chdir(out)
  behavior,contract=core.evaluate_four([(name,[actor],1)],scale,offset,out)
 finally:
  core.SEEDS=oldseeds;os.chdir(prior_cwd);restore_rng(prior_rng)
 result={'label':name,'device':'npu:0','seeds':list(seeds),'behavior':behavior[name],'contract':contract,'env_steps':sum(e['length'] for e in behavior[name]['episodes']),'wall_time_seconds':time.time()-start,'actor_updates_during_evaluation':0,'critic_updates':0,'command':'python '+str(HERE/'run_acceptance.py')+' '+sys.argv[1]}
 assert contract['parallel_envs_initialized']==contract['parallel_envs_used']==contract['parallel_envs_closed']==4
 dump(out/'invocation.json',result)
 print('SIM_DONE',json.dumps({k:v for k,v in result.items() if k not in ['behavior','contract']}),behavior[name]['success_count'],flush=True)
 return result
def preparation():
 # Read FULL documents and outputs; no inference from remembered values.
 for p in [PREV/'FINAL_REPORT.md',PREV/'final_summary.json',TEST/'mean_multi_collapse_diagnosis/round1/result.json',TEST/'mean_multi_collapse_diagnosis/round1/contract.json',TEST/'mean_multi_collapse_diagnosis/core.py']:
  content=read(p) if p.suffix=='.json' else p.read_text()
  print('READ_EXISTING',str(p),hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
 dump(HERE/'safety_before.json',qg.hashes())
 device,ready,ref,critic,scale,offset,bank,probe,contexts,probeobs,base,replay=qg.prepare()
 old=read(PREV/'design.json')
 assert qg.arrays_hash(bank)==old['bank_hash'] and qg.arrays_hash([probe])==old['drift_probe_hash'] and qg.arrays_hash([contexts])==old['q_probe_hash']
 torch.manual_seed(INITIAL_SEED);torch.npu.manual_seed_all(INITIAL_SEED);np.random.seed(INITIAL_SEED);random.seed(INITIAL_SEED)
 initial=rng()
 bankpath=HERE/'fixed_batch_bank.pth'
 torch.save({'testing_only':True,'bank':bank,'probe':probe,'contexts':contexts,'initial_rng':initial},bankpath)
 stream=np.arange(MAX_UPDATES,dtype=np.int64)%64;np.save(HERE/'batch_index_stream.npy',stream)
 design={'branch':'mean2q/multi_q ONLY','device':'npu:0','block_actor_updates':BLOCK,'max_proposed_actor_updates':MAX_UPDATES,'acceptance_rule':'success>=3/4 AND simulator errors=0; no Q/reporting feedback','reporting_seeds':list(REPORTING),'seed_selection_groups':[list(s) for s in SELECTION_GROUPS],'seed_selection_rule':'existing formal readiness confirms only disjoint20003/20004; first group includes those plus20010/20011, then consecutive ascending disjoint groups; readiness actor ONLY until first4/4; max20groups; lock first4/4, never change','bank_hash':old['bank_hash'],'q_probe_hash':old['q_probe_hash'],'drift_probe_hash':old['drift_probe_hash'],'initial_rng_hash':rnghash(initial),'initial_seed':INITIAL_SEED,'stream':'saved before any candidate; arange(10000)%64; proposal cursor advances on reject; no success-guided selection','rollback':'restore Actor parameters, optimizer state, accepted update counter, RNG; advance ONLY proposal batch cursor','schedule':'EXACT prior core.schedule indexed by ACCEPTED update count plus candidate local update; reject restores count; no nominal env steps generated','q_matching_tolerance':.05,'crossing_rule':'first accepted Q>=95% target: save before/after accepted snapshots, choose nearest by Q ONLY; report mismatch even if outside5%; evaluate once; never train to fit','stop_rule':'stop on accepted collapse Q within5%; otherwise predeclared10000proposal cap','critic_and_target':'readiness frozen; no Critic/target updates','control':'reuse exact q_gain_matched references, independently verified bank/probe hashes and first250 candidate Q','reporting_set_never_used_in_acceptance':True}
 dump(HERE/'design.json',design)
 dump(HERE/'source_existing.json',{'q_gain_matched':read(PREV/'final_summary.json'),'frozen_baseline':read(TEST/'mean_multi_collapse_diagnosis/round1/result.json')})
 return device,ready,ref,critic,scale,offset,bank,probe,contexts,probeobs,base,stream,initial
def select():
 device,ready,ref,critic,scale,offset,bank,probe,contexts,probeobs,base,stream,initial=preparation()
 records=[]
 for i,seeds in enumerate(SELECTION_GROUPS,1):
  assert not set(seeds)&set(REPORTING)
  x=evaluate(ref,seeds,'SELECTION_'+str(i).zfill(2),scale,offset);records.append(x)
  if x['behavior']['success_count']==4:
   dump(HERE/'acceptance_seeds.json',{'seeds':list(seeds),'reporting_seeds':list(REPORTING),'selected_group':i,'selection_records':records,'readiness_baseline':x,'locked':True})
   print('ACCEPTANCE_LOCKED',seeds,flush=True);return
 dump(HERE/'selection_failed.json',{'records':records,'classification':'MIXED_OR_INCONCLUSIVE','reason':'No4/4 readiness set found within predeclared20groups; no Actor experiment launched'})
 print('SELECTION_FAILED',flush=True)
def experiment():
 design=read(HERE/'design.json');chosen=read(HERE/'acceptance_seeds.json');seeds=chosen['seeds']
 device,ready,ref,critic,scale,offset,bank,probe,contexts,probeobs,base,replay=qg.prepare()
 saved=torch.load(HERE/'fixed_batch_bank.pth',map_location='cpu',weights_only=False)
 assert qg.arrays_hash(bank)==design['bank_hash'] and qg.arrays_hash([contexts])==design['q_probe_hash']
 bank=saved['bank'];stream=np.load(HERE/'batch_index_stream.npy')
 actor=copy.deepcopy(ref);optimizer=torch.optim.Adam(actor.parameters(),lr=0.)
 optimizer.load_state_dict(copy.deepcopy(ready['actor_optimizer']));assert not optimizer.state
 restore_rng(saved['initial_rng']);cfg=ready['config'];envs,lrs=core.schedule(cfg,MAX_UPDATES)
 refs=read(PREV/'stage_a.json')['references'];targets=read(PREV/'stage_a.json')['targets'];ch=core.module_hash(critic)
 accepted_count=0;proposed=0;accepted_blocks=0;rejected_blocks=0;records=[];milestones={};started=time.time()
 start=metrics(actor,ref,critic,contexts,device,scale,offset,probeobs,base,'accepted_start',0)
 start['checkpoint']=actor_save(actor,optimizer,0,'accepted_start',rng())
 start['acceptance']=chosen['readiness_baseline'];start['reporting']=refs['START']['behavior']
 current=start
 with (HERE/'proposal_logs.jsonl').open('x') as log,(HERE/'actor_updates.jsonl').open('x') as trace:
  for block in range(1,41):
   before_actor=copy.deepcopy(actor.state_dict());before_optim=copy.deepcopy(optimizer.state_dict());before_rng=rng();before_count=accepted_count;before=current
   before_hash=core.module_hash(actor);before_optimizer=copy.deepcopy(optimizer.state_dict())
   batch_ids=stream[proposed:proposed+BLOCK].tolist();rng_before=rnghash(before_rng)
   for j,idx in enumerate(batch_ids,1):
    u=accepted_count+j;actor.train()
    for pg in optimizer.param_groups:pg['lr']=lrs[u-1]
    optimizer.zero_grad(set_to_none=True)
    loss,_=core.production_actor_loss(actor,critic,bank[idx],scale,offset,cfg,device);loss.backward()
    assert bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
    torch.nn.utils.clip_grad_norm_(actor.parameters(),cfg['actor_max_grad_norm']);optimizer.step()
    trace.write(json.dumps({'proposal_block':block,'proposed_update':proposed+j,'candidate_accepted_update_index':u,'batch_index':idx,'lr':lrs[u-1],'loss':float(loss)},allow_nan=False)+'\n')
   proposed+=BLOCK;candidate_rng=rng()
   candidate=metrics(actor,ref,critic,contexts,device,scale,offset,probeobs,base,'candidate_'+str(block).zfill(2),accepted_count+BLOCK)
   if block==1:assert abs(candidate['q']['all']['q1_gain']-targets['LOW'])<1e-7
   acceptance=evaluate(actor,seeds,'ACCEPTANCE_'+str(block).zfill(2),scale,offset)
   # ONLY acceptance observations are used here. Reporting is downstream, read-only.
   decision='ACCEPT' if acceptance['behavior']['success_count']>=3 and acceptance['behavior']['sim_errors']==0 else 'REJECT'
   if decision=='ACCEPT':
    accepted_count+=BLOCK;accepted_blocks+=1;restore_rng(candidate_rng)
    candidate['checkpoint']=actor_save(actor,optimizer,accepted_count,'accepted_block_'+str(block).zfill(2),rng())
    candidate['acceptance']=acceptance;current=candidate;rollback_verified=None
   else:
    rejected_blocks+=1;actor.load_state_dict(before_actor,strict=True);optimizer.load_state_dict(before_optim);restore_rng(before_rng)
    accepted_count=before_count;current=before
    rollback_verified=core.module_hash(actor)==before_hash and torch_equal(optimizer.state_dict(),before_optimizer) and rnghash(rng())==rng_before
    assert rollback_verified
   row={'block':block,'proposed_updates':proposed,'candidate_metrics':candidate,'candidate_q1_minus_previous_accepted':candidate['q']['all']['q1_gain']-before['q']['all']['q1_gain'],'candidate_q2_minus_previous_accepted':candidate['q']['all']['q2_gain']-before['q']['all']['q2_gain'],'acceptance':acceptance,'decision':decision,'accepted_cumulative_updates':accepted_count,'accepted_cumulative_q':current['q']['all'],'accepted_metrics':current,'batch_indices':batch_ids,'batch_indices_sha256':hashlib.sha256(np.asarray(batch_ids,dtype=np.int64).tobytes()).hexdigest(),'rng_hash_before_proposal':rng_before,'rollback_verified':rollback_verified,'accepted_blocks':accepted_blocks,'rejected_blocks':rejected_blocks}
   log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();trace.flush();records.append(row)
   print('DECISION',json.dumps({'block':block,'proposed':proposed,'accepted':accepted_count,'candidate_q1':candidate['q']['all']['q1_gain'],'acceptance_success':acceptance['behavior']['success_count'],'decision':decision,'accepted_q1':current['q']['all']['q1_gain']}),flush=True)
   if decision=='ACCEPT':
    for label,target in targets.items():
     if label not in milestones and current['q']['all']['q1_gain']>=.95*target:
      near=min([before,current],key=lambda r:abs(r['q']['all']['q1_gain']-target))
      eval_actor=copy.deepcopy(ref);eval_actor.load_state_dict(torch.load(near['checkpoint'],map_location='cpu',weights_only=False)['actor'])
      reporting=evaluate(eval_actor,REPORTING,'REPORTING_'+label,scale,offset)
      err=abs(near['q']['all']['q1_gain']-target)/target
      m={'target':label,'target_q1':target,'metrics':near,'reporting':reporting,'relative_matching_error':err,'matched_within5percent':err<=.05,'crossing_before_checkpoint':before['checkpoint'],'crossing_after_checkpoint':current['checkpoint'],'selected_by':'nearest Q ONLY, never reporting success'}
      milestones[label]=m;dump(HERE/('milestone_'+label.lower()+'.json'),m)
      print('REPORT_MILESTONE',label,err,reporting['behavior']['success_count'],flush=True)
   assert core.module_hash(critic)==ch
   dump(HERE/('progress_block_'+str(block).zfill(2)+'.json'),{'proposed_actor_updates':proposed,'accepted_actor_updates':accepted_count,'rejected_actor_updates':proposed-accepted_count,'current':current,'milestones':milestones})
   if 'COLLAPSE' in milestones and milestones['COLLAPSE']['matched_within5percent']:break
 result={'start':start,'records':records,'milestones':milestones,'current_accepted':current,'proposed_actor_updates':proposed,'accepted_actor_updates':accepted_count,'rejected_actor_updates':proposed-accepted_count,'proposal_blocks':len(records),'accepted_blocks':accepted_blocks,'rejected_blocks':rejected_blocks,'critic_updates':0,'critic_hash_unchanged':core.module_hash(critic)==ch,'runtime_seconds':time.time()-started,'reporting_used_for_acceptance':False,'stop_reason':'matched collapse target' if 'COLLAPSE' in milestones and milestones['COLLAPSE']['matched_within5percent'] else '10000proposal budget reached','formal_training_stopped':True}
 dump(HERE/'round1_result.json',result);print('ROUND1_DONE',proposed,accepted_count,flush=True)
if __name__=='__main__':{'select':select,'experiment':experiment}[sys.argv[1]]()
