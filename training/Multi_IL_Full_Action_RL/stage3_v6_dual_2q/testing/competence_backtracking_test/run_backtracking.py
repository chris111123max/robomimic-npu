"""Testing-only competence line search. No formal training, Critic update or production edit."""
import sys,os,json,copy,time,hashlib,subprocess,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'closed_loop_acceptance_test'
spec=importlib.util.spec_from_file_location('prior_acceptance',OLD/'run_acceptance.py')
acc=importlib.util.module_from_spec(spec);spec.loader.exec_module(acc)
acc.qg.HERE=HERE
core=acc.core
import torch
import numpy as np
ACCEPTANCE=(20003,20004,20010,20011)
REPORTING=(20008,20002,20005,20007)
FRACTIONS=(1.,.5,.25,.125,.0625)
BLOCK=250;MAX_UPDATES=10000
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):return core.dump(p,x)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(actor,ref,critic,contexts,device,scale,offset,probeobs,base,name,count):
 return acc.qg.measure(actor,ref,critic,contexts,device,scale,offset,probeobs,base,name,count)
def save(actor,opt,count,name,rngstate,equivalent,fullupdates):
 p=HERE/(name+'.pth');assert not p.exists()
 torch.save({'testing_only':True,'actor':actor.state_dict(),'actor_optimizer':opt.state_dict(),'schedule_accepted_proposal_updates':count,'fraction_weighted_update_equivalent':equivalent,'full_accepted_optimizer_updates':fullupdates,'rng':rngstate,'source_env_steps':140000,'semantics':'partial: parameters interpolated; optimizer and RNG rollback_to_preproposal; nominal schedule count advances250 per admitted proposal'},p)
 return str(p)
def optimizer_steps(opt):
 return sorted(set(float(x['step'].detach().cpu()) if torch.is_tensor(x['step']) else float(x['step']) for x in opt.state.values() if 'step' in x))
def evaluate(actor,seeds,name,scale,offset):
 oldrng=acc.rng();oldcwd=Path.cwd();oldseeds=core.SEEDS
 out=HERE/'evaluations'/name;out.mkdir(parents=True,exist_ok=False);t=time.time()
 try:
  core.SEEDS=tuple(seeds);os.chdir(out)
  behavior,contract=core.evaluate_four([(name,[actor],1)],scale,offset,out)
 finally:
  core.SEEDS=oldseeds;os.chdir(oldcwd);acc.restore_rng(oldrng)
 result={'label':name,'device':'npu:0','seeds':list(seeds),'behavior':behavior[name],'contract':contract,'env_steps':sum(x['length'] for x in behavior[name]['episodes']),'wall_time_seconds':time.time()-t,'actor_updates_during_evaluation':0,'critic_updates':0,'command':'python -u '+str(HERE/'run_backtracking.py')+' experiment'}
 assert contract['test_valid'] and all(contract[k]==4 for k in ('parallel_envs_initialized','parallel_envs_used','parallel_envs_closed'))
 dump(out/'invocation.json',result)
 print('SIM_DONE',name,behavior[name]['success_count'],result['env_steps'],flush=True)
 return result
def fraction_name(f):return str(f).replace('.','p')
def main():
 assert not (HERE/'preregistration.json').exists()
 # Full reads of all requested historical artifacts, immutable source copies by reference/hash.
 evidence={}
 for name in ['FINAL_REPORT.md','final_summary.json','round1_result.json','round2_analysis.json','design.json']:
  p=OLD/name;data=p.read_text();obj=json.loads(data) if p.suffix=='.json' else data
  evidence[name]={'path':str(p),'sha256':sha(p),'bytes':len(p.read_bytes())}
 old=read(OLD/'final_summary.json');design=read(OLD/'design.json')
 assert old['classification']=='COMPETENCE_GATE_SUPPRESSES_Q_OPTIMIZATION'
 assert old['round1_analysis']['counts']['proposed_actor_updates']==10000
 assert old['round1_analysis']['rejection_statistics']['rejected']['both_q_improving_count']==38
 assert tuple(old['round1_analysis']['acceptance_seeds'])==ACCEPTANCE
 assert tuple(old['round1_analysis']['reporting_seeds'])==REPORTING
 dump(HERE/'prior_evidence_read.json',evidence)
 before=acc.qg.hashes();dump(HERE/'safety_before.json',before)
 fusion=Path.cwd()/'fusion_result.json'
 dump(HERE/'fusion_runtime_before.json',{'path':str(fusion),'exists':fusion.exists(),'sha256':sha(fusion) if fusion.exists() else None,'git_status':subprocess.check_output(['git','status','--short','--','fusion_result.json'],text=True).strip(),'policy':'runtime side effect recorded only; never edit/delete/restore/clean'})
 device,ready,ref,critic,scale,offset=core.setup()
 saved=torch.load(OLD/'fixed_batch_bank.pth',map_location='cpu',weights_only=False)
 bank=saved['bank'];probe=saved['probe'];contexts=saved['contexts'];initial=saved['initial_rng']
 assert acc.qg.arrays_hash(bank)==design['bank_hash']
 assert acc.qg.arrays_hash([probe])==design['drift_probe_hash']
 assert acc.qg.arrays_hash([contexts])==design['q_probe_hash']
 startstate=torch.load(OLD/'accepted_start.pth',map_location='cpu',weights_only=False)
 assert acc.torch_equal({k:v.cpu() for k,v in ref.state_dict().items()},startstate['actor'])
 assert acc.torch_equal(ready['actor_optimizer'],startstate['actor_optimizer'])
 assert acc.rnghash(initial)==design['initial_rng_hash']
 stream=np.load(OLD/'batch_index_stream.npy')
 assert np.array_equal(stream,np.arange(MAX_UPDATES)%64)
 np.save(HERE/'batch_index_stream.npy',stream)
 dump(HERE/'fixed_inputs_manifest.json',{'bank_file':str(OLD/'fixed_batch_bank.pth'),'bank_file_sha256':sha(OLD/'fixed_batch_bank.pth'),'bank_hash':design['bank_hash'],'drift_probe_hash':design['drift_probe_hash'],'q_probe_hash':design['q_probe_hash'],'source_stream_sha256':sha(OLD/'batch_index_stream.npy'),'copied_stream_sha256':sha(HERE/'batch_index_stream.npy'),'initial_rng_hash':design['initial_rng_hash'],'starting_actor_hash':core.module_hash(ref),'source_ready_step':140000,'source_ready_actor_updates':0})
 reg={'branch':'mean2q/multi_q ONLY','device':'npu:0','acceptance_seeds':list(ACCEPTANCE),'reporting_seeds':list(REPORTING),'parallel_envs':4,'block_actor_updates':BLOCK,'max_proposed_actor_updates':MAX_UPDATES,'fractions':list(FRACTIONS),'fraction_order':'full first; only competence failure triggers .5,.25,.125,.0625; stop at first competence-safe fraction; if its Q1 increment is not positive, reject whole proposal without searching for smaller safe fraction','acceptance_rule':'success>=3/4 AND sim_errors=0 AND fixed-probe block Q1 increment>0','objective':'unchanged production_actor_loss maximize probability-weighted component-mean Q1; original clipping/Adam; no gradient/loss anchor','parameter_interpolation':'named parameters theta_old+f*(theta_full-theta_old); nonparameter buffers restored from old; exact full uses theta_full','BACKTRACK_OPTIMIZER_STATE_SEMANTICS':'rollback_to_preproposal','full_accept_optimizer_semantics':'keep full candidate Adam state','partial_accept_rng_semantics':'rollback_to_preproposal','full_accept_rng_semantics':'keep candidate RNG','full_reject_semantics':'rollback parameters, optimizer, RNG and schedule count; verify equality','batch_stream_semantics':'source arange(10000)%64 saved before proposals; every decision advances global cursor250; no success-guided retry','schedule_semantics':'reuse exact prior core.schedule; nominal accepted-proposal update count advances250 for FULL_ACCEPT or PARTIAL_ACCEPT; FULL_REJECT leaves count unchanged. This is a schedule counter, NOT a claim of250 accepted full optimizer steps. Adam internal step stays preproposal for partial; record separately. Fraction-weighted equivalents recorded diagnostically only.','critic_target_replay':'readiness Critic frozen; target weights remain untouched in source checkpoint, no target network/TD calculation/update required in Actor-only test; replay bank fixed','q_probe':'same256history10contexts, scaling and reset semantics as prior experiment','matching_tolerance':.05,'reporting_rule':'once at first accepted gain>=95%target; nearest before/after accepted checkpoint by Q ONLY; flag unmatched if>5%; reporting never enters decisions','stop':'matched collapse target reporting completed OR40blocks/10000proposedupdates','twin_disagreement_diagnostic':'record any negative blockQ2; TWIN_DISAGREEMENT when blockQ2 < -abs(blockQ1); diagnostic only, never acceptance rule','source_manifest':str(HERE/'fixed_inputs_manifest.json'),'runtime_fusion_policy':'record runtime side effect only, never edit/clean fusion_result.json'}
 dump(HERE/'preregistration.json',reg)
 probeobs=torch.as_tensor(probe['observations'],device=device,dtype=torch.float32);base=core.actor_outputs(ref,probeobs,device)
 actor=copy.deepcopy(ref);opt=torch.optim.Adam(actor.parameters(),lr=0.)
 opt.load_state_dict(copy.deepcopy(ready['actor_optimizer']));assert not opt.state
 acc.restore_rng(initial);cfg=ready['config'];envs,lrs=core.schedule(cfg,MAX_UPDATES)
 refs=read(HERE.parent/'q_gain_matched_path_test/stage_a.json')['references'];targets=read(HERE.parent/'q_gain_matched_path_test/stage_a.json')['targets']
 critic_hash=core.module_hash(critic);count=0;equivalent=0.;fullupdates=0;proposed=0;records=[];milestones={};t0=time.time()
 current=metric(actor,ref,critic,contexts,device,scale,offset,probeobs,base,'accepted_start',count)
 current['checkpoint']=save(actor,opt,count,'accepted_start',acc.rng(),equivalent,fullupdates)
 current['acceptance']=old['seeds']['readiness_baseline'];current['reporting']=refs['START']['behavior']
 with (HERE/'proposal_logs.jsonl').open('x') as log,(HERE/'actor_updates.jsonl').open('x') as trace:
  for block in range(1,41):
   ba=copy.deepcopy(actor.state_dict());bo=copy.deepcopy(opt.state_dict());br=acc.rng();bc=count;before=current;bh=core.module_hash(actor);optstepsbefore=optimizer_steps(opt)
   ids=stream[proposed:proposed+BLOCK].tolist()
   for j,idx in enumerate(ids,1):
    u=count+j;actor.train()
    for pg in opt.param_groups:pg['lr']=lrs[u-1]
    opt.zero_grad(set_to_none=True)
    loss,_=core.production_actor_loss(actor,critic,bank[idx],scale,offset,cfg,device);loss.backward()
    assert bool(torch.isfinite(loss)) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in actor.parameters())
    torch.nn.utils.clip_grad_norm_(actor.parameters(),cfg['actor_max_grad_norm']);opt.step()
    trace.write(json.dumps({'block':block,'proposed_update':proposed+j,'schedule_update_index':u,'batch_index':idx,'lr':lrs[u-1],'loss':float(loss)},allow_nan=False)+'\n')
   proposed+=BLOCK;fa=copy.deepcopy(actor.state_dict());fo=copy.deepcopy(opt.state_dict());fr=acc.rng()
   full=metric(actor,ref,critic,contexts,device,scale,offset,probeobs,base,'block_'+str(block).zfill(2)+'_full',count+BLOCK)
   full['checkpoint']=save(actor,opt,count+BLOCK,'proposal_block_'+str(block).zfill(2)+'_full',fr,equivalent+250,fullupdates+250)
   if block==1:assert abs(full['q']['all']['q1_gain']-targets['LOW'])<1e-7
   full_eval=evaluate(actor,ACCEPTANCE,'BLOCK_'+str(block).zfill(2)+'_F1p0',scale,offset)
   def attempt(f,m,e):
    q=m['q']['all'];pq=before['q']['all']
    inc={k:q[k]-pq[k] for k in ('q1_gain','q2_gain','qmean_gain')}
    return {'fraction':f,'metrics':m,'evaluation':e,'block_q_increment':inc,'competence_safe':e['behavior']['success_count']>=3 and e['behavior']['sim_errors']==0,'q1_improving':inc['q1_gain']>0,'twin_q2_decreases':inc['q2_gain']<0,'twin_disagreement_flag':inc['q2_gain'] < -abs(inc['q1_gain'])}
   attempts=[attempt(1.,full,full_eval)];selected=None;reason=None
   if attempts[0]['competence_safe']:
    if attempts[0]['q1_improving']:selected=attempts[0]
    else:reason='FULL_COMPETENCE_SAFE_BUT_Q1_NOT_IMPROVING'
   else:
    for f in FRACTIONS[1:]:
     actor.load_state_dict(ba,strict=True)
     with torch.no_grad():
      for n,p in actor.named_parameters():p.copy_(ba[n]+f*(fa[n]-ba[n]))
     m=metric(actor,ref,critic,contexts,device,scale,offset,probeobs,base,'block_'+str(block).zfill(2)+'_f'+fraction_name(f),count)
     e=evaluate(actor,ACCEPTANCE,'BLOCK_'+str(block).zfill(2)+'_F'+fraction_name(f),scale,offset)
     at=attempt(f,m,e);attempts.append(at)
     if at['competence_safe']:
      if at['q1_improving']:selected=at
      else:reason='FIRST_COMPETENCE_SAFE_FRACTION_Q1_NOT_IMPROVING'
      break
   if selected is not None:
    f=selected['fraction'];count+=BLOCK;equivalent+=BLOCK*f
    if f==1.:
     decision='FULL_ACCEPT';fullupdates+=BLOCK;actor.load_state_dict(fa,strict=True);opt.load_state_dict(fo);acc.restore_rng(fr);optimverified=acc.torch_equal(opt.state_dict(),fo)
    else:
     decision='PARTIAL_ACCEPT';opt.load_state_dict(bo);acc.restore_rng(br);optimverified=acc.torch_equal(opt.state_dict(),bo)
     assert acc.rnghash(acc.rng())==acc.rnghash(br)
     with torch.no_grad():
      assert all(torch.equal(p,ba[n]+f*(fa[n]-ba[n])) for n,p in actor.named_parameters())
    assert optimverified
    current=selected['metrics'];current['schedule_accepted_proposal_updates']=count;current['acceptance']=selected['evaluation']
    current['checkpoint']=save(actor,opt,count,'accepted_block_'+str(block).zfill(2),acc.rng(),equivalent,fullupdates)
    rollbackverified=None
   else:
    f=None;decision='FULL_REJECT';actor.load_state_dict(ba,strict=True);opt.load_state_dict(bo);acc.restore_rng(br);count=bc;current=before
    rollbackverified=core.module_hash(actor)==bh and acc.torch_equal(opt.state_dict(),bo) and acc.rnghash(acc.rng())==acc.rnghash(br);assert rollbackverified
   row={'block':block,'proposed_updates':proposed,'preproposal_schedule_accepted_update_count':bc,'full_candidate':attempts[0],'backtracking_attempts':attempts[1:],'selected_fraction':f,'decision':decision,'rejection_reason':reason,'schedule_accepted_proposal_updates':count,'fraction_weighted_update_equivalent':equivalent,'full_accepted_optimizer_updates':fullupdates,'accepted_q':current['q']['all'],'accepted_metrics':current,'block_q_retained_fraction':selected['block_q_increment']['q1_gain']/attempts[0]['block_q_increment']['q1_gain'] if selected and attempts[0]['block_q_increment']['q1_gain']>0 else None,'selected_q_increment':selected['block_q_increment'] if selected else None,'optimizer_steps_before':optstepsbefore,'optimizer_steps_full_candidate':sorted(set(float(v['step'].detach().cpu()) if torch.is_tensor(v['step']) else float(v['step']) for v in fo['state'].values() if 'step' in v)),'optimizer_steps_after_decision':optimizer_steps(opt),'partial_optimizer_rollback_verified':optimverified if decision=='PARTIAL_ACCEPT' else None,'full_reject_rollback_verified':rollbackverified,'rng_hash_before':acc.rnghash(br),'rng_hash_candidate':acc.rnghash(fr),'rng_hash_after':acc.rnghash(acc.rng()),'batch_indices':ids,'batch_indices_sha256':hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()}
   log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();trace.flush();records.append(row)
   print('DECISION',json.dumps({'block':block,'proposed':proposed,'full_success':full_eval['behavior']['success_count'],'fraction':f,'decision':decision,'accepted_q1':current['q']['all']['q1_gain'],'selected_success':selected['evaluation']['behavior']['success_count'] if selected else None}),flush=True)
   if selected:
    for label,target in targets.items():
     if label not in milestones and current['q']['all']['q1_gain']>=.95*target:
      near=min([before,current],key=lambda x:abs(x['q']['all']['q1_gain']-target))
      eval_actor=copy.deepcopy(ref);eval_actor.load_state_dict(torch.load(near['checkpoint'],map_location='cpu',weights_only=False)['actor'])
      reporting=evaluate(eval_actor,REPORTING,'REPORTING_'+label,scale,offset)
      err=abs(near['q']['all']['q1_gain']-target)/target
      m={'target':label,'target_q1':target,'metrics':near,'reporting':reporting,'relative_matching_error':err,'matched_within5percent':err<=.05,'proposed_updates_at_crossing':proposed,'selected_fraction_profile_at_crossing':[r['selected_fraction'] for r in records],'crossing_before_checkpoint':before['checkpoint'],'crossing_after_checkpoint':current['checkpoint'],'selected_by':'nearest Q ONLY; reporting has no training/acceptance feedback'}
      milestones[label]=m;dump(HERE/('milestone_'+label.lower()+'.json'),m)
      print('REPORT_MILESTONE',label,err,reporting['behavior']['success_count'],flush=True)
   assert core.module_hash(critic)==critic_hash
   dump(HERE/('progress_block_'+str(block).zfill(2)+'.json'),{'proposed_updates':proposed,'current_accepted':current,'milestones':milestones,'decisions':[r['decision'] for r in records],'selected_fractions':[r['selected_fraction'] for r in records]})
   if milestones.get('COLLAPSE',{}).get('matched_within5percent'):break
 result={'records':records,'milestones':milestones,'current_accepted':current,'total_proposed_actor_updates':proposed,'total_proposal_blocks':len(records),'schedule_accepted_proposal_updates':count,'fraction_weighted_update_equivalent':equivalent,'full_accepted_optimizer_updates':fullupdates,'full_accept_count':sum(r['decision']=='FULL_ACCEPT' for r in records),'partial_accept_count':sum(r['decision']=='PARTIAL_ACCEPT' for r in records),'full_reject_count':sum(r['decision']=='FULL_REJECT' for r in records),'critic_updates':0,'target_critic_updates':0,'critic_hash_unchanged':core.module_hash(critic)==critic_hash,'runtime_seconds':time.time()-t0,'reporting_used_for_decision':False,'formal_training_stopped':True,'stop_reason':'matched collapse target' if milestones.get('COLLAPSE',{}).get('matched_within5percent') else '40blocks/10000proposal budget reached','targets':targets,'preregistration':reg}
 dump(HERE/'round1_result.json',result);print('ROUND1_DONE',proposed,json.dumps(result['current_accepted']['q']['all']),flush=True)
if __name__=='__main__':
 assert sys.argv[1]=='experiment'
 main()
