"""Completed-result analysis only; never simulate/train or modify production."""
import sys,json,hashlib,subprocess,math
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'closed_loop_acceptance_test'
CONTROL=HERE.parent/'q_gain_matched_path_test'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 assert not p.exists(),p
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def mean(v):return float(np.mean(v)) if v else None
def median(v):return float(np.median(v)) if v else None
def finite(x):
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 return not isinstance(x,float) or math.isfinite(x)
def derive():
 r=read(HERE/'round1_result.json');rows=r['records'];mil=r['milestones'];q=r['current_accepted']['q']['all']
 logs=[json.loads(l) for l in (HERE/'proposal_logs.jsonl').read_text().splitlines()]
 trace=[json.loads(l) for l in (HERE/'actor_updates.jsonl').read_text().splitlines()]
 assert logs==rows and len(trace)==r['total_proposed_actor_updates']==250*len(rows)
 stream=np.load(HERE/'batch_index_stream.npy');previous=read(HERE/'accepted_start_metrics.json')['q']['all'];count=0
 for i,row in enumerate(rows):
  assert row['block']==i+1 and row['proposed_updates']==250*(i+1)
  assert row['batch_indices']==stream[i*250:(i+1)*250].tolist()
  assert row['batch_indices_sha256']==hashlib.sha256(np.asarray(row['batch_indices'],dtype=np.int64).tobytes()).hexdigest()
  assert row['preproposal_schedule_accepted_update_count']==count
  for j,t in enumerate(trace[i*250:(i+1)*250]):
   assert t['block']==i+1 and t['proposed_update']==i*250+j+1 and t['schedule_update_index']==count+j+1 and t['batch_index']==row['batch_indices'][j] and math.isfinite(t['loss'])
  attempts=[row['full_candidate']]+row['backtracking_attempts']
  assert [a['fraction'] for a in attempts]==[1.,.5,.25,.125,.0625][:len(attempts)]
  if row['full_candidate']['competence_safe']:assert len(attempts)==1
  for at in attempts:
   assert abs(at['block_q_increment']['q1_gain']-(at['metrics']['q']['all']['q1_gain']-previous['q1_gain']))<1e-12
   assert at['evaluation']['seeds']==[20003,20004,20010,20011]
  if row['decision']=='FULL_REJECT':
   assert row['full_reject_rollback_verified'] and row['accepted_q']==previous and row['rng_hash_before']==row['rng_hash_after']
  else:
   count+=250
   selected=attempts[-1]
   assert selected['fraction']==row['selected_fraction'] and selected['competence_safe'] and selected['q1_improving']
   assert all(not a['competence_safe'] for a in attempts[:-1])
   if row['decision']=='PARTIAL_ACCEPT':
    assert row['partial_optimizer_rollback_verified'] and row['optimizer_steps_before']==row['optimizer_steps_after_decision'] and row['rng_hash_before']==row['rng_hash_after']
  assert row['schedule_accepted_proposal_updates']==count
  previous=row['accepted_q']
 failed=[x for x in rows if not x['full_candidate']['competence_safe']]
 rescued=[x for x in failed if x['decision']=='PARTIAL_ACCEPT']
 partial=[x for x in rows if x['decision']=='PARTIAL_ACCEPT']
 retained=[x['block_q_retained_fraction'] for x in rescued if x['block_q_retained_fraction'] is not None]
 hist={str(f):sum(x['selected_fraction']==f for x in rows) for f in [1.,.5,.25,.125,.0625]}
 hist['none']=sum(x['selected_fraction'] is None for x in rows)
 positive_full=sum(max(0.,x['full_candidate']['block_q_increment']['q1_gain']) for x in rows)
 total_full=sum(max(0.,x['full_candidate']['metrics']['q']['all']['q1_gain']) for x in rows)
 collapse=mil.get('COLLAPSE');fraction=q['q1_gain']/r['targets']['COLLAPSE']
 rescue_fraction=len(rescued)/len(failed) if failed else None
 secondary='Q_ASCENT_DIRECTIONS_OFTEN_HAVE_LOCAL_SAFE_REGIONS' if rescue_fraction is not None and rescue_fraction>=.5 else 'Q_ASCENT_DIRECTIONS_OFTEN_LOCALLY_INCOMPATIBLE_WITH_COMPETENCE' if failed and r['full_reject_count']/len(failed)>.5 else None
 if collapse and collapse['matched_within5percent'] and collapse['reporting']['behavior']['success_count']>=3:
  classification='COMPETENCE_AWARE_BACKTRACKING_BREAKS_Q_GAIN_COLLAPSE_LINK';decision='CONTINUE_CASE_C_DISJOINT_HELDOUT'
 elif collapse and collapse['matched_within5percent'] and collapse['reporting']['behavior']['success_count']==0:
  classification='BACKTRACKING_ONLY_DELAYS_COLLAPSE';decision='CONTINUE_LOGS_GENERALIZATION_LIMIT'
 elif fraction<.2:
  classification='BACKTRACKING_SUPPRESSES_Q_OPTIMIZATION';decision='CONTINUE_CASE_B_LOGS_ONLY' if r['full_reject_count']>=len(failed)/2 and failed else 'CONTINUE_CASE_A_LOGS_ONLY'
 elif secondary:
  classification=secondary;decision='CONTINUE_CASE_A_LOGS_ONLY' if secondary.endswith('SAFE_REGIONS') else 'CONTINUE_CASE_B_LOGS_ONLY'
 else:classification='MIXED_OR_INCONCLUSIVE';decision='STOP'
 attempts=[at for row in rows for at in [row['full_candidate']]+row['backtracking_attempts']]
 a={'classification':classification,'secondary_mechanistic_label':secondary,'next_round_decision':decision,'counts':{k:r[k] for k in ['total_proposed_actor_updates','total_proposal_blocks','full_accept_count','partial_accept_count','full_reject_count','schedule_accepted_proposal_updates','fraction_weighted_update_equivalent','full_accepted_optimizer_updates']},'effective_accepted_parameter_proposals':r['full_accept_count']+r['partial_accept_count'],'full_competence_failed_count':len(failed),'full_q_improving_competence_failed_count':sum(x['full_candidate']['q1_improving'] for x in failed),'backtracking_rescued_count':len(rescued),'rescue_fraction_of_failed_full':rescue_fraction,'selected_fraction_histogram':hist,'partial_q_retained_fraction':{'mean':mean(retained),'median':median(retained),'min':min(retained) if retained else None,'max':max(retained) if retained else None},'accepted_q':q,'q1_fraction_collapse_target':fraction,'accepted_q_efficiency':q['q1_gain']/positive_full if positive_full>0 else None,'efficiency_denominator_positive_full_block_delta_q1':positive_full,'alternative_efficiency_denominator_global_full_candidate_gain':total_full,'alternative_global_gain_efficiency':q['q1_gain']/total_full if total_full else None,'milestones':{k:{'matched':v['matched_within5percent'],'relative_error':v['relative_matching_error'],'actual_q1':v['metrics']['q']['all']['q1_gain'],'reporting_success':v['reporting']['behavior']['success_count'],'proposed_updates_at_crossing':v['proposed_updates_at_crossing']} for k,v in mil.items()},'attempt_count':len(attempts),'twin_disagreement_count':sum(x['twin_disagreement_flag'] for x in attempts),'sim_error_count':sum(x['evaluation']['behavior']['sim_errors'] for x in attempts),'audit':{'proposal_rows_fully_read':len(logs),'actor_update_rows_fully_read':len(trace),'fixed_stream_verified':True,'exact_fraction_order_verified':True,'largest_competence_safe_fraction_selected':True,'q1_positive_acceptance_verified':True,'partial_optimizer_and_rng_rollback_verified':True,'full_reject_rollback_verified':True,'reporting_used_for_decision':False}}
 return a,r
def round1():
 a,r=derive();write(HERE/'round1_analysis.json',a);p=HERE/'round1';p.mkdir(exist_ok=False)
 (p/'ROUND_REPORT.md').write_text('# ROUND1\n\nPrevious conclusion: whole-block competence gate suppresses Q learning,38/38rejected proposals improve both twins.\n\nQuestion: can same-direction backtracking find useful competence-safe steps and accumulate matched Q?\n\nWhy next most informative: distinguish full-step unsafe from direction locally incompatible, without changing Critic/replay/objective.\n\nExperiment type: offline Actor proposals plus4-env simulator.\n\nSimulator used? yes.\n\n4-env contract: each completed invocation initialized/used/closed4real parallel workers; audited in final.\n\nResult:\n'+json.dumps(a,indent=2)+'\n\nCausal interpretation: within-block comparisons hold proposal direction fixed. Long-run intervention also includes explicitly preregistered optimizer rollback and schedule semantics, so not a pure step-size-only guarantee. Finite fraction grid and4fixed acceptance seeds cannot establish universal directional incompatibility.\n\nDoes this explain formal collapse? only supports the measured Q/competence and step-acceptance mechanism; no new formal training reproduction or unique upstream Critic defect.\n\nDecision: '+a['next_round_decision']+'\n')
 print(json.dumps(a,indent=2),flush=True)

def final():
 a=read(HERE/'round1_analysis.json');r=read(HERE/'round1_result.json')
 round2=read(HERE/'round2_analysis.json') if (HERE/'round2_analysis.json').exists() else None
 classification=a['classification'];secondary=a['secondary_mechanistic_label']
 if round2 and 'updated_classification' in round2:classification=round2['updated_classification']
 reg=read(HERE/'preregistration.json');rows=r['records']
 invocations=[read(p) for p in sorted(HERE.glob('evaluations/*/invocation.json'))]
 incomplete=[{'label':d.name,'status':'INVALID_INCOMPLETE_EXCLUDED','cause':'container replacement' if d.name=='BLOCK_19_F1p0' else 'libGL missing before environment startup','env_steps':'UNKNOWN_NO_COMPLETE_RESULT','graceful_four_worker_close_verified':False} for d in sorted((HERE/'evaluations').iterdir()) if not (d/'invocation.json').exists()]
 assert {x['label'] for x in incomplete}=={'BLOCK_19_F1p0','BLOCK_19_F1p0_RECOVERED'}
 assert 'RECOVERY_EXACT_ACTOR_ADAM_RNG_VERIFIED' in (HERE/'recovery.log').read_text()
 assert 'RECOVERY_REUSE_EXACT_VERIFIED_BLOCK19_NO_REUPDATE' in (HERE/'recovery2.log').read_text()
 assert all(v['device']=='npu:0' and v['contract']['test_valid'] and all(v['contract'][k]==4 for k in ['parallel_envs_initialized','parallel_envs_used','parallel_envs_closed']) for v in invocations)
 before=read(HERE/'safety_before.json')
 after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before};assert before==after
 historical=read(HERE/'prior_evidence_read.json')
 assert all(hashlib.sha256(Path(v['path']).read_bytes()).hexdigest()==v['sha256'] for v in historical.values())
 manifest=read(HERE/'fixed_inputs_manifest.json')
 assert hashlib.sha256(Path(manifest['bank_file']).read_bytes()).hexdigest()==manifest['bank_file_sha256']
 all_json={}
 for p in HERE.rglob('*.json'):
  data=read(p);assert finite(data),p;all_json[str(p.relative_to(HERE))]=len(p.read_bytes())
 jsonl={}
 for p in HERE.rglob('*.jsonl'):
  n=0
  with p.open() as f:
   for line in f:assert finite(json.loads(line));n+=1
  jsonl[str(p.relative_to(HERE))]=n
 assert sum(n for k,n in jsonl.items() if 'trajectories' in k)==sum(v['env_steps'] for v in invocations)
 processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
 residual=[s for s in processes.splitlines() if any(v in s for v in ['run_backtracking.py','resume_backtracking','train_stage3','multiprocessing.spawn','run_secondary.py']) and 'ps -eo' not in s and 'bash -' not in s]
 assert not residual,residual
 npu=subprocess.check_output(['npu-smi','info'],text=True);assert 'No running processes found' in npu
 (HERE/'npu_smi_final.txt').write_text(npu)
 fusion=HERE.parents[5]/'fusion_result.json'
 # authoritative repository root is five parents above this testing directory.
 repo=Path('/data/home/3220251075/lerobot_workspace/robomimic')
 fusion=repo/'fusion_result.json'
 fr={'exists':fusion.exists(),'sha256':hashlib.sha256(fusion.read_bytes()).hexdigest() if fusion.exists() else None,'git_status':subprocess.check_output(['git','status','--short','--','fusion_result.json'],cwd=repo,text=True).strip(),'policy':'NPU runtime side effect recorded only; no explicit edit/delete/restore/clean'}
 write(HERE/'fusion_runtime_after.json',fr)
 safety={'protected_source_and_formal_hashes_unchanged':True,'protected_hash_count':len(after),'after_hashes':after,'historical_requested_outputs_unchanged':True,'frozen_bank_file_unchanged':True,'critic_hash_unchanged':r['critic_hash_unchanged'],'new_critic_updates':0,'target_critic_updates':0,'only_npu0_used':True,'every_completed_valid_simulator_invocation_real_parallel4':True,'incomplete_attempts_excluded':incomplete,'all_completed_valid_eval_workers_closed':True,'incomplete_historical_worker_closure_not_verified':True,'all_worker_exitcodes_zero':all(all(x==0 for x in v['contract']['worker_exitcodes']) for v in invocations),'residual_processes':residual,'no_residual_npu_process':True,'all_json_and_jsonl_fully_read':True,'jsonl_read_counts':jsonl,'numeric_outputs_finite':True,'production_edits':[],'formal_checkpoint_writes':[],'formal_training_resumed':False,'fusion_runtime_before':read(HERE/'fusion_runtime_before.json'),'fusion_runtime_after':fr,'testing_bug_fixes':[],'testing_recovery_changes':['resume_backtracking.py: exact replay checkpoint validation and append-only continuation','resume_backtracking2.py: reuse exactly verified pending proposal, no repeated update','analyze_recovery.py: distinguish incomplete attempts, physical execution and unique proposal budget'],'recovered_actor_adam_rng_exact_verified':True,'environment_recovery':'existing setup_robot_env.sh restored missing system libraries; production code unchanged'}
 write(HERE/'safety_final.json',safety)
 q=a['accepted_q'];mil=r['milestones'];endpoint=r['current_accepted']
 endpoint_reporting=[(k,v) for k,v in mil.items() if v['metrics']['checkpoint']==endpoint['checkpoint']]
 costs={'proposal_blocks':len(rows),'proposed_actor_optimizer_steps':r['total_proposed_actor_updates'],'full_accept_count':r['full_accept_count'],'partial_accept_count':r['partial_accept_count'],'full_reject_count':r['full_reject_count'],'effective_accepted_parameter_proposals':a['effective_accepted_parameter_proposals'],'nominal_schedule_accepted_proposal_updates':r['schedule_accepted_proposal_updates'],'full_accepted_optimizer_updates':r['full_accepted_optimizer_updates'],'fraction_weighted_update_equivalent_diagnostic_only':r['fraction_weighted_update_equivalent'],'simulator_invocations':len(invocations),'simulator_episodes':len(invocations)*4,'simulator_env_steps':sum(v['env_steps'] for v in invocations),'summed_simulator_wall_seconds':sum(v['wall_time_seconds'] for v in invocations),'round1_wall_seconds':r['runtime_seconds'],'preparation_through_round1_elapsed_wall_seconds':(HERE/'round1_result.json').stat().st_mtime-(HERE/'safety_before.json').stat().st_mtime,'new_critic_updates':0,'formal_env_and_optimizer_steps':0}
 costs.update({'unique_proposed_actor_updates':r['total_proposed_actor_updates'],'additional_recovery_replayed_actor_updates':250,'total_actor_optimizer_executions_including_exact_replay':r['total_proposed_actor_updates']+250,'invalid_incomplete_simulator_attempts':incomplete,'simulator_invocations_count_is_completed_valid_only':True,'calendar_elapsed_includes_container_downtime':True,'round1_wall_seconds_definition':'completed pre-interruption interval estimated from saved file mtimes plus successful continuation measured by clock; excludes downtime and failed recovery','interrupted_attempt_env_steps':'UNKNOWN_NOT_IN_VALID_TOTAL'})
 rescued=a['backtracking_rescued_count'];failed=a['full_competence_failed_count'];freq=a['rescue_fraction_of_failed_full']
 answer=f"Not usually within the tested fraction grid: same-direction backtracking found a Q-improving competence-safe fraction for only {rescued}/{failed} competence-failed full proposals (rescue fraction {freq if freq is not None else 'N/A'})."
 if classification=='COMPETENCE_AWARE_BACKTRACKING_BREAKS_Q_GAIN_COLLAPSE_LINK':
  longanswer='It also reached collapse-matched Q gain with held-out success>=3/4, supporting a causal benefit of the preregistered competence-aware acceptance/step control.'
 elif classification=='BACKTRACKING_ONLY_DELAYS_COLLAPSE':
  longanswer='At collapse-matched Q gain, held-out success is0/4: the intervention only delays collapse and acceptance-set competence does not guarantee generalization.'
 elif classification=='BACKTRACKING_SUPPRESSES_Q_OPTIMIZATION':
  longanswer='It did NOT preserve substantial Q learning: final accepted gain is below20%of collapse target. No matched-Q collapse-prevention claim is supported.'
 else:longanswer=f"Accepted Q1 progress reached {100*a['q1_fraction_collapse_target']:.2f}%of the collapse target (above the preregistered20%suppression cutoff), but stalled after block{max(x['block'] for x in rows if x['decision']!='FULL_REJECT')}. Collapse-matched gain was not reached; sustained substantial accumulation without collapse and collapse prevention were NOT established. Local early usability is not long-run competence preservation."
 interpretation={'PRIMARY_MECHANISM':'Approximate Twin-Q ascent is not an independent certificate of closed-loop competence. Within-proposal tests distinguish tested local safe fractions from harmful full steps.','EXECUTION_CHANNEL':'Parameter-space movement in the same250-update production Q1 proposal direction; frozen Critic/replay/history/scaling; common evaluation RNG; no auxiliary loss or gradient change.','WHAT_BACKTRACKING_CHANGED':'Selected the largest tested competence-safe fraction with positive blockQ1. Partial commits restore preproposal Adam/RNG state; nominal warmup counter advances as preregistered. This combined testing intervention is not step-size alone.','WHETHER_Q_LEARNING_WAS_PRESERVED':{'accepted_q':q,'collapse_target_fraction':a['q1_fraction_collapse_target'],'efficiency':a['accepted_q_efficiency'],'efficiency_denominator':'sum positive FULL BLOCK Q1 increments versus previous accepted Actor, not repeatedly counted total gain from START'},'WHETHER_COLLAPSE_WAS_PREVENTED_OR_ONLY_DELAYED':longanswer,'causal_limits':'4fixed acceptance seeds, finite fraction grid>=0.0625and finite40block budget. Failure at tested fractions does not prove every infinitesimal movement is harmful. Subsequent proposal directions differ from the prior whole-block-gate branch after a partial commit; rescue counts refer to this experiment, not paired re-tests of the old38directions. No conclusion about a unique Critic architecture, bootstrap or Bellman defect.'}
 if classification=='COMPETENCE_AWARE_BACKTRACKING_BREAKS_Q_GAIN_COLLAPSE_LINK':
  nextstep='If independently confirmed by disjoint held-out evaluation, validate this unchanged acceptance intervention in one narrowly scoped testing-only online continuation; not run here.'
 else:nextstep='Do not deploy or resume formal training. The sole suggested next test is a separately preregistered fixed-direction test below fraction0.0625at the plateau Actor to resolve the finite-grid boundary; do not run it or tune this completed experiment here.'
 summary={'answer':answer+' '+longanswer,'classification':classification,'secondary_mechanistic_label':secondary,'rounds_used':1+int(round2 is not None),'round1_analysis':a,'round1_result':r,'round2_analysis':round2,'costs':costs,'interpretation':interpretation,'endpoint_acceptance_success':endpoint['acceptance']['behavior']['success_count'],'endpoint_heldout_reporting_evaluated':bool(endpoint_reporting),'endpoint_heldout_reporting':{k:v['reporting']['behavior'] for k,v in endpoint_reporting},'preregistration':reg,'invocations':invocations,'safety':safety,'next_step_only':nextstep,'formal_training_remains_stopped':True}
 write(HERE/'final_summary.json',summary)
 lines=['# Competence-aware backtracking','',''+answer+' '+longanswer,'','Primary label: '+classification+'. Secondary mechanistic label: '+str(secondary)+'. Rounds used: '+str(summary['rounds_used'])+'.','','## Seeds','','Acceptance=[20003,20004,20010,20011]; held-out reporting=[20008,20002,20005,20007]. Reporting never selects fraction, acceptance, rollback or threshold. Readiness4/4acceptance and4/4reporting reused with unchanged source semantics.','','## Proposal blocks','','Full/partial deltaQ1 columns below are BLOCK increments versus PREVIOUS accepted Actor, not global gain. Raw total-from-START gains are retained for every attempt in JSON.','','|Block|Full candidate block deltaQ1|Full success|Selected fraction|Selected block deltaQ1|Selected success|Decision|','|---|---:|---:|---:|---:|---:|---|']
 for row in rows:
  full=row['full_candidate'];selected=([full]+row['backtracking_attempts'])[-1] if row['selected_fraction'] is not None else None
  lines.append('|'+ '|'.join([str(row['block']),f"{full['block_q_increment']['q1_gain']:.10g}",str(full['evaluation']['behavior']['success_count'])+'/4',str(row['selected_fraction']),f"{selected['block_q_increment']['q1_gain']:.10g}" if selected else '--',str(selected['evaluation']['behavior']['success_count'])+'/4' if selected else '--',row['decision']])+'|')
 control=read(CONTROL/'stage_a.json')['references']
 lines+=['','## Q milestones','','|Q target|Unrestricted success|Backtracked success|Backtracked deltaQ1|Proposed updates at crossing|Selected-fraction profile|','|---|---:|---:|---:|---:|---|']
 for label in ['START','LOW','MID','COLLAPSE']:
  ref=control[label];beh=ref.get('behavior') or read(CONTROL/'stage_c.json')['behavior']['UNRESTRICTED_LOW']
  if label=='START':vs=['4/4','0','0','none']
  elif label in mil:
   m=mil[label];vs=[str(m['reporting']['behavior']['success_count'])+'/4'+(' UNMATCHED' if not m['matched_within5percent'] else ''),f"{m['metrics']['q']['all']['q1_gain']:.10g}",str(m['proposed_updates_at_crossing']),str(dict(Counter(str(f) for f in m['selected_fraction_profile_at_crossing'])))]
  else:vs=['NOT EVALUATED','NOT REACHED','--','--']
  lines.append('|'+ '|'.join([label,str(beh['success_count'])+'/4']+vs)+'|')
 lines+=['','Match tolerance5%; checkpoint chosen by nearest Q before/after crossing only. Overshoots are flagged UNMATCHED and never repaired by retraining. Current endpoint acceptance success='+str(summary['endpoint_acceptance_success'])+'/4; endpoint held-out evaluated='+str(summary['endpoint_heldout_reporting_evaluated'])+'. Never substitute earlier milestone held-out success for endpoint.','','## Safe fractions, Q retention and progress','',json.dumps(a,indent=2),'','## Endpoint drift','',json.dumps(endpoint['drift'],indent=2),'','## Preregistered semantics','',json.dumps(reg,indent=2),'','## Mechanism and limits','',json.dumps(interpretation,indent=2),'','## Round2','',json.dumps(round2,indent=2),'','## Actual execution cost','',json.dumps(costs,indent=2),'','Every COMPLETED VALID simulator invocation (two incomplete attempts separately listed in safety):']
 for v in invocations:lines+=['',json.dumps({k:v[k] for k in ['label','device','seeds','env_steps','wall_time_seconds','contract','command']})]
 lines+=['','## ONE NEXT STEP (not executed)','',nextstep,'','## Safety','',json.dumps(safety,indent=2),'','FORMAL TRAINING REMAINS STOPPED']
 (HERE/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
 print('FINAL_DONE',classification,json.dumps(costs),flush=True)
if __name__=='__main__':{'round1':round1,'final':final}[sys.argv[1]]()
