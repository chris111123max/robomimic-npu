"""Analyze actual independent acceptance results; no training or simulation."""
import sys,json,hashlib,subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/'q_gain_matched_path_test'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 assert not p.exists(),p
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def derive():
 r=read(HERE/'round1_result.json');selection=read(HERE/'acceptance_seeds.json')
 rows=r['records'];rejected=[x for x in rows if x['decision']=='REJECT'];accepted=[x for x in rows if x['decision']=='ACCEPT']
 def stat(items):
  return {'count':len(items),'mean_candidate_total_delta_q1':float(np.mean([x['candidate_metrics']['q']['all']['q1_gain'] for x in items])) if items else None,'mean_candidate_block_delta_q1':float(np.mean([x['candidate_q1_minus_previous_accepted'] for x in items])) if items else None,'mean_candidate_block_delta_q2':float(np.mean([x['candidate_q2_minus_previous_accepted'] for x in items])) if items else None,'q1_improving_count':sum(x['candidate_q1_minus_previous_accepted']>0 for x in items),'both_q_improving_count':sum(x['candidate_q1_minus_previous_accepted']>0 and x['candidate_q2_minus_previous_accepted']>0 for x in items)}
 stats={'rejected':stat(rejected),'accepted':stat(accepted),'rejection_fraction':len(rejected)/len(rows)}
 if rejected:
  stats['rejected']['q1_improving_fraction']=stats['rejected']['q1_improving_count']/len(rejected)
  stats['rejected']['both_q_improving_fraction']=stats['rejected']['both_q_improving_count']/len(rejected)
 targets=read(PREV/'stage_a.json')['targets'];q=r['current_accepted']['q']['all'];mil=r['milestones']
 collapse=mil.get('COLLAPSE')
 if collapse and collapse['matched_within5percent'] and collapse['reporting']['behavior']['success_count']>=3:
  classification='INDEPENDENT_CLOSED_LOOP_ACCEPTANCE_CAUSALLY_PREVENTS_COLLAPSE';decision='CONTINUE_CASE_A_HELDOUT_GENERALIZATION'
 elif any(m['metrics'].get('acceptance',{}).get('behavior',{}).get('success_count',0)>=3 and m['reporting']['behavior']['success_count']==0 for m in mil.values()):
  classification='ACCEPTANCE_SET_OVERFITTING_OR_POOR_GENERALIZATION';decision='CONTINUE_CASE_C_DISJOINT_HELDOUT'
 elif q['q1_gain']<.2*targets['COLLAPSE']:
  classification='COMPETENCE_GATE_SUPPRESSES_Q_OPTIMIZATION';decision='CONTINUE_CASE_B_LOGS_ONLY'
 elif collapse and collapse['matched_within5percent'] and collapse['reporting']['behavior']['success_count']==0:
  classification='CLOSED_LOOP_ACCEPTANCE_DOES_NOT_BREAK_COLLAPSE';decision='CONTINUE_CASE_C_DISJOINT_HELDOUT'
 elif mil.get('MID',{}).get('matched_within5percent') and q['q1_gain']>=targets['MID'] and mil['MID']['reporting']['behavior']['success_count']>=3 and mil['MID']['reporting']['behavior']['success_count']>read(PREV/'stage_a.json')['references']['MID']['behavior']['success_count']:
  classification='CLOSED_LOOP_ACCEPTANCE_STRONGLY_DELAYS_COLLAPSE_WITH_REAL_Q_PROGRESS';decision='CONTINUE_CASE_A_HELDOUT_GENERALIZATION'
 else:classification='MIXED_OR_INCONCLUSIVE';decision='STOP_NO_CLEAN_MATCHED_RESULT'
 analysis={'classification':classification,'decision':decision,'rejection_statistics':stats,'accepted_q':q,'accepted_q1_fraction_collapse_target':q['q1_gain']/targets['COLLAPSE'],'counts':{k:r[k] for k in ['proposed_actor_updates','accepted_actor_updates','rejected_actor_updates','proposal_blocks','accepted_blocks','rejected_blocks']},'acceptance_seeds':selection['seeds'],'reporting_seeds':selection['reporting_seeds'],'milestones':{k:{'actual_q1':v['metrics']['q']['all']['q1_gain'],'relative_error':v['relative_matching_error'],'matched':v['matched_within5percent'],'reporting_success':v['reporting']['behavior']['success_count']} for k,v in mil.items()},'rollback_all_verified':all(x['rollback_verified'] for x in rejected),'reporting_feedback_used':False}
 return analysis,r
def round1():
 a,r=derive();write(HERE/'round1_analysis.json',a)
 out=HERE/'round1';out.mkdir(exist_ok=False)
 (out/'ROUND_REPORT.md').write_text('# ROUND1\n\nPrevious conclusion: local output damping only delays collapse at matched Q gain.\n\nCurrent question: can independent competence acceptance retain substantial Q improvement AND held-out behavior?\n\nWhy next: isolate block acceptance as the only intervention; Critic/replay/objective fixed.\n\nExperiment type: existing-data baseline, offline Actor proposals, exactly-four-env simulator checks.\n\nMethod: '+json.dumps(read(HERE/'design.json'))+'\n\nResult:\n'+json.dumps(a,indent=2)+'\n\nCausal status: decision intervention is causal; success retention WITHOUT Q progress is not proof of learning-preserving collapse prevention.\n\nDoes this change formal-collapse explanation? '+('It supports real competence signals rejecting Q-improving damage, but not sufficient progress-preserving repair.' if a['classification']=='COMPETENCE_GATE_SUPPRESSES_Q_OPTIMIZATION' else 'See matched-target and held-out evidence; do not infer a unique upstream Critic defect.')+'\n\nDecision: '+a['decision']+'\n')
 print('ROUND1_ANALYSIS',json.dumps(a),flush=True)
def finite(x):
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 return not isinstance(x,float) or bool(np.isfinite(x))
def final():
 a=read(HERE/'round1_analysis.json');r=read(HERE/'round1_result.json');sel=read(HERE/'acceptance_seeds.json');design=read(HERE/'design.json');control=read(PREV/'stage_a.json')['references']
 round2=read(HERE/'round2_analysis.json') if (HERE/'round2_analysis.json').exists() else None
 cls=a['classification']
 invocations=[read(p) for p in HERE.glob('evaluations/*/invocation.json')]
 if (HERE/'round2_result.json').exists():invocations.extend(read(HERE/'round2_result.json')['invocations'])
 assert all(i['contract']['parallel_envs_initialized']==i['contract']['parallel_envs_used']==i['contract']['parallel_envs_closed']==4 and i['contract']['test_valid'] for i in invocations)
 before=read(HERE/'safety_before.json');after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before};assert before==after
 alljson={str(p.relative_to(HERE)):read(p) for p in HERE.rglob('*.json')};assert all(finite(x) for x in alljson.values())
 jsonl={}
 for p in HERE.rglob('*.jsonl'):
  n=0
  with p.open() as f:
   for l in f:assert finite(json.loads(l));n+=1
  jsonl[str(p.relative_to(HERE))]=n
 process=subprocess.check_output(['ps','-eo','pid,args'],text=True)
 residual=[s for s in process.splitlines() if ('run_acceptance.py' in s or 'train_stage3' in s or 'multiprocessing.spawn' in s or 'run_secondary.py' in s) and 'bash -' not in s and 'ps -eo' not in s]
 assert not residual,residual
 npu=subprocess.check_output(['npu-smi','info'],text=True);(HERE/'npu_smi_final.txt').write_text(npu);assert 'No running processes found' in npu
 costs={'simulator_invocations':len(invocations),'simulator_episodes':4*len(invocations),'simulator_env_steps':sum(i['env_steps'] for i in invocations),'summed_simulator_wall_seconds':sum(i['wall_time_seconds'] for i in invocations),'round1_wall_seconds':r['runtime_seconds'],'proposed_actor_updates':r['proposed_actor_updates'],'accepted_actor_updates':r['accepted_actor_updates'],'rejected_actor_updates':r['rejected_actor_updates'],'new_critic_updates':0,'formal_env_and_optimizer_steps':0}
 safety={'production_and_formal_hashes_unchanged':True,'hash_count':len(after),'after_hashes':after,'device':'npu:0 ONLY','every_invocation_real_parallel_four':True,'all_workers_closed':True,'worker_exitcodes_all_zero':all(all(v==0 for v in i['contract']['worker_exitcodes']) for i in invocations),'residual_processes':residual,'no_residual_npu_process':True,'jsonl_fully_read':jsonl,'all_numeric_outputs_finite':True,'fusion_result_explicitly_edited':False,'fusion_result_repository_state_unchanged':False,'fusion_result_readonly_audit':read(HERE/'fusion_result_readonly_audit.json'),'repository_state_safety_exception':True,'formal_training_remains_stopped':True,'production_edits':[]}
 write(HERE/'safety_final.json',safety)
 if cls=='COMPETENCE_GATE_SUPPRESSES_Q_OPTIMIZATION':
  answer='Independent real closed-loop acceptance did NOT demonstrate collapse prevention with substantial Q progress: it rejects Q-improving proposals, but mainly suppresses optimization.'
  nextstep='No production change. If further validation is wanted, test one independently specified competence criterion that can preserve meaningful accepted Q progress; do not retune this completed test.'
 elif cls=='INDEPENDENT_CLOSED_LOOP_ACCEPTANCE_CAUSALLY_PREVENTS_COLLAPSE':
  answer='At collapse-matched fixed-Q improvement, independent competence acceptance preserves held-out success and supports a causal benefit in this frozen test.'
  nextstep='Validate the unchanged acceptance intervention in a narrowly scoped testing-only online continuation; not executed.'
 elif cls=='ACCEPTANCE_SET_OVERFITTING_OR_POOR_GENERALIZATION':
  answer='Acceptance competence does not generalize cleanly to held-out reporting competence; this fixed acceptance gate did not provide a transferable collapse-prevention guarantee.'
  nextstep='Do not deploy the gate; independently specify a broader held-out competence validation protocol before any further training test.'
 else:
  answer='Independent acceptance produced no clean matched-Q, held-out competence-preservation result; the requested learning-preserving causal claim remains unproven.'
  nextstep='Resolve only the documented matching/generalization ambiguity before any production change; not executed.'
 summary={'classification':cls,'answer':answer,'rounds_used':1+int(round2 is not None),'round1_analysis':a,'round1_result':r,'round2_analysis':round2,'seeds':sel,'design':design,'costs':costs,'invocations':invocations,'safety':safety,'next_step_only':nextstep,'formal_training_remains_stopped':True}
 write(HERE/'final_summary.json',summary)
 lines=['# Independent closed-loop acceptance test','',answer,'','Final classification: **'+cls+'**. Rounds used: '+str(summary['rounds_used'])+'.','','## Seeds and separation','',json.dumps({'acceptance':sel['seeds'],'reporting':sel['reporting_seeds'],'selection':sel['selected_group'],'readiness_acceptance_success':4}),'','Only the first readiness-only4/4 group was locked. No candidate tested during selection. Reporting never affects acceptance, rollback or batch selection.','','## Proposal blocks','','|Block|Proposed updates|Candidate deltaQ1|Acceptance success|Decision|Accepted cumulative deltaQ1|','|---|---:|---:|---:|---|---:|']
 for x in r['records']:
  lines.append('|'+ '|'.join([str(x['block']),str(x['proposed_updates']),f"{x['candidate_metrics']['q']['all']['q1_gain']:.10g}",str(x['acceptance']['behavior']['success_count'])+'/4',x['decision'],f"{x['accepted_cumulative_q']['q1_gain']:.10g}"])+'|')
 lines+=['','## Q milestones','','|Q target|Branch|Actual deltaQ1|Accepted updates|Reporting success|Weighted drift|Mean RMS|','|---|---|---:|---:|---:|---:|---:|']
 for label,ref in control.items():
  beh=ref.get('behavior') or read(PREV/'stage_c.json')['behavior']['UNRESTRICTED_LOW']
  d=ref['drift'];lines.append('|'+ '|'.join([label,'UNRESTRICTED',f"{ref['q']['all']['q1_gain']:.10g}",str(ref['actor_updates']),str(beh['success_count'])+'/4',f"{d['weighted_action_l2_mean']:.10g}",f"{d['component_mean_rms']:.10g}"])+'|')
  if label=='START':
   lines.append('|START|CLOSED_LOOP_ACCEPTED|0|0|4/4|0|0|')
  elif label in r['milestones']:
   m=r['milestones'][label];v=m['metrics'];d=v['drift']
   lines.append('|'+ '|'.join([label+(' (UNMATCHED)' if not m['matched_within5percent'] else ''),'CLOSED_LOOP_ACCEPTED',f"{v['q']['all']['q1_gain']:.10g}",str(v['actor_updates']),str(m['reporting']['behavior']['success_count'])+'/4',f"{d['weighted_action_l2_mean']:.10g}",f"{d['component_mean_rms']:.10g}"])+'|')
  else:lines.append('|'+label+'|CLOSED_LOOP_ACCEPTED|NOT REACHED|--|NOT EVALUATED|--|--|')
 lines+=['','Crossing checkpoints selected by nearest Q ONLY; mismatches outside5% are explicitly NOT matched evidence. Reporting evaluations occur only once per first crossing, never per candidate.','','## Accepted endpoint and learning suppression','',json.dumps({'q':r['current_accepted']['q']['all'],'drift':r['current_accepted']['drift'],'counts':a['counts'],'q1_fraction_collapse_target':a['accepted_q1_fraction_collapse_target']},indent=2),'','## Rejected candidates','',json.dumps(a['rejection_statistics'],indent=2),'','Block Q increments are candidate minus PREVIOUS accepted Actor, not total gain from START. Rejected gains are never counted as accepted progress. Q-improving rejected proposals demonstrate twin-Q agreement is not a competence certificate; frozen behavior preservation alone cannot prove a progress-preserving repair.','','## Round reports','',a['decision']]
 if round2:lines+=['',json.dumps(round2,indent=2)]
 lines+=['','## Execution semantics and costs','',json.dumps(design,indent=2),'',json.dumps(costs,indent=2),'','Existing control/results read and reused with exact batch/probe hashes. New Actor proposals optimize unchanged production Q1 objective; no BC/mean/parameter anchor and no gradient intervention. Critic/target/replay frozen; optimizer and update/RNG state fully rollback on reject, while predetermined proposal batch cursor advances. All checkpoints are synchronous testing-only accepted states, no learner catch-up ambiguity.','', 'Each simulator invocation:']
 for i in invocations:lines+=['',json.dumps({k:i[k] for k in ['label','device','seeds','env_steps','wall_time_seconds','contract','command']})]
 lines+=['','## Causal limits','',answer,'','No claim about a unique upstream bootstrap, Bellman, architecture, RNN or optimizer defect. No formal/online training reproduction. This is a finite fixed-seed acceptance experiment; repeated acceptance checks can overfit the acceptance set, hence reporting separation.','','## ONE NEXT STEP (not executed)','',nextstep,'','## Safety','',json.dumps(safety,indent=2),'','FORMAL TRAINING REMAINS STOPPED']
 (HERE/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
 print('FINAL_REPORT_DONE',cls,json.dumps(costs),flush=True)
if __name__=='__main__':{'round1':round1,'final':final}[sys.argv[1]]()
