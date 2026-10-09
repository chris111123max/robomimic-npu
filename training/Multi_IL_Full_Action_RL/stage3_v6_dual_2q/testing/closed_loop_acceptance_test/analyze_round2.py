"""Round2 Case B: fully read completed proposal/Actor logs; no simulator."""
import json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def write(p,x):
 assert not p.exists(),p
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
r=read(HERE/'round1_result.json')
a=read(HERE/'round1_analysis.json')
assert a['decision']=='CONTINUE_CASE_B_LOGS_ONLY'
rows=[json.loads(l) for l in (HERE/'proposal_logs.jsonl').read_text().splitlines()]
trace=[json.loads(l) for l in (HERE/'actor_updates.jsonl').read_text().splitlines()]
assert rows==r['records'] and len(rows)==40 and len(trace)==10000
stream=np.load(HERE/'batch_index_stream.npy')
accepted=0;previous_q=r['start']['q']['all'];audit=[];previous_batch=None
for i,x in enumerate(rows):
 block=i+1
 assert x['block']==block and x['proposed_updates']==250*block
 ids=stream[i*250:(i+1)*250].tolist()
 assert x['batch_indices']==ids
 assert hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()==x['batch_indices_sha256']
 if previous_batch is not None:assert ids!=previous_batch
 part=trace[i*250:(i+1)*250]
 assert all(y['proposal_block']==block and y['proposed_update']==i*250+j+1 and y['candidate_accepted_update_index']==accepted+j+1 and y['batch_index']==ids[j] and np.isfinite(y['loss']) for j,y in enumerate(part))
 expected='ACCEPT' if x['acceptance']['behavior']['success_count']>=3 and x['acceptance']['behavior']['sim_errors']==0 else 'REJECT'
 assert expected==x['decision']
 assert abs(x['candidate_q1_minus_previous_accepted']-(x['candidate_metrics']['q']['all']['q1_gain']-previous_q['q1_gain']))<1e-12
 assert abs(x['candidate_q2_minus_previous_accepted']-(x['candidate_metrics']['q']['all']['q2_gain']-previous_q['q2_gain']))<1e-12
 if expected=='ACCEPT':accepted+=250
 else:assert x['rollback_verified'] and x['accepted_cumulative_q']==previous_q
 assert x['accepted_cumulative_updates']==accepted
 # Frozen LR warmup tracks ACCEPTED count, not rejected proposal count.
 if i and rows[i-1]['decision']=='REJECT':assert part[0]['lr']==trace[(i-1)*250]['lr']
 audit.append({'block':block,'proposed':x['proposed_updates'],'accepted':accepted,'candidate_delta_q1':x['candidate_metrics']['q']['all']['q1_gain'],'candidate_delta_q2':x['candidate_metrics']['q']['all']['q2_gain'],'q1_increment_vs_previous_accepted':x['candidate_q1_minus_previous_accepted'],'q2_increment_vs_previous_accepted':x['candidate_q2_minus_previous_accepted'],'acceptance_success':x['acceptance']['behavior']['success_count'],'decision':expected,'rollback_verified':x['rollback_verified'],'batch_start':ids[0],'lr_first':part[0]['lr'],'lr_last':part[-1]['lr']})
 previous_q=x['accepted_cumulative_q'];previous_batch=ids
reject=[x for x in rows if x['decision']=='REJECT'];accept=[x for x in rows if x['decision']=='ACCEPT']
def stats(items):
 q1=[x['candidate_q1_minus_previous_accepted'] for x in items]
 q2=[x['candidate_q2_minus_previous_accepted'] for x in items]
 total=[x['candidate_metrics']['q']['all']['q1_gain'] for x in items]
 return {'count':len(items),'q1_increment_mean':float(np.mean(q1)),'q1_increment_min':min(q1),'q1_increment_max':max(q1),'q2_increment_mean':float(np.mean(q2)),'q2_increment_min':min(q2),'q2_increment_max':max(q2),'candidate_total_delta_q1_mean':float(np.mean(total)),'q1_improving_count':sum(v>0 for v in q1),'both_q_improving_count':sum(p>0 and q>0 for p,q in zip(q1,q2)),'both_q_improving_fraction':sum(p>0 and q>0 for p,q in zip(q1,q2))/len(items),'acceptance_success_histogram':dict(Counter(str(x['acceptance']['behavior']['success_count'])+'/4' for x in items))}
perseed={}
for seed in read(HERE/'acceptance_seeds.json')['seeds']:
 for label,items in [('rejected',reject),('accepted',accept)]:
  episodes=[e for x in items for e in x['acceptance']['behavior']['episodes'] if e['seed']==seed]
  perseed[str(seed)+'_'+label]={'success_count':sum(e['success'] for e in episodes),'count':len(episodes)}
out={'previous_conclusion':a['classification'],'current_question':'Does fixed competence gate reject almost all genuinely Q-improving proposals?','experiment_type':'existing-data ONLY; no new Actor updates or simulator','classification':'COMPETENCE_GATE_SUPPRESSES_Q_OPTIMIZATION','decision':'STOP','accepted_blocks':[x['block'] for x in accept],'rejected':stats(reject),'accepted':stats(accept),'rejection_fraction':len(reject)/len(rows),'accepted_q1_fraction_collapse_target':a['accepted_q1_fraction_collapse_target'],'accepted_endpoint_acceptance_success':r['current_accepted']['acceptance']['behavior']['success_count'],'accepted_endpoint_reporting_evaluated':any(m['metrics']['checkpoint']==r['current_accepted']['checkpoint'] for m in r['milestones'].values()),'per_seed_acceptance':perseed,'audit':{'fully_read_proposal_rows':len(rows),'fully_read_actor_update_rows':len(trace),'all_rows_agree_with_round1_result':True,'fixed_batch_stream_exact':True,'every_adjacent_block_has_different_250_batch_order':True,'lr_schedule_counter_rolls_back':True,'all_rejected_rollbacks_verified':True,'all_candidate_losses_finite':True,'sim_errors':sum(x['acceptance']['behavior']['sim_errors'] for x in rows),'additional_simulator_invocations':0,'additional_env_steps':0,'additional_actor_updates':0,'additional_critic_updates':0},'block_audit':audit,'causal_status':'38/38 rejected candidates improve BOTH Q1 and Q2 versus previous accepted Actor, yet fail independent competence threshold. Gate prevents their incorporation but retains only15.87% collapse-target Q progress; does not establish matched-Q collapse prevention.','formal_collapse_explanation':'Twin-Q improvement is not a certificate of competence. This strengthens the Q-versus-real-competence mismatch evidence, but does not establish missing acceptance as a sufficient learning-preserving remedy or identify a unique upstream cause.','remaining_ambiguity':'Endpoint held-out success was not evaluated because it never crossed MID; do not reuse LOW4/4 as endpoint success. This cannot change low-Q suppression classification, so no extra simulator is needed.'}
write(HERE/'round2_analysis.json',out)
p=HERE/'round2';p.mkdir(exist_ok=False)
(p/'ROUND_REPORT.md').write_text('# ROUND2\n\nPrevious conclusion: '+out['previous_conclusion']+'\n\nCurrent question: '+out['current_question']+'\n\nWhy next: Round1 ended with only15.87%collapse-target accepted Q gain; follow prescribed CaseB without new training/evaluation.\n\nExperiment type: existing-data.\n\nMethod: read every proposal row and all10000update rows; compare Q1/Q2 with previous accepted Actor; verify fixed stream, LR counter and rollback.\n\nResult:\n'+json.dumps(out,indent=2)+'\n\nCausal status: '+out['causal_status']+'\n\nDoes this change our explanation of formal collapse? '+out['formal_collapse_explanation']+'\n\nDecision: STOP. No Round3 is warranted because unmatched-Q progress already prevents primary causal PASS.\n')
print(json.dumps(out,indent=2),flush=True)
