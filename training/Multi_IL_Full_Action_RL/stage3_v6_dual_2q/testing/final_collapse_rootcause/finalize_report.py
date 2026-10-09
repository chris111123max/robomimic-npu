import json,hashlib,time
from pathlib import Path
P=Path(__file__).resolve().parent
R={i:json.loads((P/f'round{i}/result.json').read_text()) for i in range(1,7)}
A={i:json.loads((P/f'round{i}/analysis.json').read_text()) for i in range(1,7)}
S=json.loads((P/'safety_after.json').read_text())
assert all(S[k]['unchanged'] for k in ['production_hashes','formal_checkpoint_hashes','input_hashes','historical_outputs']) and S['all_4_4_4'] and S['no_NPU_process'] and not S['residual_matching_processes']
manifest=[];trace_counts={};errors=[]
for i in range(1,7):
 for f in sorted((P/f'round{i}').glob('*.json')):
  parsed=json.loads(f.read_text());manifest.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
 for f in sorted((P/f'round{i}').glob('*.jsonl')):
  count=0
  with f.open() as fp:
   for line in fp:
    row=json.loads(line);count+=1
  trace_counts[str(f)]=count
 log=(P/f'round{i}/experiment.log').read_text()
 if 'Traceback (most recent call last)' in log or 'ERR99999' in log:errors.append(str(P/f'round{i}/experiment.log'))
assert not errors
original=json.loads((P.parent/'mean_multi_collapse_diagnosis/round1/contract.json').read_text());assert R[5]['lrs']==original['lrs'];assert trace_counts[str(P/'round5/updates.jsonl')]==1250
costs={}
for i,r in R.items():
 costs[str(i)]={k:r[k] for k in ['episodes','env_steps','wall_seconds','actor_updates','critic_updates']} if i in [1,2,4] else dict(r['cost'])
 costs[str(i)]['simulator_invocations']=1 if i in [1,2,4,5,6] else 0
 costs[str(i)]['existing_data_only']=False
costs['0']=json.loads((P/'round0/cost.json').read_text());totals={k:sum(c.get(k,0) for c in costs.values()) for k in ['episodes','env_steps','actor_updates','critic_updates','wall_seconds','simulator_invocations']}
summary={'rounds_executed':6,'round0_existing_only':True,'stop_reason':'SIX_ROUND_CAP; upstream source not uniquely identified; no validated corrective intervention','upstream_root_cause':'INCONCLUSIVE','failure_mechanism':'Q-positive policy optimization produces competence-damaging executed-action and closed-loop state/history changes; reproduced with ready and original MC Critics','failure_mechanism_confidence':'HIGH at policy level; no general local derivative sign-reversal claim','upstream_root_cause_confidence':'LOW_TO_MEDIUM for precise provenance; MEDIUM for shared counterfactual-value weakness family','top_1':'History-conditioned counterfactual supervision / function-approximation generalization error; state/history vs conditional-action support not separated','top_2':'Continuation-specific value mismatch: mixed behavior MC / approximate BC target value not valid for changed recurrent-policy closed-loop continuation','stage3_TD_bootstrap_necessary':False,'later_online_replay_or_critic_evolution_necessary':False,'production_fix_ready':False,'validated_signal_correction':False,'production_fix_direction':None,'final_disjoint_seeds':R[6]['seeds'],'final_behavior':R[6]['behavior'],'final_q_metrics':A[6]['qmetrics'],'factorial_old_prefix':A[1]['effects'],'factorial_bad_prefix':A[2]['effects'],'conditional_mc':A[4]['branches'],'value_semantics':A[3],'mc_source_substitution':A[5],'round_costs':costs,'total_new_cost':totals,'simulator_invocation_definition':'one evaluator invocation / exactly-four-worker pool;5 pools total, not per-step calls','cost_time_scope':'sum measured program time; excludes SSH/reading/editing latency','safety':S,'evidence_paths':{str(i):{'raw_result':str(P/f'round{i}/result.json'),'analysis':str(P/f'round{i}/analysis.json'),'report':str(P/f'round{i}/ROUND_REPORT.md')} for i in range(1,7)},'full_readback_manifest':manifest,'fully_read_JSONL_row_counts':trace_counts,'runtime_errors':errors,'actor_lr_schedule_exactly_matches_control':True,'formal_training':'STOPPED','no_strength_reduction_solution':True}
with (P/'final_summary.json').open('x') as f:json.dump(summary,f,indent=2,allow_nan=False)
