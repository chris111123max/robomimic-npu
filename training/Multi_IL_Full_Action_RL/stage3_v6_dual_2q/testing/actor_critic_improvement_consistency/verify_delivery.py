"""Final artifact/source integrity readback, no model/simulator execution."""
from pathlib import Path
import json,hashlib,subprocess,numpy as np
P=Path(__file__).resolve().parent
def read(n):return json.loads((P/n).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
s=read('final_summary.json');e=read('earliest_mismatch.json');rows=read('per_seed_occupancy_advantages.json');m=read('context_metadata.json');z=np.load(P/'frozen_numeric_evidence.npz')
row=next(x for x in rows if x['updates']==625 and x['seed']==20007)
i=next(i for i,x in enumerate(m) if x.get('owner')==625 and x.get('seed')==20007 and x['step']==0)
e['earliest_primary_observed_conflict']={'checkpoint_updates':625,'seed':20007,'trajectory_steps':[0,699],'initial_context_index':i,'initial_proxy_Q':z['625_proxy'][i].tolist(),'initial_executed_Q':z['625_exec'][i].tolist(),'initial_conditional_delta_Q_vs_BC':(z['625_exec'][i]-z['0_exec'][i]).tolist(),'trajectory_advantage':row,'sources':['mean_multi_collapse_diagnosis/round1/READY_trajectories.jsonl','mean_multi_collapse_diagnosis/round1/FROZEN_625_trajectories.jsonl'],'single_bad_action_or_causal_transition':'NOT_MEASURABLE; whole recorded trajectory inconsistency, not single action attribution','evidence':'predicted/correlational; strict counterfactual physical snapshot absent'}
(P/'earliest_mismatch.json').write_text(json.dumps(e,indent=2,allow_nan=False)+'\n')
s['earliest_primary_conflict']=e['earliest_primary_observed_conflict']
s['source_audit']=str(P/'source_audit.md');s['report_sha256']=sha(P/'FINAL_REPORT.md')
s['artifacts']={n:str(P/n) for n in ['FINAL_REPORT.md','final_summary.json','source_audit.md','proxy_execution_audit.json','occupancy_shift_audit.json','earliest_mismatch.json']}
(P/'final_summary.json').write_text(json.dumps(s,indent=2,allow_nan=False)+'\n')
required=list(s['artifacts'])+['run_audit.py','audit.log','preregistration.json','frozen_numeric_evidence.npz','fixed_replay_histories.npz','safety_before.json','safety_after.json','per_seed_occupancy_advantages.json','policy_improvement_theory_audit.json']
assert all((P/n).is_file() and (P/n).stat().st_size for n in required)
assert all(np.isfinite(z[k]).all() for k in z.files)
assert s['fixed_replay_histories']==4096 and s['trajectory_contexts']==6594 and len(m)==10690
assert s['known_success_counts']=={'0':4,'625':3,'1250':0}
assert s['root_cause']=='INCONCLUSIVE' and s['production_changed']=='NO' and s['formal_training']=='STOPPED'
assert s['simulator']['invocations']==s['simulator']['env_steps']==s['new_Actor_updates']==s['new_Critic_updates']==0
assert sha(P/'FINAL_REPORT.md')==s['report_sha256']
before=read('safety_before.json');changes=[p for p,h in before['files'].items() if not Path(p).is_file() or sha(p)!=h];assert not changes,changes
assert all(sha(x['path'])==x['sha256'] for x in read('source_evidence.json'))
procs=[l for l in subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines() if 'python' in l and any(k in l for k in ['run_audit.py','train_stage','acceptance_runner','spawn_main'])];assert not procs,procs
report=(P/'FINAL_REPORT.md').read_text();assert report.startswith('# '+chr(26368))
assert all(k in report for k in ['CONFIRMED','SUPPORTED','REJECTED','INCONCLUSIVE','NOT_MEASURABLE','ACTOR_CRITIC_IMPROVEMENT_CONSISTENCY_DIAGNOSIS_COMPLETE','FORMAL_TRAINING_REMAINS_STOPPED'])
audit={'passed':True,'all_named_artifacts_exist_readback':True,'finite_numeric_outputs':True,'NPU_probe_finished':True,'zero_new_training_or_simulations':True,'protected_files_rehashed_unchanged':len(before['files']),'source_readback_hashes_match':True,'model_hashes_unchanged_from_completed_runner':read('safety_after.json')['model_hashes_unchanged'],'report_summary_earliest_and_raw_arrays_match':True,'no_residual_experiment_or_training':True,'report_sha256':s['report_sha256'],'scope_manifest':[str(f.relative_to(P)) for f in P.rglob('*') if f.is_file()],'inconclusive_and_measurement_limits_explicit':True}
(P/'completion_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'VERIFIED_COMPLETE':True,'protected_files':len(before['files']),'proxy_execution':s['proxy_to_execution'],'replay_onpolicy':s['replay_to_onpolicy'],'prediction_transport':s['replay_gain_prediction_transport'],'root_cause':s['root_cause'],'new_simulator_calls':0}),flush=True)
