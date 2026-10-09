"""Round2 logs-only analysis: no torch, simulation, optimizer."""
import json,hashlib,subprocess,statistics,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
 reg=read(HERE/'preregistration.json');r1=read(HERE/'round1_result.json');rows=r1['results'];assert len(rows)==4
 logs=[json.loads(l) for l in (HERE/'fraction_logs.jsonl').read_text().splitlines()]
 assert logs==[a for r in rows for a in r['attempts']]
 inv=[];readcounts={}
 for p in sorted(HERE.rglob('*.json')):read(p);readcounts[str(p.relative_to(HERE))]='FULL'
 for p in HERE.rglob('*trajectories.jsonl'):
  data=[json.loads(l) for l in p.read_text().splitlines()];readcounts[str(p.relative_to(HERE))]=len(data)
  assert all(isinstance(x['finite_mc_return'],(int,float)) for x in data)
 for p in sorted((HERE/'evaluations').glob('*/invocation.json')):
  x=read(p);c=x['contract'];assert c['test_valid'] and all(c[k]==4 for k in ('parallel_envs_initialized','parallel_envs_used','parallel_envs_closed'))
  assert x['device']=='npu:0' and x['seeds']==reg['seeds'] and x['behavior']['sim_errors']==0
  assert all(v==0 for v in c['worker_exitcodes']);inv.append(x)
 safe=[r for r in rows if r['max_safe_fraction'] is not None];no=[r for r in rows if r['max_safe_fraction'] is None]
 material=[r for r in safe if r['safe_retention']['q1_gain']>=.05]
 twin_failed=[r for r in no if all(r['attempts'][-1]['block_delta_q'][k]>r['noise_floor'] for k in ('q1_gain','q2_gain')) and r['attempts'][-1].get('evaluation') and r['attempts'][-1]['evaluation']['behavior']['success_count']<3]
 if len(safe)==2:label='HETEROGENEOUS_LOCAL_GEOMETRY'
 elif len(material)>2:label='COMPETENCE_SAFE_TRUST_REGION_EXISTS_BUT_IS_VERY_SMALL'
 elif len(safe)>2 and len(safe)-len(material)>2:label='SAFE_REGION_EXISTS_ONLY_AT_NEGLIGIBLE_Q_PROGRESS'
 elif len(twin_failed)>2:label='LOCAL_CRITIC_Q_GRADIENT_COMPETENCE_MISALIGNMENT'
 else:label='INCONCLUSIVE'
 tables=[]
 for r in rows:
  a=next((a for a in r['attempts'] if a['safe']),None)
  tables.append({'proposal':r['block'],'block_stage':{6:'early',17:'middle',29:'late',40:'final'}[r['block']],'old_actor_state_hash':r['old_state_hash'],'old_actor_parameter_drift_L2':r['old_metrics']['drift']['parameter_l2'],'full_delta_Q1':r['full_block_delta_q']['q1_gain'],'success_at_0p0625':r['source']['historical_success_at_0p0625'],'tested_fractions':[a['fraction'] for a in r['attempts']],'no_safe_down_to_1_over_256':not any(a['safe'] for a in r['attempts'][:4]) and len(r['attempts'])>=4,'successes':[a['evaluation']['behavior']['success_count'] if a.get('evaluation') else None for a in r['attempts']],'max_safe_fraction':r['max_safe_fraction'],'safe_delta_Q1':a['block_delta_q']['q1_gain'] if a else None,'retention':r['safe_retention'],'classification':r['classification']})
 groups=collections.defaultdict(list)
 for t in tables:groups[t['old_actor_state_hash']].append(t['proposal'])
 trend={'proposal_table':tables,'distinct_old_actor_states':len(groups),'state_groups':dict(groups),'safe_radius_pattern':'heterogeneous' if len({r['max_safe_fraction'] for r in rows})>1 else 'stable at tested grid' if safe else 'absent within tested grid','does_competence_safe_radius_shrink_as_actor_moves_away_from_BC':'INCONCLUSIVE','reason':'Only two accepted old Actor states: block6 versus17/29/40. Middle/late/final have exactly the same old Actor; block index does not imply cumulative drift. Four directions on repeated seeds cannot establish a systematic drift-dependent trend.','round2_env_steps':0,'round2_optimizer_steps':0}
 production_after={p:sha(p) for p in reg['production_hashes']};formal_after={p:sha(p) for p in reg['formal_checkpoint_hashes']}
 historical_ok=all(sha(v['path'])==v['sha256'] for v in reg['evidence'].values());input_ok=all(sha(p)==h for p,h in reg['input_hashes'].items())
 ps=subprocess.check_output(['ps','-eo','pid,args'],text=True)
 relevant=[l for l in ps.splitlines() if any(s in l for s in ('run_subfraction.py run','multiprocessing.spawn','stage3_v6_train','launch_stage3'))]
 npu_status=subprocess.check_output(['npu-smi','info'],text=True)
 (HERE/'npu_safety_final.txt').write_text(npu_status)
 no_npu_process='No running processes found' in npu_status
 fusion=ROOT/'fusion_result.json'

 safety={'production_files_checked':len(production_after),'production_source_unchanged':production_after==reg['production_hashes'],'formal_checkpoints_checked':len(formal_after),'formal_checkpoints_unchanged':formal_after==reg['formal_checkpoint_hashes'],'historical_reports_and_raw_logs_unchanged':historical_ok,'historical_checkpoint_and_bank_unchanged':input_ok,'critic_hash_unchanged':r1['critic_hash_before']==r1['critic_hash_after'],'actor_optimizer_steps':0,'critic_optimizer_steps':0,'target_updates':0,'formal_training_resumed':False,'only_logical_npu0_used':True,'all_simulator_invocations_real_parallel4':True,'all_eval_workers_closed':True,'all_worker_exitcodes_zero':True,'no_residual_npu_process':no_npu_process,'residual_relevant_processes':relevant,'fusion_after':{'sha256':sha(fusion) if fusion.exists() else None,'git_status':subprocess.check_output(['git','status','--short','--','fusion_result.json'],cwd=ROOT,text=True).strip(),'policy':'NPU runtime side effect recorded only; never edit/delete/restore/clean'},'fully_read_files':readcounts,'testing_bug_fixes':[]}
 assert safety['production_source_unchanged'] and safety['formal_checkpoints_unchanged'] and historical_ok and input_ok and safety['critic_hash_unchanged'] and not relevant and no_npu_process
 cost={'selected_proposals':4,'subfractions_probed':len(logs),'offline_Q_probes':12+len(logs),'real_four_env_evaluations':len(inv),'episodes':sum(len(x['behavior']['episodes']) for x in inv),'simulator_env_steps':sum(x['env_steps'] for x in inv),'simulator_wall_seconds':sum(x['wall_time_seconds'] for x in inv),'round1_wall_seconds':r1['runtime_seconds'],'existing_data_reads_only':'40 historical proposal rows and10000 update rows, no rerun','new_training_updates':0}
 aggregate={'selected':4,'safe_found':len(safe),'no_safe_found':len(no),'material_retention_ge5pct':len(material),'safe_fraction_histogram':dict(collections.Counter(str(r['max_safe_fraction']) for r in rows)),'median_max_safe_fraction_safe_only':statistics.median(r['max_safe_fraction'] for r in safe) if safe else None,'median_Q1_retention_safe_only':statistics.median(r['safe_retention']['q1_gain'] for r in safe) if safe else None,'censoring':'Max safe fraction is largest tested successful grid point, not continuous threshold. Unsafe cases censored below final tested fraction, not zeros.'}
 answer='Smaller steps recovered competence in '+str(len(safe))+'/4 fixed rejected proposals; '+str(len(material))+'/4 retained at least5%full candidate Q1 gain. '
 if label=='SAFE_REGION_EXISTS_ONLY_AT_NEGLIGIBLE_Q_PROGRESS':answer+='Mathematical safety usually exists but Q progress is below the preregistered5%material threshold. This supports an extremely restrictive competence-safe region, not practical safety of production Q ascent.'
 elif label=='LOCAL_CRITIC_Q_GRADIENT_COMPETENCE_MISALIGNMENT':answer+='Most remained twin-Q-improving but competence-failing at the smallest tested fraction: local misalignment in this finite grid, not proof for all infinitesimal steps.'
 else:answer+='See individual proposal geometry; finite-grid behavior cannot generalize to every direction.'
 result={'answer':answer,'classification':label,'rounds_used':2,'aggregate':aggregate,'proposal_table':tables,'round2':trend,'costs':cost,'safety':safety,'invocations':inv,'confidence':'Moderate for these four exact proposals on fixed four seeds; low for population competence or drift causality. No heldout tests, frozen Critic, no reproduction of online training collapse.','next_step_only':'Before production changes, validate each proposal at its first safe fraction or final failed fraction on one preregistered disjoint seed set; do not resume formal training.','formal_training_remains_stopped':True}
 dump(HERE/'round2_analysis.json',trend);dump(HERE/'safety_final.json',safety);dump(HERE/'final_summary.json',result)
 lines=['# Subfraction local gradient test','',answer,'','Primary label: '+label+'. Rounds:2, Round2 read-only.','','|Proposal|Stage|Success@0.0625|Tested fractions (successes)|Max safe f|Safe deltaQ1|Q1 retention|Classification|','|---|---|---:|---|---:|---:|---:|---|']
 for t in tables:
  tested=', '.join(str(f)+' ('+str(s)+'/4)' for f,s in zip(t['tested_fractions'],t['successes']))
  lines.append('|'+str(t['proposal'])+'|'+t['block_stage']+'|'+str(t['success_at_0p0625'])+'/4|'+tested+'|'+str(t['max_safe_fraction'])+'|'+str(t['safe_delta_Q1'])+'|'+(str(100*t['retention']['q1_gain'])+'%' if t['retention'] else '--')+'|'+t['classification']+'|')
 for title,obj in [('Aggregate',aggregate),('Early vs late / actual drift',trend),('Cost and authenticity',cost),('Safety',safety)]:lines+=['','## '+title,'','```json',json.dumps(obj,indent=2),'```']
 lines+=['','## Interpretation','',result['confidence'],'',aggregate['censoring'],'','All Q1/Q2/Qmean retentions and simulator contracts are in final_summary.json. Exact old/full checkpoints loaded directly, no250update replay. Repeated identical old Q probes calibrate noise before line search.','',result['next_step_only'],'','FORMAL TRAINING REMAINS STOPPED']
 (HERE/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
 print('FINAL',json.dumps({k:result[k] for k in ('answer','classification','aggregate','proposal_table','costs','confidence')}),flush=True)
if __name__=='__main__':main()
