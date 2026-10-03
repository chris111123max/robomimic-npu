"""Full read of existing experiment JSON/JSONL; no model/environment execution."""
import json,math,hashlib,subprocess
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;O=H/'results_20261003';REG=json.loads((H/'preregistration.json').read_text())
steps=REG['checkpoint_eval_steps'];summary={'preregistration':REG,'experiment_status':json.loads((O/'experiment_status.json').read_text()),'branches':{},'comparisons':[]}
finite_count=0

def check(x):
 global finite_count
 if isinstance(x,float):assert math.isfinite(x);finite_count+=1
 elif isinstance(x,dict):
  for v in x.values():check(v)
 elif isinstance(x,list):
  for v in x:check(v)

def stats(rows):
 if not rows:return None
 ratios=np.array([r['anchor_RL_grad_ratio'] for r in rows]);lambdas=np.array([r['lambda'] for r in rows]);ds=np.array([r['D'] for r in rows]);gains=np.array([r['same_critic_optimizer_step_RL_gain'] for r in rows])
 return {'count':len(rows),'cumulative_same_critic_RL_gain':float(gains.sum()),'RL_gain_mean':float(gains.mean()),'negative_RL_gain_fraction':float(np.mean(gains<0)),'anchor_RL_ratio_median':float(np.median(ratios)),'anchor_RL_ratio_p90':float(np.quantile(ratios,.9)),'anchor_RL_ratio_p95':float(np.quantile(ratios,.95)),'fraction_ratio_gt_1':float(np.mean(ratios>1)),'lambda_median':float(np.median(lambdas)),'lambda_p90':float(np.quantile(lambdas,.9)),'lambda_p95':float(np.quantile(lambdas,.95)),'fraction_lambda_zero':float(np.mean(lambdas==0)),'fraction_lambda_at_max':float(np.mean(lambdas>=REG['lambda_max']-1e-10)),'D_median':float(np.median(ds)),'D_p95':float(np.quantile(ds,.95)),'fraction_D_above_safe':float(np.mean(ds>REG['D_safe'])),'fraction_D_above_hard':float(np.mean(ds>=REG['D_hard'])),'max_std_RL_grad_norm':max(r['std_RL_grad_norm'] for r in rows)}
for branch in ('BASELINE','ADAPTIVE'):
 g=O/branch/'mean2q/multi_q';path=g/'actor_update_diagnostics.jsonl'
 raw=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
 for r in raw:check(r)
 # Production final catch-up may execute after a saved step checkpoint. Resuming the
 # specific saved checkpoint can repeat a counter. Keep the latest effective path;
 # do not double-count discarded terminal catch-up optimizer operations.
 unique={r['actor_updates']:r for r in raw};rows=list(sorted(unique.values(),key=lambda r:r['actor_updates']))
 results={}
 for step in steps:
  p=g/'preservation_diagnostics'/f'step_{step:07d}.json'
  if not p.exists():continue
  d=json.loads(p.read_text());check(d);m=d['fixed_140K_critic_probe'];ev=d['closed_loop']
  selected=[r for r in rows if r['actor_updates']<=d['actor_updates']]
  results[str(step)]={'checkpoint_diagnostic_JSON':str(p),'actor_updates':d['actor_updates'],'critic_updates':d['critic_updates'],'success_count':ev['success_count'],'mean_length':ev['mean_length'],'sim_error_count':ev['sim_error_count'],'execution_drift':m['execution_drift'],'weighted_early':m['early'],'weighted_mid':m['mid'],'weighted_late':m['late'],'weighted_final':m['final'],'component_RMS':m['gmm']['component_rms'],'categorical_KL':m['gmm']['categorical_kl_ref_current'],'mode_rank_change_fraction':float(np.mean(m['gmm']['rank_change_fraction_by_timestep'])),'top1_change_fraction':float(np.mean(m['gmm']['top1_change_fraction_by_timestep'])),'parameter_drift':m['parameter_drift'],'hidden_L2_by_timestep':m['hidden_drift']['l2_mean_by_timestep'],'fixed_140K_score_gain':m['RL_improvement'],'controller_statistics':stats(selected),'episodes':ev['episodes']}
  trace=g/'preservation_diagnostics'/f'trajectory_{step:07d}.jsonl'
  if trace.exists():
   for line in trace.read_text().splitlines():check(json.loads(line))
 logs=list(g.glob('*metrics.jsonl'))
 for p in logs:
  for l in p.read_text().splitlines():check(json.loads(l))
 summary['branches'][branch]={'milestones':results,'raw_actor_update_rows':len(raw),'unique_effective_actor_update_rows':len(rows),'discarded_duplicate_terminal_catchup_rows':len(raw)-len(rows),'statistics_all_observed_effective_updates':stats(rows)}
for step in steps:
 a=summary['branches']['BASELINE']['milestones'].get(str(step));b=summary['branches']['ADAPTIVE']['milestones'].get(str(step))
 if not a or not b:continue
 ga=a['controller_statistics'];gb=b['controller_statistics'];delta=a['fixed_140K_score_gain']
 row={'env_steps':step,'BASELINE_success':a['success_count'],'ADAPTIVE_success':b['success_count'],'BASELINE_execution_drift':a['execution_drift'],'ADAPTIVE_execution_drift':b['execution_drift'],'fixed_140K_score_RL_retention':b['fixed_140K_score_gain']/delta if abs(delta)>1e-12 else None,'cumulative_optimizer_step_RL_retention':gb['cumulative_same_critic_RL_gain']/ga['cumulative_same_critic_RL_gain'] if ga and gb and abs(ga['cumulative_same_critic_RL_gain'])>1e-12 else None}
 summary['comparisons'].append(row)
summary['all_read_JSON_floating_fields_finite']=True;summary['finite_fields_read']=finite_count
summary['production_files_unchanged']={f:hashlib.sha256(Path(f).read_bytes()).hexdigest()==v for f,v in REG['production_sha256'].items()}
assert all(summary['production_files_unchanged'].values())
p=O/'analysis_summary.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');tmp.replace(p)
print(json.dumps({'status':summary['experiment_status'],'comparisons':summary['comparisons'],'controller_stats':{b:v['statistics_all_observed_effective_updates'] for b,v in summary['branches'].items()},'finite_fields_read':finite_count},indent=2))
