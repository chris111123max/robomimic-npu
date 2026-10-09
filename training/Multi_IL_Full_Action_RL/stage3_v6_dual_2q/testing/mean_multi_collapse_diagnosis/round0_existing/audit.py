import json, pathlib, torch, numpy as np, subprocess
HERE=pathlib.Path(__file__).resolve().parent
TEST=HERE.parents[1]
RUN=pathlib.Path('/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_readiness_v2_multi_mean_random_formal_20260929_130945')
BR=RUN/'mean2q/multi_q'
def read(p): return json.loads(p.read_text())
def lines(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
prior=TEST/'actor_collapse_diagnosis/results'
closed=read(prior/'closed_loop.json')
rollouts={}
for r in closed['rounds']:
 for x in r['episodes']: rollouts.setdefault(x['checkpoint'],[]).append(x)
rows=[]
reference=torch.load(BR/'checkpoints/critic_ready.pth',map_location='cpu',weights_only=False)['actor']
for f in sorted((BR/'checkpoints').glob('*.pth')):
 x=torch.load(f,map_location='cpu',weights_only=False)
 print(f.name, list(x),flush=True)
 row={'path':str(f),'env_steps':x.get('env_steps'),'actor_updates':x.get('actor_updates'),'critic_updates':x.get('updates',x.get('critic_updates')),'keys':list(x)}
 opt=x.get('actor_optimizer',x.get('actor_optimizer_state_dict',{}))
 row['actor_adam_state_entries']=len(opt.get('state',{}))
 row['actor_adam_steps']=sorted(set(float(v['step']) for v in opt.get('state',{}).values() if 'step' in v))
 if 'actor' in x:
  row['actor_state_drift_l2']=(sum(float((x['actor'][k].double()-reference[k].double()).square().sum()) for k in reference))**.5
 rows.append(row)
metrics=lines(BR/'train_metrics.jsonl'); episodes=lines(BR/'episode_metrics.jsonl'); ready=lines(BR/'readiness_metrics.jsonl')
fields=('env_steps','updates','actor_updates','actor_rl_loss','actor_q','actor_grad_norm','actor_lr','q1_mean','q2_mean','td_target_mean','online_samples','offline_batch_fraction','online_batch_fraction')
selected=[]
for s in (140000,145000,150000,160000,170000,180000,190000,200000,280000):
 r=min(metrics,key=lambda r:abs(r['env_steps']-s)); selected.append({k:r[k] for k in fields if k in r})
bins=[]
for lo,hi in ((0,140000),(140000,150000),(150000,160000),(160000,170000),(170000,180000),(180000,190000),(190000,200000),(200000,280000)):
 es=[r for r in episodes if lo<r['env_steps']<=hi]
 bins.append({'range':[lo,hi],'episodes':len(es),'successes':sum(bool(r['success']) for r in es),'sim_errors':sum(bool(r['sim_error']) for r in es),'mean_length':float(np.mean([r['length'] for r in es])) if es else None})
history={}
for name in ('actor_init','critic_ready','step200k','step280k','last'):
 x=read(prior/f'offline_{name}.json')
 history[name]={'env_steps':x['env_steps'],'actor_updates':x['actor_updates'],'critic_updates':x['critic_updates'],'parameter_drift':x['parameter_drift'],'a2_summary':{k:v['all_steps'] for k,v in x['a2_behavior'].items()},'a3_summary':x['a3_q_action']['all'],'historical_four_seeds':rollouts.get(name)}
reports={}
for d in ('actor_objective_alignment','actor_temporal_supervision','actor_init_anchor','adaptive_policy_preservation'):
 for p in (TEST/d).glob('FINAL_REPORT*.md'): reports[str(p)]=p.read_text()
# Read existing mechanism experiment results in full; no random new experiment.
for d in ('results_real_geometry_20261003_retry1','results_real_module_online_20261003'):
 p=TEST/'actor_optimizer_diagnostics'/d/'analysis_summary.json'
 if p.exists(): history[d]=read(p)
replay=np.load(BR/'checkpoints/critic_ready.sequences.npy',allow_pickle=True).item()
replay_structure={k:({'keys':list(v),'episodes':len(v.get('episodes',[]))} if isinstance(v,dict) else {'type':type(v).__name__}) for k,v in replay.items()}
result={'branch':'mean2q/multi_q','checkpoints':rows,'historical_results':history,'formal_metric_samples':selected,'online_episode_bins':bins,'gate_open_records':[r for r in ready if r.get('readiness_pass_streak',0)>=3 or r.get('critic_ready')],'replay_structure':replay_structure,'prior_reports':reports,'no_new_environment_steps':True,'production_status':subprocess.check_output(['git','status','--short'],text=True)}
(HERE/'round0_existing_evidence.json').write_text(json.dumps(result,indent=2,allow_nan=False))

report="""# Round 0: existing evidence
Earliest saved competent formal checkpoint: critic_ready 140K; earliest saved collapsed: 200K. No formal checkpoint within this interval. Historical four-seed data are prior evidence, not a new 4-env contract claim. Episode metrics have changing seeds and exploration.
Strongest hypothesis: shared learned-Q ascent lacks local behavioral validity; component mean drift accumulates before outright failure. Q1-only divergence is later amplification. Round 1 tests frozen-ready Q ascent with four-worker behavioral validation, removing online replay/Critic coevolution as a necessary cause.
No new simulator in Round 0.
"""
(HERE/'ROUND0_REPORT.md').write_text(report)
print(json.dumps({'checkpoint_rows':rows,'formal_metric_samples':selected,'episode_bins':bins,'replay_structure':replay_structure},indent=2),flush=True)
