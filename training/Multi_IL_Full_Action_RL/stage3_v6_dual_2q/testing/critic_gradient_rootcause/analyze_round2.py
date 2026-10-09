import json,numpy as np
from pathlib import Path
from run_round1 import HERE,read,dump
out=HERE/'round2';r=read(out/'result.json');reg=read(out/'preregistration.json');contexts=read(out/'contexts.json');base=r['branches']['0'];table=[]
for i,c in enumerate(contexts):
 rows=[]
 for f in (1/512,1/256,1/128):
  plus=r['branches'][str(f)];minus=r['branches'][str(-f)];p=plus['probes'][i];m=minus['probes'][i];z=base['probes'][i]
  dq=[p[k]-m[k] for k in ('q1','q2')];dj=plus['episodes'][i]['mc_from_fork']-minus['episodes'][i]['mc_from_fork']
  row={'epsilon':f,'q1_delta_pm':dq[0],'q2_delta_pm':dq[1],'q1_slope':dq[0]/(2*f),'q2_slope':dq[1]/(2*f),'mc_delta_pm':dj,'mc_slope':dj/(2*f),'q1_plus_vs_zero':p['q1']-z['q1'],'q2_plus_vs_zero':p['q2']-z['q2'],'mc_plus_vs_zero':plus['episodes'][i]['mc_from_fork']-base['episodes'][i]['mc_from_fork'],'success_pm0':[plus['episodes'][i]['success'],minus['episodes'][i]['success'],base['episodes'][i]['success']],'length_pm0':[plus['episodes'][i]['length'],minus['episodes'][i]['length'],base['episodes'][i]['length']],'clipped':p['clipped'] or m['clipped'],'direct_sign_mismatch':min(dq)>reg['q_floor'] and dj < -reg['mc_floor'] and not(p['clipped'] or m['clipped'])}
  traces=[]
  for branch in (plus,minus):
   raw=[json.loads(l) for l in Path(branch['trace']).read_text().splitlines()];selected={t['timestep']:t for t in raw if t['seed']==c['seed']};zrows=[json.loads(l) for l in Path(base['trace']).read_text().splitlines()];zz={t['timestep']:t for t in zrows if t['seed']==c['seed']};dist=[float(np.linalg.norm(np.asarray(t['next_observation_flat'])-np.asarray(zz[k]['next_observation_flat']))) for k,t in selected.items() if k>=c['timestep'] and k in zz];traces.append({'common_postfork_steps':len(dist),'observation_deviation_max':max(dist),'observation_deviation_final_common':dist[-1]})
  row['trajectory_deviation_pm']=traces;rows.append(row)
 table.append({'seed':c['seed'],'timestep':c['timestep'],'proposal':c['proposal_block'],'autograd_directional':base['probes'][i]['autograd_directional'],'tiny_scaled_norm':c['tiny_scaled_norm'],'full_scaled_norm':c['full_scaled_norm'],'any_mismatch':any(x['direct_sign_mismatch'] for x in rows),'rows':rows})
count=sum(x['any_mismatch'] for x in table)
classification='LOCAL_Q_GRADIENT_SIGN_MISALIGNMENT' if count>=2 else 'STATE_CONDITIONAL_LOCAL_SIGN_MISMATCH' if count else 'LOCAL_DERIVATIVE_SIGN_NOT_ESTABLISHED'
decision='CONTINUE_TO_ROUND_3_DIRECTIONAL_SUPPORT' if count else 'CONTINUE_TO_ROUND_3_HORIZON'
a={'classification':classification,'mismatching_contexts':count,'decision':decision,'table':table,'env_contract':r['env_contract'],'env_steps':r['env_steps'],'wall_seconds':r['wall_seconds'],'limitations':['Four selected contexts, not population; observed-history identity, not serialized hidden simulator state','Finite MC is discrete sparse-return pathwise response; flat MC does not establish correct gradient','Readiness BC continuation estimates a single-action derivative, not persistent theta_old improvement','Q finite differences below floor treated unresolved; no sign claim from numerical noise']}
dump(out/'analysis.json',a)
text='# Round2: action-space local derivatives\n\nPrevious: all late tiny parameter proposals lose one old-successful independent seed; old policies already impaired.\nQuestion: does the learned-Q direction oppose actual single-action return locally?\nWhy: parameter-level loss alone cannot identify an action derivative.\nExisting data sufficient? NO; exact local paired interventions were missing.\nEvidence: CAUSAL action intervention, limited-context mechanistic inference.\n\n'
text+='Classification: '+classification+'; mismatching contexts='+str(count)+'; Decision: '+decision+'\n\n'
text+='|seed|t|epsilon|delta Q1(+/-)|delta Q2(+/-)|delta MC(+/-)|length + / - / 0|sign mismatch|\n|---|---|---|---|---|---|---|---|\n'
for x in table:
 for v in x['rows']:text+=f"|{x['seed']}|{x['timestep']}|{v['epsilon']:.8f}|{v['q1_delta_pm']:.9g}|{v['q2_delta_pm']:.9g}|{v['mc_delta_pm']:.9g}|{v['length_pm0']}|{v['direct_sign_mismatch']}|\n"
text+='\nRuled out: no unconditional global gradient conclusion from behavior value ranking. Top hypothesis: '+('local directional supervision mismatch; provenance not yet established' if count else 'persistent policy composition / horizon dependence; local sparse-return identification incomplete')+'.\nNo fixes, no optimizer steps.\n'
with (out/'ROUND_REPORT.md').open('x') as f:f.write(text)
print(json.dumps(a,indent=2,allow_nan=False))
