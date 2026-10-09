"""CPU-only interpretation/readback of the completed offline pass. No inference."""
from pathlib import Path
import json,hashlib,subprocess,numpy as np
P=Path(__file__).resolve().parent;RL=P.parents[2]
def read(n):return json.loads((P/n).read_text())
def write(n,d):(P/n).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
s=read('final_summary.json');o=read('occupancy_shift_audit.json');e=read('earliest_mismatch.json');th=read('policy_improvement_theory_audit.json');meta=read('context_metadata.json')
z=np.load(P/'frozen_numeric_evidence.npz');per_seed=[]
for u in [0,625,1250]:
 for ep in o['occupancy_results'][str(u)]['known_closed_loop']:
  ids=np.array([i for i,x in enumerate(meta) if x.get('owner')==u and x.get('seed')==ep['seed']]);w=np.array([.99**meta[i]['step'] for i in ids]);delta=z[f'{u}_exec'][ids]-z['0_exec'][ids]
  per_seed.append({'updates':u,'seed':ep['seed'],'history_count':len(ids),'predicted_finite_advantage_sum':(w[:,None]*delta).sum(0).tolist(),'uniform_mean_predicted_advantage':delta.mean(0).tolist(),'realized_start_MC':ep['start_MC'],'BC_realized_start_MC':next(x['start_MC'] for x in o['occupancy_results']['0']['known_closed_loop'] if x['seed']==ep['seed']),'success':ep['success'],'warning':'Conditional Q estimates on own histories, not true advantage or pointwise causality.'})
write('per_seed_occupancy_advantages.json',per_seed)
s['per_seed_occupancy_advantages']=per_seed
s['classifications']={'proxy_execution_collapse_explanation':{'level':'REJECTED','scope':'Tested0/625/1250 checkpoints and4096 reconstructed replay histories; actual sampling keeps Q gain. Global policy accuracy not established.'},'same_history_new_occupancy_predicted_gain':{'level':'CONFIRMED','scope':'Measured network values, Gaussian integration uncertainty only.'},'actor_specific_early_support_exit':{'level':'INCONCLUSIVE','scope':'At625, calibrated history/prior-action/combined outside fraction0; embedding0.51% below calibration1%. Some1250 metrics rise after collapse. No demonstrated prior onset.'},'improvement_value_inconsistency':{'level':'SUPPORTED','scope':'Positive new-occupancy predicted advantages accompany reduced realized performance; mathematical exact advantages not verified.'},'unique_upstream_root_cause':{'level':'INCONCLUSIVE'}}
e['per_seed_own_advantages']=per_seed
e['local_negative_example_warning']='The625 seed20005 t2 Q1-negative/Q2-positive point belongs to a successful trajectory; it is NOT the full-collapse origin. Global phase means are positive. No unique bad transition causally identified.'
s['replay_to_onpolicy']='INCONCLUSIVE'
# Read-only current source boundary masks; do not call known matched masks bugs.
notes=[]
for rel,a,b in [('stage3_v5_rgmm_td3/stage3_v5_replay.py',48,62),('stage3_v5_rgmm_td3/stage3_v5_replay.py',170,184)]:
 f=RL/rel;lines=f.read_text().splitlines();notes.append('\n'+str(f)+':'+str(a)+'\n```python\n'+'\n'.join(f'{i+1}: {x}' for i,x in enumerate(lines) if a<=i+1<=b)+'\n```\n')
with (P/'source_audit.md').open('a') as f:f.write('\n## Finite-horizon masks\nCurrent offline and online implementations explicitly use terminated OR truncated as TD terminal. This matches finite stop/no bootstrap at time-limit. Historical checkpoint accuracy still not proven.\n'+''.join(notes))
th['current_source_time_limit_contract']='CONFIRMED current offline and online TD mask=terminated OR truncated; no positive bootstrap across boundary. Not identified error.'
th['critical_logic']='Exact old-policy advantage positive under correct new-policy occupancy would imply improvement under matched sufficient state/time semantics. Ordinary continuation dependence alone cannot explain its opposite. The measured network outputs have not been proven exact/sufficient; no empirical identity claim.'
write('policy_improvement_theory_audit.json',th);write('earliest_mismatch.json',e)
extra='\n## Final evidence interpretation\n\n'+th['critical_logic']+'\n\n'+th['current_source_time_limit_contract']+'\n\nREJECTED within measured scope: proxy-to-execution Q loss as collapse explanation. SUPPORTED: predicted policy-improvement value inconsistent with recorded outcomes. INCONCLUSIVE: unique upstream cause.625 support distances show no threshold excess on history/prior-action/combined groups;1250 excess can be consequence, not a proved early cause.\n\n'+e['local_negative_example_warning']+'\n\nPer-seed finite predicted sums and real outcomes: per_seed_occupancy_advantages.json.\n'
with (P/'FINAL_REPORT.md').open('a') as f:f.write(extra)
write('final_summary.json',s)
for x in per_seed:print('PER_SEED',x['updates'],x['seed'],x['predicted_finite_advantage_sum'],x['realized_start_MC'],x['success'])
