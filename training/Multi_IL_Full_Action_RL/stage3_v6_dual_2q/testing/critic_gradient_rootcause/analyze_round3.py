import json
from pathlib import Path
from run_round1 import HERE,read,dump
out=HERE/'round3';r=read(out/'result.json');table=[];trace_rows=0
for h,b in r['branches'].items():
 raw=[json.loads(l) for l in Path(b['trace']).read_text().splitlines()];trace_rows+=len(raw)
 for i,e in enumerate(b['episodes']):
  z=r['zero_reused']['episodes'][i]
  p=[{'q1_delta':b['probes'][i]['q1']-r['zero_reused']['probes'][i]['q1'],'q2_delta':b['probes'][i]['q2']-r['zero_reused']['probes'][i]['q2'],'clipped':b['probes'][i]['clipped']}] if h=='1' else b['probes'][i]
  both=sum(min(x['q1_delta'],x['q2_delta'])>1e-7 for x in p)
  table.append({'horizon':int(h),'seed':e['seed'],'length':e['length'],'baseline_length':z['length'],'success':e['success'],'baseline_success':z['success'],'mc':e['mc_from_fork'],'baseline_mc':z['mc_from_fork'],'delta_mc':e['mc_from_fork']-z['mc_from_fork'],'q_positive_injections':both,'injections':len(p),'q_positive_fraction':both/len(p),'q1_delta_sum':sum(x['q1_delta'] for x in p),'q2_delta_sum':sum(x['q2_delta'] for x in p),'clipped_injections':sum(x['clipped'] for x in p),'horizon_returns':e['horizon_returns']})
negative=[x for x in table if x['delta_mc']<-1e-9];flip=[]
for seed in [20028,20029,20030,20031]:
 one=next(x for x in table if x['horizon']==1 and x['seed']==seed)
 for x in table:
  if x['seed']==seed and x['horizon']>1 and one['delta_mc']>=0 and x['delta_mc']<0:flip.append(x)
classification='LOCAL_VALUE_IMPROVEMENT_LONG_HORIZON_COMPETENCE_MISMATCH' if any(x['q_positive_fraction']>=.9 for x in flip) else 'HORIZON_DEPENDENT_NONMONOTONE_RESPONSE' if negative else 'FIXED_LOCAL_DIRECTION_PERSISTENCE_NOT_SUFFICIENT'
a={'classification':classification,'table':table,'negative_cases':negative,'nonnegative_H1_to_negative_longer_cases':flip,'fully_read_trace_rows':trace_rows,'env_contract':r['env_contract'],'new_env_steps':r['new_env_steps'],'wall_seconds':r['wall_seconds'],'decision':'CONTINUE_TO_ROUND_4','round4_question':'distinguish local directional supervision deficit versus sustained-policy composition mismatch; choose minimal offline audit after inspecting this table','limitations':['Constant executed displacement, not complete learned Actor','Instantaneous paired Q on own history is NOT value of the whole intervention','Sparse reward and nonmonotone outcomes prevent declaring universal local wrong derivative','No claim Q-positive every injection unless separately counted']}
dump(out/'analysis.json',a)
s='# Round3: direction persistence\n\nPrevious: no multiple-context central sign mismatch; one-sided nonmonotone finite harm at seed20031.\nQuestion: same action displacement over H1/4/8/16/32; does composition alter outcome?\nWhy: isolated actions and complete learned policy do not have equivalent continuation.\nExisting data sufficient? NO for independent contexts; H1 and zero reused, only H4/8/16/32 new.\nEvidence: CAUSAL bounded action-duration intervention; instantaneous Q comparisons MECHANISTIC.\n\nClassification: '+classification+'\n\n|seed|H|length / zero|delta MC|positive twin-Q injections|success|\n|---|---|---|---|---|---|\n'
for x in table:s+=f"|{x['seed']}|{x['horizon']}|{x['length']}/{x['baseline_length']}|{x['delta_mc']:.9g}|{x['q_positive_injections']}/{x['injections']}|{x['success']}|\n"
s+='\nWhat was ruled out: simple monotone duration/drift explanation only if contradicted by table; no claim that every local step is wrong. Current top hypothesis: state-dependent, continuation-sensitive Q optimization rather than a universally incorrect infinitesimal slope. Decision: CONTINUE_TO_ROUND_4; final offline provenance audit only, no mitigation.\n'
with (out/'ROUND_REPORT.md').open('x') as f:f.write(s)
print(json.dumps(a,indent=2,allow_nan=False))
