import json,sys
from pathlib import Path
from helpers import HERE,read,dump
p=HERE/(sys.argv[1] if len(sys.argv)>1 else 'round1')
r=read(p/'result.json')
branches=r['branches'];seeds=[e['seed'] for e in next(iter(branches.values()))['episodes']]
effects=[]
for i,seed in enumerate(seeds):
 vals={n:b['episodes'][i]['mc_from_fork'] for n,b in branches.items()}
 rr=vals['REF_REF'];br=vals['BAD_REF'];rb=vals['REF_BAD'];bb=vals['BAD_BAD']
 q=branches['REF_REF']['probe']['actions']
 effects.append({'seed':seed,'mc':vals,'current_action_effect_under_ref':br-rr,'continuation_effect_under_ref_action':rb-rr,'interaction':bb-br-rb+rr,'q_bad_minus_ref':{k:q['BAD'][k][i]-q['REF'][k][i] for k in ['q1','q2','qmean']},'single_action_harm':br<rr-1e-12,'continuation_harm':rb<rr-1e-12,'combined_harm':bb<rr-1e-12})
summary={'effects':effects,'episodes':{n:[{'seed':e['seed'],'success':e['success'],'length':e['length'],'mc':e['mc_from_fork'],'min_payload_z':e['payload_min_z_postfork']} for e in b['episodes']] for n,b in branches.items()},'strict_state_hash_all_branches_match':r['strict_state_hash_all_branches_match'],'env_contract':r['env_contract'],'cost':{k:r[k] for k in ['episodes','env_steps','wall_seconds','actor_updates','critic_updates']},'limitations':['Fork selection is observational; selected seed is a microscope, not independent confirmation.','Raw sampled actions were explicitly projected to feasible executable domain; Q comparison uses executed actions.','Snapshots cover simulator arrays, controller-visible fields, RNG and caches; opaque unexposed simulator fields are not serialized.','Current Q is not conditioned on the counterfactual continuation policy; identical Q across continuations is not by itself proof of an erroneous Bellman implementation.']}
dump(p/'analysis.json',summary)
print(json.dumps(summary,indent=2))
