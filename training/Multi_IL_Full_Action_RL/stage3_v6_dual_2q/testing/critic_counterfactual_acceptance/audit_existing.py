import json,pathlib,hashlib,collections,re
HERE=pathlib.Path(__file__).resolve().parent;TEST=HERE.parent;RL=HERE.parents[2]
prior=['critic_counterfactual_identifiability','final_collapse_rootcause','critic_gradient_rootcause','q_gain_matched_path_test','mean_multi_iterative_rootcause'];reports={}
for name in prior:
 p=TEST/name;reports[name]={}
 for f in ['FINAL_REPORT.md','final_summary.json']:
  s=(p/f).read_text();reports[name][f]={'path':str((p/f).resolve()),'sha256':hashlib.sha256(s.encode()).hexdigest(),'bytes':len(s.encode())}
  if f.endswith('.json'):reports[name][f]['conclusions']={k:v for k,v in json.loads(s).items() if isinstance(v,(str,bool,int,float,type(None)))}
used=set();invent=[]
for p in TEST.rglob('final_summary.json'):
 if HERE in p.parents:continue
 s=p.read_text();seeds=set(map(int,re.findall(r'\b2\d{4}\b',s)));used.update(seeds);invent.append({'file':str(p.resolve()),'episode_seed_literals':sorted(seeds)})
reserved=next(list(range(start,start+4)) for start in range(20040,29996,4) if not used.intersection(range(start,start+4)))
source_ranges={'stage2_2_history_aware_critic/sequence_dataset.py':[(16,19),(41,61)],'stage2_2_history_aware_critic/train_stage2_2.py':[(56,65),(99,109)],'stage2_2_history_aware_critic/history_critic.py':[(13,29),(37,55)],'stage3_v5_rgmm_td3/stage3_v5_history_critic.py':[(27,84)],'stage3_v5_rgmm_td3/stage3_v5_actor.py':[(278,331)],'stage3_v6_dual_2q/stage3_v6_agent.py':[(105,160)],'stage3_v6_dual_2q/stage3_v6_readiness.py':[(65,145)],'stage1_rollout_collection/collect_multi_il_rollouts.py':[(338,369)],'stage3_v6_dual_2q/train_stage3_v6_vector.py':[(744,756)]}
sources={}
for file,ranges in source_ranges.items():
 p=RL/file;lines=p.read_text().splitlines();sources[str(p.resolve())]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'regions':[{'start':a,'end':b,'text':'\n'.join(lines[a-1:b])} for a,b in ranges]}
old=json.loads((TEST/'critic_counterfactual_identifiability/round2/existing_pairs.json').read_text());reused=[dict(x,acceptance_classification='INVALID_COUNTERFACTUAL' if not x['strict_pair'] else 'SEMANTICS_MISMATCH',reason='approximate state' if not x['strict_pair'] else 'historical continuation samples tiny Gaussian residual;new primary benchmark uses exact TD component-mean continuation;one-RNG evidence exploratory') for x in old['pairs']]
audit={'reports_read':reports,'production_source_audit':sources,'reserved_independent_seeds':reserved,'reported_seeds_inventory':invent,'existing_pairs':reused,'historical_new_strict_ties':8,'remaining_question':'Conditional action advantages under a fixed semantically matching continuation;existing history/action sensitivity already established,no repeat experiment.','MC_semantics':'Mixture of three collection-policy realized MC labels without policy identifier;fixed BC continuation not an exact value target.','ready_semantics':'Readiness actor/target_actor equals BC source;categorical probabilities and component means define target continuation;mean2q target,finite gamma .99 return at executed candidate includes immediate reward;zero beyond success/termination/truncation.'}
(HERE/'existing_audit.json').write_text(json.dumps(audit,indent=2));print('AUDIT_DONE',len(reports),'reports',len(sources),'sourcefiles','reserved',reserved)
