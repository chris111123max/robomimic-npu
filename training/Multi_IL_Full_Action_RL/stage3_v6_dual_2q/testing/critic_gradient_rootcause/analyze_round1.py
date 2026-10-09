"""Read complete Round1 JSON/trajectories, then make a result-driven next-round decision."""
import json,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'round1'
def read(p):return json.loads(Path(p).read_text())
def main():
 r=read(OUT/'result.json');reg=read(OUT/'preregistration.json');behavior=r['evaluation']['behavior'];identities=r['identities'];table=[]
 for identity,st in zip(identities,r['states']):
  b=st['block'];candidate=behavior['BLOCK_'+str(b).zfill(2)];old=behavior[identity['old_control_label']]
  pairs=[]
  for c,o,h in zip(candidate['episodes'],old['episodes'],st['historical_evaluation']['behavior']['episodes']):
   assert c['seed']==o['seed']
   pairs.append({'seed':c['seed'],'old_success':o['success'],'new_success':c['success'],'old_length':o['length'],'new_length':c['length'],'delta_finite_MC':c['start_mc_return']-o['start_mc_return'],'lost_success':o['success'] and not c['success'],'gained_success':c['success'] and not o['success']})
  table.append({'block':b,'fraction':st['fraction'],'historical_acceptance_success':st['historical_evaluation']['behavior']['success_count'],'independent_success':candidate['success_count'],'old_independent_success':old['success_count'],'historical_episodes':st['historical_evaluation']['behavior']['episodes'],'independent_episodes':candidate['episodes'],'paired_old_control':pairs,'lost_success_count':sum(x['lost_success'] for x in pairs),'gained_success_count':sum(x['gained_success'] for x in pairs),'block_delta_Q':identity['block_delta_q']})
 bad=[t for t in table if t['block']!=6];badcount=sum(t['independent_success']<=2 for t in bad)
 if badcount>=2:
  criterion='YES';label='CROSS_SEED_GENERALIZED_Q_GRADIENT_MISALIGNMENT';decision='CONTINUE_TO_ROUND_2';reason='At least2/3late fixed Q-positive states remain <=2/4 against fresh readiness4/4.'
 elif all(t['independent_success']==4 for t in bad):
  criterion='NO';label='SEED_SPECIFIC_COMPETENCE_MISALIGNMENT';decision='STOP';reason='All three late states recover4/4 on independent seeds: original small-step failure does not generalize to this set.'
 else:
  criterion='PARTLY';label='HETEROGENEOUS_STATE_AND_SEED_DEPENDENT_MISALIGNMENT';decision='CONTINUE_TO_ROUND_2';reason='Mixed state/seed outcomes require local conditional analysis; no universal wrong-gradient claim.'
 incremental=sum(t['lost_success_count']>t['gained_success_count'] for t in bad)
 ambiguity='Candidate-versus-BC generalization is NOT identical to incremental damage of this tiny proposal. Two theta_old controls quantify already accumulated competence loss; compare paired old/candidate outcomes before calling a tiny direction destructive.'
 traces={}
 for p in (OUT/'evaluations').rglob('*trajectories.jsonl'):
  data=[json.loads(l) for l in p.read_text().splitlines()];traces[str(p.relative_to(OUT))]=len(data)
 a={'criterion_generalizes':criterion,'classification':label,'decision':decision,'reason':reason,'independent_seeds':r['seeds'],'readiness_success':4,'late_bad_count':badcount,'late_incrementally_worse_than_old_count':incremental,'control_caveat':ambiguity,'table':table,'fully_read_trajectory_rows':traces,'actor_updates':0,'critic_updates':0,'env_contract':r['evaluation']['contract']}
 with (OUT/'analysis.json').open('x') as f:json.dump(a,f,indent=2,allow_nan=False)
 lines=['# Round1 — cross-seed generalization','','Previous strongest conclusion: subfraction misalignment on four repeatedly used acceptance seeds.','Current question: generalization to readiness-qualified independent seeds.','Why: previous fixed-seed evidence is not a policy-level guarantee.','Existing data sufficient? NO. No historical disjoint4successful set found; readiness-only ascending seed selection preregistered.','Experiment: exact saved old/full weights; fixed fractions; frozen readiness Critic/replay/Qprobe; no training; exactly4parallel envs.','','Results: '+criterion+' by task criterion. '+label+'.','',reason,'',ambiguity,'','|Block|Fraction|Acceptance success|Independent success|Independent theta_old success|','|---|---:|---:|---:|---:|']
 for t in table:lines.append(f"|{t['block']}|{t['fraction']}|{t['historical_acceptance_success']}/4|{t['independent_success']}/4|{t['old_independent_success']}/4|")
 lines+=['','Evidence type: paired policy-level CAUSAL intervention plus NEGATIVE controls; not yet a measured action derivative.','What ruled out: '+('pure acceptance-seed specificity for the candidate-versus-BC deterioration' if criterion=='YES' else 'unqualified universal local misalignment claim'),'Current top hypothesis: '+('value/ranking improvement can fail to constrain policy-level derivative; whether action slope itself is wrong remains unproven' if decision!='STOP' else 'seed-dependent competence sensitivity'),'Decision: '+decision,'',json.dumps(a,indent=2)]
 (OUT/'ROUND_REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(a,allow_nan=False),flush=True)
if __name__=='__main__':main()
