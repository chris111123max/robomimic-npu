import sys,json,itertools,collections,math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest,t as student
HERE=Path(__file__).resolve().parent
from readiness_metrics import classify,acceptance_status

def analyze(stage):
 out=HERE/stage;d=json.loads((out/'result.json').read_text());reg=json.loads((out/'preregistration.json').read_text());eps=d['episodes'];threshold=reg['mc_identifiability_threshold'];qe=reg['q_direction_epsilon'];pairs=[]
 groups=collections.defaultdict(dict)
 for e in eps:groups[e['seed']].setdefault(e['branch'],{})[e['future_rng']]=e
 for seed,branches in groups.items():
  for a,b in itertools.combinations(reg['unique_candidate_branches'],2):
   futures=reg['future_rng_seeds'];aa=[branches[a][x] for x in futures];bb=[branches[b][x] for x in futures]
   assert all(x['pair']==y['pair'] for x,y in zip(aa,bb))
   delta=np.array([y['mc_from_fork']-x['mc_from_fork'] for x,y in zip(aa,bb)]);mean=float(delta.mean());n=len(delta);sd=float(delta.std(ddof=1)) if n>1 else None
   ci=[mean-float(student.ppf(.975,n-1))*sd/math.sqrt(n),mean+float(student.ppf(.975,n-1))*sd/math.sqrt(n)] if n>1 else None
   nonzero=delta[np.abs(delta)>threshold];pos=int((nonzero>0).sum());p=float(binomtest(pos,len(nonzero),.5).pvalue) if len(nonzero) else None
   tie=bool(np.all(np.abs(delta)<=threshold));identifiable=bool(not tie and p is not None and p<=reg['alpha'] and ci is not None and (ci[0]>threshold or ci[1]<-threshold))
   heads={}
   for c in ['ready','MC']:
    heads[c]={}
    for h in ['Q1','Q2','Qmean']:
     dq=bb[0]['q'][c][h]-aa[0]['q'][c][h]
     label=classify(True,c=='ready',dq,delta,identifiable,threshold,qe)
     realized=[]
     for x in delta:
      realized.append('REAL_RETURN_TIE' if abs(x)<=threshold else ('STATISTICALLY_UNRESOLVED' if abs(dq)<=qe else ('CORRECT_RANKING' if dq*x>0 else 'WRONG_RANKING')))
     if c=='MC':realized=['SEMANTICS_MISMATCH']*len(delta)
     heads[c][h]={'delta_q':dq,'classification':label,'advantage_error':abs(dq-mean) if c=='ready' else None,'realized_direction_descriptive':realized,'semantics_matched':c=='ready'}
   pair={'seed':seed,'group':aa[0]['group'],'fork':aa[0]['fork'],'actions':[a,b],'future_rng_seeds':futures,'delta_g_samples':delta.tolist(),'delta_g_mean':mean,'mean_t_interval_95_approx':ci,'sign_test_two_sided_p':p,'real_return_tie':tie,'identifiable':identifiable,'critics':heads,'twin_wrong_realizations':int(sum(x<-threshold and heads['ready']['Q1']['delta_q']>qe and heads['ready']['Q2']['delta_q']>qe for x in delta))}
   pairs.append(pair)
 summary={}
 for label,subset in [('all',pairs)]+[(g,[x for x in pairs if x['group']==g]) for g in sorted({x['group'] for x in pairs})]:
  valid=[x for x in subset if x['identifiable']];z={'context_count':len({x['seed'] for x in subset}),'unique_action_pairs':len(subset),'identifiable_pairs':len(valid),'real_return_tie_pairs':sum(x['real_return_tie'] for x in subset),'unresolved_non_tie_pairs':sum(not x['real_return_tie'] and not x['identifiable'] for x in subset),'tie_fraction':sum(x['real_return_tie'] for x in subset)/len(subset),'ready':{}}
  for h in ['Q1','Q2','Qmean']:
   count=collections.Counter(x['critics']['ready'][h]['classification'] for x in subset);correct=count['CORRECT_RANKING'];z['ready'][h]={'classifications':dict(count),'pairwise_accuracy':correct/len(valid) if valid else None,'denominator':len(valid),'mean_advantage_error_all_pairs':float(np.mean([x['critics']['ready'][h]['advantage_error'] for x in subset])),'mean_advantage_error_identifiable':float(np.mean([x['critics']['ready'][h]['advantage_error'] for x in valid])) if valid else None}
  z['twin_wrong_realizations']=sum(x['twin_wrong_realizations'] for x in subset);z['twin_common_wrong_identifiable']=sum(all(x['critics']['ready'][h]['classification']=='WRONG_RANKING' for h in ['Q1','Q2']) for x in valid);summary[label]=z
 result={'stage':stage,'pairs':pairs,'summary':summary,'conditional_mean_interval_note':'Small-sample t intervals are approximate and assume iid future-stream differences; paired sign test tests median,not mean;intersection is conservative evidence guard,not an exact nonparametric confidence bound on the mean. No production gate threshold derived.','MC':'All candidate comparisons SEMANTICS_MISMATCH with mixed behavior target;no MC accuracy or absolute advantage error asserted.','B_READINESS':'Exact A alias,excluded as independent pair.'}
 result['testing_readiness']=acceptance_status(pairs)
 (out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(summary['all'],indent=2))
if __name__=='__main__':analyze(sys.argv[1])
