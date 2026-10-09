"""Testing-only semantic and evidence guards; never opens production Actor gate."""
import numpy as np

def classify(strict,matched,dq,delta_g,identified,ge=1e-4,qe=1e-6):
 if not strict:return 'INVALID_COUNTERFACTUAL'
 if not matched:return 'SEMANTICS_MISMATCH'
 if len(delta_g)==0:return 'STATISTICALLY_UNRESOLVED'
 if np.all(np.abs(delta_g)<=ge):return 'REAL_RETURN_TIE'
 if not identified or abs(dq)<=qe:return 'STATISTICALLY_UNRESOLVED'
 return 'CORRECT_RANKING' if dq*float(np.mean(delta_g))>0 else 'WRONG_RANKING'

def acceptance_status(pair_results):
 valid=[p for p in pair_results if p['identifiable']]
 wrong=sum(any(p['critics']['ready'][h]['classification']=='WRONG_RANKING' for h in ['Q1','Q2']) for p in valid)
 return {'status':'NOT_CERTIFIED','reason':'REPEATED_SIGN_CONFLICT_UNDER_PER_PAIR_EXPLORATORY_CRITERIA' if wrong else ('NO_IDENTIFIABLE_ADVANTAGE' if not valid else 'LIMITED_NO_FAILURE_EVIDENCE_NOT_A_GENERAL_PASS'),'identified_pairs':len(valid),'wrong_pairs':wrong,'production_gate_changed':False,'needs_independent_context_coverage_and_predeclared_power':True}
