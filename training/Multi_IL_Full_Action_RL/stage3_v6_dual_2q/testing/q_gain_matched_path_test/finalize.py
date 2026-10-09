"""Read all actual outputs and generate final report, no experiments."""
import sys,json,time,hashlib,subprocess
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,pearsonr
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 assert not p.exists(),p
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def finite(x):
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 if isinstance(x,float):return bool(np.isfinite(x))
 return True
def main():
 a=read(HERE/'stage_a.json');b=read(HERE/'stage_b.json');c=read(HERE/'stage_c.json');design=read(HERE/'design.json')
 audit=read(HERE/'safety_before.json');after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in audit}
 assert audit==after
 assert c['contract']['test_valid']
 assert all(c['contract'][k]==4 for k in ['parallel_envs_initialized','parallel_envs_used','parallel_envs_closed'])
 table=[];comparisons={}
 for label,ref in a['references'].items():
  ref['behavior']=c['behavior']['UNRESTRICTED_LOW'] if label=='LOW' else ref['behavior']
  table.append({'target':label,'branch':'UNRESTRICTED','metrics':ref,'behavior':ref['behavior'],'behavior_source':'NEW Stage C' if label=='LOW' else 'EXISTING exact same four-env evaluator/seeds'})
  if label=='START':
   table.append({'target':label,'branch':'LOCAL_CONSTRAINED','metrics':ref,'behavior':ref['behavior'],'behavior_source':'SAME shared START'})
  elif label in b['matches']:
   r=b['matches'][label];beh=c['behavior']['CONSTRAINED_'+label]
   table.append({'target':label,'branch':'LOCAL_CONSTRAINED','metrics':r,'behavior':beh,'behavior_source':'NEW Stage C'})
   x=np.load(HERE/('unrestricted_'+label.lower()+'_probe.npz'))
   y=np.load(HERE/('constrained_'+label.lower()+'_probe.npz'))
   corr={}
   for twin in ('q1','q2'):
    dx=x[twin+'_expected_current']-x[twin+'_expected_init']
    dy=y[twin+'_expected_current']-y[twin+'_expected_init']
    corr[twin]={'spearman':float(spearmanr(dx,dy).statistic),'pearson':float(pearsonr(dx,dy).statistic)}
   comparisons[label]={'q1_relative_error':r['relative_matching_error'],'q2_relative_error':abs(r['q']['all']['q2_gain']/ref['q']['all']['q2_gain']-1),'qmean_relative_error':abs(r['q']['all']['qmean_gain']/ref['q']['all']['qmean_gain']-1),'context_correlations':corr,'unrestricted_success':ref['behavior']['success_count'],'constrained_success':beh['success_count']}
 if not b['reached_collapse']:
  r=b['unmatched_endpoint'];table.append({'target':'UNMATCHED_ENDPOINT','branch':'LOCAL_CONSTRAINED','metrics':r,'behavior':c['behavior']['CONSTRAINED_UNMATCHED_ENDPOINT'],'behavior_source':'NEW Stage C'})
  classification='CONSTRAINT_PREVENTS_MATCHED_Q_PROGRESS'
 elif comparisons['COLLAPSE']['unrestricted_success']==0 and comparisons['COLLAPSE']['constrained_success']>=3:
  classification='Q_MATCHED_OPTIMIZATION_PATH_CAUSAL'
 elif comparisons['COLLAPSE']['unrestricted_success']==0 and comparisons['COLLAPSE']['constrained_success']==0:
  classification='Q_MATCHED_COLLAPSE_REAPPEARS'
 else:classification='MIXED_OR_INCONCLUSIVE'
 json_files={str(p.relative_to(HERE)):read(p) for p in HERE.rglob('*.json')}
 jsonl_audit={}
 for p in HERE.rglob('*.jsonl'):
  count=0
  with p.open() as f:
   for l in f:
    row=json.loads(l);assert finite(row),(p,count);count+=1
  jsonl_audit[str(p.relative_to(HERE))]=count
 assert all(finite(v) for v in json_files.values())
 process_output=subprocess.check_output(['ps','-eo','pid,args'],text=True)
 residual=[l for l in process_output.splitlines() if ('run_test.py' in l or 'train_stage3' in l or 'multiprocessing.spawn' in l) and 'ps -eo' not in l and 'bash -' not in l]
 npu=subprocess.check_output(['npu-smi','info'],text=True)
 (HERE/'npu_smi_final.txt').write_text(npu)
 safety={'production_and_formal_hashes_unchanged':audit==after,'hash_count':len(after),'after_hashes':after,'formal_training_resumed':False,'device':'npu:0 only','parallel_envs_initialized':4,'parallel_envs_used':4,'parallel_envs_closed':4,'worker_exitcodes':c['contract']['worker_exitcodes'],'residual_experiment_processes':residual,'npu_no_running_processes':'No running processes found' in npu,'jsonl_read_audit':jsonl_audit,'all_numeric_outputs_finite':True,'fusion_result_unchanged_existing_D':not (HERE.parents[4]/'fusion_result.json').exists(),'testing_bug_fixed':'star import overwrote HERE; reset new wrapper HERE after imports; newly generated evidence file moved into authorized new directory; no historical file overwritten'}
 assert not residual,residual
 assert safety['npu_no_running_processes']
 dump(HERE/'safety_final.json',safety)
 if classification=='Q_MATCHED_OPTIMIZATION_PATH_CAUSAL':
  answer='At matched fixed-Critic Q progress, the local-constraint path preserves substantially more success; reduced Q progress alone cannot explain the prior Round5 rescue.'
  nextstep='Validate this exact local constraint in one narrowly controlled testing-only online continuation, before considering production changes; not executed.'
 elif classification=='Q_MATCHED_COLLAPSE_REAPPEARS':
  answer='Round5 survival is consistent with delayed collapse from reduced learning: after matching Q progress, the constrained Actor also loses all four successes.'
  nextstep='Test whether an explicit real-behavior acceptance criterion can prevent damage at comparable Q progress; not executed.'
 elif classification=='CONSTRAINT_PREVENTS_MATCHED_Q_PROGRESS':
  answer='Round5 survival cannot be separated from Q-progress suppression: the unchanged constraint did not attain collapse-matched Q within the predeclared cap.'
  nextstep='No new mechanism conclusion; first assess whether a larger fixed computational budget is warranted for this exact matched-Q comparison, without changing the constraint; not executed.'
 else:
  answer='Matched-Q comparison is mixed and cannot cleanly distinguish safer path from delayed collapse.'
  nextstep='Repeat only the same matched-Q comparison on an independently fixed evaluation seed set; not executed.'
 summary={'classification':classification,'answer':answer,'table':table,'comparisons':comparisons,'stage_a':a,'stage_b':b,'stage_c':c,'design':design,'learning_suppression_check':{'constrained_learned':b['actor_updates']>0,'reached_collapse_q':b['reached_collapse'],'final_q':b['last_q'],'update_count':b['actor_updates'],'interpretation':answer},'safety':safety,'confidence_scope':'These four selected BC-success seeds, frozen Critic/replay mechanism only; no unique upstream Critic-defect claim and no formal/online validation','next_step_only':nextstep,'formal_training_remains_stopped':True}
 dump(HERE/'final_summary.json',summary)
 lines=['# Q-GAIN MATCHED CAUSAL COMPARISON','',answer,'','Final classification: **'+classification+'**.','','## Main comparison','','|Q target|Branch|Actual deltaQ1|deltaQ2|deltaQmean|Actor updates|Success|Weighted action drift|Component-mean RMS|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for row in table:
  r=row['metrics'];q=r['q']['all'];d=r['drift'];beh=row['behavior']
  lines.append('|'+ '|'.join([row['target'],row['branch'],f"{q['q1_gain']:.10g}",f"{q['q2_gain']:.10g}",f"{q['qmean_gain']:.10g}",str(r['actor_updates']),str(beh['success_count'])+'/4',f"{d['weighted_action_l2_mean']:.10g}",f"{d['component_mean_rms']:.10g}"])+'|')
 lines+=['','Primary matching metric: mean expected Q1 gain on the SAME 256 fixed replay contexts using the SAME frozen readiness Critic, aligned10-history and identical action scaling. The probe never moves. Tolerance predeclared5%; no post-result tuning.','','## Matching and context-wise improvement']
 for label,x in comparisons.items():lines+=['',label+': '+json.dumps(x)]
 lines+=['','## LEARNING SUPPRESSION CHECK','',json.dumps(summary['learning_suppression_check'],indent=2),'','Drift diagnostics (parameter L2, common-RNG sampled drift, weighted drift, component mean RMS, logits RMS, categorical KL and hidden L2):']
 for row in table:lines+=['',row['target']+'/'+row['branch']+': '+json.dumps(row['metrics']['drift'])]
 lines+=['','## Fixed-probe distributions','']
 for row in table:lines+=['',row['target']+'/'+row['branch']+': '+json.dumps(row['metrics']['delta_distribution'])]
 lines+=['','Positive fractions and context correlations are reported to distinguish broad improvement from outlier-dominated means; no new upstream mechanism probe was run.','','## Per-seed closed-loop results','']
 for row in table:lines+=['',row['target']+'/'+row['branch']+' ('+row['behavior_source']+'): '+json.dumps(row['behavior']['episodes'])]
 lines+=['','## Exact intervention and provenance','',json.dumps(design,indent=2),'','Stage A read all prior JSON/report content, recomputed existing625/1250 actor fixed-probe Q and exactly reproduced old results. Only missing250 reference weights were reconstructed (250 new Actor updates). Unrestricted1250 was previously reevaluated0/4 in Round4. No full unrestricted retraining.','Stage B offline Actor-only optimization from SAME readiness actor/optimizer/replay/bank: '+str(b['actor_updates'])+' updates, Critic0; matches saved synchronously AFTER accepted parameter step. This is not online training.','Stage C launched ONE exactly-four-worker pool and evaluated checkpoints SEQUENTIALLY; no optimizer during rollout. Existing START/MID/COLLAPSE evaluations were reused, LOW and constrained evaluations are new.','','## Actual execution and cost','',c['command'],'','Simulator initialized/used/closed4/4/4; worker exits '+str(c['contract']['worker_exitcodes'])+'. New environment steps='+str(c['total_env_steps'])+'; Actor updates during simulator=0; simulator wall-clock seconds='+str(c['runtime_seconds'])+'.','Stage B wall-clock seconds='+str(b['elapsed_seconds'])+'. Total NEW offline Actor updates='+str(250+b['actor_updates'])+'; new Critic updates=0; formal env/optimizer steps=0.','','## Interpretation and limitations','',answer,'','MatchingQ, twin agreement and local-path intervention can establish a path-dependent mechanism but cannot identify any unique upstream Critic defect. Four selected seeds are not a population estimate. More updates deliberately allowed; nominal env count is not the matching criterion. Schedule extension reuses prior interpolation with saturated endpoint as predeclared, not a claim to reproduce future formal training.','If both paths collapse at this matched endpoint, that supports reduced-learning delay as the Round5 explanation, not proof all possible paths necessarily collapse or unique causal attribution to Q targets.','','## ONE NEXT STEP (not executed)','',nextstep,'','## Safety','',json.dumps(safety,indent=2),'','FORMAL TRAINING REMAINS STOPPED']
 (HERE/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
 print('FINAL',classification,json.dumps(comparisons),flush=True)
 print('REPORT',HERE/'FINAL_REPORT.md')
if __name__=='__main__':main()
