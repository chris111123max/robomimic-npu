"""Read all generated results, validate finite data and safety, write final report."""
import sys,json,math,hashlib,subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
r1=read(HERE/'round1/result.json');r2=read(HERE/'round2/result.json');r3=read(HERE/'round3/result.json');r4=read(HERE/'round4/result.json')
count=[0];files=[]
def finite(x):
 if isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
 elif isinstance(x,(float,int)) and not isinstance(x,bool):
  count[0]+=1;assert math.isfinite(x)
for p in HERE.rglob('*.json'):
 finite(read(p));files.append(str(p))
for p in HERE.rglob('*.jsonl'):
 for l in p.read_text().splitlines():finite(json.loads(l))
 files.append(str(p))
for p in HERE.rglob('*.npz'):
 a=np.load(p)
 assert all(np.isfinite(a[k]).all() for k in a.files)
old=read(HERE/'safety_before.json')
new={k:hashlib.sha256(Path(k).read_bytes()).hexdigest() for k in old}
assert old==new
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True)
formal=[l for l in processes.splitlines() if ('train_stage3_v6_vector.py' in l or 'train_stage3_v5_vector.py' in l) and 'grep' not in l]
diagnostics=[l for l in processes.splitlines() if ('diagnose.py round' in l or 'run_fork.py' in l) and 'grep' not in l]
assert not formal and not diagnostics,(formal,diagnostics)
npu=subprocess.check_output(['npu-smi','info'],text=True,timeout=30)
assert 'No running processes found' in npu,npu
(HERE/'npu_smi_final.txt').write_text(npu)
for r,key in [(r3,'env_contract'),(r4,'env_contract')]:
 c=r[key];assert c['test_valid']
 assert all(c[k]==4 for k in ['parallel_envs_initialized','parallel_envs_used','parallel_envs_closed'])
 assert all(x==0 for x in c['worker_exitcodes'])
b=r3['branches']['BASELINE']['metrics']['1250']
i=r3['branches']['SUPPORT_RADIAL_REJECTION']['metrics']['1250']
retention={k:i['q']['all'][k]/b['q']['all'][k] for k in ['q1_gain','q2_gain','qmean_gain']}
retention['sampled_action_drift']=i['policy_drift']['sampled_action_l2_mean']/b['policy_drift']['sampled_action_l2_mean']
retention['weighted_action_drift']=i['policy_drift']['weighted_action_l2_mean']/b['policy_drift']['weighted_action_l2_mean']
retention['parameter_drift']=i['parameter_drift_unique_l2']/b['parameter_drift_unique_l2']
safety={'production_and_formal_checkpoint_hashes_unchanged':old==new,'sha256_file_count':len(old),'hashes':new,'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'fusion_result_exists':Path('fusion_result.json').exists(),'formal_processes':formal,'diagnostic_processes':diagnostics,'npu_processes':0,'numeric_fields_checked':count[0],'all_finite':True,'new_actor_updates':2500,'new_critic_updates':0,'diagnostic_environment_steps':sum(e['length'] for br in r3['behavior'].values() for e in br['episodes'])+sum(e['length'] for br in r4['branches'].values() for e in br['episodes']),'formal_environment_steps':0,'formal_optimizer_steps':0}
summary={'primary_mechanism':'UNCONSTRAINED_Q_ASCENT_WITH_CUMULATIVE_POLICY_DISPLACEMENT','confirmed_execution_channel':'component-mean / executed-action displacement','upstream_origin_verdict':'INCONCLUSIVE','confidence':'MEDIUM','strong_local_causal_evidence_from_previous_rounds':True,'new_support_rejection_rescue':False,'rounds':4,'round0_completed':True,'stop_reason':'four-round cap; no clean upstream causal rescue; no tuning','round1_successful_contexts':r1['contexts'],'round1_q_landscape':r1['scores'],'round2_support':r2['metrics'],'round3_final':r3,'round3_retention':retention,'round4_single_action_fork':r4,'unsupported_primary_causes':['bootstrap uniquely causes collapse','online feedback necessary','all initial local gradients destructive','conditional replay distance is sufficient root-cause predictor'],'next_only':'paired same-state finite-MC test of short action blocks, limited contexts, to locate onset of cumulative harm (not executed)','safety':safety,'report':str(HERE/'FINAL_REPORT.md'),'formal_training_remains_stopped':True}
for name,value in [('safety_final.json',safety),('final_summary.json',summary)]:
 p=HERE/name;assert not p.exists();p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
forklines=[]
for j,e in enumerate(r4['branches']['1.0']['episodes']):
 e0=r4['branches']['0.0']['episodes'][j]
 forklines.append(f"|{e['seed']}|{e0['length']} -> {e['length']}|{r4['branches']['1.0']['paired_delta_qmean'][j]:+.8f}|{r4['branches']['1.0']['paired_delta_mc'][j]:+.8f}|SUCCESS|")
landscape=[]
for row in r1['scores']['sampled']:
 landscape.append(f"|{row['alpha']}|{row['qmean']:.8f}|{row['delta_q1']:+.8f}|{row['delta_q2']:+.8f}|{row['both_increase_fraction']:.2%}|")
txt=f"""# Stage3-v6 mean2q / multi_q upstream root-cause diagnosis

Stage3 mean2q/multi_q collapses because repeated unconstrained learned-Q ascent accumulates policy/action displacement that destroys successful closed-loop behavior; this diagnosis does NOT establish off-support Critic extrapolation or bootstrap bias as the uniquely primary upstream origin.

Best-supported mechanism: **UNCONSTRAINED_Q_ASCENT_WITH_CUMULATIVE_POLICY_DISPLACEMENT**.
Overall confidence: **MEDIUM**. The narrower upstream question remains **INCONCLUSIVE**, not HIGH.
Round0 + four new rounds completed; STOP at the cap. No production algorithm change or formal training.

## Round summary

|Round|Hypothesis|Main test|Result|Causal status|
|---|---|---|---|---|
|0|Integrate formal and prior evidence|Read checkpoint/log metadata and historical tests|Ready actor intact; formal collapse150-160K|Prior frozen and retraction causal evidence retained|
|1|Both critics reward destructive displacement|Six-point sampled/weighted lines on180 successful reference contexts|Joint Q gain179/180 sampled and180/180 weighted atalpha1|Landscape correlational; reused prior action rollback causal|
|2|Drift exits local replay action support|Conditional NN proxies K8/32/64|Sampled NN distance rises; only14.4% outside K32 proxy atalpha1|Correlation, not necessary/sufficient support boundary|
|3|Reject off-support outward mean gradient|Paired baseline/intervention,1250 updates each, frozen Critic/replay|Both0/4; intervention retains96.5% Q gain|No upstream rescue; tested proxy/rejection insufficient|
|4|Already wrong local ordering vs cumulative damage|Same-state one-action fork, then BC continuation|All alpha0/.25/1 retain4/4; alpha1 improves2seeds, worsens1, flat1|Local intervention; rejects blanket immediate-harm claim at these states|

## Formal timeline and causal background

Only mean2q/multi_q from formal run:
{RUN if 'RUN' in globals() else '/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_readiness_v2_multi_mean_random_formal_20260929_130945'}

Formal ready:140000env,0actor,34750critic updates.
Formal200K:3743actor/49722critic;280K:8750/69750;last315904:10994/78726.
No original formal150K/160K checkpoints. Earlier testing baseline615/1246 actor updates at150K/160K had2/4 and0/4; label explicitly testing, not original formal artifacts.
Formal completed online success:140-150K10/11,150-160K2/15,160-200K0/58; last success153973.
Formal280K evaluation0/10,all700,truncated,no simulator errors.

Previous exact mean frozen-ready experiment: actor625 gives3/4,1250 gives0/4 while fixed replay and Critic. Q1 gain+.00529342,Q2+.00533751;128/128 successful contexts joint gain. Critic/replay coevolution is NOT necessary.
Previous executed-action rollback of IDENTICAL failed weights: alpha.25/.5/.75 yields3/4,1/4,0/4. This is strong causal evidence for accumulated execution displacement, not an upstream Critic repair.

Historical consistency:
- Objective alignment: ready Q1 vs mean/min gradientcos approximately.991/.990; not Q1-only disagreement at onset.
- Temporal supervision: CURRENT,ALL_ALIGNED,FREEZE_RNN previously all collapse; final-token-only or RNN updates not necessary.
- Whole-policy/adaptive preservation: protect success but retain approximately.75-.97% learned-Q gain, near-freeze; do not count as clean upstream repair.
- Adam geometry: prior random2q tests also collapse under global-step-matched SGD; background only, no new random experiment or mean-specific claim.
- Module sensitivity/component-mean preservation: prior random evidence identifies a strong output channel, not a proven mean upstream cause; protection delays not permanently solves. No fresh module/optimizer tests.
- Exact latest mean frozen-Critic and action-retraction results anchor this report.

## Round1: actual drift landscape on successful trajectories

180 aligned ten-token windows from four successful READY trajectories. Fixed observations, executed-action history and progress for all alphas.
Sampled pairs use common RNG within probe (20261003). Prior full-policy behavioral interpolation uses20007; values here are NOT exact realized return estimates for all candidate actions.
Weighted line provides a second representative action; neither equals the full GMM expected production objective. Recorded MC belongs ONLY to actual reference trajectory.

|alpha|Qmean|deltaQ1|deltaQ2|joint positive|
|---|---:|---:|---:|---:|
{chr(10).join(landscape)}

Weighted alpha1 Q1 gain+.00739786,Q2+.00720601,180/180joint positive.
Thus joint learned optimism along the real destructive displacement is widespread; small twin disagreement does not validate action improvement.
However successful reference-state action scores and full-policy success on newly visited states are different measurements; no direct per-state true-Q sign claimed here.

## Round2: behavior-support proxy

Frozen readiness replay + actual three offline sources,64 production-shaped batches, deduplicated transitions. Query neighborhoods by standardized current observation distance plus normalized episode progress.
Action distances normalized by production action scale; local density calibration uses leave-one-neighbor-out nearest action distance p95.
No exact manifold/density estimator, no exhaustive replay search, no claim state matching is exact.

K32 sampled reference -> failed:
- nearest local replay action mean distance .123616 -> .192435;
- outside NN-p95 proxy 2.78% ->14.44%;
- outside broad local convex ball3.89% ->8.89%.
K8 and K64 show similar distance increase. Weighted distance .227929 -> .253442, weaker.
Most measured actions remain within this proxy. This is NOT evidence that a sharply defined support exit is the necessary trigger.

The initial unexecuted code draft used a convex ball; Round2 showed it covers separated action modes/empty regions. Before any intervention run, chose union of local nearest-action balls, K32, empirical NN-p95 radius. Single intervention, no strength sweep.

## Round3: one selective causal intervention, no rescue

Same ready actor, same64 batches, same production Q1 objective, same LR mapping from actual formal update counts, same frozen Critic. Baseline and support-gradient rejection run sequentially on npu:0.
For each mean output outside the conditional replay support proxy, remove ONLY the gradient component that would move farther from the nearest local replay action. Inward/tangential mean gradients and probability gradients remain.
This is NOT an anchor loss, whole-policy freeze, objective replacement or production change.
1250 updates each, implied schedule endpoint160047,not actual online continuation. New Critic updates0; therefore no claim to preserved TD learning or full online rescue.

|Branch|Success|Qmean gain|sampled drift|weighted drift|unique parameter L2|
|---|---:|---:|---:|---:|---:|
|Baseline|0/4|{b['q']['all']['qmean_gain']:.8f}|{b['policy_drift']['sampled_action_l2_mean']:.6f}|{b['policy_drift']['weighted_action_l2_mean']:.6f}|{b['parameter_drift_unique_l2']:.6f}|
|Support rejection|0/4|{i['q']['all']['qmean_gain']:.8f}|{i['policy_drift']['sampled_action_l2_mean']:.6f}|{i['policy_drift']['weighted_action_l2_mean']:.6f}|{i['parameter_drift_unique_l2']:.6f}|

123314/400000mean-gradient vectors have outward component removed (30.83%); average norm removed7.25%.
Q gain retention {retention['qmean_gain']:.2%};sampled drift {retention['sampled_action_drift']:.2%};parameter drift {retention['parameter_drift']:.2%}.
NOT LEARNING_SUPPRESSION; nevertheless no rescue. Endpoint-only evaluation cannot claim precise collapse timing unchanged.
This rejects sufficiency of this density proxy/output-gradient rejection. It does not falsify every off-support mechanism: nearest-state proxy approximate, GMM probabilities unchanged, parameter updates can couple outputs, and output-gradient projection is not a strict constraint on the optimizer's actual parameter step. No optimizer-geometry experiment or tuning followed.

## Round4: true single-action fork advantage

Fixed t199 chosen once, no seed-specific tuning. Before the fork all reference histories exactly match across branches (max observed-history difference0).
Only one action is interpolated along the same failed Actor direction; then original BC continues on its own visited observations. Same-RNG action sampling at20007,learned Q scored on the EXACT fork actions.
Same physical initial seeds and deterministic action prefixes; no serialized simulator-state restoration.
Each alpha0,.25,1 run uses all four genuinely concurrent workers.

All branches4/4 success. Alpha.25 all episode lengths and finite-MC returns unchanged despite joint Q increases.
Alpha1 true remaining finite-MC deltas:
|seed|baseline -> fork length|learned deltaQmean|real deltaMC|success|
|---|---:|---:|---:|---|
{chr(10).join(forklines)}

Mean alpha1 real deltaMC {np.mean(r4['branches']['1.0']['paired_delta_mc']):+.8f};all four learned Q1/Q2 increases.
Two improvements,one adverse duration effect,one tie. Cannot call the local gradient universally destructive.
This is a clean LOCAL action intervention, but only four fixed states/single-seed continuations, not a population estimator or entire-policy causal rescue.
Contrast: same failed actor continuously executed previously0/4; one isolated failed-actor action followed by BC4/4. Sustained/coordinated displacement matters; single local Q ordering alone does not explain whole collapse.

## Four-layer attribution

PRIMARY UPSTREAM CAUSE:
- Confirmed at mechanism level: repeated Q ascent lacks demonstrated behavior-improvement validity and accumulates harmful policy displacement.
- More specific off-support action extrapolation as the origin: remains INCONCLUSIVE. Proxy correlation and failed rejection are insufficient.
- No proof initial local gradients universally wrong; data also fit initially tolerable/useful perturbations followed by cumulative loss of the policy's valid neighborhood.

ACTOR-LEVEL MECHANISM:
- Repeated unconstrained ascent of approximate Q increases learned values on reference/replay states while the induced closed-loop policy loses competence.
- State/return readiness ranking is not a certificate for action counterfactual ranking or multi-step policy improvement (inference consistent with tests).

EXECUTION CHANNEL:
- Component-mean/executed-action displacement; prior identical-weight rollback causal dose response. Not parameter L2 alone.

DOWNSTREAM AMPLIFIERS:
- Online replay shift/later Critic evolution are possible amplifiers, NOT necessary triggers.
- Bootstrap/Q-scale and later twin divergence remain candidate amplifiers; no new causal bootstrap test, no primary attribution.
- RNN/logit effects not separately established in this round, no extra module claim.

## Causal boundary and STOP

We have causal evidence for frozen learned-Q optimization sufficiency and action displacement, plus a negative support-rejection intervention and mixed local-MC fork.
We do NOT have a successful new upstream intervention that preserves meaningful learning and rescues collapse.
Therefore overall MEDIUM; unique upstream provenance INCONCLUSIVE. Do not upgrade to HIGH or claim BOOTSTRAP_IS_PRIMARY_ROOT_CAUSE.
Round4 cap reached. No fifth round, no intervention-strength tuning, no fix.

Only useful next experiment (NOT executed): limited same-state finite-MC forks for short action blocks, pairing small and cumulative displacement, to test when repeated action replacement turns local tolerability into real harm. This should distinguish cumulative multi-step policy damage from uniformly wrong initial action gradients, not reopen Adam/module/anchor sweeps.

## Environment contracts, safety and artifacts

Round3:device=npu:0,initialized=used=closed=4,seeds[20008,20002,20005,20007],worker exitcodes[0,0,0,0],sim errors0.
Round4:same contract,4/4/4,all worker exitcodes0,sim errors0.
Round1/2:no NEW simulation; old behavioral outputs reused with documented valid4-worker contracts.
Exactly one experiment active at a time,branches sequential.
New diagnostic env steps {safety['diagnostic_environment_steps']},Actor updates2500,Critic updates0,20diagnostic episodes;formal env/optimizer steps0.
{len(old)} production/formal checkpoint SHA256 hashes unchanged;all JSON/JSONL fully parsed,{count[0]}numeric fields finite,NPZ finite.
HEAD {safety['head']};fusion_result.json absent/D at entry and finish,untouched.
No formal/diagnostic training process;NPU process list empty.
All new code/results under this testing directory. Pre-execution testing-only wildcard import output-path override corrected; no runtime test failure or production repair.

Results:
- round0/existing_evidence.json
- round1/result.json,successful_contexts.npz
- round2/result.json,support_raw.npz
- round3/result.json,offline.json,closed_loop.json,branch traces and trajectories
- round4/result.json,three fork trajectory files
- each round ROUND_REPORT.md
- final_summary.json,safety_before.json,safety_final.json,npu_smi_final.txt

FORMAL TRAINING REMAINS STOPPED
"""
(HERE/'FINAL_REPORT.md').write_text(txt)
(HERE/'round4/ROUND_REPORT.md').write_text("""# Round 4
Hypothesis: readiness Critic locally ranks destructive actions higher before accumulated policy displacement.
Test: same-state single-action fork at t199,alpha0/.25/1,then unchanged BC continuation; four known-good seeds in true four-worker pool.
Results: all branches4/4. Smallalpha.25 MC deltas allzero;alpha1 two improve,one worsens,one tie. Both Qs increase4/4. History matching exact before fork.
Explains original collapse: points toward cumulative sustained policy displacement, not universally harmful isolated local action. Does not establish every initial gradient reliable.
Before collapse: reference states and ready Critic, no new optimizer updates.
Causal status: clean local execution intervention,mixed MC evidence; global upstream cause remains INCONCLUSIVE. More than one action might be needed to induce failure; not tested.
Next single test: finite-MC short action blocks,not executed under four-round cap.
Contract:npu:0,seeds20008/20002/20005/20007,initialized=used=closed4,worker exits0,sim errors0.
DECISION: STOP
Reason: fourth round completed; no clean upstream rescue; no tuning or fifth experiment.
""")
print(json.dumps({'summary':str(HERE/'final_summary.json'),'report':str(HERE/'FINAL_REPORT.md'),'retention':retention,'safety':{k:v for k,v in safety.items() if k!='hashes'},'round4_mean_delta_mc':float(np.mean(r4['branches']['1.0']['paired_delta_mc']))},indent=2))
