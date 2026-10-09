"""Final read/analysis and safety verification, no experiment."""
import json,math,hashlib,subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
TEST=HERE.parent
def read(p):return json.loads(p.read_text())
rs={str(n):read(HERE/f'round{n}/result.json') for n in range(6)}
base=read(TEST/'mean_multi_collapse_diagnosis/round1/result.json')['metrics']['1250']
r4=rs['4'];r5=rs['5'];m4=r4['metrics']['1250'];m5=r5['offline']['metrics']['1250']
qret4=m4['q']['all']['qmean_gain']/base['q']['all']['qmean_gain']
ret5={'qgain':r5['qgain_retention'],'sampled_action':m5['policy_drift']['sampled_action_l2_mean']/base['drift']['sampled_action_l2_mean'],'weighted_action':m5['policy_drift']['weighted_action_l2_mean']/base['drift']['weighted_action_l2_mean'],'component_mean':m5['policy_drift']['component_mean_rms']/base['drift']['component_mean_rms']}
before=read(HERE/'safety_before.json');after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before};assert before==after
numbercount=[0]
def finite(x):
 if isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,list):
  for v in x:finite(v)
 elif isinstance(x,(int,float)) and not isinstance(x,bool):
  numbercount[0]+=1;assert math.isfinite(x)
files=[]
for p in HERE.rglob('*.json'):
 finite(read(p));files.append(str(p))
for p in HERE.rglob('*.jsonl'):
 for line in p.read_text().splitlines():finite(json.loads(line))
 files.append(str(p))
for p in list(HERE.rglob('*.npz'))+list(HERE.rglob('*.npy')):
 arrays=np.load(p)
 if p.suffix=='.npy':assert np.isfinite(arrays).all()
 else:
  for k in arrays.files:
   if arrays[k].dtype.kind in 'biufc':assert np.isfinite(arrays[k]).all()
workers=[]
for n in (1,4,5):
 c=rs[str(n)]['env_contract']
 assert c['test_valid'] and all(c[k]==4 for k in ['parallel_envs_initialized','parallel_envs_used','parallel_envs_closed'])
 assert all(e==0 for e in c['worker_exitcodes'])
 workers.extend(c['worker_pids'])
ps=subprocess.check_output(['ps','-eo','pid,args'],text=True)
running=[l for l in ps.splitlines() if any(s in l for s in ['train_stage3_v6_vector.py','train_stage3_v5_vector.py','run_blocks.py round','state_batch_intervention.py','local_trust_bound.py','probe_execution.py']) and 'grep' not in l]
assert not running,running
pids={int(l.strip().split()[0]) for l in ps.splitlines()[1:] if l.strip()}
assert not pids.intersection(workers)
npu=subprocess.check_output(['npu-smi','info'],text=True,timeout=30)
assert 'No running processes found' in npu
(HERE/'npu_smi_final.txt').write_text(npu)
envsteps=sum(e['length'] for br in rs['1']['branches'].values() for e in br['episodes'])
envsteps+=sum(e['length'] for br in r4['behavior'].values() for e in br['episodes'])
envsteps+=sum(e['length'] for br in r5['behavior'].values() for e in br['episodes'])
safety={'sha256_files':len(before),'production_and_formal_checkpoint_unchanged':before==after,'hashes':after,'new_actor_updates':2500,'new_critic_updates':0,'new_environment_steps':envsteps,'new_episodes':20,'formal_environment_steps':0,'formal_optimizer_steps':0,'workers_closed':len(workers),'remaining_worker_pids':[],'remaining_training_processes':running,'npu_processes':0,'numeric_fields_checked':numbercount[0],'finite':True,'fusion_result_exists':Path('fusion_result.json').exists(),'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
summary={'verdict':'INCONCLUSIVE','mechanistic_label':'LOCAL_Q_ASCENT_GLOBAL_CLOSED_LOOP_MISALIGNMENT','confidence':'MEDIUM for established mechanism; unique upstream cause unresolved','rounds':5,'round0_completed':True,'all_rounds_result_driven':True,'round1_duration_monotonic_hypothesis':'NOT_SUPPORTED','round4_state_sampling_intervention':'NO_RESCUE','round5_classification':'LEARNING_SUPPRESSION_WITH_PARTIAL_BEHAVIOR_PRESERVATION','round4_qgain_retention':qret4,'round5_retention':ret5,'round_results':rs,'primary_upstream':'Unresolved: repeated approximate replay-Q improvement not a certificate for new closed-loop policy improvement; cannot identify unique bootstrap/support defect','actor_mechanism':'unconstrained repeated approximate-Q optimization changes policy; induced competence can decrease','execution_channel':'component-mean/executed-action displacement; displacement norm alone not sufficient predictor','closed_loop':'path/continuation-dependent recovery or failure, not uniform duration threshold','amplifiers':'later Critic/replay/after-readiness bootstrap evolution not necessary; initial bootstrap bias not causally resolved','next_only':'matched-Qgain unrestricted versus local-bound comparison, identical optimizer/batches; evaluate4env and report drift to separate learning delay from safer path (NOT EXECUTED)','safety':safety,'report':str(HERE/'FINAL_REPORT.md'),'formal_training_remains_stopped':True}
for name,x in [('safety_final.json',safety),('final_summary.json',summary)]:
 p=HERE/name;assert not p.exists();p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
s=rs['3']['summaries']
blockrows=[]
for x in rs['2']['rows']:
 blockrows.append(f"|{x['block']}|{x['seed']}|{str(x['success'])}|{x['mean_delta_q1']:+.7f}|{x['mean_delta_q2']:+.7f}|{x['delta_mc']:+.7f}|")
text=f"""# Stage3-v6 mean2q / multi_q iterative root-cause diagnosis

mean2q/multi_q Actor collapses when repeated improvement of replay-conditioned approximate Q fails to translate into improvement of its changed closed-loop policy; these five rounds reinforce this surrogate/behavior mismatch but do NOT identify a unique upstream defect. **Final upstream verdict: INCONCLUSIVE.**

Best-supported mechanism: LOCAL_Q_ASCENT_GLOBAL_CLOSED_LOOP_MISALIGNMENT.
Confidence: MEDIUM for the established mechanism, not HIGH for unique upstream attribution.
Round0 + five RESULT-DRIVEN rounds completed. STOP, no sixth round, no tuning or production fix.

## How the rounds progressively narrowed the question

|Round|Previous conclusion|New question|Test type/cost|Result|Decision|
|---|---|---|---|---|---|
|0|Frozen-Q collapse and execution retraction already causal|What distinguishes one tolerable action from sustained failure?|existing-data only|Read previous fullJSON/reports/formalmetadata|Round1|
|1|Single action4/4,wholepolicy0/4|Does longer shifted-action block cause irreversible damage?|4-env rollout,8 newepisodes;0/1 reused|8steps3/4,32steps4/4;all32 faster|Reject simple duration threshold;Round2|
|2|Length/state norm nonmonotonic|Do localQ preferences predict block recovery?|existing-data only,all action traces|BothQs increase100% injectedactions,including onefailed8-block|Compare local surrogate with execution distribution;Round3|
|3|LocalQ preference not a whole-policy guarantee|Does Critic fail to recognize actual failed states?|offline frozenforward,827windows+sourceinspection|BCSpearman.991;actualfailedQ muchlower|Test old-state sampling explanation;Round4|
|4|Replay surrogategain differs from induced-statevalue|Is failed-state coverage correction enough?|offlineActorintervention1250updates+4-env8episodes|Control/interventionboth0/4;81.6% Qgainretained|Coveragechangealoneinsufficient;Round5|
|5|Relevantstatesalonecannotstopmovement/collapse|Does localpolicy-step bound protect without learning suppression?|offlineActorintervention1250updates+4-env4episodes;control reused|3/4,butonly4.46% Qgainretained|Learning-suppression confound;STOP|

Each round explicitly READ the preceding ROUND_REPORT.md and result.json before execution. Later interventions were designed only after earlier results: no five-experiment preplanned sweep.
Round2 source analysis was executed inline remotely and outputs saved here; no simulator used.
Rounds4/5 are Actor-only frozen-network MECHANISM experiments, NOT online continuation or formal RL.

## Round1/2: block interventions and negative duration evidence

Same four fixed seeds,reference prefix verified identical at t199. Inject the exact1250-updatefailedActor samples for8 or32steps,then return to unchanged BC on its own observations.
Length1 and baseline reused previous same-statefork; no duplicate baseline.
Success:0shift4/4;1shift4/4;8shift3/4;32shift4/4.
8-block lengths410/409/480/700;32-block401/400/471/415 vsbaseline460/423/482/444.

All32-blockepisodes finish faster thanbaseline and have positive deltaMC. The one8-blockfailure is SECONDARY: no seed/cutoff-specific follow-up.
Even muchlargerEEF/pathdeviation can represent helpful progress. No universal harm onset in1-32steps at this common phase; wholepolicyfailure cannot be reduced to duration or drift norm alone.
Exactly one common starting phase was tested; not proof all phases are tolerable.

|block|seed|success|mean localdeltaQ1|mean localdeltaQ2|true blockdeltaMC|
|---|---|---|---:|---:|---:|
{chr(10).join(blockrows)}

All160injectedactionqueries favor shiftedactions in BOTH critics.
IMPORTANT: these mean localdeltaQ values are NOT a predicted block-Q estimator. Candidate single-action values under a learned continuation are not guarantees for sustained/mixed-policy outcome. MC includes the fullblockandBCcontinuation.

Evidence classes:
- Causal: policy-block substitution changes true outcome and recovery.
- Correlational: localQ preferences vs episodeoutcome,trajectorydeviation.
- Negative evidence: monotonic duration/norm threshold and universally destructive localactions NOT_SUPPORTED.

## Round3: correct state-value ranking does not certify policy improvement

Frozen readinessCritic;827alignedwindows fromBC180,BLOCK8=199,BLOCK32=168,FAILED_FULL=280.
Both actors scored at each EXACT recorded observation/actionhistory; actual executedactionQ compared with finiteMC from that recordedpolicycontinuation.
Source checked: V6 inherits V5 Actor update;loss=-expectedQ1 onreplaycontexts,parametergradientclip,unconditionaloptimizerstep. No per-policy-output trustconstraint or realreturnacceptancecheck in that update. No production edits.

|trace|executed Qmean|actual MCmean|Spearman|failedActor-vsBC expectedQgain at same states|
|---|---:|---:|---:|---:|
|BC|{s['BC']['all']['q_executed_mean']:.6f}|{s['BC']['all']['actual_mc_mean']:.6f}|{s['BC']['all']['q_vs_mc_spearman']:.6f}|{s['BC']['all']['expected_qgain']:+.6f}|
|BLOCK8|{s['BLOCK8']['all']['q_executed_mean']:.6f}|{s['BLOCK8']['all']['actual_mc_mean']:.6f}|{s['BLOCK8']['all']['q_vs_mc_spearman']:.6f}|{s['BLOCK8']['all']['expected_qgain']:+.6f}|
|BLOCK32|{s['BLOCK32']['all']['q_executed_mean']:.6f}|{s['BLOCK32']['all']['actual_mc_mean']:.6f}|{s['BLOCK32']['all']['q_vs_mc_spearman']:.6f}|{s['BLOCK32']['all']['expected_qgain']:+.6f}|
|FAILED_FULL|{s['FAILED_FULL']['all']['q_executed_mean']:.6f}|0|undefined(constantMC)|{s['FAILED_FULL']['all']['expected_qgain']:+.6f}|

The Critic recognizes actual failed states as much lower value ON AVERAGE; it is not uniformly assigning themBC-levelhighQ. OnBCstatesfailedActor expectedQgain+.007429 vs+.000475 onactualfailedstates,about15.6x.
This is mechanistic/correlational evidence of surrogate/induced-distribution gap, NOT causal proof sampling mismatch is primary.
A frozenCritic may estimate a different future continuation than the evaluated changedActor; no claim these MC/Q comparisons isolate bootstrap error.
Same-time different-policy trajectories are NOT same-state counterfactuals after divergence. No generalization beyond fourknown-goodseeds.

## Round4: minimal fixed failed-state-batch substitution

Same readyActor140K,actorupdates0,originalfreshAdamstate. Same productionActorobjective/LRs,64-bankofflinehalf32/64 unchanged.
Only replace32onlineActorrows with280alignedwindows from preexisting failedActorvisitedtrajectories.
FixedCritic,replay;noTDupdates/no newrolloutdata fed to optimizer. This is NOT freshon-policy sampling or online causal evidence.
Baseline uses exactold1250model withsameoriginalbank/schedule; FIRST freshlyevaluatedbaseline thenintervention serially.

|branch|Actor updates|Critic updates|Qmean gain|sampled drift|weighted drift|componentmean RMS|uniqueparameter L2|success|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Originalbaseline|1250(previousoptimizationreused)|0|{base['q']['all']['qmean_gain']:.8f}|{base['drift']['sampled_action_l2_mean']:.6f}|{base['drift']['weighted_action_l2_mean']:.6f}|{base['drift']['component_mean_rms']:.6f}|.130687|0/4|
|Failedstatebatch|1250(new)|0|{m4['q']['all']['qmean_gain']:.8f}|{m4['policy_drift']['sampled_action_l2_mean']:.6f}|{m4['policy_drift']['weighted_action_l2_mean']:.6f}|{m4['policy_drift']['component_mean_rms']:.6f}|{m4['parameter_drift_unique_l2']:.6f}|0/4|

Qgainretention{qret4:.2%};sampleddrift99.04%baseline. NOT learning suppression.
Causal negative: this state-coverage intervention is insufficient at tested endpoint.
Does NOT reject everydistributionmechanism or fullon-policy refreshing;do not forcea primarysamplingcause fromRound3 correlation.

## Round5: local bound changes success but is learning suppression

Originalreadyactor,original64bank,originalobjective/schedule,1250updates.
Single heuristic epsilon7.10594e-5,chosen ONCE from known625-updateweighteddrift/625,not universal safe threshold.
Constraint reference is immediately PREVIOUSActor,not fixedBC;noanchorloss.
After each optimizerproposal,scale the parameter delta only ifmaxper-tokencomponentmean,weightedactionorprobabilityvector change exceeds epsilon;line search enforces thissamebound,no strength sweep.
Momentumstates kept fromproposals;this is diagnosticbehavior-stepclipping,not optimizercomparison.

Baseline0/4 fromRound4already freshlyevaluated;reuse validcontract/controlinstead of rerunning.
BoundedActor3/4,lengths407/700/482/542.
Qmean gain{m5['q']['all']['qmean_gain']:.8f} vsbaseline.00531547;retention{ret5['qgain']:.2%}.
Sampleddrift{m5['policy_drift']['sampled_action_l2_mean']:.6f},retention{ret5['sampled_action']:.2%};
weighteddrift{m5['policy_drift']['weighted_action_l2_mean']:.6f},retention{ret5['weighted_action']:.2%};
componentmeanRMS{m5['policy_drift']['component_mean_rms']:.6f},uniqueparameterL2{m5['parameter_drift_unique_l2']:.6f}.
1227/1250updatesclipped,averageproposalstepfraction9.706%.

Classification: **LEARNING_SUPPRESSION_WITH_PARTIAL_BEHAVIOR_PRESERVATION**.
AlthoughActor technicallyupdatesandQgainpositive,it sacrifices95.54%surrogategain andabout96%actionmovement. This is NOT a clean upstreamrescue with preservedsubstantivelearning.
Causal effect: strongstepdampingchangesendpointcompetence. Unresolvedconfound: reducedeffectivelearningamount/delay versus trulysafer path.
No precise collapse-time measurement,permanentprevention,onlinebenefit or3Mfixclaimed. No furthertuning.

## Root-cause layering

PRIMARY UPSTREAM CAUSE:
Unique cause remains INCONCLUSIVE. Candidate: approximate replay-Q improvement lacks a certificate for changed policy's entire closed-loop outcome; absence of an output/return-based acceptance mechanism permits repeated policychange.
Source evidence supports the missingcheck,not its uniquelysufficientupstreamattribution. Cannot assert action-supportextrapolation or bootstrap is primary.

ACTOR-LEVEL MECHANISM:
Repeated local approximate-Qascent can improve thesurrogate while degrading inducedpolicy. Aggregatepolicychange is insufficiently tied to actualreturn/recovery.

EXECUTION CHANNEL:
Componentmean/executedaction displacement; earlier samefailedweightsretraction causal. ParameterL2,EEFdistance oractiondistance alone notuniversalharm predictors.

CLOSED-LOOP CONSEQUENCE:
Outcome depends onvisitedpathandfuturecontroller/recovery. Some displacements accelerate successfulbehavior;otherslosecompetence. Thisrounddoesnotestablishonegeneralrecoveryboundary.

AMPLIFIERS:
LaterCriticevolution/online replay shift/after-readinessbootstrapevolution are NOT necessary,frozenreadyexperimentalreadycollapses. Theyremainpossibleamplifiers.
InitialCriticbootstrapbias BEFOREreadiness is not ruledout or causallyproven. NoBOOTSTRAP_PRIMARY_CAUSEclaim.
No newRNN/mean-only/logits-only/random2q/Adam-vsSGD study.

## Integration with historical evidence

- Objectivealignment: readyQ1/mean/mingradients nearlysame;initialerrornotQ1-only.
- Temporal CURRENT/ALL_ALIGNED/FREEZE_RNN previouslycollapse;no temporal/RNN-only necessity.
- Wholepolicyanchor/adaptivepreservation protectbutalmostfreezelearning;Round5shows relatedconfound evenwithoutBCanchor.
- Adamgeometry/equal-stepoptimizer priorbackgrounddoesnotmakeAdamnecessary;actualrandomorigin remainsbackground,notmean-specificnewtest.
- Module sensitivity/MEAN-only/RNN-onlyonline/componentmeanpreservation backgroundsupports outputchannel,notuniqueupstreamorigin;donotrerunorovergeneralize.
- MeanfrozenreadyCritic/replay already4/4->3/4->0/4,soonlinefeedbacknotneeded.
- Actionretraction causalrescue;offsupportproxy correlationplusfailedgradientrejection not primaryproof.
- Currentblocksaddimportantnegativeevidence: even32failedActoractionsinonephasecanimproveall4episodes,so donotcallallactions/destructiveaccumulationbylengthuniformlyharmful.

## Evidence grade and formal scope

Strongest causal facts: frozen-QActoroptimization can destroy competence;executedactionretractionrescues;block substitutionchangesoutcomes.
New causal intervention negative: failed-statebatchdoesnotrescue.
New causal butconfounded: localstepboundrescues3/4onlywith95.5%Qgainreduction.
Correlations: supportdistance,executionQ/MCgap,trajectorystate/EEFdrift.
OverallmechanisticconfidenceMEDIUM;uniqueupstreamverdictINCONCLUSIVE.
AllsamplesfourselectedBC-successseeds,notpopulationestimates.
Formalcollapse window150-160K and0/10at280Kremainconsistent;nonewformalcontinuationperformed.
Do notlabelanofflineActorprobeasrealonlinecausaltraining.

## ONE NEXT STEP (not executed)

Matched-Qgain unrestrictedActor versuslocal-boundActor comparison using identicaloriginaloptimizer/batches,startready. Match achievedsurrogategain rather than countofnominalupdates,then evaluateexact4env andreportaction/componentdrift.
This resolves Round5's single remainingkeyambiguity: merelearningdelay versus genuinelysafer optimizationpath. No strengthsweep,noAdamcomparison,nootherdirections.

## Safety, cost and paths

Device=npu:0 ONLY; experiments SERIAL.
Round1,4,5 eachinitialized=used=closed4,seeds[20008,20002,20005,20007],simerrors0,all12workers exit0.
Round2 existingdataonly;Round3 offlineforwardonly.
20newdiagnosticepisodes,{envsteps}newenvironmentsteps;2500newtestingActorupdates,0Criticupdates;formalenv/optimizersteps0.
{len(before)}production/formalcheckpointSHA256hashesunchanged;formaltrainingneverresumed.
AllJSON/JSONLfullyread,{numbercount[0]}numericfieldsfinite;numericNPY/NPZfinite.
Noresidualtraining/simulatorworker/NPUprocess.
HEAD{safety['head']};fusion_result.jsonexistingD/absentuntouched;noGitdestructiveoperationorcleanup.
Allnewcode/checkpoint/results underthis testingdirectory. No runtimebugrequiredrepair;allnewPythonpy_compilePASS.

Artifacts:
- round0/result.json and ROUND_REPORT.md
- round1/design.json,result.json,BLOCK_8/32_trace.jsonl,ROUND_REPORT.md
- round2/result.json,ROUND_REPORT.md
- round3/result.json,actual_execution_values.npz,ROUND_REPORT.md
- round4/design.json,offline.json,result.json,closed_loop.json,updates.jsonl,failed_window_order.npy,state_aware_actor.pth(testing-only),trajectories,ROUND_REPORT.md
- round5/design.json,offline.json,result.json,closed_loop.json,updates.jsonl,trajectories,ROUND_REPORT.md
- final_summary.json,safety_before/final.json,npu_smi_final.txt

FORMAL TRAINING REMAINS STOPPED
"""
(HERE/'FINAL_REPORT.md').write_text(text)
print(json.dumps({'verdict':summary['verdict'],'confidence':summary['confidence'],'round5_retention':ret5,'cost':{k:v for k,v in safety.items() if k not in ['hashes']},'report':summary['report']},indent=2))
