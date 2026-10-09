"""Third and final bounded offline semantic analysis; no simulator/training."""
import sys,json,hashlib,shutil,subprocess,time
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent; TEST=HERE.parent; RL=TEST.parents[1]; ROOT=RL.parents[1]
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,d):
 p=HERE/name
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
old=TEST/'critic_counterfactual_identifiability'; pairs=read(HERE/'round1_reused_pairs.json'); probe=read(HERE/'round2_frozen_semantics_probe.json')
contracts=[]; extra=[]
for r,f in [('round2','strict_result.json'),('round3','strict_result.json'),('round4','result.json')]:
 p=old/r/f; x=read(p); c=x['env_contract']
 assert c['initialized']==c['used']==c['closed']==4 and c['exitcodes']==[0]*4
 assert x['actor_updates']==x['critic_updates']==0 and x['model_hashes_unchanged']
 contracts.append({'source':str(p),'sha256':digest(p),'env_contract':c})
for p in [old/'run_round3.py',old/'run_round4.py',old/'round3/preregistration.json',old/'round4/preregistration.json',TEST/'critic_counterfactual_acceptance/final_summary.json',TEST/'critic_counterfactual_acceptance/old_readiness_at140000.json',TEST/'final_collapse_rootcause/final_summary.json',TEST/'mean_multi_collapse_diagnosis/final_summary.json']:
 dest=HERE/'supporting_evidence'/p.parent.name/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 extra.append({'source':str(p),'copy':str(dest),'sha256':digest(p)})
import h5py
cfg=read(RL/'stage2_2_history_aware_critic/stage2_2_config.json'); data=[]
for name in ['bc_rnn','bc_transformer','bc_gmm']:
 p=Path(cfg['dataset_root'])/name/'transitions.hdf5'
 with h5py.File(p,'r') as h:
  data.append({'policy':name,'path':str(p),'attrs':{k:str(v) for k,v in h.attrs.items()},'episodes':len(h['episodes'])})
reward=Path('/data/home/3220251075/lerobot_workspace/miniconda3/envs/robosuite_npu/lib/python3.10/site-packages/robosuite/environments/manipulation/two_arm_transport.py')
lines=reward.read_text().splitlines()
dump('semantic_support.json',{'datasets':data,'prior_env_contracts':contracts,'evidence_manifest':extra,'reward_source':{'path':str(reward),'sha256':digest(reward),'lines_226_264':'\n'.join(f'{i+1}: {s}' for i,s in enumerate(lines) if 226<=i+1<=264)}})
# Exact finite MDP illustration: this is algebra, NOT a robot experiment.
g=.99; qsafe=g**5; qrisk=g
rows=[]
for theta in [0.,.25,.5,.75,1.]:
 surrogate=.99*((1-theta)*qsafe+theta*qrisk)+.01*(1-theta)
 value=(1-theta)*qsafe+theta*g*(1-theta)
 rows.append({'theta':theta,'exact_old_policy_surrogate':surrogate,'true_updated_policy_start_value':value,'success_probability':1-theta*theta})
slope=.99*(qrisk-qsafe)-.01
assert slope>0 and rows[-1]['true_updated_policy_start_value']==0 and rows[-1]['success_probability']==0
assert all(rows[i+1]['exact_old_policy_surrogate']>rows[i]['exact_old_policy_surrogate'] for i in range(4))
dump('round3_exact_semantic_counterexample.json',{'purpose':'Disprove logical necessity of critic action-misranking from full-policy collapse alone; NOT robot causal reproduction.','gamma':g,'MDP':'s0 Safe -> d0 -> d1 -> d2 -> d3 -> d4 -> terminal, reward1 only at d4; s0 Risk -> s1; s1 Good -> terminal reward1; Bad -> terminal reward0; all other rewards0.','old_policy':'Safe at s0, Good at s1','shared_updated_policy':'P(Risk|s0)=theta; P(Bad|s1)=theta','replay_weights':{'s0':.99,'s1':.01},'exact_old_Q':{'s0_Safe':qsafe,'s0_Risk':qrisk,'s1_Good':1.,'s1_Bad':0.},'surrogate_slope':slope,'rows':rows,'robot_shared_coupling_and_visitation_not_proven':True})
print('ROUND3 COMPLETE semantic analysis; final reporting',flush=True)
accept=read(TEST/'critic_counterfactual_acceptance/final_summary.json'); rootcause=read(TEST/'final_collapse_rootcause/final_summary.json'); collapse=read(TEST/'mean_multi_collapse_diagnosis/final_summary.json')
discovery=[p for p in pairs if p['group']=='discovery']; independent=[p for p in pairs if p['group']=='independent']
answers={
 '1_actual_continuation':{'status':'CONFIRMED','MC':'Stage2.2 multi_q regresses finite recorded returns from bc_rnn, bc_transformer, bc_gmm episode continuations. It is observational E_D[G|compressed_history,a], not a uniquely specified fixed BC policy or post1250 policy; source-policy posterior can depend on history/action.','Ready':'At env140000 Actor updates0 and target Actor equals BC source. mean2q bootstraps categorical expectation of component-mean actions from this BC target, averaged over twin target Q. Intended BC-continuation value with function approximation and nonstationary/partial-history limitations; not proven exact Q^BC. Frozen checkpoint cannot be interpreted as Q^post1250.'},
 '2_updated_actor_invalidates_old_value':{'status':'SUPPORTED','confirmed_scope':'Identical fork state/history/action and paired RNG, changing only t>fork continuation changed realized return in five negative cases and one positive case. Discovery BC4/4 vs CURRENT1/4; independent BC2/4 vs CURRENT0/4.','limit':'CONFIRMED realized continuation effect; expected-value effect and generalization beyond eight contexts remain SUPPORTED, not a universal confidence claim. Old policy Q remains defined; applicability to a different whole policy fails, not proof old-policy Q itself is wrong.'},
 '3_conditional_supervision_stronger':{'status':'INCONCLUSIVE','answer':'Absence of paired conditional alternative-action supervision is confirmed from source; whether it causes unreliable conditional advantage or is stronger than continuation/visitation mismatch is unresolved. Completed acceptance: seed20005 5/6 negative, one positive, mean CI crosses zero; no identified twin common wrong pair. Do not infer absence of real misranking from non-identifiability.'},
 '4_full_collapse_specific_upstream_error':{'status':'INCONCLUSIVE','answer':'No unique code/label/target error with causal rescue of full collapse established. Strongest observed mechanism: later updated actions destroy successful completion even after an unchanged BC fork action, while replay-context old-policy Q surrogate improves. Exact logical counterexample shows this pattern is possible with perfect old-policy Q, so collapse alone does not identify erroneous action supervision.','prior_exclusions':'Original MC frozen source also yielded0/4 after normal1250 Actor updates; later Stage3 bootstrap/replay/critic evolution is not necessary. No correction has been shown to rescue full collapse.'},
 '5_unconfirmed':{'status':'INCONCLUSIVE','questions':['Whether true fixed-continuation expected action advantages are wrongly ranked in collapse-driving states','Whether replay weighting/shared policy parameter coupling and changed visitation specifically produce the complete robot collapse','Whether compressed ten-step history aliases task state/source behavior','How much continuation transport mismatch versus function-approximation error contributes','Generalization to other contexts or many random streams; intentionally not expanded']}}
summary={'status':'COMPLETE_STOPPED','upstream_root_cause':'INCONCLUSIVE','rounds':3,'round1':'source/strict completed rollout reuse','round2':'frozen npu:0 semantic consistency probe on the same eight contexts','round3':'bounded offline interpretation and exact logical counterexample, not a robot simulator experiment','new_simulator_invocations':0,'new_simulator_episodes':0,'new_seeds_or_contexts_or_random_streams':0,'reused_strict_pairs':8,'reused_trajectories':16,'answers':answers,'continuation_pairs':pairs,'Ready_Q_surrogate_both_positive_contexts':sum(all(x['Q_expectations']['ready']['CURRENT_minus_BC'][k]>0 for k in ['Q1','Q2']) for x in probe['probe']),'ready_actor_and_target_equal_BC':True,'rnn_parity_max_abs':max(v['native_vs_vectorized_mean_max_abs'] for x in probe['probe'] for v in x['distributions'].values()),'clip_expected_Q_max_delta':max(abs(v[a][k]) for x in probe['probe'] for v in x['Q_expectations'].values() for a in ['BC','CURRENT'] for k in ['clip_Q1_delta','clip_Q2_delta']),'remaining_two_candidates':['H2: Conditional intervention advantage not identified by observational multi-policy/TD supervision; function approximation/history aliasing','H3: Frozen old-continuation replay-weighted actor surrogate fails to control changes in future policy, state visitation and shared policy outputs'],'known_prior_control':{'ready_frozen_success_counts':collapse['frozen_success_counts'],'MC_source_1250_success':rootcause['mc_source_substitution']['behavior']['MC_SOURCE_1250']['success_count'],'stage3_bootstrap_necessary':rootcause['stage3_TD_bootstrap_necessary'],'later_online_replay_or_critic_evolution_necessary':rootcause['later_online_replay_or_critic_evolution_necessary']},'prior_action_acceptance':{'unique_pairs':accept['unique_strict_action_pairs'],'identifiable_pairs':accept['identifiable_pairs'],'identified_twin_common_wrong':accept['twin_common_wrong_identifiable'],'independent_identified_twin_common_wrong':accept['independent_twin_common_wrong_identifiable'],'seed20005_expected_advantage_result':accept['seed20005_expected_advantage_result']},'actor_updates':0,'critic_updates':0,'production_modified':False,'formal_checkpoints_modified':False,'formal_training':'STOPPED','recommendations_proposed_only':['If later authorized, distinguish H2 versus H3 with policy-matched conditional advantages and a mechanistic intervention whose rescue tests the same normal-strength collapse. No additional seeds/streams/contexts/training executed now.'],'report':str(HERE/'FINAL_REPORT.md'),'final_markers':['CRITIC CONTINUATION SEMANTICS DIAGNOSIS COMPLETE','FORMAL TRAINING REMAINS STOPPED']}
# Actual current-state safety, not an assertion based on intent.
baseline=read(HERE/'safety_before.json'); changes=[]
for f,h in baseline['files'].items():
 p=Path(f)
 if not p.exists() or digest(p)!=h:changes.append(f)
assert not changes,changes
assert (ROOT/'fusion_result.json').exists()==baseline['fusion_exists']
live=[]
for line in subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines():
 if 'python' in line and any(s in line for s in ['acceptance_runner.py','run_round2.py','run_round3.py','run_round4.py','run_diagnosis.py','train_stage3','spawn_main']):live.append(line)
assert not live,live
safety={'baseline_files_checked':len(baseline['files']),'unchanged':True,'changed':changes,'fusion_existence_unchanged':True,'live_simulator_or_training_or_probe_processes':live,'prior_workers_all4_closed_exit0':True,'prior_contracts':contracts,'finalizer_is_offline_reporting_only':True}
dump('safety_after.json',safety); summary['safety']=safety
rows='\n'.join('| {} | {} | {} | {} / {} | {:.9f} | {:.9f} | {:+.9f} |'.format(x['group'],x['seed'],x['fork'],x['BC_length'],x['CURRENT_length'],x['BC_mc'],x['CURRENT_mc'],x['delta_mc']) for x in pairs)
qrows='\n'.join('| {} | {:+.9f} | {:+.9f} | {:+.9f} | {:+.9f} |'.format(x['seed'],x['Q_expectations']['ready']['CURRENT_minus_BC']['Q1'],x['Q_expectations']['ready']['CURRENT_minus_BC']['Q2'],x['Q_expectations']['MC']['CURRENT_minus_BC']['Q1'],x['Q_expectations']['MC']['CURRENT_minus_BC']['Q2']) for x in probe['probe'])
report=f'''# Stage3-V6 mean2q / multi_q: Critic continuation-policy diagnosis

Status: COMPLETE_STOPPED. Unique upstream cause of full Actor collapse: INCONCLUSIVE.
Three bounded analysis rounds completed. No new simulator calls, episodes, seeds,
contexts, or random streams. No Actor/Critic training or production fixes.

## 1. Which continuation does each Critic evaluate?

CONFIRMED source/checkpoint semantics. Stage2.2 multi_q trains on bc_rnn,
bc_transformer and bc_gmm episode returns G_t=sum_j(0.99^j*r_(t+j)), without
bootstrap. Both heads regress recorded-action labels. Input is a ten-step
sliding zero-state compressed history: observation59, previous action14, t/700.
The first predecessor action in each window is zero; candidate current action
enters the Q head separately, not the recurrent history.

Its appropriate interpretation is observational E_D[G|h,a], combining recorded
source-policy continuations. Source-policy posterior weights can depend on h,a.
This is neither a uniquely specified BC continuation nor per-step random mixing
of the three policies nor the post1250 Actor. Interpreting it as do(a) value under
one specified continuation requires additional identification assumptions.
Compressed history has not been proved sufficient for the task state.

At readiness env140000, Actor updates0, Ready Actor and target Actor both equal
the BC source, verified tensor by tensor. Its mean2q target is:
y=r+0.99*(1-terminal)*0.5*(sum_k p_k Q1target(h_next,mu_k)
                              +sum_k p_k Q2target(h_next,mu_k)).
Thus its intended evaluation continuation is categorical BC target component
means, with approximation error and partial-history/nonstationarity limitations.
It is not Q of the weighted average action, nor proven exact Q^BC. Actual eval
execution includes Gaussian std1e-4 and clipping; the target uses component means.
A frozen Ready checkpoint does not automatically become Q^CURRENT when Actor
changes. During actual subsequent TD training the target policy may evolve;
this statement concerns the frozen checkpoint, not all future training phases.

## 2. Strict fixed-action continuation evidence

Reused completed identifiability round2/3/4. Round3 BC_strict names the BC fork
action; its t>fork continuation is CURRENT/post1250, verified in the runner.

| Group | Seed | Fork | BC / CURRENT length | BC G | CURRENT G | Delta G |
|---|---:|---:|---:|---:|---:|---:|
{rows}

Discovery: BC4/4 vs CURRENT1/4 successes. Independent preregistered fixed fork199
seeds20036-20039: BC2/4 vs CURRENT0/4. Eight valid continuation pairs, sixteen
reused trajectories: five negative, one positive, two zero return changes.
These are not eight new alternative-current-action comparisons.

Recomputed discounted returns from raw JSONL and matched all episode/fork labels.
Verified observation, next observation, executed action and reward through and
including the fork; matching histories, current candidates, Q values and saved
physical/history/actor/RNG hashes. Independent records match both Actor hidden
states. Current action14 is feasible in [-1,1]. Historical contracts show exactly
four true parallel workers, all four closed with exit0, no model updates.
The snapshots cover exposed physics/controller/cache/environment RNG, not a claim
about unknown opaque internals. Discovery baseline and intervention were separate
completed invocations, with matched prefix and snapshot evidence.

CONFIRMED for these realized paired streams: changing only future policy can
turn successful completion into failure even when current action is unchanged.
SUPPORTED: old-continuation scores cannot be transported unchanged into the
return of the new whole policy. This changes applicability, not the definition
of old-policy Q; it does not prove current-action ranking under old continuation
is wrong. One coupled stream per context does not identify expected advantage
or population generality. Seed20031 improves; two independent cases already fail.
'''
report+=f'''
## 3. Frozen npu:0 consistency checks

Used only the same eight histories, no simulator/optimizer or trajectory sampling.
Native sequential RNN inference and production vectorized suffix inference match
with max mean and category-probability errors0. The successor reset metadata +1
mapping is consistent. Production uses next observations; this offline probe
checked equal-input reset/forward parity, not a newly simulated successor target.
All loaded model hashes remain unchanged. Clipping component means has zero
expected-Q effect at these eight contexts; this is not a universal exclusion.

CURRENT-minus-BC categorical current-action surrogate under the frozen networks:

| Seed | Ready Delta Q1 | Ready Delta Q2 | MC Delta Q1 | MC Delta Q2 |
|---|---:|---:|---:|---:|
{qrows}

Ready both heads prefer CURRENT at8/8; MC does not show the same universal pattern.
This table changes the CURRENT ACTION DISTRIBUTION, while section2 keeps current
action fixed and changes FUTURE POLICY. Combining them does not establish the
ranking of one strict counterfactual action pair. They show how a replay-context
old-continuation surrogate can diverge from updated whole-policy performance.

## 4. Is deficient conditional supervision the stronger cause?

INCONCLUSIVE: it remains a candidate, not established stronger. Source confirms
no paired same-history alternative-action fixed-continuation advantage labels.
Absence of explicit paired supervision alone does not prove conditional Q wrong.

Completed action acceptance remains unchanged: seed20005/fork199 BC->post1250
current-action pair has Ready DeltaQ1=+0.002381831, DeltaQ2=+0.001315147.
Six streams yield five negative and one positive DeltaG, mean-0.005680973,
approx95% mean t interval[-0.025309737,+0.013947791], two-sided sign p0.21875.
The preregistered identification rule fails. Of24 unique action pairs, only3
are identified, all correctly ranked; independent set has no identified twin
common wrong pair. This proves neither expected misranking nor correctness of
unresolved pairs. Original MC source continuation does not match the BC
continuation in that acceptance; do not label its scores wrong on that basis.

Readiness tests behavior-data Q/MC Spearman, relative TD health, finite values
and scales. It does not test conditional action advantage or updated-policy
continuation transport. Actual140000 gate Spearman is approximately0.708;
restricted BC-success probe correlation approximately0.990 is not that gate.
Consequently passing readiness does not eliminate either remaining candidate.

## 5. Specific upstream error explaining full collapse?

INCONCLUSIVE: none established with a causal correction rescuing full collapse.
Strongest directly observed mechanism: changed subsequent actions destroy
completion after an unchanged BC fork, while old-policy replay Q can rise.
Original frozen Stage2.2 MC after normal1250 Actor updates also yielded0/4;
Ready-frozen controls report success counts[4,3,0]. Later Stage3 TD bootstrap,
online replay or Critic evolution are not necessary for failure. Removing TD
bootstrap is not a validated fix.

Source Actor objective is -mean(sum p_k Q1(h,mu_k)); TD target uses twin mean.
This is the actual definition, not automatically a bug. Environment reward is
sparse success indicator and matches the saved traces. Discounted G also measures
completion timing: lower G with both policies successful is not success collapse.
No reward-sign, transition-indexing or current-action-history leakage error has
been established as the cause of full collapse.

Round3 provides an exact finite-MDP logical counterexample with perfect old Q:
a shared policy parameter increases replay-weighted old-Q surrogate monotonically,
yet the updated whole policy ends at success0. Risk at s0 needs future Good at s1;
the shared parameter also switches s1 to Bad, and replay gives s1 small weight.
This refutes the inference that full-policy collapse necessarily proves old-Q
action misranking. It is algebra, NOT robot causal reproduction; actual robot
state weighting and parameter coupling were not proved by this illustration.

Remaining TWO candidates, not uniquely ordered:
H2: observational supervision fails to identify conditional intervention advantage,
with function approximation or compressed-history aliasing.
H3: replay-weighted frozen old-continuation optimization does not control changed
future shared policy outputs and state visitation.
Ordinary continuation dependence itself is not a specific implementation bug.
'''
report+=f'''
## 6. Uncertainties, artifacts, and stop

Unconfirmed: wrong expected fixed-continuation advantages in collapse-driving
states; relative H2/H3 contributions; history sufficiency; robot-specific shared
parameter coupling/visitation explanation for complete collapse; generalization
to further contexts or random streams. Scope intentionally not expanded.

Proposed ONLY for a separately authorized future task: policy-matched conditional
advantages and a mechanism intervention whose rescue tests the same normal-strength
collapse. No new training, gating, fix, or learning-rate/anchor/strength investigation
is executed here.

All new artifacts are in {HERE}: FINAL_REPORT.md, final_summary.json, testing-only
runners and logs, source_audit.json, semantic_support.json, round1_reused_pairs.json,
round2_frozen_semantics_probe.json, round3_exact_semantic_counterexample.json,
raw_evidence copies with SHA256 manifest, supporting_evidence, safety_before/after.
Original historical artifacts remain intact.

Final safety: {len(baseline['files'])} baseline files unchanged; production/formal
checkpoints/history not modified; no remaining simulator/training/probe process.
Historical workers closed normally; none created this phase. Stop immediately
following reporting; await user instruction.

CRITIC CONTINUATION SEMANTICS DIAGNOSIS COMPLETE

FORMAL TRAINING REMAINS STOPPED
'''
p=HERE/'FINAL_REPORT.md'
assert not p.exists();p.write_text(report,encoding='utf-8')
summary['report_sha256']=digest(p);dump('final_summary.json',summary)
assert read(HERE/'final_summary.json')['upstream_root_cause']=='INCONCLUSIVE'
assert all(f'## {i}.' in p.read_text() for i in range(1,7))
print(json.dumps({'status':'COMPLETE_STOPPED','pairs':8,'new_simulator_runs':0,'safety_files':len(baseline['files']),'report_bytes':p.stat().st_size,'summary':str(HERE/'final_summary.json')}),flush=True)
