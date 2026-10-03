# Stage3-v6 testing-only adaptive policy-preservation real short run

Primary result: `POLICY_FROZEN_NOT_SOLVED`
Answer: **NO — evidence does not support it**
The real BASELINE reproduces 4/4 at140K -> 0/4 by160K and through200K. ADAPTIVE preserves4/4 through190K and3/4 at200K, reducing fixed-data execution drift from1.3149064343 to0.0019763096 (99.8497%). However checkpoint-matched cumulative Actor optimizer-step improvement is only0.000534530729 versus0.055071733892:0.970608% retention. The common fixed140K Critic score gains are0.000139725191 versus0.018682705238:0.747885% retention. Both fail the intended materially-more-learning criterion; the historical2.65% is contextual, not a matched real-training fixed-anchor control. The ADAPTIVE controller has grad-ratio median1.18034/p90=3.78552/p95=5.99990,60.6369% updates above1,lambda=0 only2.35483%,lambda median30.38426 andp95=66.63427(max). This is functionally near-locked policy behavior, not literally frozen weights. Std gradients and std parameter drift remainzero. Choose only Actor update frequency / optimizer-step magnitude as the next direction; no further anchor retuning or experiment. Evidence is limited tofour preselected baseline-success seeds and matched staged-resume semantics, not a population success estimate. A testing-only stage-integrity assertion initially confused ready_step140000 with normal gate_open_step140001; only that assertion was corrected, completed BASELINE160K reused, and no gate/controller/production modification was made. Final cleanup audit found no testing trainer, orchestrator, vector child or formal trainer processes; allsix stage logs contain completed vector-env closure and alleight per-stage diagnostic-env closures, no Traceback or ERR99999. Source checkpoint SHA, preregistration SHA,production sourceSHA andHEAD remainunchanged. Formal training remainsstopped.

## BASELINE REPRODUCTION

|env step|BASELINE success|ADAPTIVE success|BASE drift|ADAPT drift|fixed-score RL retention|step-improvement RL retention|
|---:|---:|---:|---:|---:|---:|---:|
|140000|4/4|4/4|0|0|N/A|N/A|
|150000|2/4|4/4|0.0550756001|0.0020157527|9.19274%|13.0627%|
|160000|0/4|4/4|0.218858862|0.0019706147|2.45657%|4.78177%|
|170000|0/4|4/4|0.436520004|0.00306968577|1.16637%|3.12142%|
|180000|0/4|4/4|0.713244645|0.00239498738|0.697066%|1.89198%|
|190000|0/4|4/4|1.05251983|0.00102330012|0.699916%|1.32579%|
|200000|0/4|3/4|1.31490643|0.00197630963|0.747885%|0.970608%|

140K success is reused from the verified identical140K Actor and previous4-seed original trajectories, not reevaluated or assumed from a different checkpoint. Startup Actor/init equality, env140K, zero Actor updates and empty Adam state were verified.

## ADAPTIVE CONTROLLER

`D_mu = E_batch,time sum_k p_ref[t,k] ||mu_current[t,k]-mu_ref[t,k]||_2^2`
`D_p = E_batch,time KL(p_ref || p_current)`
`D = D_mu + beta * D_p`
`lambda(D) = lambda_max * clamp((stop_gradient(D)-D_safe)/(D_hard-D_safe),0,1)`
`L_total = L_production_RL + lambda(D) * D`

- beta=0.396452595479; D_safe=7.76283564669e-07; D_hard=3.10513425868e-06; lambda_max=66.6342692658.
- Derivation fixed BEFORE rollout: D_safe = prior WEAK1000 mean combined distance on fixed192 contexts; D_hard=4*D_safe (2x RMS behavior budget), both chosen before any new rollout; no retuning.
- D_safe RMS-equivalent=.00088107; D_hard RMS-equivalent=.00176214. These are combined-distance scales, not a claimed hard bound on categorical-sampled action displacement.
- lambda_max is previous gradient-calibrated WEAK scale; no success-conditioned retuning, no lambda grid. No supplementary branch was executed.
- Fixed deep-copy140K reference eval(), requires_gradFalse, outside optimizer and Polyak; immutable hash verified periodically. Targets are reference Actor means/probabilities, never replay/expert actions; no std or parameter anchor.
- Full10 timesteps on EXACTLY the production RL Actor batch with identical zero-init semantics. Production RL remains final-token probability-weighted Q1, full BPTT.

## EXECUTION / SAFETY CONTRACT

- Repo/current HEAD=2419e0008a016ba0fee842b48f078edb186c0bcf; no git sync, no production edits, no formal run resumed.
- Source checkpoint: `/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_readiness_v2_multi_mean_random_formal_20260929_130945/mean2q/multi_q/checkpoints/critic_ready.pth`.
- Production16-env async collector/learner, startup_parallelism4 (4+4+4+4), original replay, mean2q target, optimizer, target networks, scheduler, gate, Critic LR, Actor warmup, UTD=.25, policy_delay4, gradclip10, Polyak unchanged.
- Testing-only agent Actor function is derived from the inspected actual production function with only preservation-loss/observational logging additions. Trainer main is derived with only testing run label, checkpoint and observational diagnostic hooks. Source SHA256 asserted at phase startup.
- Only one NPU is visible on this server. NPU0 sequential BASELINE then ADAPTIVE at each common stage160K/180K/200K. All vector environments close before the next process starts.
- Matched production resume semantics: partial episodes discarded/reset; generations advance at common staged resumes. This is not uninterrupted historical training or an exact replay of old simulation states. That limitation applies to both branches.
- Saved step checkpoints/evaluations occur at the production checkpoint boundary before terminal pending-credit catch-up. Raw actor logs may include catch-up operations discarded when the next phase resumes the explicit saved step checkpoint. Retention follows the latest effective update-counter path and does not double-count such discarded operations; counts are reported below.
- Fixed-seed evaluation is a testing-only observational hook, not a change to the production readiness gate or formal evaluation FSM. Learner Python/NumPy/Torch/NPU RNG is restored after each diagnostic.
- All output/checkpoint writes are guarded to the testing directory. No original training checkpoint is overwritten. Runtime cwd is under testing to contain runtime-generated files.

## TRAINING CURVES / CHECKPOINT TABLE

|step|branch|Actor updates|Critic updates|success|execution drift|fixed140K score improvement|lambda median|ratio median|
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|140000|BASELINE|0|34750|4/4|0|0|N/A|N/A|
|140000|ADAPTIVE|0|34750|4/4|0|0|N/A|N/A|
|150000|BASELINE|615|37210|2/4|0.0550756001|0.00136334635|0|0|
|150000|ADAPTIVE|622|37239|4/4|0.0020157527|0.000125328952|16.7539486|0.703860965|
|160000|BASELINE|1246|39732|0/4|0.218858862|0.00519353058|0|0|
|160000|ADAPTIVE|1245|39728|4/4|0.0019706147|0.000127582549|22.4011761|0.838677252|
|170000|BASELINE|1862|42198|0/4|0.436520004|0.0106967725|0|0|
|170000|ADAPTIVE|1860|42191|4/4|0.00306968577|0.000124764105|25.1479857|0.917834738|
|180000|BASELINE|2487|44699|0/4|0.713244645|0.0162478499|0|0|
|180000|ADAPTIVE|2499|44744|4/4|0.00239498738|0.000113258204|27.27179|1.00618582|
|190000|BASELINE|3110|47190|0/4|1.05251983|0.020315174|0|0|
|190000|ADAPTIVE|3124|47244|4/4|0.00102330012|0.000142189121|29.336537|1.11183892|
|200000|BASELINE|3747|49736|0/4|1.31490643|0.0186827052|0|0|
|200000|ADAPTIVE|3737|49696|3/4|0.00197630963|0.000139725191|30.3842591|1.18034391|

## LEARNING CAPACITY / CONTROLLER OCCUPANCY

Primary improvement: sum of Q1 expected score AFTER minus BEFORE each actual Actor optimizer step, on the SAME batch and SAME current Critic within that step. It does not credit Critic weight updates as Actor improvement. Compare ADAPTIVE/BASELINE on the matched interval.
Secondary improvement: fixed140K Critic, shared fixed192 histories, Actor score gain over140K. It provides a common fixed scoring probe and is closer to previous frozen-critic retention, but this run has a moving training Critic and a different training trajectory/update count. Historical2.65% is contextual, not an additional real-training fixed-anchor control.

|branch|evaluated step|cumulative per-step improvement|ratio median|p90|p95|fraction ratio>1|fraction lambda=0|lambda median|max-boundary fraction|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|BASELINE|150000|0.001266218722|0|0|0|0%|100%|0|80.325%|
|BASELINE|160000|0.004704520106|0|0|0|0%|100%|0|90.289%|
|BASELINE|170000|0.009688146412|0|0|0|0%|100%|0|93.502%|
|BASELINE|180000|0.01753877848|0|0|0|0%|100%|0|95.135%|
|BASELINE|190000|0.03442101926|0|0|0|0%|100%|0|96.109%|
|BASELINE|200000|0.05507173389|0|0|0|0%|100%|0|96.771%|
|ADAPTIVE|150000|0.0001654028893|0.703861|1.204954|1.330869|22.026%|13.826%|16.75395|0%|
|ADAPTIVE|160000|0.0002249591053|0.8386773|1.497249|1.749788|35.02%|7.0683%|22.40118|0.56225%|
|ADAPTIVE|170000|0.0003024078906|0.9178347|1.721636|2.192036|42.796%|4.7312%|25.14799|1.4516%|
|ADAPTIVE|180000|0.0003318302333|1.006186|2.285377|3.762172|50.86%|3.5214%|27.27179|3.8415%|
|ADAPTIVE|190000|0.0004563517869|1.111839|3.297386|5.252542|57.266%|2.8169%|29.33654|6.306%|
|ADAPTIVE|200000|0.0005345307291|1.180344|3.785522|5.999898|60.637%|2.3548%|30.38426|7.5729%|
- BASELINE: raw Actor update rows=3767; effective unique rows=3750; discarded duplicate terminal-catchup rows=17. Full details remain in raw JSONL.
- ADAPTIVE: raw Actor update rows=3756; effective unique rows=3750; discarded duplicate terminal-catchup rows=6. Full details remain in raw JSONL.

20% RL retention is an interpretation reference, NOT a tuned acceptance threshold or production gate. An apparent success rescue with only1–3% retention/long-term ratio>1 must not be presented as a balanced repair.

## POLICY PRESERVATION / GMM

|branch|step|component RMS|categorical KL|mode-rank changes|top1 changes|RNN RMS|std RMS|
|---|---:|---:|---:|---:|---:|---:|---:|
|BASELINE|140000|0|0|0%|0%|0|0|
|BASELINE|150000|0.00757603699|0.00296519283|4.6875%|1.66667%|2.12878966e-05|0|
|BASELINE|160000|0.0295483104|0.0433570074|15.7812%|5.05208%|8.45535732e-05|0|
|BASELINE|170000|0.0619803242|0.219454387|30.4688%|10.0521%|0.000183513696|0|
|BASELINE|180000|0.105817665|0.367108491|36.0938%|14.7396%|0.000305281866|0|
|BASELINE|190000|0.163002756|0.589360642|43.5938%|20.0521%|0.000466409693|0|
|BASELINE|200000|0.198763272|1.20467195|65.5729%|28.8021%|0.000596427612|0|
|ADAPTIVE|140000|0|0|0%|0%|0|0|
|ADAPTIVE|150000|0.000288308582|3.78948504e-07|0.104167%|0%|9.30380275e-06|0|
|ADAPTIVE|160000|0.000272492302|3.5731766e-07|0.104167%|0.0520833%|1.74074707e-05|0|
|ADAPTIVE|170000|0.000300197573|3.77981164e-06|0.15625%|0.0520833%|2.40259621e-05|0|
|ADAPTIVE|180000|0.000268738969|2.70123813e-06|0.208333%|0.0520833%|2.81235445e-05|0|
|ADAPTIVE|190000|0.000307777729|6.61154249e-07|0.104167%|0%|3.07818566e-05|0|
|ADAPTIVE|200000|0.00029579772|6.16483181e-07|0.0520833%|0.0520833%|3.35957975e-05|0|

Every Actor update logs lambda, D_mu/D_p/D, unit/scaled anchor gradient norms, RL gradient norm, ratio, losses, LR and same-Critic pre/post score. At log intervals it also records common-RNG execution drift, weighted-mean drift, hidden drift, parameter RMS, entropy, top1 probability/mode histogram, mode rank changes and component pairwise distances.

## CLOSED LOOP / EARLIEST DIVERGENCE

Same known-success seeds20008,20002,20005,20007; horizon700; production GMM executor, categorical/Normal sampling, eval std1e-4, no external noise. All4 slots consume RNG every timestep, even after termination; inactive actions discarded. Four diagnostic environments initialized serially and closed after each evaluation.

|branch|step|seed|result + length|earliest action>1e-3|earliest EEF>1mm|
|---|---:|---:|---|---:|---:|
|BASELINE|140000|20008|SUCCESS 460|None|None|
|BASELINE|140000|20002|SUCCESS 423|None|None|
|BASELINE|140000|20005|SUCCESS 482|None|None|
|BASELINE|140000|20007|SUCCESS 444|None|None|
|BASELINE|150000|20008|SUCCESS 402|0|3|
|BASELINE|150000|20002|SUCCESS 448|0|3|
|BASELINE|150000|20005|FAIL 700|0|3|
|BASELINE|150000|20007|FAIL 700|0|4|
|BASELINE|160000|20008|FAIL 700|0|2|
|BASELINE|160000|20002|FAIL 700|0|2|
|BASELINE|160000|20005|FAIL 700|0|2|
|BASELINE|160000|20007|FAIL 700|0|2|
|BASELINE|170000|20008|FAIL 700|0|2|
|BASELINE|170000|20002|FAIL 700|0|2|
|BASELINE|170000|20005|FAIL 700|0|2|
|BASELINE|170000|20007|FAIL 700|0|2|
|BASELINE|180000|20008|FAIL 700|0|2|
|BASELINE|180000|20002|FAIL 700|0|2|
|BASELINE|180000|20005|FAIL 700|0|1|
|BASELINE|180000|20007|FAIL 700|0|2|
|BASELINE|190000|20008|FAIL 700|0|1|
|BASELINE|190000|20002|FAIL 700|0|1|
|BASELINE|190000|20005|FAIL 700|0|1|
|BASELINE|190000|20007|FAIL 700|0|1|
|BASELINE|200000|20008|FAIL 700|0|1|
|BASELINE|200000|20002|FAIL 700|0|1|
|BASELINE|200000|20005|FAIL 700|0|1|
|BASELINE|200000|20007|FAIL 700|0|1|
|ADAPTIVE|140000|20008|SUCCESS 460|None|None|
|ADAPTIVE|140000|20002|SUCCESS 423|None|None|
|ADAPTIVE|140000|20005|SUCCESS 482|None|None|
|ADAPTIVE|140000|20007|SUCCESS 444|None|None|
|ADAPTIVE|150000|20008|SUCCESS 460|38|232|
|ADAPTIVE|150000|20002|SUCCESS 409|37|55|
|ADAPTIVE|150000|20005|SUCCESS 503|48|55|
|ADAPTIVE|150000|20007|SUCCESS 443|50|60|
|ADAPTIVE|160000|20008|SUCCESS 460|49|261|
|ADAPTIVE|160000|20002|SUCCESS 438|36|56|
|ADAPTIVE|160000|20005|SUCCESS 509|18|55|
|ADAPTIVE|160000|20007|SUCCESS 546|58|60|
|ADAPTIVE|170000|20008|SUCCESS 460|26|228|
|ADAPTIVE|170000|20002|SUCCESS 428|5|57|
|ADAPTIVE|170000|20005|SUCCESS 503|16|56|
|ADAPTIVE|170000|20007|SUCCESS 546|4|59|
|ADAPTIVE|180000|20008|SUCCESS 460|8|232|
|ADAPTIVE|180000|20002|SUCCESS 400|19|55|
|ADAPTIVE|180000|20005|SUCCESS 481|18|89|
|ADAPTIVE|180000|20007|SUCCESS 501|9|87|
|ADAPTIVE|190000|20008|SUCCESS 401|32|58|
|ADAPTIVE|190000|20002|SUCCESS 409|9|57|
|ADAPTIVE|190000|20005|SUCCESS 503|14|55|
|ADAPTIVE|190000|20007|SUCCESS 428|27|60|
|ADAPTIVE|200000|20008|SUCCESS 399|36|92|
|ADAPTIVE|200000|20002|FAIL 700|38|60|
|ADAPTIVE|200000|20005|SUCCESS 482|36|92|
|ADAPTIVE|200000|20007|SUCCESS 549|19|57|

Selected deviations at0,1,2,3,5,9,10,20,50,100 and later steps are in each milestone JSON. Every diagnostic action, reward and EEF position is in trajectory_STEP.jsonl. Comparisons stop where the original reference episode terminated successfully; no invented reference beyond its termination.

## ACTUAL CHECKPOINT / JSON PATHS

- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/checkpoints/140K_start.pth` (20042116 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/checkpoints/last.pth` (36452180 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/checkpoints/latest.pth` (36458352 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/checkpoints/step_0150000.pth` (36463300 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/checkpoints/step_0160000.pth` (36463300 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/checkpoints/step_0170000.pth` (36465156 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/checkpoints/step_0180000.pth` (36465156 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/checkpoints/step_0190000.pth` (36465156 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/checkpoints/step_0200000.pth` (36465156 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/preservation_diagnostics/step_0140000.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/preservation_diagnostics/step_0150000.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/preservation_diagnostics/step_0160000.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/preservation_diagnostics/step_0170000.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/preservation_diagnostics/step_0180000.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/preservation_diagnostics/step_0190000.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/preservation_diagnostics/step_0200000.json`
- Per-update diagnostics: `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/BASELINE/mean2q/multi_q/actor_update_diagnostics.jsonl`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/checkpoints/140K_start.pth` (20042116 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/checkpoints/last.pth` (36452180 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/checkpoints/latest.pth` (36458352 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/checkpoints/step_0150000.pth` (36463300 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/checkpoints/step_0160000.pth` (36463300 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/checkpoints/step_0170000.pth` (36465156 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/checkpoints/step_0180000.pth` (36465156 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/checkpoints/step_0190000.pth` (36465156 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/checkpoints/step_0200000.pth` (36465156 bytes)
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/preservation_diagnostics/step_0140000.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/preservation_diagnostics/step_0150000.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/preservation_diagnostics/step_0160000.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/preservation_diagnostics/step_0170000.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/preservation_diagnostics/step_0180000.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/preservation_diagnostics/step_0190000.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/preservation_diagnostics/step_0200000.json`
- Per-update diagnostics: `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/ADAPTIVE/mean2q/multi_q/actor_update_diagnostics.jsonl`

- Preregistration: `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/preregistration.json`
- Full read/analysis JSON: `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/analysis_summary.json`
- Source integrity: `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/source_integrity.json`
- Experiment status: `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/adaptive_policy_preservation/results_20261003/experiment_status.json`

## VERDICT / NEXT STEP

`POLICY_FROZEN_NOT_SOLVED` — NO — evidence does not support it
The real BASELINE reproduces 4/4 at140K -> 0/4 by160K and through200K. ADAPTIVE preserves4/4 through190K and3/4 at200K, reducing fixed-data execution drift from1.3149064343 to0.0019763096 (99.8497%). However checkpoint-matched cumulative Actor optimizer-step improvement is only0.000534530729 versus0.055071733892:0.970608% retention. The common fixed140K Critic score gains are0.000139725191 versus0.018682705238:0.747885% retention. Both fail the intended materially-more-learning criterion; the historical2.65% is contextual, not a matched real-training fixed-anchor control. The ADAPTIVE controller has grad-ratio median1.18034/p90=3.78552/p95=5.99990,60.6369% updates above1,lambda=0 only2.35483%,lambda median30.38426 andp95=66.63427(max). This is functionally near-locked policy behavior, not literally frozen weights. Std gradients and std parameter drift remainzero. Choose only Actor update frequency / optimizer-step magnitude as the next direction; no further anchor retuning or experiment. Evidence is limited tofour preselected baseline-success seeds and matched staged-resume semantics, not a population success estimate. A testing-only stage-integrity assertion initially confused ready_step140000 with normal gate_open_step140001; only that assertion was corrected, completed BASELINE160K reused, and no gate/controller/production modification was made. Final cleanup audit found no testing trainer, orchestrator, vector child or formal trainer processes; allsix stage logs contain completed vector-env closure and alleight per-stage diagnostic-env closures, no Traceback or ERR99999. Source checkpoint SHA, preregistration SHA,production sourceSHA andHEAD remainunchanged. Formal training remainsstopped.
Only recommend Actor update frequency / optimizer-step magnitude as the next direction. No adaptive-threshold retuning or additional training is executed this round.

- Final orchestration status: COMPLETE.
- Full JSON/JSONL finite numeric fields read: 903248.
- Production source SHA256 unchanged: True.
- Final process/environment cleanup is checked separately; no testing trainer may be left running at completion.

FORMAL TRAINING REMAINS STOPPED
