# Actor-to-140K behavior-anchor causal diagnosis

Verdict: **YES — strong evidence**
Primary: `GENERAL_UNCONSTRAINED_ACTOR_DRIFT`
Mechanism: `INSUFFICIENT_POLICY_PRESERVATION`
The prespecified WEAK anchor restores at least 3/4 seeds while parameters still move and the frozen-score RL objective improves. This is short-horizon, four-seed mechanism evidence, not proof of a balanced formal-training fix.

## PRODUCTION ACTOR CONTRACT / BASELINE

- Source: `/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_readiness_v2_multi_mean_random_formal_20260929_130945/mean2q/multi_q/checkpoints/critic_ready.pth`; env_steps=140000, Actor updates=0, Adam state entries=0.
- RNNGMMActorNetwork; horizon=10, burn-in=0, zero-state reset at each aligned block; modes=5, action dimensions=14; full 10-step BPTT.
- V6 inherits the actual V5 actor_update: negative probability-weighted Q1 on the final token only. No changes to RL aggregation or temporal credit assignment.
- Means are tanh(raw means) in normalized coordinates (`use_tanh=False` denotes no distribution wrapper, not unsquashed means). Environment action = normalized action * scale + offset.
- Adam: LR initially0, betas=(0.9,0.999), eps=1e-8, weight_decay=0; all branches restore the same checkpoint optimizer state; policy_delay=4, max grad norm=10, production UTD=.25.
- RL gradients reach RNN/mean/logits, not std. Encoder has no independently trainable parameters in this checkpoint.
- Production execution: BatchedGMMExecutor, recurrent reset every10 steps, categorical component sample plus Normal sample, eval std=1e-4, no external noise. Same prior evaluator and four seeds in the same order.
- Training-mode std remains learned softplus+min_std but neither the RL objective nor anchor uses it.
- All original scale/offset values are recorded in `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_init_anchor/results_20261003/experiment_contract.json`.

## LR / REPLAY CONTRACT

**APPROXIMATE_PRODUCTION_LR_REPLAY**: exact scheduler formula, approximate update-to-env mapping.
- Formula: actor_lr = 2e-6 * min(1,max(0,env_steps-140000)/140000).
- Replay first LR=4e-10 at interpolated env=140028; update1000 LR=2.299571429e-07 at env=156097.
- Actual first sparse log anchors: Actor update13/env140368, update38/env140816, update63/env141200. Logging is every100 Critic updates. Async collection and delayed update-credit consumption prevent exact per-update env timestamps from being reconstructed from UTD/policy_delay alone.
- Therefore no claim of exact stochastic production replay or exact per-update production LR. All branches use the SAME saved interpolated sequence and deterministic batches.
- Fixed data: `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_temporal_supervision/results_20261003/fixed_temporal_contexts.npz`; hash=bb38dbff40c2194f0b59d88d3a887b73fa4705a6fccca5136f56df704516fc77. 192 aligned Actor 10-step contexts, histories stored as19 steps; 64 offline success,64 online success,64 online failure.
- Testing batch64 (not production256), fixed repeating4 batches: 32 offline +16 online success +16 online failure; same data and sequence as prior CURRENT.
- BASELINE reproduces previous CURRENT exactly: segment differences={'early': 0.0, 'mid': 0.0, 'late': 0.0, 'final': 0.0}.

## ANCHOR FORMULATION

`L_mu = mean_batch,time sum_k p_ref[t,k] * sum_dim (mu_current[t,k]-mu_ref[t,k])^2`
`L_p = mean_batch,time KL(p_ref || p_current)`
`L_anchor = L_mu + beta*L_p`
`L_total = L_RL + lambda*L_anchor`

- Targets are the fixed deep-copy 140K Actor, NOT replay/expert actions. Reference eval(), requires_grad=False, no optimizer membership, no Polyak; same observations/zero-state semantics as current Actor; full10 timesteps.
- No std anchor, no parameter-L2 anchor. Mode matching is index-matched; nearest-reference-component mapping was identity in every observed milestone (no permutation ambiguity).

## LAMBDA_CALIBRATION_METHOD

25 deterministic production-RL virtual probe updates at unmodified LR; median ratios after each small update; all main branches reset to original 140K empty Adam
- Probe maximum normalized component L2=0.0006126568769 <=1e-3.
- beta=0.396452595479: median full-Actor ||g_mu||/||g_KL|| across25 post-probe states; balances component/probability gradient norms, not loss values.
- lambda = requested ratio / median(||g_mu+beta*g_KL||/||g_RL||). Full active-parameter L2 norm, equivalent ratio to RMS on the same parameter set. Mean/probability gradient cross terms are included.
- Probe main-policy states are discarded; every main branch restarts from identical140K parameters and empty Adam. No LR amplification; one sweep only. Calibration procedures, per-update losses/norms/cross terms and group ratios are preserved in calibration.json.

| branch | lambda_mu | beta | effective lambda_KL | total grad ratio | RNN | mean | logits |
|---|---:|---:|---:|---:|---:|---:|---:|
|BASELINE|0|0.396452595|0|0|0|0|0|
|WEAK|66.6342693|0.396452595|26.417329|0.1|0.314441|0.0638878|0.171257|
|MEDIUM|199.902808|0.396452595|79.251987|0.3|0.943324|0.191664|0.513772|
|STRONG|666.342693|0.396452595|264.17329|1|3.14441|0.638878|1.71257|

These are CALIBRATION ratios, not constant update-time ratios. At update1000 (gradients measured just before that step):

|branch|total actual ratio|RNN|mean|logits|
|---|---:|---:|---:|---:|
|BASELINE|0|0|0|0|
|WEAK|1.16991|2.20568|1.06609|1.28249|
|MEDIUM|1.40089|3.70545|1.08509|2.16216|
|STRONG|2.24758|6.57836|1.57878|4.69401|

## OFFLINE RESULTS

Execution drift is common-RNG production sampled normalized action mean L2 across all10 steps. EARLY/MID/LATE/FINAL below are weighted-mixture-mean L2 for comparability with previous task; do not conflate these two metrics.

|branch|updates|execution drift|early0:3|mid3:7|late7:9|final9|KL(ref||current)|RNN RMS|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|BASELINE|0|0|0|0|0|0|0|0|
|BASELINE|1|1.9099339e-07|1.9568603e-07|2.7809251e-07|2.7961708e-07|2.8723543e-07|4.4674441e-10|1.4073262e-10|
|BASELINE|10|1.6380233e-05|1.5425269e-05|2.3916806e-05|2.7223538e-05|2.731846e-05|-1.0562255e-09|1.3393587e-08|
|BASELINE|25|0.001187126|8.9287267e-05|0.0001432535|0.0001667342|0.00016759083|1.3676689e-08|8.0769875e-08|
|BASELINE|50|0.0014139758|0.00030750755|0.00049576628|0.0005786044|0.00058157394|1.7585615e-07|2.8073862e-07|
|BASELINE|100|0.0021589423|0.0010714543|0.0017297681|0.0020195814|0.0020293385|2.1575953e-06|9.8043946e-07|
|BASELINE|250|0.01153018|0.0059809598|0.0096608752|0.011233294|0.011277913|6.7632069e-05|5.4540191e-06|
|BASELINE|500|0.042948391|0.022753603|0.036697565|0.042132918|0.042291796|0.0009705334|2.0567183e-05|
|BASELINE|750|0.097579742|0.051406383|0.082684463|0.093336624|0.093878796|0.0049785498|4.5942503e-05|
|BASELINE|1000|0.1639432|0.091184997|0.14643988|0.16324648|0.16375048|0.016027753|8.1170564e-05|
|WEAK|0|0|0|0|0|0|0|0|
|WEAK|1|1.9086763e-07|1.955102e-07|2.7754211e-07|2.8082704e-07|2.9164909e-07|1.9576476e-10|1.4073374e-10|
|WEAK|10|1.6094631e-05|1.4461216e-05|2.2790001e-05|2.6131581e-05|2.6312468e-05|-7.1691063e-10|1.331128e-08|
|WEAK|25|8.0092392e-05|6.2820073e-05|0.00010197782|0.00012292914|0.00012833252|6.9346784e-09|7.7602469e-08|
|WEAK|50|0.001277304|0.00013126804|0.00021609572|0.00026630039|0.00028647835|2.6511048e-08|2.5164628e-07|
|WEAK|100|0.0013965765|0.00018813445|0.00033125337|0.00041374977|0.00045939233|3.8130832e-08|8.062241e-07|
|WEAK|250|0.0014857695|0.00017198537|0.00041745062|0.00059159927|0.0007341011|7.4122066e-08|3.4327562e-06|
|WEAK|500|0.0015298451|0.00014734604|0.00044026568|0.00071959448|0.00098535177|1.1906318e-07|8.6113878e-06|
|WEAK|750|0.001684895|0.00015136457|0.00047500026|0.00083134822|0.0011672288|1.6615844e-07|1.4471622e-05|
|WEAK|1000|0.00047449029|0.00017817415|0.00047520682|0.00087380418|0.0012785719|1.8730661e-07|2.0730208e-05|
|MEDIUM|0|0|0|0|0|0|0|0|
|MEDIUM|1|1.9077676e-07|1.9547145e-07|2.779409e-07|2.8106114e-07|2.969156e-07|-1.450409e-10|1.4072443e-10|
|MEDIUM|10|1.5555334e-05|1.3132461e-05|2.0881592e-05|2.4120229e-05|2.4467355e-05|2.9236356e-10|1.3158109e-08|
|MEDIUM|25|6.2085244e-05|4.5899009e-05|7.4705739e-05|9.1220966e-05|9.7179303e-05|2.6191026e-09|7.2709533e-08|
|MEDIUM|50|0.00010289601|6.7204395e-05|0.00011174401|0.00013699505|0.00015060775|3.8452694e-09|2.2500102e-07|
|MEDIUM|100|0.00012379273|6.4543731e-05|0.00013893546|0.00018437534|0.0002160171|5.6313742e-09|6.8804119e-07|
|MEDIUM|250|0.00014582985|5.8048888e-05|0.00016233094|0.00024318247|0.00031318054|1.4326651e-08|2.3022384e-06|
|MEDIUM|500|0.0012546616|6.7289375e-05|0.00018750103|0.00030232579|0.00041381527|2.1207317e-08|5.252772e-06|
|MEDIUM|750|0.0003048144|7.4962024e-05|0.00021523318|0.00035518115|0.00048499988|3.7940002e-08|8.3425274e-06|
|MEDIUM|1000|0.00017662725|5.6951277e-05|0.00016098492|0.00032998797|0.00050003818|2.1805206e-08|1.1324078e-05|
|STRONG|0|0|0|0|0|0|0|0|
|STRONG|1|1.9074189e-07|1.940892e-07|2.814401e-07|2.8077511e-07|2.9163449e-07|-1.1556213e-11|1.4075281e-10|
|STRONG|10|1.4042612e-05|1.0993987e-05|1.715879e-05|2.0182475e-05|2.0935577e-05|1.2017307e-10|1.2645271e-08|
|STRONG|25|3.440836e-05|2.3947603e-05|3.7685821e-05|4.5716496e-05|4.9701057e-05|1.7806226e-09|6.2908795e-08|
|STRONG|50|3.9064755e-05|2.1920026e-05|4.3874045e-05|5.7238614e-05|6.5673907e-05|1.6198097e-10|1.8611702e-07|
|STRONG|100|4.2739069e-05|2.005492e-05|4.9926682e-05|7.0822333e-05|8.8123786e-05|1.7376169e-09|4.6092861e-07|
|STRONG|250|4.9568888e-05|2.0332514e-05|6.4161672e-05|0.00010347231|0.00013199242|3.4464228e-09|1.3001771e-06|
|STRONG|500|6.2544387e-05|3.5823912e-05|8.4611493e-05|0.00012915154|0.00016717151|5.412748e-09|2.631428e-06|
|STRONG|750|5.4303415e-05|2.0915854e-05|5.5482647e-05|0.00010850693|0.00015832698|2.2186581e-09|3.8042048e-06|
|STRONG|1000|7.2937042e-05|6.3605137e-05|0.0001231803|0.00016457253|0.00021185792|1.7787753e-08|4.8797471e-06|

Tiny negative KL values around1e-9 at near-identical checkpoints arise from float32 probability normalization/roundoff; no nonfinite values. Actual gradients use production float32; no clipping of the underlying diagnostic KL data.

## RL VS PRESERVATION

|branch|frozen-score RL improvement|drift reduction|RL retention|total parameter L2|
|---|---:|---:|---:|---:|
|BASELINE|0.004508343525|0.00000%|100.00000%|0.11654896|
|WEAK|0.0001192757918|99.71058%|2.64567%|0.029546639|
|MEDIUM|4.672480645e-05|99.89226%|1.03641%|0.016156087|
|STRONG|1.538843753e-05|99.95551%|0.34133%|0.0069772688|

Initial fixed-data final-token Q1=0.1499640197. We treat Q1 only as the identical frozen optimization score; do not interpret its correctness or analyze Critic-side mechanisms.
- WEAK is not literally frozen: RNN RMS=2.0730e-5 and total parameter L2=.0295466 (about25.35% of baseline L2), positive increasing RL gain. However it retains only2.65% of baseline score gain, and its late gradient ratio exceeds1. This is a major preservation/optimization tradeoff.
- The prespecified heuristic offline eligibility (>=30% drift reduction, >=10% RL retention) selected NO qualified weak/medium branch. The fallback WEAK is tested as a diagnostic, not declared an already-balanced repair. STRONG is the fixed upper bound. These heuristic cutoffs are not an arbitrary combined score or a production gate.
- Actual offline selection: `{'selected': 'WEAK', 'eligible': [], 'reason': 'no weak/medium qualified; WEAK diagnostic and STRONG upper bound, not an effective-anchor claim', 'branches': ['ORIGINAL_140K', 'BASELINE', 'WEAK', 'STRONG'], 'success_conditioned_retuning': False}`.

## CLOSED LOOP

Same fixed seeds20008,20002,20005,20007; horizon700; no external noise. All4 slots consume identical categorical/Gaussian RNG each timestep, including completed slots; inactive actions are discarded. Each group builds4 environments serially and closes all4 before the next group. No formal rollout/training.

|branch|success|mean length|sim errors|earliest action divergence per seed|earliest EEF>1mm per seed|
|---|---:|---:|---:|---|---|
|ORIGINAL_140K|4/4|452.25|0|[None, None, None, None]|[None, None, None, None]|
|BASELINE|0/4|700.00|0|[0, 0, 0, 0]|[2, 2, 2, 3]|
|WEAK|3/4|529.25|0|[36, 43, 48, 52]|[232, 56, 55, 60]|
|STRONG|4/4|483.50|0|[68, 59, 58, 60]|[290, 84, 55, 60]|

Action divergence threshold: normalized action vector L2>1e-3. EEF threshold: either arm pre-step position distance>1mm. Null for ORIGINAL means not applicable (self-reference).

|seed|ORIGINAL140K|BASELINE|WEAK|STRONG|
|---|---|---|---|---|
|20008|SUCCESS 460|FAIL 700|SUCCESS 460|SUCCESS 460|
|20002|SUCCESS 423|FAIL 700|FAIL 700|SUCCESS 409|
|20005|SUCCESS 482|FAIL 700|SUCCESS 513|SUCCESS 519|
|20007|SUCCESS 444|FAIL 700|SUCCESS 444|SUCCESS 546|

### Selected closed-loop action / EEF deviations

|branch|seed|step|normalized action L2|arm0 EEF m|arm1 EEF m|
|---|---:|---:|---:|---:|---:|
|BASELINE|20008|0|0.035461551|0|0|
|BASELINE|20008|1|0.066688079|5.1301458e-05|0.00037841596|
|BASELINE|20008|2|0.11255155|0.00034531253|0.0011016138|
|BASELINE|20008|3|0.11390489|0.0010447221|0.0019489732|
|BASELINE|20008|5|0.11627485|0.0030003943|0.0036084297|
|BASELINE|20008|9|0.15713553|0.0082303927|0.0065606837|
|BASELINE|20008|10|0.045240413|0.0098749677|0.0072988588|
|BASELINE|20008|20|0.057295121|0.015593233|0.010144288|
|BASELINE|20008|50|0.082825799|0.016611623|0.009902199|
|BASELINE|20008|100|0.16930339|0.075063803|0.027606599|
|BASELINE|20008|200|2.0928189|0.069718913|0.10720643|
|BASELINE|20008|300|0.2634991|0.089178823|0.12621493|
|BASELINE|20008|400|2.1292666|0.53749804|0.4156687|
|BASELINE|20002|0|0.035554461|0|0|
|BASELINE|20002|1|0.072436102|6.6681871e-05|0.00039240073|
|BASELINE|20002|2|0.10966501|0.00041212595|0.0011417416|
|BASELINE|20002|3|0.11677024|0.001147388|0.0019916118|
|BASELINE|20002|5|0.12196772|0.0032917105|0.0035545834|
|BASELINE|20002|9|0.075789124|0.0076792672|0.006054966|
|BASELINE|20002|10|0.031908043|0.0083615966|0.006620199|
|BASELINE|20002|20|0.059991589|0.0080825932|0.0082136261|
|BASELINE|20002|50|0.05293418|0.026251979|0.015364011|
|BASELINE|20002|100|0.18831938|0.085204867|0.021213022|
|BASELINE|20002|200|0.12990297|0.055098469|0.061575068|
|BASELINE|20002|300|2.0437439|0.15358476|0.13757644|
|BASELINE|20002|400|2.2339258|0.45037039|0.3625652|
|BASELINE|20005|0|0.035919462|0|0|
|BASELINE|20005|1|0.073465913|6.3209492e-05|0.00041459684|
|BASELINE|20005|2|0.11172489|0.00040434885|0.0012395476|
|BASELINE|20005|3|0.11731323|0.0011390378|0.0022076043|
|BASELINE|20005|5|0.11447998|0.0032032313|0.0041025873|
|BASELINE|20005|9|0.080045071|0.0074455059|0.0070459643|
|BASELINE|20005|10|0.033711082|0.0081526367|0.0076703988|
|BASELINE|20005|20|0.066498896|0.0085012531|0.0064144161|
|BASELINE|20005|50|0.080341462|0.014635185|0.016290972|
|BASELINE|20005|100|0.16996357|0.065921186|0.038848969|
|BASELINE|20005|200|2.9165044|0.056616212|0.3768159|
|BASELINE|20005|300|2.0871296|0.51187085|0.56684385|
|BASELINE|20005|400|2.353729|0.53047682|0.50597976|
|BASELINE|20007|0|0.034846887|0|0|
|BASELINE|20007|1|0.054063319|7.4313516e-05|0.00039311525|
|BASELINE|20007|2|0.10686795|0.0001385685|0.00097319259|
|BASELINE|20007|3|0.12879162|0.00066946057|0.0017865059|
|BASELINE|20007|5|0.14036759|0.002999526|0.0034423192|
|BASELINE|20007|9|0.13173386|0.0096936699|0.0056643673|
|BASELINE|20007|10|0.03939121|0.011375045|0.0060611498|
|BASELINE|20007|20|0.070300965|0.014605907|0.0083465371|
|BASELINE|20007|50|0.12824275|0.017018447|0.0095001589|
|BASELINE|20007|100|0.1127538|0.0070967099|0.035906223|
|BASELINE|20007|200|2.0597556|0.041362394|0.076517571|
|BASELINE|20007|300|2.0443387|0.11055667|0.52445565|
|BASELINE|20007|400|2.2466738|0.37053036|0.38505678|
|WEAK|20008|0|4.0911163e-05|0|0|
|WEAK|20008|1|0.00024143245|5.4086488e-07|2.6977661e-07|
|WEAK|20008|2|0.00026700823|1.2418941e-06|5.0056754e-06|
|WEAK|20008|3|0.00031144906|4.7606951e-06|9.5601389e-06|
|WEAK|20008|5|0.00018458153|1.270725e-05|1.2037084e-05|
|WEAK|20008|9|0.00039129642|1.1310845e-05|1.1535857e-05|
|WEAK|20008|10|0.00011927777|1.2068092e-05|1.1911359e-05|
|WEAK|20008|20|7.7928393e-05|1.9238536e-05|2.3088461e-05|
|WEAK|20008|50|0.00077009103|7.2144382e-05|5.6939151e-05|
|WEAK|20008|100|0.00039633539|6.9285033e-05|4.6029944e-05|
|WEAK|20008|200|0.003756872|0.00043453977|0.00029176005|
|WEAK|20008|300|0.004981303|0.0038372771|0.0016258916|
|WEAK|20008|400|0.0035255835|0.0043667817|0.0036601814|
|WEAK|20002|0|5.9269364e-05|0|0|
|WEAK|20002|1|0.00024304819|2.7184662e-07|6.7259929e-07|
|WEAK|20002|2|0.00016321682|8.1583759e-07|4.5999253e-06|
|WEAK|20002|3|0.00022838572|2.4876732e-06|8.0278474e-06|
|WEAK|20002|5|0.00026677483|6.3613851e-06|1.1007674e-05|
|WEAK|20002|9|0.00027585868|1.2290275e-05|1.4614796e-05|
|WEAK|20002|10|8.2155716e-05|1.2977888e-05|1.5447843e-05|
|WEAK|20002|20|8.6361568e-05|7.3689253e-06|1.2522994e-05|
|WEAK|20002|50|0.0011957557|5.2573579e-05|2.0445526e-05|
|WEAK|20002|100|0.0056658462|0.0020004716|0.00042212648|
|WEAK|20002|200|0.13395525|0.030869655|0.010545262|
|WEAK|20002|300|2.1340501|0.27618949|0.033972345|
|WEAK|20002|400|2.8631019|0.16646442|0.54830491|
|WEAK|20005|0|5.0815976e-05|0|0|
|WEAK|20005|1|0.00027277826|3.320302e-07|6.4892528e-07|
|WEAK|20005|2|0.00020113976|8.9403761e-07|5.0871563e-06|
|WEAK|20005|3|0.0002310993|2.3891519e-06|8.9875347e-06|
|WEAK|20005|5|0.00032669467|5.772872e-06|1.4705242e-05|
|WEAK|20005|9|0.00049101327|1.3445015e-05|2.7537113e-05|
|WEAK|20005|10|0.00012265797|1.7156607e-05|3.0183191e-05|
|WEAK|20005|20|7.9007885e-05|1.5371895e-05|2.218395e-05|
|WEAK|20005|50|0.00014406076|2.297417e-05|5.2339445e-05|
|WEAK|20005|100|0.0034584082|0.0018110143|0.0002497815|
|WEAK|20005|200|0.0045264922|0.0018106372|0.0027781097|
|WEAK|20005|300|0.081307106|0.016702877|0.039338543|
|WEAK|20005|400|2.3341275|0.069856317|0.066082846|
|WEAK|20007|0|5.5837321e-05|0|0|
|WEAK|20007|1|0.00016149954|3.0989773e-07|5.7369209e-07|
|WEAK|20007|2|0.00019480103|1.8923783e-06|1.2629476e-06|
|WEAK|20007|3|0.00029473958|8.4549491e-07|5.7183884e-06|
|WEAK|20007|5|0.00031248539|7.3102663e-06|1.2220467e-05|
|WEAK|20007|9|0.00077228253|2.4009336e-05|9.1380735e-06|
|WEAK|20007|10|8.5912335e-05|2.7383521e-05|7.1534765e-06|
|WEAK|20007|20|0.00013142293|2.221118e-05|5.3463031e-06|
|WEAK|20007|50|0.00043021878|2.0151481e-05|2.6214494e-05|
|WEAK|20007|100|0.0046565471|0.0022580737|0.0011489747|
|WEAK|20007|200|0.021347578|0.0027839043|0.0022391568|
|WEAK|20007|300|0.020577067|0.010174912|0.0033188714|
|WEAK|20007|400|0.0068361281|0.012105185|0.0051634872|
|STRONG|20008|0|1.2139304e-05|0|0|
|STRONG|20008|1|3.5708914e-05|1.4742469e-07|1.1994058e-07|
|STRONG|20008|2|3.7785591e-05|4.5448085e-07|2.8613893e-07|
|STRONG|20008|3|4.8158987e-05|3.7660297e-07|5.2540811e-07|
|STRONG|20008|5|3.8133969e-05|1.0010152e-06|1.2659348e-06|
|STRONG|20008|9|8.500986e-05|2.2585732e-06|2.4608189e-06|
|STRONG|20008|10|1.2077601e-05|2.853075e-06|2.8320088e-06|
|STRONG|20008|20|3.191038e-05|4.7566995e-06|3.8280244e-06|
|STRONG|20008|50|6.7262542e-05|7.7287583e-06|1.1150862e-05|
|STRONG|20008|100|0.00032924073|0.00015797888|4.2205495e-05|
|STRONG|20008|200|0.00015008095|4.2473148e-05|4.9772876e-05|
|STRONG|20008|300|0.0012814421|0.00096981241|0.00033086639|
|STRONG|20008|400|0.0063335177|0.0028394521|0.0010584471|
|STRONG|20002|0|9.5769995e-06|0|0|
|STRONG|20002|1|4.3824337e-05|8.5018039e-08|4.9305033e-08|
|STRONG|20002|2|4.3507339e-05|4.6183115e-07|2.0816551e-07|
|STRONG|20002|3|4.0692817e-05|7.7249617e-07|7.0401878e-07|
|STRONG|20002|5|5.1898618e-05|1.3813395e-06|1.5947109e-06|
|STRONG|20002|9|5.6355573e-05|3.0561453e-06|1.8586035e-06|
|STRONG|20002|10|1.2672456e-05|3.4908795e-06|2.0099093e-06|
|STRONG|20002|20|1.6763662e-05|3.1863659e-06|3.7401053e-06|
|STRONG|20002|50|5.9456353e-05|8.5111944e-06|5.4269633e-06|
|STRONG|20002|100|0.018440146|0.0027162914|0.0022829996|
|STRONG|20002|200|0.11220248|0.042934479|0.010173814|
|STRONG|20002|300|0.1098316|0.082810031|0.038147122|
|STRONG|20002|400|0.074152085|0.28529924|0.029618897|
|STRONG|20005|0|1.0273985e-05|0|0|
|STRONG|20005|1|4.8312955e-05|9.5533118e-08|8.3650398e-08|
|STRONG|20005|2|5.7788141e-05|5.2114563e-07|3.3340045e-07|
|STRONG|20005|3|4.636585e-05|9.1414063e-07|1.0456647e-06|
|STRONG|20005|5|5.2387666e-05|1.4926837e-06|2.7399706e-06|
|STRONG|20005|9|8.2884816e-05|3.0345575e-06|4.9254148e-06|
|STRONG|20005|10|1.6126706e-05|3.6595368e-06|5.19453e-06|
|STRONG|20005|20|2.5063924e-05|5.1559785e-06|2.0183236e-06|
|STRONG|20005|50|4.6038448e-05|5.8743978e-06|4.5215922e-06|
|STRONG|20005|100|0.0031520618|0.0015333447|0.00023700243|
|STRONG|20005|200|0.0041725891|0.0015898131|0.0026292079|
|STRONG|20005|300|0.084043075|0.016427698|0.039686897|
|STRONG|20005|400|2.3334249|0.070147284|0.063670546|
|STRONG|20007|0|9.5472073e-06|0|0|
|STRONG|20007|1|4.5738608e-05|7.9889463e-08|3.2384153e-08|
|STRONG|20007|2|3.9092451e-05|3.8173257e-07|8.5588888e-07|
|STRONG|20007|3|3.7304362e-05|4.336636e-07|1.7314045e-06|
|STRONG|20007|5|4.1405523e-05|7.1735056e-07|2.5179317e-06|
|STRONG|20007|9|7.3716108e-05|2.290323e-06|1.1630633e-06|
|STRONG|20007|10|1.7967232e-05|2.6794679e-06|5.8310525e-07|
|STRONG|20007|20|1.7914678e-05|2.9732306e-06|2.2857534e-06|
|STRONG|20007|50|0.00016149263|5.7505021e-06|5.9314453e-06|
|STRONG|20007|100|0.0057355096|0.0021640409|0.0015552133|
|STRONG|20007|200|0.063370013|0.020725654|0.015311959|
|STRONG|20007|300|0.17535897|0.041679528|0.07132252|
|STRONG|20007|400|2.8588718|0.17638631|0.38280066|

All actual actions/EEF/rewards for every diagnostic frame are saved in closed_loop_steps.jsonl. Paired deviations only exist while the corresponding ORIGINAL episode is still alive; no invented original reference after its success termination.

## GMM ATTRIBUTION

|branch|component mean RMS|normalized categorical logits RMS|KL|mode-rank changes|top1 changes|mean top1 probability|mode histogram|
|---|---:|---:|---:|---:|---:|---:|---|
|BASELINE|0.022242816|0.47810292|0.016027753|11.30208%|4.37500%|0.89642349|[665, 793, 0, 344, 118]|
|WEAK|0.00024849877|0.0038679861|1.8730661e-07|0.10417%|0.00000%|0.89485838|[664, 808, 0, 315, 133]|
|MEDIUM|0.00010048149|0.001383728|2.1805206e-08|0.05208%|0.00000%|0.89486292|[664, 808, 0, 315, 133]|
|STRONG|3.7025207e-05|0.0005137797|1.7787753e-08|0.00000%|0.00000%|0.89486489|[664, 808, 0, 315, 133]|

- Nearest-reference mode mapping remains identity, including BASELINE. No mode permutation mechanism is supported.
- Combined anchor protects both component locations and mode probabilities; no factorial mean-only/logits-only causal attribution is claimed. Probability drift is already suppressed, so the conditional extra mechanism test is not justified and was not run.
- No std gradient or parameter drift in any branch; no independently trainable encoder group.

### Parameter drift at1000

|branch|group|L2|RMS|relative L2|
|---|---|---:|---:|---:|
|BASELINE|mean|0.0158177573|9.44112399e-05|0.0017567656|
|BASELINE|std|0|0|0|
|BASELINE|logits|0.00436960775|9.75854938e-05|0.00025539416|
|BASELINE|rnn|0.115387889|8.11705636e-05|0.00110320131|
|BASELINE|total|0.116548959|8.08326653e-05|0.00107341967|
|WEAK|mean|0.00210969533|1.25921107e-05|0.000234308829|
|WEAK|std|0|0|0|
|WEAK|logits|0.000362535764|8.09643191e-06|2.11894345e-05|
|WEAK|rnn|0.0294689948|2.07302078e-05|0.000281747364|
|WEAK|total|0.0295466394|2.04921059e-05|0.000272125501|
|MEDIUM|mean|0.00135945345|8.11415195e-06|0.000150984809|
|MEDIUM|std|0|0|0|
|MEDIUM|logits|0.000185144988|4.13480251e-06|1.08213257e-05|
|MEDIUM|rnn|0.0160977253|1.13240778e-05|0.000153907241|
|MEDIUM|total|0.0161560872|1.12050729e-05|0.000148798083|
|STRONG|mean|0.000747517083|4.46169538e-06|8.30213965e-05|
|STRONG|std|0|0|0|
|STRONG|logits|6.59220455e-05|1.47222262e-06|3.85300156e-06|
|STRONG|rnn|0.00693679694|4.87974711e-06|6.6321375e-05|
|STRONG|total|0.00697726876|4.83909281e-06|6.42608701e-05|

### Per-timestep execution and hidden drift at1000

|branch|step|sampled action L2|weighted mean L2|hidden L2|
|---|---:|---:|---:|---:|
|BASELINE|0|0.0726837673|0.0528190089|0.0265092367|
|BASELINE|1|0.165693512|0.099769628|0.0845167067|
|BASELINE|2|0.129518281|0.120966356|0.112541985|
|BASELINE|3|0.14154346|0.133640756|0.127991156|
|BASELINE|4|0.216448669|0.143629233|0.139485459|
|BASELINE|5|0.16923719|0.150791863|0.150392475|
|BASELINE|6|0.142131777|0.157697657|0.161829774|
|BASELINE|7|0.17541989|0.162306379|0.173366622|
|BASELINE|8|0.211520877|0.16418658|0.185182145|
|BASELINE|9|0.215234576|0.16375048|0.196093576|
|WEAK|0|5.05619039e-05|7.88164758e-05|0.0024027724|
|WEAK|1|0.000125802616|0.000179726851|0.00529010927|
|WEAK|2|0.000202783077|0.000275979125|0.00652654251|
|WEAK|3|0.000270996462|0.000338631091|0.006853595|
|WEAK|4|0.000349858122|0.000414840419|0.0069408545|
|WEAK|5|0.000462815292|0.000518184707|0.0070159449|
|WEAK|6|0.000581325558|0.000629171045|0.00714647501|
|WEAK|7|0.000717125445|0.000766174531|0.00734136564|
|WEAK|8|0.000892353981|0.000981433829|0.00767003084|
|WEAK|9|0.00109128049|0.00127857191|0.00806773775|
|MEDIUM|0|2.56098736e-05|2.59023962e-05|0.000919162771|
|MEDIUM|1|5.48658169e-05|5.94844227e-05|0.00222817365|
|MEDIUM|2|7.78412925e-05|8.54670113e-05|0.00285427431|
|MEDIUM|3|9.61841819e-05|0.000110709231|0.00304189218|
|MEDIUM|4|0.000117697388|0.000135617944|0.00307998429|
|MEDIUM|5|0.00015488069|0.000174110444|0.00309636473|
|MEDIUM|6|0.000205486621|0.000223502062|0.00312864861|
|MEDIUM|7|0.000265316944|0.000284655996|0.00318529038|
|MEDIUM|8|0.000341183186|0.000375319935|0.00329736802|
|MEDIUM|9|0.000427206525|0.000500038184|0.00345142749|
|STRONG|0|1.90469307e-05|2.76899543e-05|0.000320016794|
|STRONG|1|3.52527063e-05|6.89685936e-05|0.000770178723|
|STRONG|2|4.36982597e-05|9.41568628e-05|0.000997638358|
|STRONG|3|4.93581423e-05|0.000109361585|0.00107225233|
|STRONG|4|5.77406066e-05|0.000119116888|0.00108833111|
|STRONG|5|6.53452441e-05|0.00012601646|0.00109383925|
|STRONG|6|8.05152646e-05|0.000138226263|0.00110277414|
|STRONG|7|9.92201685e-05|0.000154214397|0.00111776935|
|STRONG|8|0.000124022354|0.000174930659|0.00115028486|
|STRONG|9|0.000155170747|0.000211857924|0.00120103496|

Per-mode/per-action-dimension component drift, per-timestep mean absolute/max action drift, hidden output drift, entropy, mode ranks, every milestone and every measured gradient ratio are fully preserved in offline_results.json and output NPZ files.

## VERDICT / SCOPE

**YES — strong evidence**
Primary: `GENERAL_UNCONSTRAINED_ACTOR_DRIFT`.
The prespecified WEAK anchor restores at least 3/4 seeds while parameters still move and the frozen-score RL objective improves. This is short-horizon, four-seed mechanism evidence, not proof of a balanced formal-training fix.
- Four prespecified branches ×1000 virtual Actor updates, no Critic updates; one25-update discarded calibration probe; four closed-loop groups only. No success-conditioned tuning or extra experiment.
- Reference hash/norm/output invariance checked at36 main milestones; frozen Critic hash unchanged; all test losses/gradients/model outputs finite; production hashes unchanged.
- Diagnostic env steps=8660; episodes=16. No formal environment steps or training checkpoint writes.
- Limits:4 selected baseline-success seeds,1000 virtual updates,fixed64-batch states,approximate LR timestamps,frozen RL score; no claim about3M learning or broad success distribution.

## NEXT STEP (recommendation only)

Only recommend a testing-only short real Stage3-v6 replay/training branch with the weakest tested preservation anchor, keeping all other algorithm factors fixed. Explicitly measure RL retention because the current calibration strongly suppresses optimization. Do not resume formal training now.

## ARTIFACTS

- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_init_anchor/results_20261003/analysis_summary.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_init_anchor/results_20261003/experiment_contract.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_init_anchor/results_20261003/calibration.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_init_anchor/results_20261003/schedule.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_init_anchor/results_20261003/offline_results.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_init_anchor/results_20261003/closed_loop.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_init_anchor/results_20261003/closed_loop_steps.jsonl`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_init_anchor/results_20261003/safety_audit_start.json`

Testing code: `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_init_anchor/run_anchor.py` and `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_init_anchor/run_anchor_closed_loop.py`. Testing-only Actor checkpoints are saved ONLY in the results directory and explicitly tagged testing_only, not in TARGET_RUN.

FORMAL TRAINING REMAINS STOPPED
