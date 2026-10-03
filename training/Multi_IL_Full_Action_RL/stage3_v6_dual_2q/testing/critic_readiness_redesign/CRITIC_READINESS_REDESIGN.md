# Stage3-v6 Multi Critic readiness: offline audit and v2 design

Date: 2026-09-29. Repository HEAD: `aa2180b`. This is a test-only, offline audit. Environment steps executed: 0; optimizer steps executed: 0. No production code, checkpoint, simulator, training process, threshold, or actor gate was changed.

Sources: the two formal `multi_q/readiness_metrics.jsonl` files, their step0/100K/200K/300K checkpoints and 100K frozen replay snapshots, Stage3-v5/v6 readiness/agent/schedule/config source, and the official PIRLNav README/config/scheduler/trainer. The two branches use different branch-local fixed sets (mean: 117 episodes; random: 118), each with 256 fixed TD sequences. Within each branch the 100K/200K/300K stored sequence hashes agree. Both step0 Critic state dictionaries match the Stage2.2 Multi 5K source exactly (maximum tensor difference 0). Comparisons over time within a branch are paired; direct mean-versus-random numeric differences are not perfectly paired.

## 1. Why the current gate fails

Both formal runs reached 300K with 74,750 Critic updates and zero Actor updates, then entered `CRITIC_NOT_READY`; this is a deliberate timeout, not a crash. All 21 readiness checks per branch had `overall_pass=false`. Mean: twin p95 passed 0/21 and TD plateau 1/21. Random: twin p95 passed 2/21 (100K, 110K) and TD plateau 2/21 (240K, 300K). By contrast, data counts, AUC, twin median, finite, OOD and Q separation passed every check; Spearman passed mean 20/21 and random 21/21; Q-scale passed 19/21 after the three-check history became available. The gate is dominated by two conditions that do not directly answer whether a small Actor warmup is safe.

The old TD series is noisy rather than monotonically improving: mean Qmin TD MAE was 0.002486 at 100K and 0.002407 at 300K (-3.16%); random was 0.003356 to 0.002062 (-38.56%). The old plateau test can reject genuine improvement, but these particular 21-point curves also oscillate. At 300K objective-aligned member residuals are low relative to step0 (table below), and both ranking series recover/improve by 300K.

## 2. Exact production gate audit

`stage3_v5_schedule.py::CriticHandoff._flags` treats every listed flag as hard; `submit_readiness` requires *all* flags false for 3 consecutive checks 10K apart. The first two checks fail Q-scale stability because the three-check window is incomplete. `fail_if_timed_out` changes `CRITIC_ONLY` to `CRITIC_NOT_READY` at 300K if no handoff occurred. A passing check moves to `ACTOR_WARMUP`, with an Actor learning rate ramp from 0 toward 2e-6 and continued Critic training; `policy_delay=4`.

| Current condition | Actual calculation, data and threshold | Current role |
|---|---|---|
| Online coverage | Env steps >=100K; cumulative completed >=150, success >=30, failure >=30. These counts are from online collection, not the fixed diagnostic episodes. | Hard |
| Spearman | `corr(rank(Qmin(s_t,a_t)), rank(finite MC return_t))` over all transitions of the frozen, branch-local complete episodes; threshold >=0.70. Qmin is min(Q1,Q2) on executed actions. | Hard |
| AUC | Rank AUC of success labels against *episode mean Qmin* on those same frozen episodes; >=0.80. Positive `delta_q` (success mean episode Qmin minus failure mean) is another hard flag. | Hard |
| Twin disagreement | On 256 frozen sequence final transitions, `d=abs(Q1-Q2)/(abs(Q1)+abs(Q2)+1e-8)`; median <=0.10 **and** p95 <=0.25 in one combined flag. | Hard |
| TD residual and plateau | `td_mae=mean(abs(min(Q1,Q2)-y_diag))` on the same frozen 256 sequences. Last 3 readiness rows: `c=abs(TD_t-TD_(t-20K))/max(abs(TD_(t-20K)),1e-12)`. Plateau needs c<=0.05, `TD_t<=1.05*TD_(t-20K)`, and no worsening. | Hard |
| TD worsening | Last 3 old Qmin TD values: true if `TD_t>1.10*TD_(t-20K)`, or all three strictly increase and c>0.05. A true value fails separately from plateau. | Hard |
| Q mean/std stability | Last 3 diagnostic rows, using `abs(mean(Qmin))` and `std(Qmin)`. Both adjacent relative changes must be <=0.05 for mean and <=0.08 for std. Combined as `q_scale_stable`. | Hard |
| Finite | Checks diagnostic Q1/Q2/Qmin, episode Qmin values and returns, TD target, and OOD outputs. The readiness flag does not directly inspect loss or gradients; `critic_update` separately checks loss finite before backward, but no explicit post-backward gradient-finite readiness test was found. | Hard, incomplete scope |
| OOD | Perturb normalized actions on fixed sequences at configured radii; compare Qmin of perturbed vs reference actions. p95 positive excess divided by reference Qmin std must be <=2. Raw max is logged but not gated. | Hard |

The frozen set is built once from branch-local stratified online reservoirs and retained in replay; the episode set and 256 sampled sequences remain fixed at later checks. For example `0.010, 0.008, 0.006` yields `c=0.40`, no worsening, and plateau FAIL even though residual improves. At 100K/200K/300K the offline rerun reproduced the logged old Qmin `td_mae` to approximately 1e-10, confirming the reconstructed diagnostic contract.

## 3. V6 objective alignment

`stage3_v6_agent.py::bellman_target` overrides the V5 target. Training uses `y = r + gamma*(1-terminal)*target_next`, then **separate** `MSE(Q1,y)+MSE(Q2,y)`. For mean2q, `target_next` is the mean of the two target-Q component-expectation values. For random2q, a dedicated selector draws Q1 or Q2 once per entire Critic update minibatch. Outside an update, `diagnostic_expectation` returns their deterministic mean and consumes no selector RNG. Thus both branches use a deterministic twin-mean `y_diag` in fixed readiness; the two branches differ in their *training* target estimator. The offline selector audit observed zero draws.

The old residual measures Qmin against that deterministic target. It is neither the V6 mean target prediction residual nor the two member losses. Define `Qbar=(Q1+Q2)/2` and `e_i=Q_i-y_diag`. The relevant offline summaries are `td_qmin_mae=mean(abs(min(Q1,Q2)-y_diag))` (legacy), `td_qmean_mae=mean(abs(Qbar-y_diag))`, `td_members_mae=(mean(abs(e_1))+mean(abs(e_2)))/2`, and `td_member_max_mae=max(mean(abs(e_1)),mean(abs(e_2)))`. Since training optimizes each member's MSE, member MAE and member max MAE are the better readiness health proxies; Qbar MAE can hide cancellation between opposing member errors. Log the per-member MSE too before setting a production threshold.

All values below use the **same fixed 256 sequences and complete episodes within each branch** at step0/100K/200K/300K. These are existing checkpoints; no 10K checkpoints were invented. TD targets are recomputed with each checkpoint's own target networks on that same input set.

| Branch | Step | old Qmin TD MAE | Qmean TD MAE | members TD MAE | worst member MAE | Qmin Spearman | Qmean Spearman | Q1 / Q2 Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean | 0 | .009289 | .003178 | .009355 | .009500 | .719775 | .714031 | .707814 / .697850 |
| mean | 100K | .002486 | .002376 | .002775 | .002965 | .691370 | .688475 | .688337 / .688516 |
| mean | 200K | .003604 | .002463 | .003076 | .003853 | .738021 | .735926 | .735026 / .736452 |
| mean | 300K | .002407 | .001550 | .002160 | .002214 | .766389 | .763578 | .763442 / .763148 |
| random | 0 | .009133 | .002974 | .009516 | .009559 | .723746 | .715623 | .708412 / .700955 |
| random | 100K | .003356 | .003585 | .003828 | .004721 | .705364 | .702993 | .702392 / .703462 |
| random | 200K | .002030 | .001608 | .002117 | .002317 | .756965 | .755813 | .754047 / .756519 |
| random | 300K | .002062 | .001494 | .002256 | .002359 | .776732 | .774698 | .773751 / .774999 |

Mean member MAE fell 77% from step0 to 300K; random fell 76%. At 300K member MAE is about 1.1% of the diagnostic target standard deviation in either branch; at step0 it is about 4.7%. Neither sequence is monotonic: mean increased 100K->200K before improving, and random's member MAE rose slightly 200K->300K. There is no evidence here of sustained catastrophic Bellman deterioration through 300K. Qmin and objective-aligned trends differ in level and some intervals; the legacy residual is a poor single proxy for both members. At step0, Qbar MAE near .003 while member MAE near .0095 demonstrates cancellation risk.

Qmean ranking tracks Qmin closely, generally 0.002-0.008 lower at these checkpoints; it is a plausible primary V6 ranking measure. At 100K the mean Qmean Spearman is .6885 and random .7030; at 300K .7636 and .7747. Four snapshots do **not** establish the first exact 10K check at which a Qmean-based gate would pass.

## 4. Redundancy and twin-tail interpretation

AUC is 1.000 (or very near it) for essentially all checks. It is not mathematically equivalent to Spearman, since it only compares success/failure episode averages, but it supplied no ongoing discrimination here. Coverage counts are prerequisites to form a useful diagnostic set; after the fixed set is constructed, repeating them adds no quality signal. Qmin mean and std stability always pass together from 120K onward and can be one Q-scale condition. TD plateau implies no TD worsening by construction, so those hard flags overlap. Twin median passes all checks, while p95 sees a distinct tail. The p95/old TD MAE correlation is weak/inconsistent across branches (Pearson mean .296, random .042); they are not the same statistic.

At 300K, the top 5% relative-disagreement samples number 13/256. Mean: ratio p95=.6303, tail median absolute `|Q1-Q2|=.00423`, tail median `|Qbar|=.00129`, tail member TD MAE=.00229. Random: ratio p95=.9102, tail median absolute difference=.00130, tail median `|Qbar|=.00023`, tail member TD MAE=.00097. A ratio near 1 often coincides with near-zero Q scale; these absolute values do not by themselves show catastrophic prediction error. Tail disagreement still matters for Actor gradients, so retain it as a warning with absolute-difference and tail TD context. The present p95<=.25 rule has no demonstrated link to unsafe small Actor updates.

## 5. PIRLNav mechanism

The [official README](https://github.com/Ram81/pirlnav) describes critic-only learning followed by gradual Actor/Critic joint training. The [official config](https://github.com/Ram81/pirlnav/blob/main/pirlnav/config.py) provides scheduled `start_critic_warmup_at`, `start_critic_update_at`, `start_actor_warmup_at`, and `start_actor_update_at`. The [LR scheduler](https://github.com/Ram81/pirlnav/blob/main/pirlnav/utils/lr_scheduler.py) adjusts Critic LR and starts Actor LR from zero; the [trainer](https://github.com/Ram81/pirlnav/blob/main/pirlnav/ppo_trainer.py) steps it by update count. Those inspected paths do not impose Spearman, AUC, twin-p95, a 5% TD plateau, or three consecutive all-metric passes. The transferable idea is a protected, gradual handoff, not its numeric schedule or PPO objective. V6 already has a low target Actor LR (2e-6), policy delay 4, Critic continuation, and Actor warmup; readiness should decide entry to that limited phase rather than certify Critic convergence.

## 6. Counterfactual replay of the existing 21-point logs

The replay uses the original values and thresholds. For B/C/D, C retains the median twin condition while removing only p95; median is true in all checks. The final minimal row is an explicitly marked **historical-log proxy**: old Qmin Spearman>=.70, old Qmin TD worsening=false, existing Qmin Q-scale stability, and finite/data coverage. It cannot claim exact future Qmean/member-metric opening times, because those quantities are not logged at every 10K check.

| Gate design | Mean first 1 / 2 / 3 consecutive PASS | Random first 1 / 2 / 3 consecutive PASS | Reason |
|---|---|---|---|
| CURRENT | never / never / never | never / never / never | p95 and plateau dominate |
| NO STRICT PLATEAU | never / never / never | never / never / never | p95 still blocks after random's first two checks, which lack Q-scale history |
| NO TWIN-P95 HARD | 120K / never / never | 240K / never / never | isolated plateau passes cannot sustain handoff |
| NO PLATEAU + P95 SOFT | 120K / 130K / 180K | 130K / 140K / 180K | remaining TD worsening occasionally interrupts |
| FINAL MINIMAL LOG PROXY | 120K / 130K / 180K | 130K / 140K / 180K | removed AUC/median/OOD/delta-Q hard gates were always passing |

With checks 10K apart, 3 consecutive PASS means the first-to-third span is 20K, plus any delay before the first pass. Under the proxy, 1 pass opens at 120K/130K, 2 passes at 130K/140K, and 3 passes at 180K/180K (mean/random). Recommend **2 consecutive passes**: it filters a one-check spike while the extra third pass delays this replay 40K-50K and Actor warmup itself ramps gradually. This recommendation is contingent on calibrating the new Qmean/member metrics.

## 7. Recommended Critic readiness v2

Four core conditions, with the data prerequisite latched once achieved:

| Core condition | Definition and source | Why hard; failure meaning | Threshold basis/status |
|---|---|---|---|
| `DATA_READY` (startup prerequisite) | Online >=100K, >=150 complete episodes with >=30 success and >=30 failure; frozen balanced complete-episode set exists. No Q aggregation. Latch once established. | Without both outcome classes and enough complete episodes, ranking cannot be assessed. | Retain current counts provisionally for sampling feasibility; both branches first satisfy them at 100K. Not a Critic quality threshold. |
| `RANK_READY` | Spearman between **Qmean** on executed actions and finite MC returns over fixed complete episodes, with episode-aware uncertainty estimate. | A Stage2-pretrained Critic losing return order can misdirect Actor optimization. | Calibrate floor and confidence rule from Stage2 step0 plus held-out episode/bootstrap variation and harmful-checkpoint controls. Step0 Qmean is .7140/.7156; current .70 floor alone is not evidence of a safe threshold. No new numeric floor asserted. |
| `TD_HEALTHY` | On the fixed sequence batch and deterministic V6 `y_diag`, monitor each member's MAE/MSE, `td_members_mae`, and `td_member_max_mae`. Fail for nonfinite or *sustained, significant* member deterioration, not lack of plateau. | Both members participate in training and later Actor Q; one bad member must not hide behind Qbar. | Calibrate absolute/normalized envelope and trend tolerance using Stage2 initialization, V6 fixed-batch noise, and known damaged controls. Four checkpoints and 21 old-Qmin rows cannot justify a numerical cutoff. |
| `NUMERIC_SAFE` | All Q members, targets, Critic loss and gradients finite; one combined rolling Qmean mean/std scale check on the fixed sequence set. | Detects numerical failure or Q-scale shock before Actor uses Q. | Finite is unconditional. Calibrate Qmean-scale change limits; old Qmin thresholds 5%/8% are only replay proxies. |

Soft warning: relative twin p95 accompanied by absolute `|Q1-Q2|` p95 and TD residual within the disagreement tail; OOD normalized p95 and a large Q-scale shift. A catastrophic twin threshold should only become hard after offline evidence links it to bad Actor gradients or failed rollouts; do not invent one now. Diagnostic only: Qmin ranking and TD, Qbar MAE, member split, AUC, `delta_q`, twin median, old plateau, Pearson and raw OOD max. Continue logging all for explanation.

The readiness finite flag currently omits direct Critic loss/gradient checks. The training update checks loss finite, but a complete v2 safety predicate would need a logged gradient-finite result; report this gap before implementing v2.

At 300K, do not automatically terminate a 3M experiment merely because a warning or an uncalibrated plateau condition fails. If the actual hard safety conditions pass, enter the protected warmup; if a genuine hard safety condition fails, keep Actor off and continue bounded Critic-only diagnosis/escalation rather than silently force it open. A timeout is an operational alert, not evidence that an unsafe Actor update has become safe.

**Direct answers:** old strict TD plateau hard gate: **NO**. Old twin p95<=.25 hard gate: **NO**. Use Qmean Spearman as proposed primary ranking metric: **YES, after calibration**. Use V6 objective-aligned per-member TD instead of old Qmin TD as health measure: **YES**. AUC hard gate: **NO**. Twin median hard gate: **NO**. Merge Q mean/std scale stability: **YES**, with Qmean-based calibration. Consecutive passes: **2** proposed. 300K timeout as unconditional training termination: **NO**.

**Final recommendation:** Critic readiness v2 core hard gates = [`DATA_READY`, `RANK_READY(Qmean)`, `TD_HEALTHY(per-member V6 Bellman)`, `NUMERIC_SAFE(finite + combined Qmean scale)`]. Keep old TD plateau as hard gate = **NO**. Keep old twin_p95 as hard gate = **NO**. With two consecutive passes, the historical-log proxy first opens mean2q at **130K** and random2q at **140K**. Exact v2 opening times cannot be established from these logs because 10K Qmean ranking and objective-aligned member TD values were not recorded; calibrate and log them before production use.

Artifacts: `analyze_checkpoint_metrics.py`, `checkpoint_metrics_mean2q.json`, `checkpoint_metrics_random2q.json`, `analyze_readiness.py`, `critic_readiness_replay.json` in this directory.
