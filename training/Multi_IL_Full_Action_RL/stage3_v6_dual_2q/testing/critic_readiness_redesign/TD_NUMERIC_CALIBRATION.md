# Stage3-v6 Critic Readiness V2 — TD / numeric offline calibration

## A. Data availability and method

- Formal run: `/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_multi_mean_random_formal_20260928`.
- Both `mean2q/multi_q/checkpoints` and `random2q/multi_q/checkpoints` have unique Critic weights only at `step0_transfer.pth`, `step_0100000.pth`, `step_0200000.pth`, `step_0300000.pth`. `latest.pth`, `last.pth`, and `critic_not_ready.pth` are terminal/alias saves, not extra 10K checkpoints. Each unique checkpoint has a `.sequences.npy` companion. There are no 110K–290K weights for objective-aligned reevaluation.
- Each branch has 21 readiness rows, 100K–300K in 10K intervals, and 747 training metric rows (median spacing after 100K: mean 392, random 384 env steps). The latter are stochastic minibatch losses, **not** fixed diagnostic errors.
- Existing branch-local episode pools/windows/actions/MC returns and diagnostic seed were reused. One frozen sequence set was held fixed across all four checkpoints within each branch: mean 117 episodes/256 sequences, SHA256 `c64f0f3f4979e4a25585b72a3f685af0984f69861a72e9761bf8fc15a9495a82`; random 118 episodes/256 sequences, SHA256 `75e8c8bf69a9ade9dd9063ff30c86bfa391ff6ece401dfe8db1e736b021cbe54`. Cross-branch sets need not match. Random-one selector RNG is not consumed in diagnostic; the expected twin-mean target is used. No environment or optimizer steps were performed.
- Baseline `step0` on the *same frozen set* is required for the proposed scale anchor. A production implementation would need to evaluate that saved weight after the set is frozen; this report does not implement it.

## B. TD calibration

Deterministic `y_diag` is mean2q expected target or random-one expectation. `E1=MAE(Q1,y_diag)`, `E2=MAE(Q2,y_diag)`, `E_member=(E1+E2)/2`, `E_max=max(E1,E2)`, and `E_qmean=MAE((Q1+Q2)/2,y_diag)`. `E_max` is the hard-gate statistic: both optimized members must remain healthy. `E_member`, individual errors, and `E_qmean` remain diagnostic; averaging Q first can hide opposite member errors. No Qmin or selector-specific target is used.

### Table 1 — fixed-set TD (relative change is E_max vs preceding *available* checkpoint)

| branch | step | E1 | E2 | E_member | E_max | E_qmean | Δ E_max |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean | 0 | .009500 | .009210 | .009355 | .009500 | .003178 | — |
| mean | 100K | .002585 | .002965 | .002775 | .002965 | .002376 | −68.80% |
| mean | 200K | .003853 | .002299 | .003076 | .003853 | .002463 | +29.97% |
| mean | 300K | .002214 | .002107 | .002160 | .002214 | .001550 | −42.55% |
| random | 0 | .009473 | .009559 | .009516 | .009559 | .002974 | — |
| random | 100K | .004721 | .002935 | .003828 | .004721 | .003585 | −50.61% |
| random | 200K | .002317 | .001918 | .002117 | .002317 | .001608 | −50.93% |
| random | 300K | .002359 | .002152 | .002256 | .002359 | .001494 | +1.84% |

Mean's 100K→200K E_max +29.97% is a **single rebound** followed by improvement; random's 200K→300K +1.84% is small. The 300K E_member values are 76.9% (mean) and 76.3% (random) below step0. The 21 old fixed-set **Qmin** TD proxy rows have adjacent absolute-relative p50/p75/p90/p95 of mean .243/.378/.489/.626 and random .219/.397/.628/.731; they are not objective-aligned V2 values. After 100K, 501 mean and 500 random logged stochastic `critic_loss_q1+critic_loss_q2` samples have adjacent absolute-relative p50/p75/p90/p95 mean .442/.741/1.537/2.587, random .438/.737/1.287/1.840; all logged loss and gradient norms are finite. These high minibatch fluctuations argue against a per-update or single-check relative-rise hard fail, but cannot set a 10K fixed-set TD quantile.

### Table 3 — TD candidate comparison and separate scale stress

Historical false-fail counts below are `mean/random` over 21 old-Qmin proxy rows; all five rules yield 0 flags at the four sparse objective-aligned checkpoints. A 3-row history is required. Synthetic columns mean whether **any** FAIL occurs in that finite-length progression, starting from two flat reference values.

| TD rule | old-Qmin historical flags | +25% for 2/3 intervals | +50% for 1/2 intervals | +100% for 1/2 intervals | merit | issue |
|---|---:|---|---|---|---|---|
| A: single +35% | 3/4 | no/no | yes/yes | yes/yes | fast | single-spike false failures |
| B: 1.5× recent best | 3/3 | yes/yes | yes/yes | yes/yes | detects drift | transient low best lingers |
| C: 3-point monotone, endpoint +50% | 1/1 | yes/yes | no/yes | no/yes | trend-based | still misflags a historical proxy rebound |
| D: two successive ≥35% rises | 0/0 | no/no | no/yes | no/yes | rejects single spike and confirmed strong worsening | slow <35%-per-check drift can evade |
| E: D plus 2× endpoint | 0/0 | no/no | no/yes | no/yes | stricter evidence | misses two 35–41% rises without demonstrated benefit |

The numeric rule below, **not any TD candidate**, rejects synthetic Q×2, Q×5, and Q×10 in both branches; this scale stress is independent of the TD sequence. The chosen TD rule is D. The +35% threshold is a **PROVISIONAL safety threshold**, chosen above the observed single +29.97% objective rebound and above the old-Qmin two-rise sequence at 250K (+26.7%, then +38.0%), while catching the specified `.0021→.0030→.0045` (+42.9%, +50%). It has zero old-proxy historical flags. This is not a statistically estimated 10K objective-aligned quantile: those weights do not exist.

Executable rule at 10K readiness checks:

```python
# E[t] = max(MAE(Q1, y_diag), MAE(Q2, y_diag)), same frozen diagnostic set.
# Until three consecutive 10K E values exist, TD_HEALTHY is PENDING (not PASS).
finite = all_finite(Q1, Q2, y_diag, E1, E2, E_member, E_max, E_qmean)
sustained_worsening = (E[t-1] > E[t-2] and E[t] > E[t-1]
                       and E[t-1] / max(E[t-2], 1e-12) >= 1.35
                       and E[t] / max(E[t-1], 1e-12) >= 1.35)
TD_HEALTHY = finite and not sustained_worsening
```

If an error is zero, the explicit strict-increase clauses prevent a zero-to-zero interval from counting as worsening; a nonfinite value fails immediately. No episode-set refresh can be mixed into the same 3-point comparison. A prolonged sub-35%-per-check drift is a stated blind spot; future real 10K V2 measurements should validate or amend it before claiming statistical calibration.

## C. Q scale / numeric calibration

Use Qmean and deterministic `y_diag` distributions on the same frozen set. For each `X∈{Qmean,y_diag}` and reference `r∈{previous 10K check, step0 on same set}` define `Z(X,r)=|μ_t−μ_r|/max(σ_r,10⁻⁶)` and `R(X,r)=max(σ_t/σ_r,σ_r/σ_t)`. Require each σ finite and `>10⁻⁶`. Let `Z_max` and `R_max` be maxima over the four X/reference pairs. `|
### Table 2 — fixed-set Qmean scale (Z/R are worst across Qmean and target, vs previous and step0)

|---|---:|---:|---:|---:|---:|---:|---:|---|| branch | step | Qmean 
| mean | 0 | .120248 | .201215 | .121105 | −.029791/.936940 | — | — | yes |
| mean | 100K | .127902 | .199780 | .128009 | −.003378/.964863 | .0380 | 1.0072 | yes |
| mean | 200K | .123836 | .199699 | .124362 | −.015294/.986626 | .0210 | 1.0153 | yes |
| mean | 300K | .123412 | .199062 | .123778 | −.005878/.982859 | .0173 | 1.0124 | yes |
| random | 0 | .115281 | .200099 | .115655 | −.012961/.982206 | — | — | yes |
| random | 100K | .130874 | .206851 | .130885 | −.001457/.998646 | .0779 | 1.0491 | yes |
| random | 200K | .127385 | .207192 | .127517 | −.005527/1.001377 | .0606 | 1.0376 | yes |
| random | 300K | .127962 | .206026 | .128434 | −.010003/.999604 | .0634 | 1.0320 | yes |

Q1/Q2 and target mean/std/min/max for every checkpoint are in the machine-readable JSON; the displayed maxima include the target. Historical worst `Z=.0779`, `R=1.0491`. Thus a `Z≤0.5`, `1.5` envelope is far outside this healthy trajectory while not requiring almost-static Q. On synthetic 300K Qmean transforms, 1.25× and 1.5× pass; 2×, 5×, 10× fail; 0.5×, 0.25×, 0.1× fail (std ratio 2,4,10). A +0.25σ translation passes, +1σ fails. Synthetic +0.5σ sits exactly on the specified boundary; IEEE rounding can put it infinitesimally over, so production should compare with a small explicit floating-point tolerance if equality is intended. Stress transforms change summary outputs only, not model weights. The 0.5/1.5 values are **PROVISIONAL**, not data-estimated failure quantiles; no real failure trajectory exists. Historical ranges and synthetic response are data-checked.R

```python
FINITE = all_finite(Q1, Q2, Qmean, y_diag, E1, E2,
                    logged_critic_loss_since_previous_check,
                    logged_critic_grad_norm_since_previous_check)
# The two logged series are present and finite in both audited runs.
# If a future runtime cannot reliably provide grad norms, omit that check
# explicitly rather than silently treating missing as finite.
SCALE_SAFE = (all_sigmas_finite_and_gt_1e_minus_6
              and Z_max <= 0.5 and R_max <= 1.5)
NUMERIC_SAFE = FINITE and SCALE_SAFE
```

`Q1/Q2` finite checks must cover **all evaluated elements**, not only finite aggregate means. A nonfinite TD target or train loss/gradient norm fails immediately. Target scale participates in Z/R to catch target explosion separately from Q; actual target-stat computation must retain the same deterministic diagnostic policy.

## D. Frozen V2 gate and counterfactual

- `DATA_READY`: env steps ≥100K, completed episodes ≥150, success ≥30, failure ≥30 (unchanged prerequisite).
- `RANK_READY`: Spearman(`Qmean`, finite MC return) ≥.70 (unchanged candidate).
- `TD_HEALTHY`: 3 consecutive 10K fixed-set `E_max` readings exist, finite, and **not** two successive rises each ≥35%.
- `NUMERIC_SAFE`: `FINITE ∧ Z_max≤.5 ∧ R_max≤1.5` with valid σ as above.
- All four conditions must PASS on **two consecutive 10K checks** before `ACTOR_WARMUP`; same formulas/constants for mean2q and random2q.
- Diagnostic-only, never hard gate: twin mean/median/p95/p99, AUC, old TD plateau, OOD stress, and other explanatory metrics. These diagnostic warnings must not force a 300K hard stop. If rank/TD is unready, remain `CRITIC_ONLY`; catastrophic `NUMERIC_SAFE` failure may trigger protection. This is a design recommendation only; production is unchanged.

The only fully real V2 component evaluations are at 0/100K/200K/300K. Their Qmean Spearman is mean `.714/.688/.736/.764`, random `.716/.703/.756/.775`; all four have finite TD/scale. The 100K mean rank is below .70. **A complete 10K V2 replay is impossible** because 110K–290K weights, and hence their objective-aligned E_max/Qmean distributions, are absent. To bound expectations, a *non-equivalent proxy replay* used 21 old readiness rows: old Qmin rank, old Qmin TD with D's 3-point rule, and old finite/scale flag. It predicts the first two full proxy passes at **130K for both branches**, with no later proxy failures. Thus estimated mean/random opening is **around 130K, confidence LOW**, not a measured V2 opening or an assertion that the old and new rank/scale quantities agree at every 10K row. Earlier than 130K is excluded by the chosen three-reading TD warmup plus two-pass requirement, assuming first TD check at 100K.

## E. Limitations, verification, and artifacts

The chosen constants are provisional safety margins. Four true weight checkpoints cannot establish 10K objective-aligned fluctuation quantiles, frequency of V2 false failures, or exact V2 gate opening. 10K training losses are not the fixed diagnostic objective. The next genuine V2 run should log frozen-set E1/E2/E_max, Qmean/target μ/σ, finite flags, and criterion decisions on each 10K check to validate these provisional margins. This offline test ran without simulator rollout, training, or optimizer steps and made no production edits, checkpoint writes, commit, or push; `fusion_result.json` was not touched.

- `calibration_checkpoint_metrics_mean2q.json`, `calibration_checkpoint_metrics_random2q.json`: real fixed-set checkpoint metrics and full Q1/Q2/Qmean/target distributions.
- `td_numeric_calibration.json`: candidate comparisons, historical proxies, train loss fluctuation, synthetic TD and Q-scale tests, counterfactual rows.
- `analyze_checkpoint_metrics.py` and `calibrate_td_numeric_gate.py`: test-only scripts; both pass `py_compile`.

**Critic Readiness V2 的最终 hard gate = DATA_READY + RANK_READY + TD_HEALTHY + NUMERIC_SAFE，连续2次通过后进入 ACTOR_WARMUP。TD_HEALTHY threshold = 连续两个 10K interval 的 E_max 各上升 ≥35%。NUMERIC_SAFE threshold = Z_max≤0.5 且 R_max≤1.5，并且所有指定量 finite。这些规则对 mean2q 和 random2q 完全共用。**
