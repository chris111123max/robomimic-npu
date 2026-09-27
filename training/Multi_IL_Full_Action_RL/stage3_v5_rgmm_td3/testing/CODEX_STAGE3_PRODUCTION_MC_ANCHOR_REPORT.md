# Final conclusion

**PARTIAL (numerical failure), not a validated repair.** The production-like Stage3 baseline (lambda=0) completed 1,000 diagnostic Critic updates and fell from Qmin Spearman 0.726384 to **0.653678**; its Qmin MAE rose from 0.033311 to **0.127714** and bias reached **−0.127105**. Both MC-anchor branches protected ranking, MAE and bias at updates 250 and 500, but **both stopped on a non-finite Critic loss at update 504**. Neither has a 1,000-update result. Therefore the test does not meet the requested sustained-through-1,000 criterion and does **not** justify moving directly to small-scale formal training.

Branch setup: Stage3-v5 Multi step0, 200K canonical replay, seed 20260926, batch 256, probe 8192, probe batch 1024, 1000 requested updates, NPU 0. The diagnostic target is `(1-lambda) * production TD + lambda * exact finite-episode G_t`. Production TD is called directly from `RecurrentGMMTD3.bellman_target`, retaining moving target Critic, production Polyak update, target Actor, categorical component-mean expectation, clipped twin minimum, horizon-10 successor context, and original terminal semantics. Only the Q1/Q2 regression label is mixed.

## Validity

The paired contract passed: all three branches start with identical online Critic, target Critic, Actor, target Actor, and Adam state; deterministic schedule and probe hashes are shared. Schedule SHA-256 is `dea995b0e1a2d19c6754e18461f70714395ba18a521e0cf09abdb2f4bc2a8296`; probe SHA-256 is `27df9f62b48a33df93df7321200145c56654f61d69ea95d237ef33e2d8a504e3`. Actor and target Actor hashes never changed. A production Polyak step followed every completed Critic update. Lambda formula reconstruction max error was **0** for each branch. The overall JSON is **PARTIAL** only because two branches did not complete and nonfinite count is 2; do not interpret them as 1,000-step successes.

| lambda | completed updates | first nonfinite update | outcome |
|---:|---:|---:|---|
| 0.00 | 1000 | none | complete |
| 0.25 | 503 | 504 | partial |
| 0.50 | 503 | 504 | partial |

## First-batch production target contract

On the full first batch (256 transitions), lambda=0 helper versus the unmodified production method had **target max/mean absolute difference 0**, terminal-mask difference **0**, successor-context max difference **0**, Actor-distribution max difference **0**, and expected-next-Q max difference **0**. An independent stepwise manual reference on eight first-batch transitions had successor-context max difference **0**, Actor-distribution max difference **3.5763e−7**, next-Q max difference **5.1223e−9**, and TD target max difference **5.1223e−9**. Full canonical finite-episode MC identity max error was **0**. Terminal flags matched terminated OR truncated on all **62,943** canonical transitions: **64 terminated**, **47 truncated**, **0 mismatches**.

The first scheduled batch happened to contain no terminal transition, so a supplementary read-only check used one actual terminated and one actual truncated canonical transition. Both passed the existing independent target-contract helper: terminated target = reward = **1.0**, truncated target = reward = **0.0**, with no bootstrap and zero target/context/Actor/next-Q discrepancy. No optimizer step was used in this supplementary check.

## Qmin Spearman trajectory

| update | lambda=0 | lambda=.25 | lambda=.50 |
|---:|---:|---:|---:|
| 0 | 0.726384 | 0.726384 | 0.726384 |
| 100 | 0.728653 | 0.716998 | 0.730882 |
| 250 | 0.700533 | 0.726036 | 0.721769 |
| 500 | 0.720782 | 0.740168 | 0.729612 |
| 1000 | 0.653678 | N/A (failed at 504) | N/A (failed at 504) |

At 250, .25/.50 exceed baseline by **+0.025503/+0.021236**. At 500, gains are **+0.019386/+0.008830**. Lambda=.25 is worse at 100, so even before failure the benefit is not uniform across every milestone.

## Qmean Spearman trajectory

| update | lambda=0 | lambda=.25 | lambda=.50 |
|---:|---:|---:|---:|
| 0 | 0.722577 | 0.722577 | 0.722577 |
| 100 | 0.723342 | 0.705148 | 0.725144 |
| 250 | 0.701487 | 0.716123 | 0.701139 |
| 500 | 0.725130 | 0.734555 | 0.722755 |
| 1000 | 0.638693 | N/A | N/A |

Qmean provides weaker evidence than Qmin, especially for lambda=.50 at 250/500.

## MC MAE / signed bias

Qmin MAE:

| update | lambda=0 | lambda=.25 | lambda=.50 |
|---:|---:|---:|---:|
| 0 | 0.033311 | 0.033311 | 0.033311 |
| 100 | 0.034313 | 0.039727 | 0.037675 |
| 250 | 0.052205 | 0.042840 | 0.043523 |
| 500 | 0.055575 | 0.037328 | 0.040403 |
| 1000 | 0.127714 | N/A | N/A |

Qmin signed bias:

| update | lambda=0 | lambda=.25 | lambda=.50 |
|---:|---:|---:|---:|
| 0 | −0.007860 | −0.007860 | −0.007860 |
| 100 | −0.013050 | −0.016873 | −0.013889 |
| 250 | −0.042444 | −0.027972 | −0.028720 |
| 500 | −0.047572 | −0.020351 | −0.023053 |
| 1000 | −0.127105 | N/A | N/A |

At 500, both anchors materially reduce negative bias and MAE; there is no observed extreme MAE/bias tradeoff through that point. No anchor result exists at 1,000.

## Target Critic trajectory

Target Qmin Spearman:

| update | lambda=0 | lambda=.25 | lambda=.50 |
|---:|---:|---:|---:|
| 0 | 0.726384 | 0.726384 | 0.726384 |
| 100 | 0.730850 | 0.735337 | 0.738754 |
| 250 | 0.724069 | 0.730972 | 0.738589 |
| 500 | 0.717232 | 0.735866 | 0.730486 |
| 1000 | 0.640415 | N/A | N/A |

At 500 the target Critic also shows ranking protection. Target Qmin MAE at 500 is **0.050012 / 0.039665 / 0.046318** for lambda=0/.25/.50, signed bias **−0.039189 / −0.023607 / −0.032720**. Baseline target alone reaches MAE **0.119684**, bias **−0.118353** at 1,000. The anchor target-Critic effect after 503 updates is unknown.

## Target-vs-MC geometry

The fixed 2,048-transition geometry subset is identical across branches. Immediate interpolation improves the *label* by construction: at step0, production-target Spearman/MAE are **0.746333/0.032585**; lambda=.25 anchored values are **0.824430/0.024439**, lambda=.50 **0.873347/0.016293**. This algebraic target improvement alone is not evidence of an improved learned Critic.

At update 500:

| lambda | production target Spearman | anchored target Spearman | production MAE | anchored MAE | lowest-return decile bias, production → anchor | highest-return decile bias, production → anchor |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.733871 | 0.733871 | 0.049787 | 0.049787 | −0.003231 → −0.003231 | −0.144322 → −0.144322 |
| 0.25 | 0.742792 | 0.818903 | 0.039666 | 0.029749 | +0.008996 → +0.006747 | −0.102076 → −0.076557 |
| 0.50 | 0.737801 | 0.864557 | 0.046794 | 0.023397 | −0.000898 → −0.000449 | −0.128447 → −0.064223 |

The MC anchor reduces high-return compression markedly; when the low-return bucket has positive bias, it also reduces the uplift. The production target itself varies by branch because each moving target Critic has learned differently, so compare production and anchored labels **within** a branch.

## Nonfinite / safety

The two anchored branches encountered `Non-finite Stage3-v5 Critic loss` on the **same update 504**. The test stopped each immediately; it did not change LR, batch, seed, lambda, optimizer, or schedule and did not rerun. A read-only inspection of scheduled batches 503–505 found **zero nonfinite values** in MC returns, replay-next actions, observations, executed actions, rewards, next observations, and terminals. This rules out an obvious NaN/Inf in those stored inputs but does not establish the cause of the loss failure.

Environment steps **0**, formal training steps **0**, Actor optimizer steps **0**, target Actor updates **0**, formal checkpoint writes **0**. No production Stage3/Stage2.2 code, existing training checkpoint, or pre-existing modified `fusion_result.json` was touched. Only the new testing script and this report were added; result JSON is in the existing run's testing output directory.

## Production implication

Current category: **D — PARTIAL**. Up to 500 updates, lambda=.25 is the strongest observed anchor (Qmin Spearman **0.740168** versus **0.720782** baseline; MAE **0.037328** versus **0.055575**; bias **−0.020351** versus **−0.047572**), and the target Critic is protected too. But both anchor branches failed numerically at 504, before the required 1,000-step endpoint. **Do not modify production or launch small-scale formal training on this evidence alone.** The result establishes a promising short-horizon signal and a decisive stability blocker, not a validated repair.
