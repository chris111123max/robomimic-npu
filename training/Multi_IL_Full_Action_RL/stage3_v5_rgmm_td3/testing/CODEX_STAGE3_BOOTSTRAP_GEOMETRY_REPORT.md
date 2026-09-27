# Stage3-v5 Multi fixed-bootstrap target geometry diagnostic

Date: 2026-09-27. Branch: `fix/stage3-v5-horizon10-critic`.

This report follows `CODEX_STAGE3_ROOT_CAUSE_REPORT.md`. All tests use Stage3 Multi step0, the 200K canonical replay sidecar, NPU 0, seed 20260926, batch size 256, and a fixed 8,192-transition probe. No environment rollout or formal training occurs. The target Critic is **fixed at step0**; continuation is the replay next action and twin **mean**, not target Actor or clipped min.

## 1. Gradient geometry

Result: `$RUN/testing/stage2_vs_stage3_readiness/multi_bootstrap_mc_gradient_geometry.json`. **PASS**: all 12 validity checks true, 64 independent batches completed, no optimizer step, no nonfinite. Online Critic, target Critic, Actor, target Actor and Adam state remained unchanged. The model has two independent Q networks; each contains a `token_encoder`, `lstm`, and `q_head`. There is no shared representation block.

On 8,192 fixed probe transitions, `delta = y_boot - G_t` has mean **+0.001581**, std **0.065386**, MAE **0.032382**, median **+0.003366**, absolute median **0.013468**, absolute p90 **0.077934**, absolute p95 **0.120773**, max absolute **0.989325**. Delta is positive for **60.73%** and negative for **39.16%** (the remainder is zero). Delta versus return: Spearman **−0.391034**, Pearson **−0.254694**. Delta versus current Qmean minus MC error: Spearman **+0.974361**, Pearson **+0.993104**. The latter near-identity is consistent with the previously measured small Bellman-closure residual, not independent proof of feedback.

Mechanically, `Q_t - y_boot = (Q_t - G_t) - delta`. Since delta strongly tracks current MC error, fixed bootstrap supervision approximately cancels part of the error signal that direct MC supervision would correct. This is consistent with the measured bootstrap gradient norm being only 0.411 of the MC norm on average; the cosine and block analysis below show that the effect is not merely a scalar reduction.

MC-return deciles, each about 819 transitions (the first five are tied at zero return, so interpret their order only as zero-return mass):

| Decile | n | MC mean | bootstrap mean | delta mean | delta median | abs delta mean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 820 | 0.000000 | 0.022934 | +0.022934 | +0.008500 | 0.023855 |
| 2 | 820 | 0.000000 | 0.019908 | +0.019908 | +0.008622 | 0.021244 |
| 3 | 819 | 0.000000 | 0.017865 | +0.017865 | +0.007817 | 0.019343 |
| 4 | 819 | 0.000000 | 0.017607 | +0.017607 | +0.008491 | 0.019356 |
| 5 | 819 | 0.000000 | 0.019894 | +0.019894 | +0.008443 | 0.021216 |
| 6 | 819 | 0.010824 | 0.015451 | +0.004627 | +0.000526 | 0.009732 |
| 7 | 819 | 0.036838 | 0.027845 | −0.008994 | −0.012491 | 0.018737 |
| 8 | 819 | 0.099414 | 0.078368 | −0.021046 | −0.027862 | 0.041744 |
| 9 | 819 | 0.252382 | 0.225022 | −0.027359 | −0.001402 | 0.068055 |
| 10 | 819 | 0.655439 | 0.625761 | −0.029678 | −0.005512 | 0.080560 |

Success-episode transitions have mean delta **−0.017774** (n=3,941); failure-episode transitions **+0.019524** (n=4,251). Progress bins 9–49, 50–99, 100–199, 200–399 and 400+ have mean deltas **+0.001892, +0.001632, +0.000943, +0.010103, −0.008823**, respectively. Thus the small overall signed mean hides a return- and outcome-dependent compression: low/zero returns are lifted and high returns lowered. This is measured on the replay distribution; it does not prove a universal bias outside it.

The two MSE losses were backpropagated separately on the same untouched online Critic and same batch; no step was taken. Across 64 independent batches:

| Parameter group | mean cosine | median cosine | mean bootstrap/MC norm | mean sign conflict |
|---|---:|---:|---:|---:|
| Entire Critic | 0.306438 | 0.334053 | 0.411083 | 0.379471 |
| Recurrent/history encoders (both twins) | 0.252656 | 0.252818 | 0.347800 | 0.383347 |
| Q1 head | 0.698578 | 0.756540 | 0.435982 | 0.320331 |
| Q2 head | −0.019435 | −0.067604 | 0.670183 | 0.402965 |

Global cosine p05/p95 is **−0.112951 / 0.631112**, with **5/64** negative-cosine batches (7.8125%). Global MC gradient norm mean is **0.238786**, bootstrap norm mean **0.088753**, and mean relative gradient difference **0.943075**. Near-zero element fractions are about **0.000982** in either gradient. Named-module mean cosines: Q1 token encoder **0.153600**, Q1 LSTM **0.199498**, Q2 token encoder **0.260874**, Q2 LSTM **0.281621**, Q1 head **0.698578**, Q2 head **−0.019435**. The largest observed direction discrepancy is Q2 head; recurrent representation gradients are also substantially misaligned. It is not a uniform output offset.

The optional one-step probe was not run: the multi-batch gradient geometry and the subsequent repeated-update interventions directly test ranking relevance.

## 2. Fixed-target interpolation dose response

Result: `$RUN/testing/stage2_vs_stage3_readiness/multi_fixed_qmean_mc_bootstrap_interpolation.json`. **PASS**: all 12 validity checks true, four 1,000-update branches completed, zero nonfinite. Identical initial online/target hashes and Adam states, exact first-batch alpha endpoints/interpolation, identical schedule and probe, frozen teacher, unchanged Actor/target Actor. Schedule SHA-256: `dea995b0e1a2d19c6754e18461f70714395ba18a521e0cf09abdb2f4bc2a8296`; probe SHA-256: `27df9f62b48a33df93df7321200145c56654f61d69ea95d237ef33e2d8a504e3`. First-batch bootstrap-versus-MC MAE: **0.034499**.

| alpha | update | Qmin Spearman | Qmean Spearman | Qmin MAE | Qmin signed bias |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | 0.726384 | 0.722577 | 0.033311 | −0.007860 |
| 0.00 | 100 | 0.752558 | 0.736759 | 0.032138 | −0.005572 |
| 0.00 | 250 | 0.720164 | 0.700983 | 0.034491 | −0.005604 |
| 0.00 | 500 | 0.715477 | 0.695440 | 0.037632 | −0.011284 |
| 0.00 | 1000 | 0.753832 | 0.740956 | 0.030655 | +0.000897 |
| 0.25 | 0 | 0.726384 | 0.722577 | 0.033311 | −0.007860 |
| 0.25 | 100 | 0.726502 | 0.708860 | 0.036249 | −0.012381 |
| 0.25 | 250 | 0.720454 | 0.711247 | 0.036055 | −0.011047 |
| 0.25 | 500 | 0.744081 | 0.736990 | 0.036939 | −0.020749 |
| 0.25 | 1000 | 0.768422 | 0.760948 | 0.028607 | −0.001392 |
| 0.50 | 0 | 0.726384 | 0.722577 | 0.033311 | −0.007860 |
| 0.50 | 100 | 0.743120 | 0.729953 | 0.035548 | −0.013485 |
| 0.50 | 250 | 0.700230 | 0.694190 | 0.034368 | −0.000694 |
| 0.50 | 500 | 0.745316 | 0.739551 | 0.032042 | −0.007933 |
| 0.50 | 1000 | 0.761740 | 0.745267 | 0.028885 | −0.004324 |
| 1.00 | 0 | 0.726384 | 0.722577 | 0.033311 | −0.007860 |
| 1.00 | 100 | 0.728896 | 0.723288 | 0.033641 | −0.002401 |
| 1.00 | 250 | 0.686032 | 0.666873 | 0.040421 | −0.001659 |
| 1.00 | 500 | 0.696652 | 0.697427 | 0.038078 | −0.016269 |
| 1.00 | 1000 | 0.704372 | 0.699461 | 0.037778 | −0.006847 |

At each nonzero milestone alpha=1 ranks below alpha=0, by **−0.023663, −0.034132, −0.018826, −0.049460** respectively. However alpha=.25 and .50 are often *better* than alpha=0, including both at 1,000. Thus pure fixed bootstrap versus pure MC shows a paired degradation, but **there is no monotone alpha dose-response**. Qmin final mean/std at alpha=0/.25/.50/1 are **0.106361/0.201820**, **0.104072/0.198568**, **0.101140/0.199529**, **0.098617/0.196385**; a simple scale or mean collapse does not explain this interpolation result.

## 3. Parameter-block causal check

Because the interpolation lacked a stable monotone dose-response, the optional block ablation was run. Result: `$RUN/testing/stage2_vs_stage3_readiness/multi_fixed_bootstrap_parameter_blocks.json`. **PASS**: all 11 validity checks true; all three branches completed 1,000 steps, zero nonfinite. It reused the interpolation schedule/probe hashes and fixed alpha=1 teacher. All branches start from the same weights and optimizer state; Actor and target Actor remain unchanged. Actual architecture has **134,272** encoder parameters and **29,186** head parameters (both twins combined).

| Trainable parameters | update 0 | 100 | 250 | 500 | 1000 | final MAE | final bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| All Critic | 0.726384 | 0.714340 | 0.685033 | 0.654014 | 0.714922 | 0.038537 | −0.015222 |
| Heads only; encoder frozen | 0.726384 | 0.729268 | 0.732621 | 0.729091 | 0.730612 | 0.032310 | −0.004785 |
| Encoder only; heads frozen | 0.726384 | −0.397340 | −0.270557 | −0.157115 | −0.234964 | 0.193358 | −0.190286 |

Freezing the encoder prevents drift in this run; updating encoder while fixing both heads is severely unstable. This implicates representation changes, but the encoder-only branch is an extreme constrained optimization problem, not a clean estimate of the encoder's contribution in normal joint training. The full alpha=1 branch also differs across independent harness runs: **0.704372** in interpolation versus **0.714922** here at 1,000 despite matching visible setup. Do not treat small between-harness differences as causal; compare paired arms within each test.

## 4. Final mechanism judgment

### Proven in these diagnostics

- The fixed step0 learned-Q bootstrap labels contain structured, return-dependent error on the canonical replay: zero/low returns are raised and high returns are lowered despite signed mean near zero.
- On the same online Critic and batch, MC and bootstrap losses induce distinctly different parameter gradients; global cosine mean is 0.306, with especially poor Q2-head and recurrent-encoder alignment.
- In one valid fixed-teacher interpolation run, pure bootstrap (alpha=1) has lower Qmin ranking than pure MC (alpha=0) at all four nonzero milestones. The earlier moving-bootstrap-versus-MC endpoint difference was independently replicated twice.
- In the block ablation, freezing the encoder preserves ranking under fixed bootstrap while full updates lose some ranking; encoder-only updates catastrophically alter output geometry.

### Strongly supported

A small global target MAE hides a *structured* label discrepancy. Because the discrepancy depends on return/outcome and aligns with current approximation error, repeated gradient projection is not equivalent to adding unbiased target noise or a harmless constant offset. It can move the recurrent representation and Q2 head in substantially different directions from direct MC supervision. This is the narrowest supported mechanism for why similar aggregate label MAE can lead to different learned ranking.

### Not yet proven

- A **stable monotone** alpha dose-response: intermediate alphas improved final ranking here, so the proposed simple linear dose mechanism fails this test.
- Unique necessity/sufficiency of recurrent drift in unconstrained training. Encoder-only collapse may be caused by frozen-head incompatibility; cross-harness full-arm variation remains unexplained.
- That the static label/gradient geometry alone fully explains the 2,000-step moving-target trajectory or production 200K/300K drift. No environment or formal training was run.

Direct answers: (1) yes, label error is structured; (2) yes, it correlates negatively with return and reverses sign across outcome groups; (3) yes, gradients are strongly misaligned; (4) Q2 head has the lowest mean cosine, with recurrent representation also low; (5) **no stable monotone interpolation dose-response**, though alpha=1 is below alpha=0 at every measured post-update milestone; (6) aggregate MAE is misleading because it masks structured error and changed parameter-gradient projection, but the full long-horizon mechanism is not closed; (7) the most credible root is **return-dependent learned-Q label bias projected through an updateable recurrent twin-Q approximator**, rather than moving target-Critic feedback alone. Its precise necessity and quantitative contribution remain uncertain.

Safety: environment steps 0; no rollout, formal RL training, production-code edit, formal checkpoint write, or modification to pre-existing `fusion_result.json`. Only three testing scripts and this report were added.
