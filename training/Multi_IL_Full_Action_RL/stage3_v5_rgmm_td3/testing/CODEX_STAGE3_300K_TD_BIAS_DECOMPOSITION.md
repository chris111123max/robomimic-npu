# Stage3-v5 300K TD target negative-bias decomposition

## Execution and contract

- Evaluation HEAD: `667546da47633eb032b9a54f6fbf7f1a56b7a14a`; device: `npu:0`; status: **PASS**.
- Two complete test executions; the second corrected only the decomposition formula text and added the incremental min component. `py_compile` passed before each run.
- Multi and RNN use their run-manifest Stage2.2 checkpoints with `full_episode_prefix_unroll_learning_mask`; each Stage3 checkpoint has `critic_context_length=11`, Actor horizon 10 and γ=0.99.
- Production successor Critic tokens: `next_observations[s:t+1]`, previous-action tokens: `actions[s:t+1]`, zero initial recurrent state. The terminal mask is `terminated OR truncated`.
- All eight checkpoints have actor updates 0. Online Actor and target Actor hashes are identical at every checkpoint and unchanged across checkpoints (`f1e203241312301b…`). Target Critic hashes change.
- Primary estimates use full 11-token, nonterminal transitions; all-transition statistics are also retained in JSON. MC returns use complete episodes. Errors are transition-weighted rather than training sampler-weighted.

The exact additive identity used is:

`production−G = (MC-id−G) + (behavior-mean−MC-id) + (behavior-min−behavior-mean) + (policy-mean−behavior-mean) + [(policy-min−policy-mean)−(behavior-min−behavior-mean)]`.

The direct min path is `production−G = (behavior-min−G) + (policy-min−behavior-min)`. The JSON stores both paths, both clipped-min terms, raw Q1/Q2 future values, and all requested error quantiles.

## Dataset and numeric validity

| Group | Dataset | Complete episodes | Transitions | Full-context nonterminal | Success / failure |
|---|---|---:|---:|---:|---:|
| multi_q | old | 300 | 181682 | 178382 | 127 / 173 |
| multi_q | late | 171 | 91651 | 89770 | 123 / 48 |
| rnn_q | old | 100 | 52474 | 51374 | 77 / 23 |
| rnn_q | late | 171 | 90781 | 88900 | 125 / 46 |

- Late data were aligned uniquely to episode logs and conservatively exclude the first completion per environment after 200K (16 episodes per group).
- Across all cells: maximum finite-MC identity error `5.655e-8`, maximum additive residual `5.961e-8`, all tensors finite. All 10,640 numeric JSON fields are finite; CSV contains 352 component rows.

## Core decomposition on training-eligible transitions

Each pair is mean bias / MAE. `Behavior` is the replay executed next action under the target Critic. `Policy − behavior` changes only the next action to the checkpoint target Actor component-mean expectation. `Min − mean` is the production policy-side clipped-min contribution.

| Group | Checkpoint | Data | Qmin(next)−Gnext | Learned mean TD | Behavior−G | Policy−behavior | Min−mean | Production−G |
|---|---|---|---:|---:|---:|---:|---:|---:|
| multi_q | step0 | old | +0.000482 | +0.007414 | +0.000477 / 0.028991 | +0.000022 / 0.004206 | -0.007148 | +0.000499 / 0.029227 |
| multi_q | step0 | late | -0.002353 | +0.006643 | -0.002329 / 0.041553 | -0.000012 / 0.001986 | -0.008973 | -0.002341 / 0.041398 |
| multi_q | 100k | old | -0.076022 | -0.074003 | -0.075262 / 0.077718 | +0.000077 / 0.001362 | -0.001389 | -0.075185 / 0.077714 |
| multi_q | 100k | late | -0.099011 | -0.095565 | -0.098021 / 0.102004 | -0.000003 / 0.001192 | -0.002452 | -0.098024 / 0.102006 |
| multi_q | 200k | old | -0.117223 | -0.114404 | -0.116050 / 0.116866 | +0.000000 / 0.001747 | -0.001836 | -0.116050 / 0.116920 |
| multi_q | 200k | late | -0.151983 | -0.147820 | -0.150464 / 0.152359 | +0.000028 / 0.001600 | -0.002639 | -0.150435 / 0.152329 |
| multi_q | 300k | old | -0.185109 | -0.180708 | -0.183257 / 0.183544 | +0.000073 / 0.002545 | -0.002787 | -0.183184 / 0.183495 |
| multi_q | 300k | late | -0.234121 | -0.229103 | -0.231780 / 0.231932 | +0.000007 / 0.002343 | -0.002679 | -0.231773 / 0.231917 |
| rnn_q | step0 | old | -0.008500 | -0.001588 | -0.008415 / 0.030386 | +0.000001 / 0.002204 | -0.006842 | -0.008414 / 0.030233 |
| rnn_q | step0 | late | -0.004008 | +0.003519 | -0.003968 / 0.041863 | -0.000001 / 0.002233 | -0.007487 | -0.003968 / 0.041699 |
| rnn_q | 100k | old | -0.127469 | -0.124966 | -0.126195 / 0.126345 | -0.000005 / 0.001403 | -0.001239 | -0.126200 / 0.126363 |
| rnn_q | 100k | late | -0.115537 | -0.112534 | -0.114382 / 0.115469 | -0.000008 / 0.001239 | -0.001852 | -0.114390 / 0.115486 |
| rnn_q | 200k | old | -0.197329 | -0.193808 | -0.195356 / 0.195375 | -0.000046 / 0.001918 | -0.001562 | -0.195402 / 0.195417 |
| rnn_q | 200k | late | -0.183370 | -0.179337 | -0.181536 / 0.181560 | -0.000025 / 0.001821 | -0.002195 | -0.181561 / 0.181580 |
| rnn_q | 300k | old | -0.233140 | -0.229264 | -0.230809 / 0.230827 | -0.000052 / 0.002251 | -0.001568 | -0.230861 / 0.230874 |
| rnn_q | 300k | late | -0.221690 | -0.217696 | -0.219473 / 0.219491 | -0.000072 / 0.002196 | -0.001780 | -0.219545 / 0.219559 |

The first large systematic negative bias is visible by the 100K checkpoint (24,750 Critic updates). The available 0/100K/200K/300K snapshots do not identify the exact update inside 0–100K. At step0 the learned mean future-Q term is close to zero, with a small pre-existing clipped-min offset; at 100K the learned future-Q term dominates.

## Late production error distribution

| Group | Checkpoint | Mean | Median | MAE | RMSE | Std | p05 | p25 | p50 | p75 | p95 | Negative fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi_q | step0 | -0.002341 | -0.011143 | 0.041398 | 0.074632 | 0.074595 | -0.090727 | -0.026405 | -0.011143 | 0.008284 | 0.115203 | 0.679626 |
| multi_q | 100k | -0.098024 | -0.063223 | 0.102006 | 0.140262 | 0.100323 | -0.282098 | -0.126495 | -0.063223 | -0.041055 | -0.025931 | 0.983224 |
| multi_q | 200k | -0.150435 | -0.113794 | 0.152329 | 0.200149 | 0.132019 | -0.391520 | -0.184580 | -0.113794 | -0.067855 | -0.039070 | 0.993227 |
| multi_q | 300k | -0.231773 | -0.168834 | 0.231917 | 0.298392 | 0.187934 | -0.642012 | -0.285223 | -0.168834 | -0.112750 | -0.046972 | 0.996647 |
| rnn_q | step0 | -0.003968 | -0.008327 | 0.041699 | 0.076484 | 0.076381 | -0.087806 | -0.029858 | -0.008327 | 0.010399 | 0.104835 | 0.635883 |
| rnn_q | 100k | -0.114390 | -0.070422 | 0.115486 | 0.161750 | 0.114359 | -0.359364 | -0.143141 | -0.070422 | -0.040672 | -0.028286 | 0.987750 |
| rnn_q | 200k | -0.181561 | -0.141748 | 0.181580 | 0.230661 | 0.142268 | -0.424956 | -0.270769 | -0.141748 | -0.065634 | -0.032585 | 0.999010 |
| rnn_q | 300k | -0.219545 | -0.140621 | 0.219559 | 0.300905 | 0.205777 | -0.727357 | -0.261309 | -0.140621 | -0.099508 | -0.031615 | 0.999134 |

Every component and scope in JSON additionally stores p05/p25/p50/p75/p95, positive and negative fractions, median, MAE, RMSE and standard deviation.

## Finite-return buckets: production target minus MC

Buckets are defined per group/dataset from positive finite MC returns on full-context nonterminal transitions. `medium` is p50–p90; `top10` is above p90. Values are mean bias, with transition counts fixed across checkpoints.

| Group | Data | Checkpoint | Zero | Positive ≤p50 | Medium p50–p90 | Top 10% |
|---|---|---|---:|---:|---:|---:|
| multi_q | old | step0 | +0.009063 (n=119197) | -0.015307 (n=29594) | -0.009749 (n=23749) | -0.052503 (n=5842) |
| multi_q | old | 100k | -0.049726 (n=119197) | -0.067460 (n=29594) | -0.192120 (n=23749) | -0.158406 (n=5842) |
| multi_q | old | 200k | -0.078607 (n=119197) | -0.108766 (n=29594) | -0.252290 (n=23749) | -0.363087 (n=5842) |
| multi_q | old | 300k | -0.131214 (n=119197) | -0.156483 (n=29594) | -0.372575 (n=23749) | -0.608914 (n=5842) |
| multi_q | late | step0 | +0.017585 (n=33072) | -0.016096 (n=28408) | -0.000498 (n=22632) | -0.057127 (n=5658) |
| multi_q | late | 100k | -0.045362 (n=33072) | -0.068513 (n=28408) | -0.191191 (n=22632) | -0.181338 (n=5658) |
| multi_q | late | 200k | -0.074407 (n=33072) | -0.109445 (n=28408) | -0.254567 (n=22632) | -0.384114 (n=5658) |
| multi_q | late | 300k | -0.126880 (n=33072) | -0.157013 (n=28408) | -0.375335 (n=22632) | -0.646001 (n=5658) |
| rnn_q | old | step0 | +0.008181 (n=15847) | -0.011887 (n=17817) | -0.023645 (n=14168) | -0.004273 (n=3542) |
| rnn_q | old | 100k | -0.055583 (n=15847) | -0.078181 (n=17817) | -0.224611 (n=14168) | -0.290043 (n=3542) |
| rnn_q | old | 200k | -0.099173 (n=15847) | -0.128827 (n=17817) | -0.320017 (n=14168) | -0.462365 (n=3542) |
| rnn_q | old | 300k | -0.094444 (n=15847) | -0.133702 (n=17817) | -0.375406 (n=14168) | -0.751742 (n=3542) |
| rnn_q | late | step0 | +0.030177 (n=31694) | -0.013022 (n=28706) | -0.033743 (n=22875) | -0.029071 (n=5625) |
| rnn_q | late | 100k | -0.045671 (n=31694) | -0.075664 (n=28706) | -0.221185 (n=22875) | -0.264910 (n=5625) |
| rnn_q | late | 200k | -0.092505 (n=31694) | -0.126773 (n=28706) | -0.311419 (n=22875) | -0.434856 (n=5625) |
| rnn_q | late | 300k | -0.089784 (n=31694) | -0.133863 (n=28706) | -0.377103 (n=22875) | -0.747209 (n=5625) |

The high-return bucket is suppressed much more than zero-return transitions by 300K: on late data Multi is `−0.646000` versus `−0.126880`, and RNN is `−0.747210` versus `−0.089780`. This magnitude and direction match the observed Q-mean drift and loss of return ordering/success-failure separation. It is a fixed-data diagnostic, not an intervention proving which training update first caused the drift.

## Evidence-ranked sources of systematic negative bias

1. **A. Learned future-Q bootstrap error — dominant.** At 300K late, `Qmin_target(replay next action)−Gnext` is Multi `−0.234121`, RNN `−0.221690`. The learned-mean TD terms are `−0.229103` and `−0.217696`, against production biases `−0.231773` and `−0.219545`. Both become clearly negative by 100K and continue to worsen.
2. **C. Clipped twin-min — small secondary offset.** At 300K late the production policy min-minus-mean term is Multi `−0.002679`, RNN `−0.001780` (about 1.2% and 0.8% of total bias). It was larger at step0 (`−0.008973` / `−0.007487`), so its growth cannot explain the drift.
3. **D. Recurrent/history boundary difference — small in the measured control.** Keeping the same successor states and zero initial LSTM but zeroing the first unavailable previous-action token changes the 300K late behavior TD mean by Multi `−0.000996`, RNN `−0.001017`. This isolates that boundary choice only; broader representation effects inside learned Q are not ruled out.
4. **B. Target-policy action substitution — negligible signed mean.** At 300K late, policy-min minus behavior-min is Multi `+0.000007` (MAE `0.002343`) and RNN `−0.000072` (MAE `0.002196`). All Actor/target Actor hashes stay fixed.
5. **E. Terminal/MC semantics — rejected as a material source.** `terminated OR truncated` matches the episode mask; each complete episode ends with one terminal transition; MC identity max error is `5.655e-8`.
6. **F. Other causes — not isolated by this read-only checkpoint comparison.** The test identifies the point where bias enters the target chain, not the first optimizer operation or data sampling event that shifted the learned target Critic.

## Mechanism and next step

The production target starts with only a tiny bias at step0 and is clearly below finite MC by 100K on both old and late data. Its dominant negative term is already present when the target Critic evaluates the replay-executed next action. Changing to the target Actor component means barely changes the mean target; clipped min adds only a small offset. The target therefore supplies increasingly low learning labels, consistent with recursive underestimation as Critic updates accumulate. This explains the direction and scale of the observed Q collapse but does not, by itself, establish why the initial target Critic shift occurred.

Prioritize the bootstrap target mechanism in subsequent controlled work. UTD may affect the rate of feedback, but this test holds UTD fixed and does not isolate it. A direct 300K resume would inherit a target whose mean error is already about −0.22 to −0.23 on late data; further training is not supported by these diagnostics without first validating the target mechanism. No training or rollout was started here.

## Artifacts and safety

- Test: `test_stage3_300k_td_bias_decomposition.py`
- Full JSON: `stage3_300k_td_bias_decomposition.json`
- Flat component CSV: `stage3_300k_td_bias_decomposition.csv`
- No production, Stage2, checkpoint, replay, reward, gamma, Actor, readiness or fusion file was edited. Environment/optimizer/rollout steps: 0.
