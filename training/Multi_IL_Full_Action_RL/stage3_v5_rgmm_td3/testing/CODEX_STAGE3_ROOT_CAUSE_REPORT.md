# Stage3-v5 Multi Critic root-cause diagnostic

RUN = `/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3/stage3v5_h10_20260923_161337`.

Repository branch: `fix/stage3-v5-horizon10-critic`. The canonical replay is the Stage3-v5 Multi 200K sidecar. Diagnostic Critic updates occur only in memory; no environment rollout or training checkpoint is produced.

## Iteration 1

Hypothesis: Repeated moving learned-Q bootstrap supervision damages the Stage2.2 MC-return ranking more than repeated optimization on the same replay.

Test: Reused the existing paired moving-qmean replay-bootstrap versus exact-MC result at `$RUN/testing/stage2_vs_stage3_readiness/multi_bootstrap_vs_oracle_mc_causal.json`; it was already present and was not rerun. Both branches start from the same Stage3 step0 weights and Adam state, use the same precomputed batches and probe, and run 2,000 Critic steps. The target Critic receives the same Polyak updates in both branches.

Validity: JSON status PASS; all 14 validity checks true. Both branches completed 2,000 updates without a non-finite failure. Initial online/target/Actor/target-Actor hashes and optimizer states match. The Actor and target Actor remain unchanged; environment steps and training checkpoint writes are zero.

Key numbers:

| Update | Moving online Qmin Spearman | Oracle online Qmin Spearman | Oracle minus moving | Moving Qmin MAE | Oracle Qmin MAE | Moving target Qmin Spearman | Oracle target Qmin Spearman |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.726384 | 0.726384 | 0.000000 | 0.033311 | 0.033311 | 0.726384 | 0.726384 |
| 100 | 0.728953 | 0.748938 | 0.019984 | 0.034990 | 0.031662 | 0.728142 | 0.739707 |
| 250 | 0.713515 | 0.754380 | 0.040866 | 0.037883 | 0.030209 | 0.724559 | 0.752267 |
| 500 | 0.686999 | 0.724648 | 0.037650 | 0.042934 | 0.035569 | 0.704498 | 0.740981 |
| 1000 | 0.648869 | 0.693346 | 0.044477 | 0.070701 | 0.037917 | 0.647179 | 0.698820 |
| 2000 | 0.642396 | 0.724335 | 0.081939 | 0.070778 | 0.035381 | 0.647400 | 0.747240 |

At 2,000 updates, moving Qmean Spearman is 0.639509 versus oracle 0.732232; moving Qmin signed bias is -0.057629 versus oracle -0.012683.

Interpretation: Generic repeated optimization on this replay produces a transient oracle dip at 1,000 updates, but its final ranking is near step0. Moving learned-Q targets cause a much larger persistent ranking and MAE regression. The earlier production-target frozen-versus-Polyak result is directionally consistent (final frozen 0.685154 versus Polyak 0.457763), but changes target Actor/min behavior relative to this isolated qmean replay test.

Decision: CONTINUE. The paired test isolates learned-Q supervision from exact MC, but does not yet separate a fixed approximate Q teacher from evolving target-Critic feedback under otherwise identical qmean replay targets.

Next hypothesis: Freezing the step0 target Critic while keeping the same qmean replay bootstrap target and batch schedule will preserve substantially more MC ranking than production Polyak feedback.

## Iteration 2

Hypothesis: Evolving target-Critic feedback, rather than a fixed approximate Q teacher, is necessary for the qmean replay-bootstrap ranking drift.

Test: New testing-only paired script `test_stage3_moving_vs_fixed_qmean_replay_bootstrap.py`. It reused the original test's step0 agent construction, Adam state, exact 2,000 x 256 schedule, 8,192-transition probe, replay-next-action qmean target, Critic update and probe metrics. The fixed arm skipped only target-Critic Polyak updates. Result: `$RUN/testing/stage2_vs_stage3_readiness/multi_moving_vs_fixed_qmean_replay_bootstrap.json`.

Validity: JSON status PASS; all 15 checks true. Initial Critic, target Critic, Actor, target Actor and optimizer states match. First-batch targets match exactly (max abs 0). The moving target changed and fixed target remained unchanged. Both arms completed 2,000 updates without non-finite failure; environment and Actor steps and training checkpoint writes were zero.

Key numbers:

| Update | Moving online Qmin Spearman | Fixed online Qmin Spearman | Fixed minus moving | Moving Qmin MAE | Fixed Qmin MAE |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.726384 | 0.726384 | 0.000000 | 0.033311 | 0.033311 |
| 100 | 0.728439 | 0.723751 | -0.004687 | 0.032893 | 0.034755 |
| 250 | 0.725076 | 0.722791 | -0.002285 | 0.032388 | 0.034771 |
| 500 | 0.728482 | 0.683587 | -0.044896 | 0.033281 | 0.040853 |
| 1000 | 0.735781 | 0.720135 | -0.015646 | 0.032090 | 0.037149 |
| 2000 | 0.726421 | 0.730892 | +0.004471 | 0.032727 | 0.034995 |

Interpretation: This run does not show meaningful final moving-versus-fixed separation. More seriously, its moving arm used the same initial online/target hashes, seed, schedule construction, probe and update path as Iteration 1 but ended at 0.726421 instead of 0.642396. A target-feedback causal statement cannot be considered isolated while this cross-run contradiction remains unexplained. Source inspection found no explicit random draw in the replay-qmean target or Critic update path used here; runtime reproducibility remains an open issue.

Decision: CONTINUE. Do not infer that moving feedback is harmless or that the earlier drop is robust from these conflicting single runs.

Next hypothesis: The 2,000-step paired bootstrap-versus-oracle trajectory is not reproducible under the current NPU diagnostic setup despite identical checkpoint, replay schedule, and CLI parameters. Repeat the original paired experiment to a new JSON path and compare full milestones with Iteration 1.

## Iteration 3

Hypothesis: The large learned-Q versus oracle-MC separation from Iteration 1 was an unrepeatable trajectory artifact.

Test: Reran the original paired bootstrap-versus-oracle test with exactly the same checkpoint, replay, NPU device, seed, 2,000 x 256 schedule and 8,192-transition probe. Wrote a separate result at `$RUN/testing/stage2_vs_stage3_readiness/multi_bootstrap_vs_oracle_mc_causal_repeat.json`, preserving the original JSON.

Validity: JSON status PASS; all 14 validity checks true. Both arms completed all 2,000 steps without non-finite failure; Actor and target Actor unchanged; environment steps and training checkpoint writes zero. Initial online and target hashes match Iteration 1.

Key numbers:

| Update | Original moving Qmin Spearman | Repeat moving Qmin Spearman | Repeat oracle Qmin Spearman | Repeat oracle minus moving | Repeat moving Qmin MAE | Repeat oracle Qmin MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.726384 | 0.726384 | 0.726384 | 0.000000 | 0.033311 | 0.033311 |
| 100 | 0.728953 | 0.728960 | 0.748938 | 0.019978 | 0.034990 | 0.031662 |
| 250 | 0.713515 | 0.713386 | 0.754433 | 0.041047 | 0.038048 | 0.030208 |
| 500 | 0.686999 | 0.677350 | 0.732045 | 0.054694 | 0.044932 | 0.035251 |
| 1000 | 0.648869 | 0.658760 | 0.749306 | 0.090545 | 0.072575 | 0.032480 |
| 2000 | 0.642396 | 0.662766 | 0.746579 | 0.083813 | 0.071554 | 0.035726 |

Repeat moving Qmean Spearman falls from 0.722577 to 0.655928, while oracle ends at 0.734930. Moving target-Critic Qmin Spearman ends at 0.662228; oracle target ends at 0.751684.

Interpretation: The direction and approximately 0.08 final paired ranking separation reproduce. Generic repeated optimization with exact MC supervision again does not cause comparable final damage. However the moving arm of Iteration 2 ended at 0.726421 with the same visible start/schedule/target mode; untracked cross-run variation or pre-run runtime setup remains. Thus the broader learned-Q-versus-MC effect is replicated, but the narrower evolving-versus-fixed teacher effect is not isolated.

Decision: CONTINUE. Use the fourth and final iteration to repeat the moving-versus-fixed qmean replay paired test with unchanged code and parameters, writing to a new JSON. This directly checks whether Iteration 2's near-zero final separation is reproducible.

## Iteration 4

Hypothesis: The lack of moving-versus-fixed separation in Iteration 2 was an unrepeatable trajectory artifact.

Test: Repeated the Iteration 2 script with identical source, checkpoint, replay, seed, schedule, NPU device and CLI arguments, writing a separate JSON at $RUN/testing/stage2_vs_stage3_readiness/multi_moving_vs_fixed_qmean_replay_bootstrap_repeat.json.

Validity: JSON status PASS; all 15 validity checks true. Both arms completed 2,000 Critic optimizer steps without non-finite failure. Initial Critic/target/Actor hashes and Adam state match; first-batch qmean replay targets have max difference 0. The fixed target Critic hash stays at step0, the moving target changes, and both Actor hashes stay fixed. Environment steps, Actor updates and training checkpoint writes are zero.

Key numbers:

| Update | Moving online Qmin Spearman | Fixed online Qmin Spearman | Fixed minus moving | Moving Qmin MAE | Fixed Qmin MAE |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.726384 | 0.726384 | 0.000000 | 0.033311 | 0.033311 |
| 100 | 0.728438 | 0.723751 | -0.004686 | 0.032894 | 0.034755 |
| 250 | 0.723659 | 0.722793 | -0.000865 | 0.032445 | 0.034772 |
| 500 | 0.729675 | 0.680667 | -0.049008 | 0.033434 | 0.044837 |
| 1000 | 0.726797 | 0.699515 | -0.027282 | 0.032131 | 0.038153 |
| 2000 | 0.727425 | 0.725203 | -0.002222 | 0.032659 | 0.035785 |

Interpretation: The final moving-versus-fixed gap is again negligible and changes sign versus Iteration 2 (+0.004471 then -0.002222 for fixed minus moving). Both moving arms in this harness remain close to the initial 0.726384 ranking, whereas the two original bootstrap-versus-oracle moving arms ended at 0.642396 and 0.662766. The fixed teacher can show transient mid-run ranking loss (0.683587 and 0.680667 at 500 updates) even though its final ranking recovers. The clean qmean replay intervention therefore does not isolate evolving target-Critic feedback as necessary for degradation.

Decision: STOP. Four diagnostic iterations have been used. No fifth test is authorized by this task. The broader learned-Q-versus-MC effect is replicated, but the deeper moving-versus-fixed mechanism remains unresolved.

# Final conclusion

## A. Iterations completed

| Iteration | Hypothesis | Test | Status | Main result | Decision |
|---|---|---|---|---|---|
| 1 | Learned-Q bootstrap versus generic repeated optimization | Existing moving-qmean versus exact-MC paired JSON | PASS | Final Qmin Spearman 0.642396 versus 0.724335 | CONTINUE |
| 2 | Evolving target Critic versus fixed approximate-Q teacher | New moving-versus-fixed qmean replay test | PASS | Final 0.726421 versus 0.730892; gap +0.004471 fixed minus moving | CONTINUE |
| 3 | Reproducibility of Iteration 1 | Same bootstrap-versus-oracle test, new output | PASS | Final 0.662766 versus 0.746579; gap +0.083813 oracle minus moving | CONTINUE |
| 4 | Reproducibility of Iteration 2 | Same moving-versus-fixed test, new output | PASS | Final 0.727425 versus 0.725203; gap -0.002222 fixed minus moving | STOP |

## B. Best causal explanation

PARTIAL SUPPORT. In two paired runs of the original isolated target test, repeated learned-Q Bellman supervision caused substantially more loss of Stage2.2 MC-return ranking than exact MC supervision under the same batch and optimizer schedule. Thus learned-Q bootstrap supervision is a supported causal contributor in that diagnostic setup. A unique root cause at the finer level of evolving target-Critic feedback was **not isolated**: two clean moving-versus-fixed qmean replay pairs showed near-zero final separation and no sustained moving-arm ranking loss.

## C. Evidence chain

- All four result JSONs report PASS, matching paired initial weights and optimizer state, complete 2,000-step arms and no non-finite failures.
- In Iterations 1 and 3, initial Qmin Spearman was 0.726384. Moving learned-Q bootstrap ended at 0.642396 and 0.662766; exact MC ended at 0.724335 and 0.746579. Oracle-minus-moving gaps were 0.081939 and 0.083813.
- The same two moving arms had final Qmin MAE 0.070778 and 0.071554, while oracle MAE was 0.035381 and 0.035726. Target-Critic Qmin Spearman also separated: moving 0.647400/0.662228, oracle 0.747240/0.751684.
- In Iterations 2 and 4, moving-versus-fixed final Qmin Spearman was 0.726421 versus 0.730892, then 0.727425 versus 0.725203. Paired gaps were only +0.004471 and -0.002222 (fixed minus moving), despite the fixed target hash remaining unchanged and the moving target hash changing.
- The original production-target frozen-versus-Polyak result (frozen 0.685154 versus Polyak 0.457763 at 2,000 updates) supports target feedback directionally, but includes the target-Actor/min target construction absent from the clean qmean replay intervention.

## D. Alternatives ruled low

- Transfer/loading, terminal/history slicing and target/online checkpoint contracts were previously audited without a material mismatch; Stage2.2 and Stage3 step0 Critic weights match.
- Stage2.2's initial Bellman-closure MAE was only 0.003357 for Qmean and 0.003931 for Qmin, versus direct MC-fit MAE 0.032040/0.032430; a large initial closure mismatch is unsupported.
- Target-Actor continuation is nearly identical to replay-next-action continuation on the canonical set. Clipped twin min worsens pessimistic bias but a mean target still drifts in the earlier paired result. Slowing tau about tenfold only slightly helped.
- Exact-MC oracle arms finished near or above initial Qmin ranking in both repeats, so generic repeated optimization alone did not produce the comparable final degradation in those runs.

## E. Remaining uncertainty

The two test harnesses produced sharply different moving-qmean trajectories despite the same visible checkpoint hash, canonical replay, seed, batch schedule and probe. Their pre-run target-contract evaluation procedures differ, and the NPU runtime may contribute numerical variation, but neither explanation was isolated. The data therefore do not prove that evolving target-Critic feedback is necessary or sufficient, nor do they determine whether fixed approximate-Q supervision can cause a persistent loss under a matched setup.

## F. Production implication

A later production change should focus on preserving MC-return ranking under learned-Q Bellman supervision. The evidence does not yet justify treating target-Critic Polyak feedback alone as the unique cause. No production code was changed in this task.
