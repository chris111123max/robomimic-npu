# Stage3-v6 Actor objective alignment — focused offline mechanism test

Run: `/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_readiness_v2_multi_mean_random_formal_20260929_130945/mean2q/multi_q`. All diagnostics used the preexisting 256 aligned frozen contexts (128 success, 128 failure); no environment, rollout, formal training, or production edit. Gradient and virtual branches used online Critic weights from the named checkpoint, production history adapter, full Actor 10-step recurrence, the same GMM forward per comparison, and frozen Critic parameters.

## ACTUAL PRODUCTION OBJECTIVES

For mean2q, the Stage3-v6 Critic target is `y = r + gamma*(1-terminal)*sum_k p_target(k|h')*[Q1_target(h', mu_target,k)+Q2_target(h', mu_target,k)]/2`, with action normalization undone before Critic evaluation. Both online Q heads regress to this target. For random2q, one target-Q member is selected for the entire Critic minibatch by the dedicated selector; this run is mean2q. The Actor instead minimizes `L = -mean_batch sum_k p_actor(k|h)*Q1_online(h, mu_actor,k)`. This is inherited V5 `component_mean_q(..., twin_min=False)`: online Q2 and both target Q heads are absent from the Actor gradient. Actor GMM means and categorical probabilities matter; learned component std does not. Critic context is the production zero-state sliding history of 10 observation/previous-executed-action/progress tokens (the previous action is zero at the sequence boundary), and Actor receives its aligned 10-step observation recurrence. The Q1-only versus mean2q objective mismatch therefore exists in production.

## GRADIENT ALIGNMENT

Cosines compare **loss gradients** on the identical Actor parameters and identical forward tensors. Norm columns are `(Q1, mean2q, twin-min)` in that order.

| checkpoint | contexts | cos(Q1,mean) | cos(Q1,min) | cos(mean,min) | full gradient norms |
|---|---|---:|---:|---:|---|
| 140K ready | all | 0.9912 | 0.9898 | 0.9977 | 0.3158 / 0.3381 / 0.3480 |
| 140K ready | success | 0.9932 | 0.9939 | 0.9992 | 0.5808 / 0.6063 / 0.6110 |
| 140K ready | failure | 0.9098 | 0.8081 | 0.9540 | 0.1204 / 0.1137 / 0.1232 |
| 200K | all | 0.8441 | 0.6697 | 0.9420 | 0.2002 / 0.2078 / 0.2293 |
| 200K | success | 0.8617 | 0.7104 | 0.9362 | 0.2505 / 0.2549 / 0.2909 |
| 200K | failure | 0.8463 | 0.7208 | 0.9471 | 0.2059 / 0.2073 / 0.2209 |
| 280K | all | 0.5148 | 0.2180 | 0.9352 | 0.0875 / 0.1341 / 0.2074 |
| 280K | success | 0.5340 | 0.2102 | 0.9231 | 0.1117 / 0.1578 / 0.2437 |
| 280K | failure | 0.6104 | 0.3600 | 0.9445 | 0.0948 / 0.1370 / 0.2003 |

| module, all contexts | 140K cos Q1/mean, Q1/min | 200K | 280K | 140K norms Q1/mean/min |
|---|---|---|---|---|
| RNN | 0.9788 / 0.9420 | 0.9196 / 0.8281 | 0.7096 / 0.5809 | 0.0481 / 0.0488 / 0.0522 |
| GMM mean | 0.9915 / 0.9909 | 0.8371 / 0.6536 | 0.4819 / 0.1667 | 0.3121 / 0.3345 / 0.3441 |
| GMM logits | 0.9993 / 0.9997 | 0.6207 / 0.2992 | 0.9068 / 0.8808 | 0.00486 / 0.00492 / 0.00491 |
| GMM std | undefined (all norms exactly 0) | same | same | 0 / 0 / 0 |
| separate encoder | absent in unique parameter list | absent | absent | N/A |

No key module had a negative all-context cosine. At 140K success contexts, the gradients are especially aligned; divergence is mainly seen after Actor updates and collapse are already underway. The std-head gradient was explicitly tested as zero for all checkpoints/objectives; the network has no separately named encoder parameters.

## ONE-STEP COUNTERFACTUAL

All three branches start from the identical 140K Actor and frozen 140K online Critic. Adam defaults `betas=(.9,.999), eps=1e-8, weight_decay=0`, clip norm 10, batch 64. The first nonzero Actor LR is *inferred* from the production warmup equation and first 16-env collector round: `2e-6*16/140000 = 2.285714e-10`; an exact first Actor-step LR was not separately logged. At this near-float32-resolution LR, one-step Q deltas are about `1e-9` and should not be interpreted beyond noise.

| objective | normalized action drift | ΔQ1 | ΔQ2 | ΔmeanQ | ΔminQ | Δnormalized replay distance |
|---|---:|---:|---:|---:|---:|---:|
| Q1-only | 2.54e-7 | +1.98e-9 | +1.20e-9 | +1.12e-9 | +1.62e-9 | +1.29e-8 |
| mean2q | 2.40e-7 | +1.86e-9 | +1.40e-9 | +1.51e-9 | +2.53e-9 | +1.20e-8 |
| twin-min | 2.60e-7 | +0.87e-9 | +0.61e-9 | +0.39e-9 | +0.20e-9 | -0.63e-8 |

Action movement is the norm of the probability-weighted normalized GMM component-mean action change. Replay distance compares this action to the last executed replay action, first transformed to the same normalized scale. This is a local replay-distance proxy, not a density estimate or environment outcome.

## MULTI-STEP COUNTERFACTUAL

The same four balanced 64-context batches cycle for all branches. Each branch uses an independent fresh Adam state and exactly the same deterministic LR schedule, reaching only `2.285714e-8` by virtual step 100. Critic is fixed at 140K. These are **not** 100 real training steps or a substitute for 140K→200K continuation.

|---|---:|---:|---:|---:|---:|---:|---:|| objective at step 100 | normalized action drift | 
| Q1-only | 0.001632 | +3.696e-5 | +3.606e-5 | +3.651e-5 | +3.678e-5 | +0.0001754 | 0.001016 |
| mean2q | 0.001672 | +3.662e-5 | +3.872e-5 | +3.767e-5 | +3.808e-5 | +0.0001530 | 0.001035 |
| twin-min | 0.001702 | +3.612e-5 | +3.855e-5 | +3.733e-5 | +3.855e-5 | +0.0001012 | 0.001028 |

At the onset, all three objectives increase both Q heads and all three move the policy slightly farther from replay actions. Q1-only is **not** the fastest action or parameter drift branch. Twin-min reduces but does not eliminate the replay-distance increase. This supports a shared learned-Q incentive to move away from demonstrated actions; it does not prove an environment success counterfactual. Since this is a frozen 140K Critic and only 100 tiny-LR steps, later target-Critic/online-Critic coevolution is untested.

## Q1 VS Q2 ATTRIBUTION

These paired values use the preexisting A3 offline results with each checkpoint's own frozen Critic and compare current Actor action with init Actor action on the same 256 contexts. `Q1-only advantage` is `ΔQ1>0 and ΔQ2<=0`; meaningful margin is `ΔQ1-ΔQ2>0.01` (1% of sparse success reward scale).

| checkpoint | contexts | mean ΔQ1 | mean ΔQ2 | Q1-only advantage | both positive | Q1-Q2 gain gap >0.01 |
|---|---|---:|---:|---:|---:|---:|
| 200K | all | +0.008906 | -0.000633 | 28.52% | 62.11% | 36.33% |
| 200K | success | +0.008902 | -0.001534 | 30.47% | 57.81% | 40.63% |
| 200K | failure | +0.008909 | +0.000268 | 26.56% | 66.41% | 32.03% |
| 280K | all | +0.018901 | -0.000906 | 44.53% | 46.48% | 48.44% |
| 280K | success | +0.018621 | -0.002461 | 46.09% | 45.31% | 63.28% |
| 280K | failure | +0.019181 | +0.000648 | 42.97% | 47.66% | 33.59% |

The absolute twin expected-Q disagreement on the current actions rises from median 0.00701→0.00878, p95 0.03398→0.04271 at 200K (init→current within its Critic), and from median 0.00678→0.01245, p95 0.03312→0.08677 at 280K. This is strong evidence of **later Q1-only exploitation**, particularly in formerly successful contexts, but timing alone does not establish it as the initial 140K trigger.

## ACTUAL DRIFT ALIGNMENT

Cosine between actual 140K→200K Actor parameter delta and negative 140K fixed-context loss gradient: Q1 `0.06665`, mean2q `0.06091`, min `0.05956`; RNN `0.16294/0.16181/0.15587`; GMM mean `0.25833/0.23168/0.22344`. The Q1 edge is tiny and all global alignments are near zero. This net delta includes 3,743 Actor Adam updates, evolving Critics, changed batches, and optimizer momentum; it is directional context only, not a causal attribution.

## DETAILED MEASUREMENTS

Per-module gradient norms on all 256 contexts (Q1 / mean2q / twin-min):

| checkpoint | module | Q1 | mean2q | twin-min |
|---|---|---:|---:|---:|
| 140K | rnn | 0.048050 | 0.048775 | 0.052165 |
| 140K | gmm_mean | 0.312066 | 0.334504 | 0.344068 |
| 140K | gmm_logits | 0.004857 | 0.004917 | 0.004907 |
| 140K | gmm_std | 0.000000 | 0.000000 | 0.000000 |
| 200K | rnn | 0.057528 | 0.069055 | 0.081392 |
| 200K | gmm_mean | 0.191716 | 0.195976 | 0.214379 |
| 200K | gmm_logits | 0.001007 | 0.001039 | 0.001702 |
| 200K | gmm_std | 0.000000 | 0.000000 | 0.000000 |
| 280K | rnn | 0.031741 | 0.053716 | 0.070345 |
| 280K | gmm_mean | 0.081511 | 0.122794 | 0.195036 |
| 280K | gmm_logits | 0.001454 | 0.004971 | 0.006428 |
| 280K | gmm_std | 0.000000 | 0.000000 | 0.000000 |

Paired Q gain distributions; full success/failure-specific gradient and gain statistics are retained in the JSON files:

| checkpoint | contexts | gain | mean | median | p25 | p75 | p95 |
|---|---|---|---:|---:|---:|---:|---:|
| 200K | all | q1_gain | +0.008906 | +0.009271 | +0.005439 | +0.013529 | +0.022375 |
| 200K | all | q2_gain | -0.000633 | +0.003399 | -0.006198 | +0.007881 | +0.020696 |
| 200K | success | q1_gain | +0.008902 | +0.009709 | +0.005125 | +0.013774 | +0.022188 |
| 200K | success | q2_gain | -0.001534 | +0.002185 | -0.014752 | +0.009746 | +0.021875 |
| 200K | failure | q1_gain | +0.008909 | +0.008583 | +0.005492 | +0.013015 | +0.022131 |
| 200K | failure | q2_gain | +0.000268 | +0.003977 | -0.002073 | +0.007693 | +0.015506 |
| 280K | all | q1_gain | +0.018901 | +0.017227 | +0.004856 | +0.030165 | +0.051049 |
| 280K | all | q2_gain | -0.000906 | -0.000348 | -0.007445 | +0.016960 | +0.038129 |
| 280K | success | q1_gain | +0.018621 | +0.018552 | +0.009251 | +0.026354 | +0.045523 |
| 280K | success | q2_gain | -0.002461 | -0.000297 | -0.009954 | +0.011400 | +0.033283 |
| 280K | failure | q1_gain | +0.019181 | +0.015813 | +0.002675 | +0.033455 | +0.051735 |
| 280K | failure | q2_gain | +0.000648 | -0.000421 | -0.005396 | +0.023390 | +0.041077 |

## ROOT CAUSE AND DECISION

**Primary, most plausible:** `SHARED_CRITIC_EXTRAPOLATION_ERROR` leading to `ACTION_DISTRIBUTION_SHIFT`. The three objectives already point almost together at 140K and all three fixed-Critic virtual branches raise both Q estimates while increasing replay distance. **Secondary:** `Q1_ONLY_POLICY_EXPLOITATION` and `ACTOR_CRITIC_OBJECTIVE_MISMATCH`—real and increasingly consequential by 200K/280K, but not demonstrated sufficient to initiate collapse. `RECURRENT_POLICY_DRIFT` is the observed policy-change channel, not proof of what started it. GMM mode collapse, transfer/checkpoint error, excessive initial Actor LR, and gate readiness are not supported as primary by this focused comparison and preexisting 6/10 unchanged 140K baseline. Gate insufficiency remains a separate possible contributor, not retested here.

**Question:** Would replacing Actor Q1-only with a dual-Q objective from the start demonstrably alleviate this collapse? **NO — current evidence does not support that prediction.** At 140K the Q1/mean and Q1/min gradient cosines are 0.991/0.990, and the controlled 100-step mean/min branches also move away from replay. This is **not** proof that a 60K-step dual-Q continuation would fail; only a paired training continuation could answer its long-horizon success rate.

**One next minimal fix experiment, not executed:** In a separate future test branch, continue from the same 140K checkpoint with one weak Actor-to-init action anchor as the only change, compare against unanchored control at 160K/200K on the same four formerly successful seeds, and retain all Critic/gate/scheduler settings. This directly tests whether constraining shared learned-Q-driven action drift prevents collapse; no such training or code change was made now.

## Artifacts and integrity

- `results/gradient_ready.json`: 140K gradients, attribution, actual-delta cosines.
- `results/gradient_200_280.json`: 200K/280K gradients and paired Q-head attribution.
- `results/virtual_adam_100_corrected.json`: all one-step and 100-step branch metrics; **use this corrected replay-scale version**.
- `results/virtual_adam_100.json`: superseded first virtual result; replay-distance field used mixed scales and must not be used. Q and action fields were unaffected. Kept only as a debugging audit trace.
- `run_gradient_alignment.py`, `run_virtual_adam.py`: testing-only source, both `py_compile` PASS.
- Formal training process check: no `train_stage3_v6_vector.py` process detected at report time. Environment steps 0, rollout steps 0, training optimizer steps 0; virtual Adam updates acted only on private in-memory Actor clones. Checkpoints untouched.

FORMAL TRAINING REMAINS STOPPED
