# Stage3-v6 Actor temporal supervision causal diagnosis — 2026-10-03

## VERDICT

Primary: **GENERAL_UNCONSTRAINED_ACTOR_DRIFT**. Answer: **NO — evidence does not support final-token-only temporal supervision as the primary cause.**
After identical 1,000 in-memory Actor Adam updates against the same frozen scoring function, CURRENT, correctly ALL_ALIGNED, and FINAL_ONLY_FREEZE_RNN each fall from the original 4/4 successful control to 0/4. Adding direct loss at every timestep does not rescue the policy. Eliminating recurrent parameter and hidden-state changes also does not rescue it. This is the requested Case D: general Actor-update drift can damage the competent policy even without recurrent-core updates. Recurrent-core drift amplifies action movement but is not necessary for failure in this test. No conclusion about Critic quality, targets, Q-head disagreement, or OOD is drawn.

## PRODUCTION ACTOR UPDATE

Current remote HEAD: `2419e0008a016ba0fee842b48f078edb186c0bcf`. V6 inherits V5 actor_update. Replay supplies one 10-observation block starting at episode step 0,10,20,..., burn-in=0. forward_train starts with zero RNN state, has no internal reset, and produces all ten GMM distributions. Only the final distribution enters `component_mean_q(..., twin_min=False)`. The loss is `L_CURRENT = -E_batch sum_k p_9,k Q1(h_9, mu_9,k_env)`. Full ten-step BPTT updates shared recurrent parameters. Earlier actions have no direct RL loss; parameter sharing lets both recurrent and output-head updates change them. Learned std has zero RL gradient. Rollout samples a categorical GMM component with low-noise std=1e-4 and resets recurrence every ten executed observations.

## THREE-BRANCH DESIGN

- CURRENT: the production final-token loss with full BPTT.
- ALL_ALIGNED: `L = -E_batch (1/10) sum_t sum_k p_t,k Q1(h_t, mu_t,k_env)`, same Actor ten-step block and zero-state reset. For block s..s+9, context h_t uses observations s+t-9..s+t (ten full sliding tokens), zero-state Critic initialization, window-local previous executed actions, and episode_step/700 progress. The last context was bitwise equal to the production CURRENT final context. No Actor-prefix history approximation was used.
- FINAL_ONLY_FREEZE_RNN: the exact CURRENT loss with all `nets.rnn.nets.*` parameters frozen; only the production-gradient output heads update. No artificial std loss.

Fixed dataset: 192 complete 19-step windows, 64 BC offline success + 64 frozen online success + 64 frozen online failure. Actor blocks start at multiples of ten >=10 so every timestep has complete prehistory. Seed 20261003, episode/start indices and immutable input arrays saved. Four fixed batches cycle; every update is batch 64 with 32 offline / 16 online success / 16 online failure. Baseline hashes, component means, probabilities, logits, hidden outputs, Q scores, and execution-matched sampled streams were exactly identical across all three clones.

Optimizer: production Adam, betas=(0.9,0.999), eps=1e-8, weight_decay=0, clip norm=10. The 140K checkpoint optimizer state is empty. Production scheduler is reused without changes. Because individual optimizer-step environment timestamps were not logged, update-to-env mapping is interpolated between actual cumulative train_metrics endpoints; this is not an exact replay of every original asynchronous update timestamp. The three branches share the identical mapping. First update maps to env140028, LR4e-10; update1000 maps to env156097, LR2.299571e-7, below target2e-6. No fixed-LR probe was needed.

## GRADIENT TIME STRUCTURE

CURRENT first production-sized fixed batch; unique named parameters, no duplicate aliases:

| module | L2 gradient | RMS gradient | gradient/parameter norm | parameters |
|---|---:|---:|---:|---:|
| mean | 0.36444705 | 0.0021752703 | 0.040476537 | 28070 |
| std | 0 | 0 | 0 | 28070 |
| logits | 0.0087834568 | 0.00019615902 | 0.00051337413 | 2005 |
| rnn | 0.094075839 | 6.6178426e-05 | 0.00089944091 | 2020800 |

No separate trainable encoder parameters exist in this loaded Actor. Std gradient=0 and std parameter drift=0 in every branch. One exact-schedule Adam step moves weighted action outputs at about 1e-7, close to float32 resolution; its early/final ratios are not strong evidence. Full timestep L2/mean-absolute/max-absolute and per-action-dimension drift are stored for every milestone in offline_results.json.

## OFFLINE RESULTS

Drift below is normalized probability-weighted mixture-mean movement, a structural diagnostic rather than the sampled execution action. EARLY=t0..2, MID=t3..6, LATE=t7..8, FINAL=t9.

| branch | updates | early | mid | late | final | RNN RMS drift |
|---|---:|---:|---:|---:|---:|---:|
| CURRENT | 0 | 0 | 0 | 0 | 0 | 0 |
| CURRENT | 1 | 1.95686e-07 | 2.78093e-07 | 2.79617e-07 | 2.87235e-07 | 1.40733e-10 |
| CURRENT | 10 | 1.54253e-05 | 2.39168e-05 | 2.72235e-05 | 2.73185e-05 | 1.33936e-08 |
| CURRENT | 25 | 8.92873e-05 | 0.000143254 | 0.000166734 | 0.000167591 | 8.07699e-08 |
| CURRENT | 50 | 0.000307508 | 0.000495766 | 0.000578604 | 0.000581574 | 2.80739e-07 |
| CURRENT | 100 | 0.00107145 | 0.00172977 | 0.00201958 | 0.00202934 | 9.80439e-07 |
| CURRENT | 250 | 0.00598096 | 0.00966088 | 0.0112333 | 0.0112779 | 5.45402e-06 |
| CURRENT | 500 | 0.0227536 | 0.0366976 | 0.0421329 | 0.0422918 | 2.05672e-05 |
| CURRENT | 1000 | 0.091185 | 0.14644 | 0.163246 | 0.16375 | 8.11706e-05 |
| ALL_ALIGNED | 0 | 0 | 0 | 0 | 0 | 0 |
| ALL_ALIGNED | 1 | 2.00983e-07 | 2.78625e-07 | 2.87821e-07 | 2.88972e-07 | 1.45233e-10 |
| ALL_ALIGNED | 10 | 2.04696e-05 | 3.18416e-05 | 3.5352e-05 | 3.48652e-05 | 1.48817e-08 |
| ALL_ALIGNED | 25 | 0.000120753 | 0.000192417 | 0.000216623 | 0.00021348 | 8.95649e-08 |
| ALL_ALIGNED | 50 | 0.000418255 | 0.0006673 | 0.000752492 | 0.00074121 | 3.10877e-07 |
| ALL_ALIGNED | 100 | 0.00145704 | 0.00232437 | 0.00261851 | 0.00257867 | 1.08306e-06 |
| ALL_ALIGNED | 250 | 0.00816782 | 0.0130277 | 0.01456 | 0.0143402 | 6.02858e-06 |
| ALL_ALIGNED | 500 | 0.0310614 | 0.0490377 | 0.053397 | 0.052703 | 2.27432e-05 |
| ALL_ALIGNED | 1000 | 0.123537 | 0.184425 | 0.189902 | 0.186793 | 8.98053e-05 |
| FINAL_ONLY_FREEZE_RNN | 0 | 0 | 0 | 0 | 0 | 0 |
| FINAL_ONLY_FREEZE_RNN | 1 | 6.59377e-08 | 7.477e-08 | 8.01542e-08 | 7.73667e-08 | 0 |
| FINAL_ONLY_FREEZE_RNN | 10 | 9.33543e-06 | 1.14887e-05 | 1.18333e-05 | 1.19326e-05 | 0 |
| FINAL_ONLY_FREEZE_RNN | 25 | 5.1704e-05 | 6.33107e-05 | 6.51459e-05 | 6.56518e-05 | 0 |
| FINAL_ONLY_FREEZE_RNN | 50 | 0.000176719 | 0.00021615 | 0.000222395 | 0.000224099 | 0 |
| FINAL_ONLY_FREEZE_RNN | 100 | 0.000612707 | 0.000749349 | 0.000770994 | 0.000776868 | 0 |
| FINAL_ONLY_FREEZE_RNN | 250 | 0.00339258 | 0.00414826 | 0.00426812 | 0.00430059 | 0 |
| FINAL_ONLY_FREEZE_RNN | 500 | 0.0127768 | 0.0156296 | 0.016085 | 0.016209 | 0 |
| FINAL_ONLY_FREEZE_RNN | 1000 | 0.050037 | 0.0612223 | 0.0630376 | 0.0635321 | 0 |

At 1000 updates, mean(t0..t8)/t9 weighted-action drift is CURRENT 0.8046, ALL_ALIGNED 0.8852, FREEZE_RNN 0.9113. ALL_ALIGNED early drift is 35.5% larger than CURRENT; FREEZE_RNN reduces early drift by 45.1% and final drift by 61.2%, but does not prevent performance collapse.

Hidden representation L2 drift, on identical observations, at update1000:

| branch | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CURRENT | 0.026509 | 0.084517 | 0.112542 | 0.127991 | 0.139485 | 0.150392 | 0.161830 | 0.173367 | 0.185182 | 0.196094 |
| ALL_ALIGNED | 0.042958 | 0.117456 | 0.153692 | 0.172462 | 0.185178 | 0.197154 | 0.210220 | 0.223974 | 0.236877 | 0.247699 |
| FINAL_ONLY_FREEZE_RNN | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

Normalized unique-parameter drift at update1000:

| branch | module | L2 | RMS | relative L2 |
|---|---|---:|---:|---:|
| CURRENT | mean | 0.0158178 | 9.44112e-05 | 0.00175677 |
| CURRENT | std | 0 | 0 | 0 |
| CURRENT | logits | 0.00436961 | 9.75855e-05 | 0.000255394 |
| CURRENT | rnn | 0.115388 | 8.11706e-05 | 0.0011032 |
| CURRENT | total | 0.116549 | 8.08327e-05 | 0.00107342 |
| ALL_ALIGNED | mean | 0.0160435 | 9.57589e-05 | 0.00178184 |
| ALL_ALIGNED | std | 0 | 0 | 0 |
| ALL_ALIGNED | logits | 0.00489332 | 0.000109281 | 0.000286004 |
| ALL_ALIGNED | rnn | 0.127663 | 8.98053e-05 | 0.00122056 |
| ALL_ALIGNED | total | 0.12876 | 8.93015e-05 | 0.00118588 |
| FINAL_ONLY_FREEZE_RNN | mean | 0.0156812 | 9.35961e-05 | 0.0017416 |
| FINAL_ONLY_FREEZE_RNN | std | 0 | 0 | 0 |
| FINAL_ONLY_FREEZE_RNN | logits | 0.0045968 | 0.000102659 | 0.000268673 |
| FINAL_ONLY_FREEZE_RNN | rnn | 0 | 0 | 0 |
| FINAL_ONLY_FREEZE_RNN | total | 0.0163411 | 1.13334e-05 | 0.000150502 |

CURRENT RNN RMS drift (8.12e-5) is not larger than mean/logits-head RMS drift (9.44e-5 / 9.76e-5). Its larger absolute L2 reflects 2,020,800 recurrent parameters; absolute L2 alone does not establish recurrent dominance.

## GMM RESULTS

Execution matching uses production forward_train_step and distribution.sample(), eval low-noise std=1e-4, identical CPU/NPU RNG seeds, and identical sampling call shapes at every timestep. Weighted means are not presented as executed actions. Raw distributions, component means, hidden outputs and sampled streams are saved as NPZ diagnostic arrays, not model checkpoints.

| branch | mean-head RMS | logits-head RMS | component mean L2 | rank change | top1 change | top1/top2 swap | sampled action L2 | entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CURRENT | 9.44112e-05 | 9.75855e-05 | 0.078831 | 11.30% | 4.38% | 4.38% | 0.163943 | 0.254010 |
| ALL_ALIGNED | 9.57589e-05 | 0.000109281 | 0.085168 | 15.68% | 6.04% | 5.94% | 0.204192 | 0.251614 |
| FINAL_ONLY_FREEZE_RNN | 9.35961e-05 | 0.000102659 | 0.060094 | 0.94% | 0.26% | 0.26% | 0.067391 | 0.255845 |

Original categorical entropy is 0.254735. Entropy remains close to baseline; the diagnostic does not show a new GMM mode-collapse mechanism. Even FREEZE_RNN, with zero hidden drift and only 0.26% top1 changes, alters component means and still fails all four episodes.

## CLOSED-LOOP RESULTS

Four environments initialized serially once and reused for the four policy branches. Seeds=(20008,20002,20005,20007), horizon700, production BatchedGMMExecutor, no external noise, reset horizon10. All four slots are sampled each timestep, discarding inactive actions, so early episode termination cannot change cross-branch RNG consumption. The ORIGINAL control reproduces all four successes under this common execution contract.

| branch | success | mean length | simulator errors | first action divergence | first EEF >1mm |
|---|---:|---:|---:|---|---|
| ORIGINAL_140K | 4/4 | 452.25 | 0 | [None, None, None, None] | [None, None, None, None] |
| CURRENT | 0/4 | 700.00 | 0 | [0, 0, 0, 0] | [2, 2, 2, 3] |
| ALL_ALIGNED | 0/4 | 700.00 | 0 | [0, 0, 0, 0] | [2, 2, 2, 2] |
| FINAL_ONLY_FREEZE_RNN | 0/4 | 700.00 | 0 | [0, 0, 0, 0] | [3, 3, 3, 3] |

Per seed (success/fail + length):

| seed | original140K | CURRENT | ALL_ALIGNED | FREEZE_RNN |
|---:|---|---|---|---|
| 20008 | SUCCESS 460 | FAIL 700 | FAIL 700 | FAIL 700 |
| 20002 | SUCCESS 423 | FAIL 700 | FAIL 700 | FAIL 700 |
| 20005 | SUCCESS 482 | FAIL 700 | FAIL 700 | FAIL 700 |
| 20007 | SUCCESS 444 | FAIL 700 | FAIL 700 | FAIL 700 |

Mean paired early closed-loop divergence (EEF distances in metres):

| branch | timestep | normalized action difference | arm0 EEF | arm1 EEF |
|---|---:|---:|---:|---:|
| CURRENT | 0 | 0.035446 | 0.000000 | 0.000000 |
| CURRENT | 1 | 0.066663 | 0.000064 | 0.000395 |
| CURRENT | 2 | 0.110202 | 0.000325 | 0.001114 |
| CURRENT | 5 | 0.123273 | 0.003124 | 0.003677 |
| CURRENT | 9 | 0.111176 | 0.008262 | 0.006331 |
| CURRENT | 10 | 0.037563 | 0.009441 | 0.006913 |
| CURRENT | 20 | 0.063522 | 0.011696 | 0.008280 |
| ALL_ALIGNED | 0 | 0.040982 | 0.000000 | 0.000000 |
| ALL_ALIGNED | 1 | 0.095170 | 0.000380 | 0.000402 |
| ALL_ALIGNED | 2 | 0.156855 | 0.001201 | 0.001139 |
| ALL_ALIGNED | 5 | 0.172924 | 0.006413 | 0.003804 |
| ALL_ALIGNED | 9 | 0.159658 | 0.014479 | 0.006813 |
| ALL_ALIGNED | 10 | 0.053920 | 0.016279 | 0.007569 |
| ALL_ALIGNED | 20 | 0.109457 | 0.018389 | 0.011556 |
| FINAL_ONLY_FREEZE_RNN | 0 | 0.031361 | 0.000000 | 0.000000 |
| FINAL_ONLY_FREEZE_RNN | 1 | 0.041629 | 0.000225 | 0.000349 |
| FINAL_ONLY_FREEZE_RNN | 2 | 0.047142 | 0.000485 | 0.000932 |
| FINAL_ONLY_FREEZE_RNN | 5 | 0.051733 | 0.001542 | 0.002637 |
| FINAL_ONLY_FREEZE_RNN | 9 | 0.048858 | 0.002708 | 0.004160 |
| FINAL_ONLY_FREEZE_RNN | 10 | 0.028425 | 0.002954 | 0.004440 |
| FINAL_ONLY_FREEZE_RNN | 20 | 0.037927 | 0.004909 | 0.004571 |

CURRENT changes initial execution at t0 (action difference0.03545); EEF divergence exceeds1mm by t2–3. ALL_ALIGNED also diverges at t0 (0.04098), EEF>1mm at t2. FREEZE_RNN diverges at t0 (0.03136), EEF>1mm at t3. Thus early execution damage is real, but it survives both temporal supervision densification and removal of recurrent updates.

## ROOT CAUSE AND NEXT STEP

Primary: **GENERAL_UNCONSTRAINED_ACTOR_DRIFT**. Secondary: recurrent-core changes amplify temporal output and mode-ranking movement, but are not necessary for the observed policy failure; shared output-head adaptation alone also damages earlier executed actions. The evidence does not justify treating FINAL_TOKEN_ONLY_TEMPORAL_SUPERVISION as the primary mechanism, nor does it justify implementing ALL_ALIGNED as a demonstrated fix.

Answer: **NO — evidence does not support it.** Both key interventions fail the same four originally successful seeds: ALL_ALIGNED0/4 and FREEZE_RNN0/4, compared with CURRENT0/4 and original4/4. This conclusion applies to this fixed-data, frozen-score Actor causal experiment. It does not replay the real 3,743 evolving-batch Actor updates through200K, and cannot establish all possible long-horizon outcomes. The exact scheduler equation is preserved; individual original update timestamps are unavailable and interpolated.

One next minimal experiment, not implemented: **Actor-to-init action anchor**. Repeat this identical CURRENT, 140K-initialized, frozen-score experiment with one weak per-timestep init-policy action anchor as the only changed training term; compare against the existing unanchored result on the same fixed contexts and four seeds. Freezing RNN alone already failed to preserve competence, so no further partial-freeze experiment is prioritized.

## ARTIFACTS AND FINAL STATE

- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_temporal_supervision/results_20261003/offline_results.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_temporal_supervision/results_20261003/closed_loop.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_temporal_supervision/results_20261003/closed_loop_steps.jsonl`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_temporal_supervision/results_20261003/sampling_manifest.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_temporal_supervision/results_20261003/fixed_temporal_contexts.npz`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_temporal_supervision/results_20261003/baseline_checks.json`
- `/data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/actor_temporal_supervision/results_20261003/schedule.json`

Critic parameter hash unchanged: True. Virtual Actor optimizer updates=1000 per branch, frozen Critic optimizer updates=0, formal training optimizer/environment steps=0. Closed-loop diagnostic evaluation used16 episodes / 10209 environment steps. All gradients finite; all sixteen episodes simulator-error-free. No model weights/checkpoints were written; NPZ files contain input/output diagnostic arrays only. All four environments closed and diagnostic process exited. Both testing scripts py_compile PASS. No extra diagnosis iterations or fixed-LR probe required. Production files, gate, Q aggregation and fusion_result.json were not edited.

FORMAL TRAINING REMAINS STOPPED
