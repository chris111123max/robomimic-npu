# Stage3-V8 QSE-GMM Final Report

Status: implementation COMPLETE; required CPU checks PASS. Formal V7 STOPPED; formal V8 NOT STARTED. No push. No simulator experiment was run.

## Required checklist

| Item | Result |
|---|---|
| GitHub actual source research (six pinned repos) | COMPLETE |
| Remote V7 source and exact340K checkpoint audit | PASS |
| V8 mathematical objective definition | COMPLETE |
| V7 Q loss / gradients / actual Adam update equivalence | PASS |
| GMM likelihood and gradients | PASS |
| Ten-step RNN supervision and executor/reset semantics | PASS |
| Verified successful experience filtering | PASS |
| Exact V7 -> V8 checkpoint compatibility | PASS |
| Independent V8 checkpoint save/restore | PASS |
| Actual combined gradient / Adam update | PASS |
| Pinned V7 loop adapter and no-training preflight | PASS |
| Optional real MuJoCo smoke | NOT_RUN |
| Formal training | STOPPED |

## Research and implemented change

Read actual actor/critic/replay Python from IBRL, AWAC/rlkit, RLPD, HIL-SERL, POMDP Baselines and ResFiT, with pinned commit URLs and license decisions in V8_GITHUB_RESEARCH.md. Borrow independent successful-episode retention, executed-action labels, normalized data likelihood and masked recurrent supervision. Do not import Q-max action selection, advantage weights, SAC/entropy, residual Actor, ensemble or new rewards.

V8 preserves the original final-H10 Q1 component-mean objective. It adds a separate all-ten-position likelihood on verified offline/online successful executed actions. Separate good data/RNG do not change the ordinary Critic replay sampler. The original Q backward, clipping, optimizer, policy delay, Polyak and V7 handoff remain the actual reused implementation.

## Exact fork and data audit

Source: /data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v7_pirlnav_schedule/stage3v6_readiness_v2_multi_mean_random_formal_20260929_130945/random2q/multi_q/checkpoints/best_success.pth

Companion: /data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v7_pirlnav_schedule/stage3v6_readiness_v2_multi_mean_random_formal_20260929_130945/random2q/multi_q/checkpoints/best_success.sequences.npy

Checkpoint SHA256: a9510da7c30ea87623717c52bc2bb3a86e5ed92f8602b2ae057154a76be4773e

Metadata: stage3-v7; env_steps340000; Critic updates84750; Actor updates12; Actor LR0; ACTOR_WARMUP. Actor is tensor-identical to Stage1 BC (hash f1e203241312301be92a504543727fb70daef33abf880ee7552218ce3ce2e873). These twelve zero-LR updates advanced optimizer bookkeeping but did not change Actor parameters. Full models/targets/optimizers/selector/global RNG/frozen readiness/offline sampler/pipeline were audited and recovered.

Offline labels: RNN77/100, Transformer40/100, GMM10/100. Five labeled successes lack complete final subtask corroboration and are excluded; accepted RNN75 +Transformer38 +GMM9 =122. Their identities remain in calibration.json. Historical successful collector-stop truncation is distinguished from timeout using the actual collector's end-reason semantics. Original data are untouched.

The initial real V8 good pool has122 offline episodes and zero online V8 successes. No new success is claimed. CPU checkpoint tests use one synthetic online episode only; those bundles are marked CPU_TEST_ONLY and training preflight rejects them. The earlier preserved initial validation bundle predates that marker and is rejected by the final strict V8 schema.

## Calibrated objective

Fixed supervised sigma: normalized action RMS residual to nearest component, floored at1% full controller input range. Range 0.02000000..0.07528995. Evaluation/online low_noise_eval=True and Gaussian std1e-4 remain unchanged.

Learned-std comparison: NLL -42.40845108, good norm 2188.20805684, scale-head norm 27.39464460.
Fixed-std comparison: NLL -32.70399475, good norm 128.97267607, scale-head norm0.

Q loss -0.180753469; lambda_good 0.00047228213199129753. Weighted good/Q gradient ratios: total14.7624%, shared RNN50%, mean head11.2290%, logits1.1529%. Combined total norm 0.420845135. The fixed coefficient keeps Q signal active on the calibration batch; future ratios remain monitored rather than assumed constant.

Fixed sigma disconnects direct std-head supervision while mean/logits and shared recurrent parameters learn. Shared-RNN changes can still change predicted learned std indirectly. The helper compares learned std diagnostically; production V8 permits fixed calibrated supervision only.

## Test evidence

- Test1: Q loss difference 0.0; gradient/parameter/Adam-state max differences 0.0/0.0/0.0. Tolerance1e-7. Test LR2e-7 is nonzero, with actual parameter change 2.38418579102e-07; this is not a zero-LR false equivalence.
- Test2: native MixtureSameFamily equality, five-component probability normalization,14D Gaussian product, learned/fixed scale paths, extremes and mask/weight normalization PASS.
- Test3: all ten positions have direct mean/logit gradients and recurrent/input BPTT. Actual executor20-step/two-reset-block max error 5.36441802979e-07, tolerance1e-5.
- Test4: all eight requested success/failure/timeout/partial/subtask/fallback cases PASS; independent RNG, dedup, exact action labels and retention after normal replay eviction PASS.
- Test5: full migration/save/resume, identical pool/global RNG draws and optimizers/targets/handoff/pipeline/offline sampler PASS. Source V7 files unchanged; overwrite guards and CPU-test isolation PASS.
- Test6: inherited gradient/Adam update equals explicitly combined gradients: max gradient/parameter/optimizer differences 0.0/0.0/0.0. All finite; actual shared RNN/mean/logit updates; std-head parameter update0.
- Adapter: exactly four pinned extension points PASS; actual default CLI returned PREPARED/STOPPED, no optimizer or simulator started.

Optional Test7 was not run because the existing executor/replay interfaces were tested offline and an additional simulator smoke was unnecessary for this task. Real NPU training, genuine online success collection, throughput and closed-loop gains remain unverified. They are not included in PASS claims.

## Artifacts and isolation

Remote code/report directory: /data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v8_qse_gmm

Branch: codex/stage3-v8-qse-gmm; HEAD 884e2bdce1ad6e04168419bd1a907387de5de3e2. Changes are uncommitted and not pushed. The pre-existing unrelated tracked modifications/deletion remain as found. All 39 audited V5/V6/V7 production Python hashes are unchanged.

Read: V8_GITHUB_RESEARCH.md, V8_ALGORITHM_DESIGN.md, README.md, stage3_v8_config_calibrated.json.
Raw numerical evidence: testing/calibration_inputs.npz, calibration.json, calibration.log, test_results.json, test_results_initial.json, test_cpu.log, test_cpu_initial.log, test_checkpoint_isolation.log, preflight.log, source_and_safety_audit.json, adapted_main_source.py.txt.
Current immutable CPU migration test bundle: /data/home/3220251075/lerobot_workspace/robomimic/training/Multi_IL_Full_Action_RL/stage3_v8_qse_gmm/testing/checkpoint_validation/migration_1791549483419786560.pth, with its .sequences.npy. Earlier test bundles/logs are retained; none are formal checkpoints.

The original V7 trainer has nested closures without injection interfaces. The small V8 adapter clones only main in memory, checks source hash, redirects Agent/replay/output/finish metadata and independent checkpoint helpers, and edits no production source. This is deliberate minimal isolation, not an unreviewed copied training framework.

No active formal/previous experiment process remains in the final process audit. No simulator worker was created by this task.

## Future acceptance and remaining questions

Do not equate maintaining BC behavior with success. At350K and400K, compare the original six successful seeds against V7's6/6 ->1/6 ->0/6 history, and evaluate all original ten seeds against BC6/10. Require real successes on formerly failed scenarios and a pre-registered independent seed set outside calibration/checkpoint selection. Record full Transport success and payload/trash flags, fixed-history Q, all-position mean/logit/probability drift, Q/good gradient ratios, online new-success counts and seed/source diversity.

The algorithm has not yet demonstrated improved real success or resolved the complete upstream cause of V7 collapse. Critic action-value limitations and insufficient supported-action supervision remain plausible contributors. Initial sigma/lambda calibration, conservative success filtering and source/seed concentration require later empirical assessment. All later training/evaluation is proposed only.

V8 IMPLEMENTATION AND CPU ACCEPTANCE COMPLETE

FORMAL STAGE3-V7 REMAINS STOPPED

FORMAL STAGE3-V8 NOT STARTED
