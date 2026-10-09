# Stage3-V8 Algorithm Design

Status: COMPLETE. Algorithm: **Q-Guided Successful Experience GMM Fine-Tuning (QSE-GMM)**. This task implements and tests the algorithm; formal training remains stopped.

## Evidence and choice

Prior paired closed-loop diagnosis recorded 340K/350K/400K success rates of 6/6, 1/6, 0/6 on the six initially successful seeds. On fixed BC histories, later GMM actions often have higher predicted Q1 and Q2 despite worse closed-loop outcomes. This supports loss of supported behavior under policy improvement; it does not uniquely prove local action misordering or identify the complete upstream cause. V8 is a testable intervention, not a declared cure.

The source research completed before SSH selects additive outcome-filtered data likelihood. Full AWAC would replace the objective and require credible advantage ordering; IBRL Q-max action selection/bootstrap has the same ordering dependence; residual policies change the architecture and execution contract. V8 borrows their data-learning principles without copying their objectives or source code.

## Preserved contracts

Actor: exact Stage1 BC RNN-GMM, canonical 59-dimensional observations, 14 controller actions, five components, two-layer 400-hidden LSTM, zero hidden at episode start and each actual step divisible by 10. low_noise_eval=True stays unchanged. Target Actor and rollout boundary snapshots remain the original implementations.

Critic: original Stage2.2 history-aware Twin-Q. A sliding context starts with zero recurrent state and zero unavailable previous action; each token uses real observation, previous executed action and episode progress. Only the final token of a complete aligned H10 Actor block matches the original pretrained Critic context. No early-position Q loss is added.

Bellman target remains random2q: select J uniformly from {1,2} exactly once per Critic update, then use reward + gamma*(1-terminal)*sum_k p'_k Q'_J(h',mu'_k). No hard min, ensemble or entropy term is introduced. Selector diagnostics do not consume its update RNG.

Retain Critic 128 offline +128 online sampling, actor Q batches from the original aligned 50/50 sampler, UTD=.25, policy_delay=4, original Adam/AdamW, clipping, and the original target Polyak rules (Critic each update; Target Actor each Critic update while actor gate is open). This is deliberately the existing rule, not a silent TD3 reinterpretation.

Retain V7 readiness, 200K frozen-Actor Critic LR decay 3e-4 ->7.5e-5, 100K Actor LR ramp 0 ->2e-6, no 300K Critic-only cutoff, original Transport reward/environment, evaluation seeds and NPU vector executor.

## Objective

For the original Q replay batch, retain exactly:

L_Q = -mean_b sum_k p_theta(k|h_b,9) Q1(ctx_b,H10, mu_theta(b,9,k)).

For a separate good-experience batch, normalize real controller commands as
a_norm=(a_env-offset)/scale and define:

log tilde_pi(a_norm|h) = logsumexp_k[
 log_softmax(logits)_k
 - sum_j ( .5*((a_norm_j-mu_kj)/sigma_j)^2 + log(sigma_j) + .5*log(2*pi)) ].

L_good = -sum_bt(mask_bt*w_bt*log tilde_pi(a_bt|h_bt))
          /sum_bt(mask_bt*w_bt).

L_V8 = L_Q + lambda_good*L_good.

This is a true five-component 14-dimensional Gaussian mixture likelihood, not weighted-mean MSE. It is a density in normalized action units. Environment-space log density would additionally subtract sum_j log(scale_j); that constant has no actor gradient and is excluded consistently from calibration and training.

Offline and online-success base weights both default to 1. Good batches use 50% each when an online pool exists and all offline when it is empty. Equal-source weighting is independent of Critic Q; no advantage weighting is enabled. Dividing by total valid weight prevents batch size or common source-weight scale from changing gradient magnitude. No padding is emitted by the sampler; binary masks are supported and empty/invalid weight reductions stop clearly.

Q gradients and good gradients use separate histories. The good loss covers all ten real Actor outputs; it never supplies early outputs to the final Critic context.

## Verified successful data

The Stage1 sources are the original bc_rnn, bc_transformer and bc_gmm transition HDF5 files recorded by V7. They contain successes and failures. Audit found success labels 77,40,10 of 100 episodes each, respectively.

The Stage1 collector latches env.is_success()['task'] and can end with termination_reason=success_collector_stop. It encodes that stop as truncated=True even though it is not a horizon failure. Therefore truncated alone cannot classify this historical dataset. V8 requires consistent episode_success labels, compatible canonical/environment/subtask schema, an explicitly completed episode, an audited successful end reason, contiguous steps from zero and final payload/trash flags corroborating full task completion. Actual timeout reason horizon is rejected. Five positively labeled episodes lack complete final corroboration and are conservatively excluded (2 RNN,2 Transformer,1 GMM), leaving 122. Their names and source paths remain in testing/calibration.json; no original label is changed. The exclusion does not prove the collector labels are wrong.

Full installed Transport success is payload_in_target_bin AND trash_in_trash_bin. V5's worker calls the official env.is_success task definition. Its original terminal/truncation rules are retained. Online V8 collection passes its won label only on finish, and the independent pool additionally requires a terminated, non-timeout, completed contiguous episode with both final subtask flags. Partial, aborted, failed, timeout and single-subtask episodes do not enter L_good. Flags alone never grant success. Subtask data remain available in recorded canonical observations.

Online labels are the exact action arrays passed by the existing executor to vector.step and ordinary replay.add. No synthesized BC/reference/Q-max action is used as a target. The adapter adds seed/episode_id provenance at the existing finish call. Existing V7 online successes are not silently relabeled into the new good pool.

GoodReplay keeps an independent copy of observations/actions/episode_steps. It uses complete windows starting at actual 0,10,20,...; tails shorter than ten are omitted, and no episode crossing occurs. It samples episodes uniformly, then aligned windows uniformly. It owns an independent NumPy Generator; it does not advance ordinary offline/online replay, target-selector, simulator or policy RNG.

Online pool capacity: 200000 transitions and at most 512 episodes; evict oldest complete episodes until both bounds hold. Oversized episodes are rejected. Content-hash dedup uses a bounded 4096-entry recent history; an identical experience can be re-admitted after that history expires, explicitly avoiding an unbounded seen-set. Offline pool is the fixed audited corpus. Success copies survive ordinary replay eviction.

Record source sample totals, accepted/rejected/duplicate/eviction counts, labels, seeds, episode IDs, unique seeds and largest seed share. No complicated seed-balancing curriculum is added. If no qualified offline or online good window exists, preparation/update stops clearly rather than looping, creating NaN or admitting failed data.

## Supervision variance and calibration

Use the real 340K Actor and fixed 64x10 good positions. Compare:
- learned std: NLL -42.408451; total good gradient norm 2188.208057; direct std-head norm 27.394645.
- fixed calibrated std: NLL -32.703995; total good norm 128.972676; direct std-head norm 0.

A negative NLL is valid for a continuous density and is not a probability error. Learned std can fit variance rather than preserve action modes, creates a new direct scale-head update absent in V7, and has much greater gradient magnitude on this actual batch. Fixed supervision std is selected. Learned std remains a diagnostic option of the likelihood helper, not the enabled production Agent mode.

For each normalized action dimension, measure RMS residual to the nearest Actor GMM component on the fixed good batch. Set supervision std to max(RMS residual,1% of the full normalized controller command interval). The audited controller input interval is [-1,1]; normalization therefore gives a floor .02/scale. The calibrated vector lies between .02 and .07528995. This scale is derived from actual action residuals and command units, not an arbitrarily inflated Gaussian. All 14 values, residuals and floors are saved.

This supervision distribution is distinct from actual execution. Evaluation/online Gaussian std remains 1e-4 with categorical sampling. Fixed sigma receives no gradient; mean/logits and the shared RNN do. Scale-head parameters remain without a direct loss update, but their outputs can change indirectly as the shared RNN changes.

## Initial auxiliary coefficient

Fixed original Q batch: 16 archived real aligned histories (16x10x59). It is read without rerunning simulator experiments. Good batch: 64 verified successful H10 windows. Calibration seed and checkpoint/input hashes are fixed; no evaluation or held-out seeds are used to select weights.

At 340K:
L_Q=-0.180753469; ||g_Q||=.412611199.
L_good=-32.703994751; ||g_good||=128.972676067.

Choose a fixed coefficient as the minimum of:
- .25*||g_Q||/||g_good|| for total gradient;
- .5*||g_Q,group||/||g_good,group|| for shared RNN, means and logits.

Result lambda_good=.00047228213199129753.
Weighted auxiliary/Q norm ratios: total .1476244, shared RNN .5, mean head .1122897, logits .0115286. Total cosine is .0627153; shared RNN .3280333; mean .0270111; logits -.8065651. Logits therefore have conflicting directions, but the auxiliary logits magnitude is only about 1.15% of its Q gradient. Combined total norm .420845135. Q signal remains active.

This is one initial calibration, not a controller or a guarantee about future ratios. The coefficient remains fixed across resume; log evolving ratios and source coverage. Later tuning/advantage weighting would require a separately authorized experiment.

## Minimal integration

V7's main function owns nested local learner/collector/evaluation closures and hard-coded imports; it has no trainer class or injection hooks. Copying its thousand-line framework would create a diverging fork. V8 instead clones that single function AST in memory and redirects four explicit extension points, each required to occur exactly once:
1. Agent import.
2. Online replay class import (inherited ordinary samplers plus finish hook).
3. Output root folder.
4. finish seed/episode provenance.

Helper bindings in the isolated namespace redirect read/config, metadata writes and checkpoint save/restore. No production file or production module global is patched. The audited V7 trainer SHA256 is required; any upstream change stops before execution and requires an explicit re-audit. The adapted function source is recorded for review.

The Agent inherits the actual V6/V7 implementation. With lambda=0 it immediately delegates without good sampling/forward/hooks. Otherwise it computes good autograd gradients on the same current parameters, adds detached first-order gradients via temporary parameter hooks during the original Q backward, and retains the original norm measurement, clipping, Adam step, counters, diagnostics and exception cleanup. A pre-step guard stops if any supervised parameter lacks an inherited Q gradient path. Fixed std avoids such an unsupported scale path. This implements exactly g_Q+lambda*g_good for ordinary first-order Adam; no second-order differentiation is claimed.

## Checkpoint and recovery

Only V7 best_success.pth metadata stage3-v7, env_steps340000, actor_updates12 and zero Actor LR is accepted for a fork. The Actor is compared tensor-for-tensor with loaded Stage1 BC; critic/targets/optimizers/selector/reference state/RNG/handoff/offline sampler/pipeline and exact companion best_success.sequences.npy are required. No 350K/400K fallback.

Restore exact Actor, Target Actor, Critic, Target Critic, Adam/AdamW states, target-selection RNG, global RNG, online replay/frozen readiness set, counters and handoff position. The reused loop restores offline sampler/pipeline credit. Retain V7 partial-episode discard/reset semantics on resume.

V8 adds algorithm version, full good pool (including source corpus copies and dedup/sample counters), its independent RNG, fixed loss config/coefficient, fork provenance and replay SHA256. V8 saves immutable binary checkpoints and companion replay files only under its own directory. LATEST.json is a metadata pointer. Existing names create a new timestamped name in the loop rather than overwriting a binary; direct save rejects existing paths. V7 paths are rejected as destinations.

CPU-validation checkpoints are explicitly CPU_TEST_ONLY, can be restored only on CPU and are rejected by training preflight. They contain synthetic cases used to test pool recovery and are not formal initialization artifacts. Unknown V8 purpose/schema fails closed. CPU saves preserve the original NPU RNG byte state without initializing NPU to read its current RNG.

## Correctness tests

Test1 uses real 340K optimizer state and an actual nonzero LR=2e-7 (350K schedule) on CPU clones; all Q-loss, gradient, parameter and Adam-state differences were exactly zero (tolerance1e-7). Parameters changed by2.3844186e-7; the good RNG was untouched.

Test2 compares logsumexp with native MixtureSameFamily at float64, five components/14 dimensions; verifies probability normalization, mean/logits/learned-scale gradients, fixed-scale disconnection, extreme logits/action/std finite values, mask rejection, batch duplication and weight normalization.

Test3 verifies direct mean/logits supervision at every position, input/shared-RNN BPTT, boundary errors, full-forward vs step semantics, and the actual BatchedGMMExecutor for two ten-step blocks and explicit reset. Max executor difference5.3644180e-7 (tolerance1e-5).

Test4 covers all eight requested successful/failed/partial/timeout/subtask/fallback cases, independent RNG, success-stop semantics, dedup and survival after ordinary replay eviction.

Test5 restores exact V7, saves immutable V8, restores models/targets/optimizers/readiness/pipeline/offline sampler and identical pool/global RNG sample sequences; confirms no source checkpoint/replay modification and CPU-test isolation.

Test6 compares the hooked production update against explicitly combined gradients and Adam. Gradient/parameter/optimizer differences are exactly zero; finite shared RNN/mean/logits changes follow the expected update. Std-head parameter change is zero.

Adapter preflight passes with exactly four extension points. Optional Test7 real MuJoCo smoke was NOT_RUN: existing executor and replay interfaces were exercised offline. NPU training, real new online success collection, throughput and closed-loop gains remain untested.

## Future acceptance: design only, no execution

Start from exact340K. Register checkpoint350K and400K comparisons before training. Preserve the original six seeds20002,20003,20004,20005,20007,20008, and the original full ten20000..20009. Avoiding the V7 6/6->1/6->0/6 loss is an early stability result, not the final success claim.

The scientific improvement endpoint must exceed BC6/10 on the full ten, including successes on previously failed seeds20000,20001,20006,20009. Report paired per-seed outcomes, uncertainty and failures; do not equate one noisy7/10 result with generalization.

Pre-register an independent20-seed set30000..30019 for a later authorized paired BC/V8 evaluation, after verifying these seeds have not been used for development or calibration. Keep them outside weight/checkpoint selection. Choose the held-out checkpoint by predeclared development criteria and evaluate it once. If evidence is inconclusive, report that rather than autonomously adding seeds.

Record both payload/trash task flags, completion stages and full task success; test recovery of payload ability lost in five350K V7 failures and acquisition of genuinely new full successes.

At every early checkpoint monitor:
- fixed-history Q1 objective and Q1/Q2 diagnostics (not a replacement objective);
- GMM mean/logit/probability drift by all ten positions;
- Q/good/combined gradient norms, group ratios/cosines and Actor parameter updates;
- source fractions, new verified online successes, distinct seeds/episode coverage, largest seed share;
- closed-loop success on known successful, original BC-failed and held-out scenarios.

A nearly unchanged Actor retaining only BC's six successes is preservation, not learning success. Higher Q alone is insufficient. Online success diversity and reproducible real full-task gains above BC are required.

## Remaining uncertainties

The complete upstream cause of V7 collapse is not uniquely identified; current evidence is compatible with critic action-value limitations and insufficient supported-action supervision. V8's objective targets the latter while preserving Q-guided improvement. This task proves local code/gradient/checkpoint correctness only.

Initial coefficient and sigma are calibrated on a bounded corpus; their usefulness during later policy/data distribution change is unproven. Conservative terminal corroboration can omit genuinely successful historical episodes with stale observation flags. Uniform episode sampling does not itself guarantee seed diversity; monitor the recorded coverage. The AST adapter is intentionally pinned and requires re-audit after V7 changes. Real NPU execution and closed-loop improvement require a subsequently authorized training/evaluation task.
