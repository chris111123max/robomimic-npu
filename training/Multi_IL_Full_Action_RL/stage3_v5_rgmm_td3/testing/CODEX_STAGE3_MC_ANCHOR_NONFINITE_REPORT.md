# Final conclusion

The first observed nonfinite is **update 503, backward_gradient**, in the **Q1 recurrent/history encoder** for both MC-anchor branches (lambda 0.25 and 0.50). The first affected parameter in named-parameter inspection is `q1.token_encoder.0.weight` (`[64, 74]`); all Q1 LSTM weight/bias gradients are also nonfinite. Target construction, online Q, and loss are finite immediately before backward. The 504 loss failure is delayed: nonfinite gradients at 503 flow through clipping and Adam into the online Critic; Polyak then contaminates the target Critic, so target Q and TD target are nonfinite at 504. This is **not** a target-value explosion or Adam-state corruption preceding backward. It is a Q1 recurrent backward numerical failure; the exact autograd kernel/internal tensor first producing NaN/Inf is not established.

## Reproducibility

The minimal replay-only reproduction completed 503 updates then failed at update 504 for **both** lambda 0.25 and 0.50, matching the original causal JSON. The deterministic schedule and probe hashes match the original. Initial online/target Critic and actor hashes, optimizer-state hash, seed 20260926, batch size 256, step0 transfer, MC returns, production Bellman target, and Polyak path were preserved. The first update-504 batch contract is finite. The diagnostic used `npu:0`, no environment rollout.

## Validity

The detailed 480–504 instrumentation altered the execution trajectory and completed 504 updates without failure. It is retained in the JSON as `traces`, **not** used as evidence for the failing trajectory. Two narrower runs preserved production updates through 502 or 503. The 503 precursor run reproduced nonfinite backward gradients; the 504 pinpoint run reproduced the original target-Q failure. Since the 503 trace executes the same production forward/target/loss/backward arithmetic but observes its intermediate tensors, its phase identification is strongly supported. It cannot prove which internal backward kernel first emitted the nonfinite element.

## First nonfinite location

| Branch | First update | First phase | First affected named gradient | Other affected block |
|---|---:|---|---|---|
| lambda 0.25 | 503 | `backward_gradient` (before clipping) | `q1.token_encoder.0.weight` `[64,74]` | Q1 token encoder and LSTM; 8 parameters, 67,110 scalar gradients |
| lambda 0.50 | 503 | `backward_gradient` (before clipping) | `q1.token_encoder.0.weight` `[64,74]` | Q1 token encoder and LSTM; 8 parameters, 67,134 scalar gradients |

The named parameter is first in deterministic inspection order, **not** proof that its autograd operation was the first to fail. All Q2/head gradients remained finite at the first preclip inspection.

## Lambda 0.25 trace

At 503 entry, online and target parameters and Adam state were finite; parameter max abs was 1.14935 and Adam `exp_avg_sq` max was 1.57748. Target Q1/Q2 max abs: 0.96606/0.96194; `y_stage3` and `y_anchor` max abs: 1/1. Online Q1/Q2 max abs: 1.00387/0.92618. Q1/Q2 MSE: 0.00145205/0.00062739 (total 0.00207944), all finite. Q1/Q2 residual RMSE: 0.03811/0.02505. After backward, 67,110 Q1 recurrent gradient scalars were nonfinite; the largest *remaining finite* gradient magnitude was 5.07e36. Clip returned nonfinite and all 163,458 gradient scalars became nonfinite. Adam state was finite before step (in the phase-continuation trace `exp_avg_sq` max 1.45989), then 326,916 state scalars and all 163,458 online parameter scalars became nonfinite after step. Target parameters were finite before Polyak, all nonfinite after it.

## Lambda 0.50 trace

At 503 entry, online and target parameters and Adam state were finite; parameter max abs was 1.15356 and Adam `exp_avg_sq` max was 5.40745. Target Q1/Q2 max abs: 0.96309/0.95875; `y_stage3` and `y_anchor` max abs: 1/1. Online Q1/Q2 max abs: 0.93788/0.93519. Q1/Q2 MSE: 0.00146686/0.00097061 (total 0.00243746), all finite. Q1/Q2 residuals and target values were not exploding. After backward, 67,134 Q1 recurrent gradient scalars were nonfinite; the largest *remaining finite* gradient magnitude was 0.19544. Clip returned nonfinite and all 163,458 gradient scalars became nonfinite. Adam state was finite before step (in the phase-continuation trace `exp_avg_sq` max 3.94544), then 326,916 state scalars and all 163,458 online parameter scalars became nonfinite after step. Target parameters were finite before Polyak, all nonfinite after it.

## 500–504 comparison

Only 503 and 504 have phase-level measurements on reproducing trajectories. The broad 480–504 trace did not reproduce the failure, so its 500–502 values must **not** be spliced into this table. Original production diagnostic records a finite update-500 probe and the branch was still running through 502, but does not contain batch-level phase values for those steps.

| Metric | 500 | 501 | 502 | 503 lambda .25 / .50 | 504 lambda .25 / .50 |
|---|---|---|---|---|---|
| Q1 online max abs | not captured | not captured | not captured | 1.00387 / 0.93788 | not reached: target already nonfinite |
| Q2 online max abs | not captured | not captured | not captured | 0.92618 / 0.93519 | not reached |
| target Q max abs (Q1/Q2) | not captured | not captured | not captured | 0.96606/0.96194 ; 0.96309/0.95875 | both Q1/Q2 entirely nonfinite |
| `y_stage3`, `y_anchor` max abs | not captured | not captured | not captured | 1/1 ; 1/1 | nonfinite by target-Q propagation |
| loss Q1 / Q2 | not captured | not captured | not captured | 0.00145205/0.00062739 ; 0.00146686/0.00097061 | production reports nonfinite loss; phase trace stops at target Q |
| grad preclip | not captured | not captured | not captured | 67,110 / 67,134 nonfinite scalars | not reached |
| grad postclip | not captured | not captured | not captured | all 163,458 nonfinite (both) | not reached |
| online parameter max abs pre-step / post-step | not captured | not captured | not captured | 1.14935/nonfinite ; 1.15356/nonfinite | already nonfinite at entry |
| Adam `exp_avg_sq` max before step | not captured | not captured | not captured | 1.45989 / 3.94544 (finite) | state nonfinite at entry |

This establishes a sudden **finite forward/loss → nonfinite backward at 503**, not a gradual target or online-Q blow-up visible in the 503 forward. It does not quantify the 500–502 batch-level trend on the exact failing trajectory.

## Parameter / optimizer state evidence

All 8 nonfinite preclip parameter gradients are `q1.token_encoder.{0,1}.{weight,bias}` and `q1.lstm.{weight_ih_l0,weight_hh_l0,bias_ih_l0,bias_hh_l0}`. The first inspected parameter, `q1.token_encoder.0.weight`, has 4,736/4,736 nonfinite gradient elements. Parameter/state were finite pre-step. The nonfinite clip norm caused all formerly finite gradients to become nonfinite; Adam then contaminated moments/parameters. Polyak propagated the online nonfinite values to target, not vice versa. At 504 entry all 28 online and all 28 target parameter tensors were nonfinite (163,458 scalars each). Target Actor output and replay batch were finite; Q1/Q2 target each had 1,280/1,280 nonfinite values and `expected_next` 256/256.

## Most likely numerical mechanism

A Q1 recurrent/history-encoder backward computation produces NaN/Inf despite a modest finite loss and finite forward activations. Global norm clipping receives nonfinite gradients and spreads NaNs to otherwise finite Q2/head gradients. Adam and then Polyak faithfully propagate the corruption. The trace does not establish whether the LSTM backward operator itself or an immediately upstream recurrent backprop tensor is the initial arithmetic source; no LR, optimizer, dtype, clipping, lambda, or target formula was changed to test a remedy.

## What is still unproven

The precise internal autograd node/operation responsible; why it occurs specifically at update 503 under NPU execution; exact 500–502 batch-level phase trends on the uninstrumented failing trajectory. Broad instrumentation prevents the original failure, so those rows cannot be claimed as paired evidence.

## Next repair implication

A next, separately authorized repair should focus on Q1 recurrent backward stability and add a nonfinite-gradient guard before clipping/optimizer/Polyak so one bad backward cannot contaminate the entire Critic. **No repair was performed in this diagnostic.** Formal training remains inappropriate until a remedy is tested and stability verified.

## Safety / artifacts

Only this report and `test_stage3_mc_anchor_nonfinite_trace.py` were created/edited under `testing/`, plus the diagnostic JSON under the run's `testing/stage2_vs_stage3_readiness/`. Environment steps: 0. Formal training steps: 0. No rollout, formal training, production-code edit, training-checkpoint write, or `fusion_result.json` edit.
