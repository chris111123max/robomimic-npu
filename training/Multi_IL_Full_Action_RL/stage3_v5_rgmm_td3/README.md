# Stage3-v5: Critic-gated TD3 handoff

This Stage3-v5 adaptation preserves the audited GMM Actor objective, twin-Q target,
action normalization, reward/terminal behavior, Polyak update, and the original
`CRITIC_ONLY -> ACTOR_WARMUP -> JOINT_RL` control logic. It changes only the
Critic representation and initialization source: both branches now load the
Stage2.2 history-aware Twin-Q architecture.

`CRITIC_ONLY` has UTD 0.25 (`update_credit += 0.25` per eligible online
transition), Actor LR 0, no Actor optimizer step, and no policy evaluation.
Readiness is considered only at 100K and then each 10K aggregate online steps.
It needs data coverage, rank correlation, success/failure AUC and Q separation,
twin-Q agreement, TD plateau, Q-scale stability, OOD action stress, and three
consecutive passes. A branch that is not ready at 300K becomes
`CRITIC_NOT_READY`; it never opens the Actor.

When ready at S, warm-up lasts `clip(S, 100K, 300K)` online transitions.
Actor LR is `2e-6 * progress`; Critic LR is
`critic_lr_ready * (1 - 0.75 * progress)`. Actor updates use `policy_delay=4`.
There is no policy evaluation during warm-up. At warm-up completion the first
ten-seed evaluation runs, then evaluations run every 100K steps for that branch.

Replay remains 256 transitions per Critic update: 128 offline + 128 online.
`rnn_q` draws the offline half from BC-RNN only. `multi_q` uses the same 128
offline total, split 43/43/42 across BC-RNN, BC-Transformer and BC-GMM, with the
remainder rotated over time.

The Critic token is `[observation_t, previous_executed_action_t, t/700]`.
Stage3 now uses the exact Stage2.2 recurrent contract: a zero-state sliding
context of at most 10 transitions, with the first previous-action token forced
to zero. Critic replay is fixed at 10 transitions. The Bellman successor uses
the corresponding shifted 10-step next-observation context and is also
zero-state at its first token. Actor BPTT still follows the BC-RNN 10-step reset
block; the RL Q objective is evaluated only at that block's final transition,
where the Actor block and Critic sliding window are exactly identical.
Readiness, OOD stress, TD diagnostics, and episode Q diagnostics all use the
same horizon-10 Critic adapter. The readiness thresholds, consecutive-pass rule,
warm-up schedule, and unfreeze criteria are unchanged.

Stage1 HDF5 replay is loaded natively from `/episodes`, retaining
`terminated`, `truncated`, and `dones`. Both a true termination and a
truncation mask the Bellman target; neither episode boundary bootstraps. The vector collector dispatches
worker steps asynchronously and consumes previous-round credits with actual
forward/backward/optimizer/Polyak work while workers simulate. Each atomic
update is followed by a readiness poll; an already-ready round waits at most
one guaranteed atomic update, not an unbounded credit drain. Credit backlog
divided by UTD defines learner lag. Dispatch is throttled before this exceeds
256 eligible transitions. Resets send all RESET requests before receiving.

Rollout uses immutable Actor snapshots, selected per environment only at its
hidden-reset boundary. Train weights never replace weights within a recurrent
block. Policy-version lag is bounded and logged. Resume discards incomplete
rollout episodes/hidden state rather than mixing old hidden with new weights.

Readiness uses independent reservoir-selection RNG (at most 64 success and
64 failure episodes), then freezes episodes, sequence indices and OOD noise.
All training RNG states are defensively restored even if diagnostics fail.
Coverage still requires 150 completed episodes and at least 30 of each class.
TD diagnostics share the training Bellman target helper. Plateau requires
small absolute relative change, not merely lack of improvement. OOD hard
gating uses **p95 excess normalized by reference Q standard deviation**;
the existing numeric threshold 2.0 is applied to this statistic, not absolute
maximum excess. Absolute/relative/normalized mean, p95 and max are logged.

Checks begin at 100K. With three history points followed by three consecutive
overall passes, the earliest usual readiness is 140K, not 120K; insufficient
coverage or failed metrics can delay it. No gate is relaxed to shorten this.

## Ordered server acceptance (no formal training)

After preparing a fresh pair, run on one card:

```bash
SCRIPT_DIR=training/Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3
python "$SCRIPT_DIR/benchmark_stage3_v5.py" --acceptance --pair-run-dir "$RUN_DIR" --group multi_q --device npu:0 --steps 4096 --warmup-steps 1024
```

This runs compilation, replay/schedule/validator, RNG/snapshot/lag and math
tests, real A/B/C/D benchmarks, A/D scaling at 2/4/8/16 envs, then a small
overlap-verified smoke. Failure stops acceptance. Measurements and logs go
under the pair's branch benchmarks directory; smoke has a separate directory.
A/B use stored Stage1 actions and no timed NPU forward/backward; C/D use the
same frozen Actor and are the matched synchronous/asynchronous comparison.
Benchmark profiling synchronizes devices for honest timing; production
profiling defaults off. CPU synthetic tests are correctness evidence only.

`collector_wait_ms` measures lag-throttle catch-up; `learner_wait_ms` records
environment wait (not an independent idle-device estimator). Aggregate and
collector throughput both count collected transitions per measured wall time.
Overlap sums per-round maximum intersections of actual worker simulation
and device-completed learner intervals, excluding replay preparation.
Effective policy delay is null in critic-only benchmarks (Actor disabled).
Source counters separately report `critic_offline_*` and `actor_offline_*`.
The historical approximately 22 steps/s baseline is user-reported, not a
matched measurement. Do not claim improvement before server C/D results.

Prepare a pair, then train a branch (do not run the former Phase-0 scripts):

```bash
python training/Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3/prepare_stage3_v5_pair.py \
  --bc-rnn-checkpoint "$BC_RNN_CHECKPOINT" --rnn-q-checkpoint "$RNN_Q_CHECKPOINT" \
  --multi-q-checkpoint "$MULTI_Q_CHECKPOINT"
python training/Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3/train_stage3_v5_vector.py \
  --group rnn_q --device npu:0 --pair-run-dir "$RUN_DIR" --critic-init-checkpoint "$RNN_Q_CHECKPOINT"
```

Use `--group multi_q` and the Multi Critic checkpoint for the other branch.
Checkpoints persist the FSM, readiness history, warm-up schedule, LRs, update
credit, replay, optimizer/model/target state, and RNG state.
