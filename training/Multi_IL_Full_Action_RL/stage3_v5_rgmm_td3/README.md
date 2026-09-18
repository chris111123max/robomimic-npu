# Stage3-v5: Critic-gated TD3 handoff

Stage3-v5 preserves V4's GMM Actor objective, twin-Q target, recurrent context,
action normalization, reward/terminal behavior and Polyak update.  It changes
only training control: `CRITIC_ONLY -> ACTOR_WARMUP -> JOINT_RL`.

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
