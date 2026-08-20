# Stage 2 critic pretraining

Critic pretraining is not implemented or started. The only implemented component is the Stage 2.0
frozen BC-RNN target-policy sanity check:

```bash
python training/Multi_IL_Full_Action_RL/stage2_critic_pretraining/validate_frozen_rnn_target.py
```

It reads five fixed episodes per Stage 1 policy, never modifies the HDF5 files, and writes only
`training/Multi_IL_Full_Action_RL/analysis/stage2/frozen_rnn_target_validation.json`.

## Formal critic pretraining

`train_stage2_critics.py` compares an untrained random twin critic, an RNN-only critic, and a
policy/episode-balanced Multi-IL critic. All variants use the same `59 + 14 -> 256 -> 256 -> 1`
twin-Q initialization and the same frozen BC-RNN target policy. The Bellman input action is always
the stored behavior action; only the next action comes from the cached RNN history replay.

Run a small server-side smoke test first:

```bash
python -u training/Multi_IL_Full_Action_RL/stage2_critic_pretraining/train_stage2_critics.py \
  --smoke-test --device npu:0
```

This uses 20 updates, batch size 32, five training seeds, two validation seeds, and writes under
`/tmp/multi_il_full_action_rl_stage2_smoke`. It does not modify the formal 50K configuration.

Formal training:

```bash
python -u training/Multi_IL_Full_Action_RL/stage2_critic_pretraining/train_stage2_critics.py \
  --device npu:0
```

The default configuration is 50,000 updates per trained critic, batch size 256, validation every
1,000 updates, gamma 0.99, and tau 0.005. Formal artifacts are stored only below
`training_runs/Multi_IL_Full_Action_RL/stage2_critic_pretraining/<timestamp>`.
