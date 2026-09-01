# Stage2-R-v2

Stage2-R-v2 is a clean restart of both recurrent critic groups after Stage2-R
exhibited non-finite LSTM gradients. It preserves the Stage2-R actor, native
pomdp-baselines Critic_RNN architecture, datasets, balanced sampling, TD target,
sequence length, gamma, tau, validation protocol, and 50,000-update budget.

The only optimization changes are shared by RNN-only and Multi-IL:

- critic learning rate: `3e-4 -> 1e-4`
- global gradient clipping: L2 norm, `max_norm=1.0`, `foreach=False`

The pre-clipping norm and interval mean, median, p95, p99, maximum, and clipped
fraction are written to training and validation metrics. A non-finite gradient
is never skipped. The current batch and pre-step critic, target, optimizer, and
diagnostics are saved below `debug_nan/update_*`, then the group stops.

Smoke:

```bash
NPU_ID=0 bash training/Multi_IL_Full_Action_RL/stage2_r_v2_recurrent_critic_pretraining/run.sh smoke
```

Formal:

```bash
NPU_ID=0 bash training/Multi_IL_Full_Action_RL/stage2_r_v2_recurrent_critic_pretraining/run.sh formal
```

Replay a saved failure batch:

```bash
python -u training/Multi_IL_Full_Action_RL/stage2_r_v2_recurrent_critic_pretraining/debug_replay_nan_batch.py \
  --failure-dir /absolute/path/to/debug_nan/update_00000000 --device cpu

python -u training/Multi_IL_Full_Action_RL/stage2_r_v2_recurrent_critic_pretraining/debug_replay_nan_batch.py \
  --failure-dir /absolute/path/to/debug_nan/update_00000000 --device npu:0 --detect-anomaly
```

Stage4 must inherit `critic_lr=1e-4` and `max_gradient_norm=1.0`. This stage
does not modify Stage3-R and never starts Stage4.
