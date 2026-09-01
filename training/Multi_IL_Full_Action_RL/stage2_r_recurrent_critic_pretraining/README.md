# Stage2-R: Recurrent SAC Native Critic Pretraining

Stage2-R is independent from Stage2 and Stage2.1. It freezes the closed-loop
best Stage3-R actor at epoch 150 and compares random, RNN-only, and balanced
Multi-IL initialization of the vendored pomdp-baselines `Critic_RNN`.

The critic is the Stage4 contract: LSTM encoder, native previous-action/reward/
observation history, native state-action shortcut, twin SAC Q heads, target
critic, and native Polyak update. Training uses the unmodified efficient
`RAMEfficient_SeqReplayBuffer` with ten-step sequences. The current Q action is
always the recorded Stage1 behavior action. The deterministic frozen Stage3-R
action is used only for the next-state target. Both terminated and truncated
transitions disable bootstrapping.

Architecture:

- state/action: 59 / 14
- embeddings (action/observation/reward): 16 / 32 / 16
- recurrent core: one-layer LSTM, hidden size 128
- shortcut: current 59D observation and 14D current action
- twin Q heads: `[256, 256] -> 1`
- sequence length: 10
- gamma / tau: 0.99 / 0.005
- target update interval: 1
- sequence batch: 26, giving about 260 valid timesteps per update
- formal budget: 50,000 updates, validation every 1,000
- Multi-IL policy sampling: balanced RNN:Transformer:GMM = 1:1:1

Smoke test (20 updates, five train seeds and two validation seeds):

```bash
NPU_ID=0 bash training/Multi_IL_Full_Action_RL/stage2_r_recurrent_critic_pretraining/run.sh smoke
```

Formal run:

```bash
NPU_ID=0 bash training/Multi_IL_Full_Action_RL/stage2_r_recurrent_critic_pretraining/run.sh formal
```

Outputs are written only below
`training_runs/Multi_IL_Full_Action_RL/stage2_r_recurrent_critic_pretraining/`.
Stage3-R is never modified and Stage4 is never started.
