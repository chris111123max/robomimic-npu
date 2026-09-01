# Stage4: Same Recurrent SAC Actor + Two Selected Critic Initializations

The default launcher starts exactly two independent online recurrent SAC jobs. Both
jobs use the same Stage3-R epoch-150 Actor, seed, empty online sequence replay,
environment budget, evaluation states, and optimization settings. The only
experimental variable is Critic initialization: Stage2-R-v2 RNN-only update
41000 or Stage2-R-v2 Multi-IL update 14000. The Random Critic implementation is
retained for later use, but the default launcher does not preflight or start it.

The online objective calls the vendored pomdp-baselines SAC loss and recurrent
Critic directly. Stage4 adds orchestration, phase scheduling, global Critic
gradient clipping, failure capture, exact evaluation, checkpointing, and replay
persistence. It does not inject demonstrations, add BC loss, use residual
actions, or perform random-action warm-up.

## Phase contract

- `0..30000`: Actor parameters have `requires_grad=False`; Critic-only updates.
- `30000..50000`: Actor is enabled and its LR increases linearly from zero to
  `3e-4`.
- `50000..1000000`: ordinary joint recurrent SAC.

Critic LR is `1e-4` in all phases, inherited from Stage2-R-v2. Critic global
gradient norm is clipped at `1.0`. Evaluation always uses deterministic
`tanh(mu)` on exact Stage1 states 10080 through 10099. Training actions are
stochastic SAC samples from the initialized Actor.

## Smoke test

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
bash training/Multi_IL_Full_Action_RL/stage4_rsac_online_finetuning/run_stage4_two_groups.sh smoke
```

The launcher first runs strict preflight checks on NPU 1 and NPU 2, then starts
the RNN-only and Multi-IL smoke jobs in the background. NPU 0 is unused.

## Formal run

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
bash training/Multi_IL_Full_Action_RL/stage4_rsac_online_finetuning/run_stage4_two_groups.sh formal
```

The command prints the run directory, two owned PIDs, and two log paths. It
does not attach `tail -f`. A finalizer creates `stage4_comparison.json`,
`stage4_comparison.md`, and `stage4_learning_curves.csv` only after all three
groups report successful completion.

Monitor or stop only this run:

```bash
bash training/Multi_IL_Full_Action_RL/stage4_rsac_online_finetuning/monitor_stage4.sh RUN_DIR
bash training/Multi_IL_Full_Action_RL/stage4_rsac_online_finetuning/stop_stage4.sh RUN_DIR
```

Resume is explicit and group-local:

```bash
bash training/Multi_IL_Full_Action_RL/stage4_rsac_online_finetuning/resume_stage4_group.sh \
  rnn_only_critic npu:1 RUN_DIR RUN_DIR/rnn_only_critic/checkpoints/step_00500000.pth
```
