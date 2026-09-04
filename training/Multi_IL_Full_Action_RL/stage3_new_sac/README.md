# Stage3-new: paired standard SAC, RNN-Q versus Multi-Q

This independent route tests whether the two Stage2-new Critic initializations
change subsequent online policy optimization.  The first formal experiment is
RNN-Q versus Multi-Q only; Random-Q is intentionally deferred.  No Stage3-R,
RSAC, Actor distillation, Actor/Critic freeze, BC loss, high UTD, ensemble, or
Stage1 rollout replay is used.

## Shared standard SAC

The Actor is a randomly initialized RLKit `TanhGaussianPolicy`: 59-D canonical
low-dimensional observation, hidden layers `[256, 256]` with ReLU, state-dependent
log standard deviation clamped to `[-20, 2]`, and a 14-D post-tanh environment
action.  `shared/actor_init.pth` is created once; both processes strict-load the
same state dict and verify its hash.  Evaluation uses deterministic `tanh(mu)`.

The online and target Twin Critics use the exact Stage2-new public builder:

```
concat(state 59, action 14)
Linear(73,256) -> LayerNorm -> ReLU
Linear(256,256) -> LayerNorm -> ReLU
Linear(256,1)
```

Stage2 checkpoints are checked for dimensions, hidden sizes, ReLU, LayerNorm,
and gamma 0.99, then loaded with `strict=True`.  The target Critic starts as an
exact copy.  Online Critic optimization uses AdamW with weight decay `1e-4`;
Actor and alpha optimizers use zero weight decay.

The TD target and Actor objective are standard clipped-double-Q SAC with
automatic entropy tuning, target entropy `-14`, initial alpha `1`, soft target
updates (`tau=0.005`), and UTD=1.  After each environment transition the code
immediately inserts one online transition and, once replay has 1000 transitions,
performs one joint SAC update.  There are no episode-end update bursts.

## Expert/offline data

Each batch is composed by sampling two independent buffers: 50% uniform
transitions from the original TwoArmTransport proficient-human low-dimensional
demonstration HDF5 and 50% uniform transitions from online replay.  Stage1
BC-RNN/Transformer/GMM rollout HDF5 files are explicitly rejected because their
information must enter Stage3 only through Critic initialization.

The repository identifies the official Transport PH low-dim source through the
Pure-IL BC-RNN checkpoint, but does not contain the server HDF5 absolute path.
`prepare_stage3_new_pair.py --expert-checkpoint ...` reads the checkpoint's real
`train.data` at runtime, validates the HDF5 environment and schema, and records
the resolved path.  `--expert-dataset` is also supported when the exact path is
already known.  There is no observation normalization or extra action scaling.
Stored robomimic demonstration `dones` are used as the offline terminal masks;
online time-limit truncations retain SAC bootstrap while genuine environment or
success terminals do not.

## Budget, evaluation, probes, and resume

The first run uses 300,000 actual environment transitions per group.  Evaluations
use 10 fixed seeds `20000..20009` at steps 0, 25k, 50k, 100k, 150k, 200k, and
300k.  Evaluation is deterministic and its transitions are never inserted into
replay.  Training reset number `k` uses seed `30000+k` in both processes.

Stage2's saved deterministic probe manifest is reused at steps 0, 1k, 5k, 10k,
25k, 50k, 100k, 200k, and 300k.  NPZ artifacts retain Q1/Q2/Qmin, deterministic
Actor actions, and both action gradients.  Pair comparison reports head-wise
absolute Q differences and Pearson/Spearman correlations, gradient cosines,
Actor-action L2 divergence, and the two success-rate learning curves.

Milestone checkpoints and `latest.pth` contain all models, optimizers, alpha,
RNG state, episode/environment context, and a separate complete online replay
snapshot.  Resume restores them.  Extending to 500k requires explicitly resuming
both groups with the same new budget; it is never automatic.

## Local synthetic smoke test

```bash
python training/Multi_IL_Full_Action_RL/stage3_new_sac/validate_stage3_new.py
```

This exercises strict Stage2 loading, identical Actors, LayerNorm placement,
target copying/soft update, automatic alpha, per-transition replay, the 1000-step
threshold, UTD accounting, 50/50 sampling, SAC backward passes, evaluation
isolation, deterministic actions, checkpoint/resume, probes, gradients, and pair
comparison.  It does not use robosuite or server artifacts.

## Formal server run

Set the already completed Stage2-new run once:

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
conda activate robosuite_npu
STAGE2_RUN=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage2_new_critic_pretraining/stage2new_formal_001
PAIR_ID=stage3new_pair_001
python training/Multi_IL_Full_Action_RL/stage3_new_sac/prepare_stage3_new_pair.py \
  --run-id "$PAIR_ID" \
  --stage2-run-dir "$STAGE2_RUN" \
  --rnn-q-checkpoint "$STAGE2_RUN/rnn_q/checkpoints/best.pth" \
  --multi-q-checkpoint "$STAGE2_RUN/multi_q/checkpoints/best.pth" \
  --expert-checkpoint "/data/home/3220251075/lerobot_workspace/training_runs/Pure IL/two_arm_transport_bc_rnn_official_ph_low_dim/two_arm_transport_bc_rnn_official_ph_low_dim/20260811151701/models/model_epoch_1000_low_dim_v15_success_0.9.pth"
```

The prepare command prints `PAIR_RUN_DIR`.  In terminal 1 / NPU 0:

```bash
PAIR_RUN_DIR=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_new_sac/stage3new_pair_001
STAGE2_RUN=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage2_new_critic_pretraining/stage2new_formal_001
python -u training/Multi_IL_Full_Action_RL/stage3_new_sac/train_stage3_new.py \
  --group rnn_q --device npu:0 --pair-run-dir "$PAIR_RUN_DIR" \
  --critic-init-checkpoint "$STAGE2_RUN/rnn_q/checkpoints/best.pth" \
  2>&1 | tee "$PAIR_RUN_DIR/rnn_q/console.log"
```

In terminal 2 / NPU 1:

```bash
PAIR_RUN_DIR=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_new_sac/stage3new_pair_001
STAGE2_RUN=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage2_new_critic_pretraining/stage2new_formal_001
python -u training/Multi_IL_Full_Action_RL/stage3_new_sac/train_stage3_new.py \
  --group multi_q --device npu:1 --pair-run-dir "$PAIR_RUN_DIR" \
  --critic-init-checkpoint "$STAGE2_RUN/multi_q/checkpoints/best.pth" \
  2>&1 | tee "$PAIR_RUN_DIR/multi_q/console.log"
```

Resume both to 500k (run both commands, one per NPU):

```bash
python -u training/Multi_IL_Full_Action_RL/stage3_new_sac/train_stage3_new.py --group rnn_q --device npu:0 --pair-run-dir "$PAIR_RUN_DIR" --critic-init-checkpoint "$STAGE2_RUN/rnn_q/checkpoints/best.pth" --resume "$PAIR_RUN_DIR/rnn_q/checkpoints/latest.pth" --total-env-steps 500000
python -u training/Multi_IL_Full_Action_RL/stage3_new_sac/train_stage3_new.py --group multi_q --device npu:1 --pair-run-dir "$PAIR_RUN_DIR" --critic-init-checkpoint "$STAGE2_RUN/multi_q/checkpoints/best.pth" --resume "$PAIR_RUN_DIR/multi_q/checkpoints/latest.pth" --total-env-steps 500000
```

After both processes reach the same budget:

```bash
python training/Multi_IL_Full_Action_RL/stage3_new_sac/compare_stage3_new_pair.py --pair-run-dir "$PAIR_RUN_DIR"
```

Formal output is isolated under
`.../stage3_new_sac/<PAIR_ID>/{shared,rnn_q,multi_q,pair_comparison}`.
