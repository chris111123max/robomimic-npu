# Stage 3B — DAgger Corrective Actor Distillation

Stage 3B starts from the Stage3A-v2 success-only Actor and distils corrective
actions from the frozen BC-RNN teacher on the history that is actually visited
by a mixed teacher/student rollout. It trains only the existing SAC-compatible
`59 -> 256 -> 256 -> Gaussian(14)` Actor. The `log_std` head remains exactly
zero-weight / `-3` bias and is frozen. No Critic, SAC update, or Stage 4 code is
used.

The Student state is concatenated in the `canonical_observation_keys` order
read from the Stage 1 BC-RNN HDF5 file. The Teacher receives the live robomimic
observation dictionary in temporal order, and its recurrent state is reset at
the start of every episode.

## Data split and targets

- DAgger collection and gradients: seeds `10000..10079` only.
- Held-out imitation validation and environment selection: seeds
  `10080..10099`; these seeds never enter corrective data or gradients.
- Corrective target: the frozen BC-RNN action on the actual mixed-policy
  history, never the action selected for execution.
- Each minibatch contains 50% successful Stage 1 BC-RNN transitions and 50%
  aggregated corrective transitions, sampled with replacement.
- Round betas are `0.7`, `0.4`, and `0.1`; every action-selection Bernoulli draw
  is independent.

## Teacher sanity and two-seed smoke test

Run this before any formal 80-seed collection. It first executes the Teacher
with `beta=1.0` on seeds 10000 and 10001 and compares the complete action arrays,
episode lengths, and success results against Stage 1. Only a PASS permits the
two-seed corrective collection and two-epoch reload test.

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic

python -u \
  training/Multi_IL_Full_Action_RL/stage3_actor_initialization/stage3b_run_rounds.py \
  --stage smoke \
  --device npu:0
```

The smoke run is written below
`training_runs/Multi_IL_Full_Action_RL/stage3b_dagger_actor_distillation/smoke_<timestamp>`.
Formal collection is not started by this command.

## Round 1 formal collection

Use a new formal run directory. The Teacher sanity is deliberately repeated in
that directory because formal collection is blocked unless its local
`teacher_sanity.json` says `PASS`.

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic

RUN_ROOT="/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3b_dagger_actor_distillation"
RUN_DIR="$RUN_ROOT/$(date +%Y%m%d_%H%M%S)"
CONFIG="training/Multi_IL_Full_Action_RL/stage3_actor_initialization/stage3b_config.json"
INITIAL_ACTOR="/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3a_v2_success_only_actor_initialization/stage3a_v2_success_only_400ep_20260827_210550/checkpoints/success_only_best_val_mse.pth"

mkdir -p "$RUN_DIR/logs"

python -u \
  training/Multi_IL_Full_Action_RL/stage3_actor_initialization/stage3b_collect_dagger.py \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --student-checkpoint "$INITIAL_ACTOR" \
  --mode teacher-sanity \
  --device npu:0

python -u \
  training/Multi_IL_Full_Action_RL/stage3_actor_initialization/stage3b_collect_dagger.py \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --student-checkpoint "$INITIAL_ACTOR" \
  --mode collect \
  --round-id 1 \
  --beta 0.7 \
  --seed-start 10000 \
  --num-seeds 80 \
  --device npu:0 \
  --output "$RUN_DIR/datasets/round1_corrective.hdf5" \
  2>&1 | tee "$RUN_DIR/logs/round1_collection.log"
```

## Round 1 formal training

```bash
python -u \
  training/Multi_IL_Full_Action_RL/stage3_actor_initialization/stage3b_train_dagger.py \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --round-id 1 \
  --input-checkpoint "$INITIAL_ACTOR" \
  --corrective-dataset "$RUN_DIR/datasets/round1_corrective.hdf5" \
  --device npu:0 \
  2>&1 | tee "$RUN_DIR/logs/round1_training.log"
```

Training runs for at most 100 epochs with patience 20. Round 2 and Round 3 must
pass all accumulated `--corrective-dataset` paths so that data is aggregated.
`stage3b_run_rounds.py --stage run-all --run-dir "$RUN_DIR"` automates the three
formal rounds, held-out evaluations, round selection, and final 100-seed
evaluation, but it must not be launched until the smoke test has been reviewed.

Stage 3B stops after producing `stage3b_shared_actor_best.pth` and its evaluation
reports. It never enters Stage 4.
