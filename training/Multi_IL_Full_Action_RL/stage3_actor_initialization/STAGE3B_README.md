# Stage 3B — DAgger Corrective Actor Distillation

Stage 3B is fully independent of all historical Stage3A Actor checkpoints.
Round 0 creates a fresh `59 -> 256 -> 256 -> Gaussian(14)` SAC-compatible Actor
inside the Stage 3B run directory and trains it on successful BC-RNN episodes.
The best held-out imitation-MSE checkpoint becomes `round0_best.pth`; DAgger
Round 1 starts from that file.

The frozen BC-RNN checkpoint is still required as the expert Teacher and as the
provenance of the Stage 1 dataset. It is not an Actor initialization checkpoint.
The Student `log_std` head stays exactly zero-weight / `-3` bias and frozen.

## Fixed data rules

- Round 0 and DAgger gradient data: seeds `10000..10079` only.
- Held-out validation/evaluation: seeds `10080..10099`; never used by gradients
  or corrective collection.
- Student state order comes from the Stage 1 HDF5 root attribute
  `canonical_observation_keys`.
- Corrective target is the frozen BC-RNN action on the actual mixed-policy
  history, never the executed action.
- DAgger minibatches are 50% original successful expert data and 50% aggregated
  corrective data.
- Round betas are `0.7`, `0.4`, and `0.1`.

## Smoke test

This creates a fresh two-epoch Round-0 Actor, performs the beta=1 Teacher
reproduction check, collects two Round-1 episodes, reloads the dataset, and runs
two Round-1 training epochs. It does not start formal collection.

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic

python -u \
  training/Multi_IL_Full_Action_RL/stage3_actor_initialization/stage3b_run_rounds.py \
  --stage smoke \
  --device npu:0
```

## Formal Round-0 training

Round 0 trains for 400 epochs and saves the lowest held-out imitation-MSE Actor.
The same command then runs the beta=1 Teacher sanity check. It stops before any
formal DAgger rollout.

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic

RUN_ROOT="/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3b_dagger_actor_distillation"
RUN_DIR="$RUN_ROOT/$(date +%Y%m%d_%H%M%S)"
CONFIG="training/Multi_IL_Full_Action_RL/stage3_actor_initialization/stage3b_config.json"
mkdir -p "$RUN_DIR/logs"

nohup python -u \
  training/Multi_IL_Full_Action_RL/stage3_actor_initialization/stage3b_run_rounds.py \
  --config "$CONFIG" \
  --stage bootstrap \
  --run-dir "$RUN_DIR" \
  --device npu:0 \
  > "$RUN_DIR/logs/bootstrap_launcher.log" 2>&1 < /dev/null &

echo "PID=$!"
echo "RUN_DIR=$RUN_DIR"
```

Outputs:

```text
checkpoints/round0_best.pth
checkpoints/round0_last.pth
round_0/training_metrics.csv
round_0/training_summary.json
teacher_sanity.json
```

## Round 1 formal collection

Run only after `teacher_sanity.json` reports `PASS`.

```bash
ROUND0="$RUN_DIR/checkpoints/round0_best.pth"

nohup python -u \
  training/Multi_IL_Full_Action_RL/stage3_actor_initialization/stage3b_collect_dagger.py \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --student-checkpoint "$ROUND0" \
  --mode collect \
  --round-id 1 \
  --beta 0.7 \
  --seed-start 10000 \
  --num-seeds 80 \
  --device npu:0 \
  --output "$RUN_DIR/datasets/round1_corrective.hdf5" \
  > "$RUN_DIR/logs/round1_collection.log" 2>&1 < /dev/null &
```

## Round 1 formal training

```bash
nohup python -u \
  training/Multi_IL_Full_Action_RL/stage3_actor_initialization/stage3b_train_dagger.py \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --round-id 1 \
  --input-checkpoint "$ROUND0" \
  --corrective-dataset "$RUN_DIR/datasets/round1_corrective.hdf5" \
  --device npu:0 \
  > "$RUN_DIR/logs/round1_training.log" 2>&1 < /dev/null &
```

Each later round must receive all corrective datasets accumulated through that
round. The runner enforces this. Final Actor selection compares Round 0 through
Round 3 using held-out environment performance, with imitation MSE only as the
last tie-breaker. Stage 3B never enters Stage 4.
