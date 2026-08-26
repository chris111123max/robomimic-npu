# Stage 3 — SAC Actor Initialization

Stage 3 distills the selected Stage 1 BC-RNN behavior dataset into a standalone,
SAC-compatible stochastic actor. It does not use a critic, Q-value, RL update,
success filter, or Stage 2 output.

## Data and split

Only the `bc_rnn/transitions.hdf5` selected by the Stage 1.5 manifest is read.
Seeds `10000..10079` are training data and `10080..10099` are validation data.
The state vector is flattened in the exact `canonical_observation_keys` order
stored at the HDF5 root; Python dictionary iteration order is never used.

The target is the 14-D action actually executed by Stage 1. Those values are
already post-tanh environment actions with nominal bounds `[-1, 1]`. The Stage 1
checkpoint path can retain sub-`1e-3` floating-point overshoot at a bound; Stage 3
records that overshoot and preserves the target verbatim instead of clipping it.
The deterministic actor output is `tanh(mu)`, so training applies neither another
tanh nor action scaling.

## Actor API

The network is `59 -> 256 -> 256`, followed by separate 14-D `mu` and
`log_std` heads. It uses RLKit's `TanhGaussianPolicy`, including deterministic
actions, reparameterized stochastic actions, and summed log probabilities with
shape `[batch, 1]`. `log_std` is clamped to `[-20, 2]`, initialized to `-3`, and
frozen during action-MSE distillation. A later SAC stage may unfreeze it.

## Smoke test

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic

python -u \
  training/Multi_IL_Full_Action_RL/stage3_actor_initialization/train_stage3_actor.py \
  --config training/Multi_IL_Full_Action_RL/stage3_actor_initialization/stage3_config.json \
  --smoke-test \
  --device npu:0
```

This runs two epochs on five training seeds and two validation seeds, reloads
the best checkpoint, and validates deterministic action, stochastic action, and
log-probability shapes and finiteness.

## Formal training

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic

LOG_ROOT=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_actor_initialization
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/formal_$(date +%Y%m%d_%H%M%S).out"

nohup python -u \
  training/Multi_IL_Full_Action_RL/stage3_actor_initialization/train_stage3_actor.py \
  --config training/Multi_IL_Full_Action_RL/stage3_actor_initialization/stage3_config.json \
  --device npu:0 \
  > "$LOG" 2>&1 < /dev/null &

echo "PID=$!"
echo "LOG=$LOG"
```

Training saves `best_val_mse.pth`, `last.pth`, and candidates at epochs 10, 25,
50, 75, and 100. It then stops. It does not automatically run rollout
evaluation or select a shared actor.

## Manual candidate evaluation

```bash
python -u \
  training/Multi_IL_Full_Action_RL/stage3_actor_initialization/evaluate_stage3_actor.py \
  --checkpoint /path/to/checkpoints/candidate_epoch_050.pth \
  --seed-start 10000 \
  --num-seeds 100 \
  --deterministic \
  --device npu:0
```

The evaluator restores Stage 1 initial simulator states and recreates the
environment from the BC-RNN checkpoint metadata. Each result is written as
`evaluation_<checkpoint>.json` in the Stage 3 run directory. Candidate choice
is deliberately manual; Stage 3 creates no `shared_actor` alias.
