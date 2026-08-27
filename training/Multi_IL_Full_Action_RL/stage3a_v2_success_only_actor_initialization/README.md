# Stage 3A-v2 — Success-Only RNN Distillation

This is an independent single-variable comparison against Stage 3A-v1. It uses
the same SAC-compatible actor, initialization seed, optimizer, hyperparameters,
state ordering, and action convention, but excludes failed BC-RNN episodes from
both training and held-out validation datasets.

The historical Stage 3A-v1 run is read-only and is never loaded as an actor
initialization. Stage 3A-v2 always creates a fresh randomly initialized student.

## Smoke test

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic

python -u \
  training/Multi_IL_Full_Action_RL/stage3a_v2_success_only_actor_initialization/train_stage3a_v2_success_only.py \
  --config training/Multi_IL_Full_Action_RL/stage3a_v2_success_only_actor_initialization/stage3a_v2_success_only_config.json \
  --smoke-test \
  --device npu:0
```

## Formal training

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic

LOG_ROOT=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3a_v2_success_only_actor_initialization
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/stage3a_v2_success_only_$(date +%Y%m%d_%H%M%S).out"

nohup python -u \
  training/Multi_IL_Full_Action_RL/stage3a_v2_success_only_actor_initialization/train_stage3a_v2_success_only.py \
  --config training/Multi_IL_Full_Action_RL/stage3a_v2_success_only_actor_initialization/stage3a_v2_success_only_config.json \
  --device npu:0 \
  > "$LOG" 2>&1 < /dev/null &

echo "PID=$!"
echo "LOG=$LOG"
```

Training stops after checkpoint and summary generation. It does not launch
environment evaluation, SAC, a critic, Stage 4, or automatic actor selection.
