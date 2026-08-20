#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/data/home/3220251075/lerobot_workspace"
PROJECT_ROOT="$WORKSPACE/robomimic/training/Multi_IL_Full_Action_RL"
SMOKE_ROOT="/tmp/multi_il_full_action_rl_stage1_smoke"
RUN_ID="smoke_$(date +%Y%m%d_%H%M%S)"

source "$WORKSPACE/miniconda3/etc/profile.d/conda.sh"
conda activate robosuite_npu

# CANN set_env.sh can read PYTHONPATH while nounset is active.
export PYTHONPATH="${PYTHONPATH:-}"
source "$WORKSPACE/Ascend/ascend-toolkit/set_env.sh"
export PYTHONPATH="$WORKSPACE/robomimic:$WORKSPACE/robomimic/rlkit:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID:-0}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$SMOKE_ROOT/datasets" "$SMOKE_ROOT/runs"
cd "$WORKSPACE/robomimic"

python -u "$PROJECT_ROOT/stage1_rollout_collection/collect_multi_il_rollouts.py" \
  --config "$PROJECT_ROOT/configs/stage1_three_policies.json" \
  --seed-list "$PROJECT_ROOT/configs/stage1_seeds.json" \
  --num-episodes 3 \
  --run-id "$RUN_ID" \
  --output-root "$SMOKE_ROOT/datasets" \
  --run-root "$SMOKE_ROOT/runs" \
  --smoke-test

python -u "$PROJECT_ROOT/stage1_rollout_collection/validate_dataset.py" \
  --dataset-root "$SMOKE_ROOT/datasets/$RUN_ID"

echo "Smoke dataset : $SMOKE_ROOT/datasets/$RUN_ID"
echo "Smoke metadata: $SMOKE_ROOT/runs/$RUN_ID"
