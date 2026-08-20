#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: NPU_ID=0 $0 NUM_EPISODES [SEED_START]"
    exit 2
fi

NUM_EPISODES="$1"
SEED_START="${2:-10000}"
WORKSPACE="/data/home/3220251075/lerobot_workspace"
PROJECT_ROOT="$WORKSPACE/robomimic/training/Multi_IL_Full_Action_RL"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
DATASET_ROOT="$PROJECT_ROOT/datasets/stage1_raw"
RUN_ROOT="$PROJECT_ROOT/runs/stage1_rollout_collection"

source "$WORKSPACE/miniconda3/etc/profile.d/conda.sh"
conda activate robosuite_npu

export PYTHONPATH="${PYTHONPATH:-}"
source "$WORKSPACE/Ascend/ascend-toolkit/set_env.sh"
export PYTHONPATH="$WORKSPACE/robomimic:$WORKSPACE/robomimic/rlkit:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID:-0}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$DATASET_ROOT" "$RUN_ROOT"
cd "$WORKSPACE/robomimic"

python -u "$PROJECT_ROOT/stage1_rollout_collection/collect_multi_il_rollouts.py" \
  --config "$PROJECT_ROOT/configs/stage1_three_policies.json" \
  --num-episodes "$NUM_EPISODES" \
  --seed-start "$SEED_START" \
  --run-id "$RUN_ID"

python -u "$PROJECT_ROOT/stage1_rollout_collection/validate_dataset.py" \
  --dataset-root "$DATASET_ROOT/$RUN_ID"

echo "Formal dataset : $DATASET_ROOT/$RUN_ID"
echo "Run metadata   : $RUN_ROOT/$RUN_ID"
