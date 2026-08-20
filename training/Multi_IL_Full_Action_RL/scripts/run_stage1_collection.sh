#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: NPU_IDS=0,1,2,3 $0 NUM_EPISODES [SEED_START]"
    exit 2
fi

NUM_EPISODES="$1"
SEED_START="${2:-10000}"
WORKSPACE="/data/home/3220251075/lerobot_workspace"
PROJECT_ROOT="$WORKSPACE/robomimic/training/Multi_IL_Full_Action_RL"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
ARTIFACT_ROOT="$WORKSPACE/training_runs/Multi_IL_Full_Action_RL/stage1_rollout_collection"
DATASET_ROOT="$ARTIFACT_ROOT/datasets"
RUN_ROOT="$ARTIFACT_ROOT/runs"
NPU_IDS="${NPU_IDS:-0,1,2,3}"

source "$WORKSPACE/miniconda3/etc/profile.d/conda.sh"
conda activate robosuite_npu

export PYTHONPATH="${PYTHONPATH:-}"
source "$WORKSPACE/Ascend/ascend-toolkit/set_env.sh"
export PYTHONPATH="$WORKSPACE/robomimic:$WORKSPACE/robomimic/rlkit:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$DATASET_ROOT" "$RUN_ROOT"
cd "$WORKSPACE/robomimic"

echo "Parallel NPU IDs: $NPU_IDS"
npu-smi info

LAUNCHER_PID=""
stop_launcher() {
    if [ -n "$LAUNCHER_PID" ] && kill -0 "$LAUNCHER_PID" 2>/dev/null; then
        kill -TERM "$LAUNCHER_PID"
        wait "$LAUNCHER_PID" || true
    fi
}
trap stop_launcher TERM INT

python -u "$PROJECT_ROOT/stage1_rollout_collection/launch_parallel_collection.py" \
  --config "$PROJECT_ROOT/configs/stage1_three_policies.json" \
  --num-episodes "$NUM_EPISODES" \
  --seed-start "$SEED_START" \
  --npu-ids "$NPU_IDS" \
  --run-id "$RUN_ID" \
  --output-root "$DATASET_ROOT" \
  --run-root "$RUN_ROOT" &

LAUNCHER_PID=$!
wait "$LAUNCHER_PID"
LAUNCHER_PID=""
trap - TERM INT

python -u "$PROJECT_ROOT/stage1_rollout_collection/validate_dataset.py" \
  --dataset-root "$DATASET_ROOT/$RUN_ID"

echo "Formal dataset : $DATASET_ROOT/$RUN_ID"
echo "Run metadata   : $RUN_ROOT/$RUN_ID"
echo "Worker logs    : $RUN_ROOT/$RUN_ID/worker_shards/logs"
