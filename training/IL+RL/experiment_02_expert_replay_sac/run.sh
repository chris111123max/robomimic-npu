#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data/home/3220251075/lerobot_workspace
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$WORKSPACE/miniconda3/etc/profile.d/conda.sh"
conda activate robosuite_npu

# CANN's generated script may read unset variables in a clean nohup shell.
set +u
source "$WORKSPACE/Ascend/ascend-toolkit/set_env.sh"
set -u

export PYTHONPATH="$WORKSPACE/robomimic:$WORKSPACE/robomimic/rlkit:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

cd "$EXPERIMENT_DIR"
exec python -u launch.py "$@"
