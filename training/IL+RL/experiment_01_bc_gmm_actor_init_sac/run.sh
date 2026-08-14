#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data/home/3220251075/lerobot_workspace
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 {stage1|stage2|all} [best_actor.pth] [--smoke-test]"
    exit 2
fi

STAGE="$1"
shift
case "$STAGE" in
    stage1|all)
        EXTRA_ARGS=("$@")
        ;;
    stage2)
        if [ "$#" -lt 1 ]; then
            echo "ERROR: stage2 requires a Stage1 best_actor.pth path"
            exit 2
        fi
        ACTOR_CHECKPOINT="$1"
        shift
        EXTRA_ARGS=(--actor-checkpoint "$ACTOR_CHECKPOINT" "$@")
        ;;
    *)
        echo "ERROR: unknown stage '$STAGE' (expected stage1, stage2, or all)"
        exit 2
        ;;
esac

source "$WORKSPACE/miniconda3/etc/profile.d/conda.sh"
conda activate robosuite_npu
source "$WORKSPACE/Ascend/ascend-toolkit/set_env.sh"

export PYTHONPATH="$WORKSPACE/robomimic:$WORKSPACE/robomimic/rlkit:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

cd "$EXPERIMENT_DIR"
exec python -u launch.py --stage "$STAGE" "${EXTRA_ARGS[@]}"
