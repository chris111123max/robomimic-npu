#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data/home/3220251075/lerobot_workspace
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$WORKSPACE/miniconda3/etc/profile.d/conda.sh"
conda activate robosuite_npu

source "$WORKSPACE/Ascend/ascend-toolkit/set_env.sh"

export PYTHONPATH="$WORKSPACE/robomimic:$WORKSPACE/robomimic/rlkit:${PYTHONPATH:-}"

# 默认 NPU 2；运行前可通过 NPU_ID=0/1/2/3 修改
export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID:-2}"

cd "$EXPERIMENT_DIR"
exec python launch.py
