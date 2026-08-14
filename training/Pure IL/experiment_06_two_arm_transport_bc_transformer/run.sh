#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/data/home/3220251075/lerobot_workspace
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$WORKSPACE/miniconda3/etc/profile.d/conda.sh"
conda activate robosuite_npu

source "$WORKSPACE/Ascend/ascend-toolkit/set_env.sh"

# Keep existing CANN entries.
export PYTHONPATH="$WORKSPACE/robomimic:$WORKSPACE/robomimic/rlkit:${PYTHONPATH:-}"

# Default to NPU 3 so Exp6 can coexist with Exp5 if Exp5 is on NPU 2.
# Override with NPU_ID=0/1/2/3 when launching.
export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID:-3}"

cd "$EXPERIMENT_DIR"
exec python launch.py
