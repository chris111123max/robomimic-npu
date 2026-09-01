#!/bin/bash
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
MODE="${1:-formal}"
NPU_ID="${NPU_ID:-0}"

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/rlkit:${PYTHONPATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

ARGS=(--device "npu:${NPU_ID}")
if [ "$MODE" = "smoke" ]; then
    ARGS+=(--smoke-test)
elif [ "$MODE" != "formal" ]; then
    echo "Usage: NPU_ID=0 bash run.sh [smoke|formal]" >&2
    exit 2
fi
exec python -u "$HERE/train_stage2_r_v2.py" "${ARGS[@]}"
