#!/bin/bash
set -euo pipefail

MODE="${1:-formal}"
SEED="${SEED:-20260901}"
ROOT="/data/home/3220251075/lerobot_workspace/robomimic"
OUT_ROOT="/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage4_rsac_online_finetuning"
SCRIPT="$ROOT/training/Multi_IL_Full_Action_RL/stage4_rsac_online_finetuning/train_stage4_group.py"

if [[ "$MODE" != "formal" && "$MODE" != "smoke" ]]; then
  echo "Usage: bash run_stage4_three_groups.sh [smoke|formal]"
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ "$MODE" == "smoke" ]]; then STAMP="smoke_$STAMP"; fi
RUN_DIR="$OUT_ROOT/$STAMP"
mkdir -p "$RUN_DIR"

ACTIVE_GROUPS=(rnn_only_critic multi_il_critic)
declare -A DEVICES=( [rnn_only_critic]="npu:1" [multi_il_critic]="npu:2" )
declare -A PIDS
SMOKE_ARG=()
if [[ "$MODE" == "smoke" ]]; then SMOKE_ARG=(--smoke-test); fi

echo "Running Stage4 preflight for RNN-only on NPU 1 and Multi-IL on NPU 2..."
for GROUP in "${ACTIVE_GROUPS[@]}"; do
  python -u "$ROOT/training/Multi_IL_Full_Action_RL/stage4_rsac_online_finetuning/preflight_stage4.py" \
    --group "$GROUP" --device "${DEVICES[$GROUP]}" --output "$RUN_DIR/preflight_$GROUP.json"
done
python - "$RUN_DIR" <<'PY'
import json,sys
import numpy as np
from pathlib import Path
run=Path(sys.argv[1]);groups=("rnn_only_critic","multi_il_critic")
items={g:json.loads((run/f"preflight_{g}.json").read_text()) for g in groups}
if len({v["actor_hash"] for v in items.values()}) != 1: raise SystemExit("Actor hash mismatch: launch forbidden")
zero=[np.asarray(v["deterministic_zero_action"]) for v in items.values()]
environment=[np.asarray(v["deterministic_seed10080_action"]) for v in items.values()]
if not all(np.allclose(zero[0],value,rtol=0.0,atol=1e-7) for value in zero[1:]): raise SystemExit("Deterministic Actor action mismatch: launch forbidden")
if not all(np.allclose(environment[0],value,rtol=0.0,atol=1e-7) for value in environment[1:]): raise SystemExit("Same-environment-state Actor action mismatch: launch forbidden")
if len({v["critic_hash"] for v in items.values()}) != 2: raise SystemExit("Critic initialization hashes are not distinct: launch forbidden")
print("PREFLIGHT PASSED: identical Actor hashes/actions; two distinct Critic initializations")
PY

for GROUP in "${ACTIVE_GROUPS[@]}"; do
  LOG="$RUN_DIR/$GROUP/train.log"
  mkdir -p "$RUN_DIR/$GROUP"
  nohup env PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
    python -u "$SCRIPT" \
    --group "$GROUP" --device "${DEVICES[$GROUP]}" --seed "$SEED" \
    --run-dir "$RUN_DIR" "${SMOKE_ARG[@]}" \
    > "$LOG" 2>&1 < /dev/null &
  PIDS[$GROUP]=$!
  echo "${PIDS[$GROUP]}" > "$RUN_DIR/$GROUP/pid"
done

python - "$RUN_DIR/pids.json" "${PIDS[rnn_only_critic]}" "${PIDS[multi_il_critic]}" <<'PY'
import json,sys
path,rnn_pid,multi_pid=sys.argv[1:]
with open(path,"w",encoding="utf-8") as stream:
    json.dump({"rnn_only_critic":int(rnn_pid),"multi_il_critic":int(multi_pid)},stream,indent=2)
PY

FINAL_LOG="$RUN_DIR/finalizer.log"
nohup python -u "$ROOT/training/Multi_IL_Full_Action_RL/stage4_rsac_online_finetuning/finalize_stage4.py" \
  --run-dir "$RUN_DIR" > "$FINAL_LOG" 2>&1 < /dev/null &
echo "$!" > "$RUN_DIR/finalizer.pid"

echo "Stage4 run directory: $RUN_DIR"
echo "RNN PID=${PIDS[rnn_only_critic]} device=npu:1 log=$RUN_DIR/rnn_only_critic/train.log"
echo "Multi PID=${PIDS[multi_il_critic]} device=npu:2 log=$RUN_DIR/multi_il_critic/train.log"
echo "Finalizer PID=$(cat "$RUN_DIR/finalizer.pid") log=$FINAL_LOG"
