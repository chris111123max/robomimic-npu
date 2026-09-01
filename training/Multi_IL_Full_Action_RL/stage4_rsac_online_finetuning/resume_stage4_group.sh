#!/bin/bash
set -euo pipefail
GROUP="${1:?Usage: bash resume_stage4_group.sh GROUP DEVICE RUN_DIR CHECKPOINT}"
DEVICE="${2:?}"
RUN_DIR="${3:?}"
CHECKPOINT="${4:?}"
ROOT="/data/home/3220251075/lerobot_workspace/robomimic"
LOG="$RUN_DIR/$GROUP/resume_$(date +%Y%m%d_%H%M%S).log"
nohup env PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  python -u "$ROOT/training/Multi_IL_Full_Action_RL/stage4_rsac_online_finetuning/train_stage4_group.py" \
  --group "$GROUP" --device "$DEVICE" --run-dir "$RUN_DIR" --resume "$CHECKPOINT" \
  > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$RUN_DIR/$GROUP/pid"
python - "$RUN_DIR/pids.json" "$GROUP" "$PID" <<'PY'
import json,sys
path,group,pid=sys.argv[1:]
with open(path,encoding="utf-8") as stream:data=json.load(stream)
data[group]=int(pid)
with open(path,"w",encoding="utf-8") as stream:json.dump(data,stream,indent=2)
PY
FINAL_LOG="$RUN_DIR/finalizer_resume_$(date +%Y%m%d_%H%M%S).log"
nohup python -u "$ROOT/training/Multi_IL_Full_Action_RL/stage4_rsac_online_finetuning/finalize_stage4.py" \
  --run-dir "$RUN_DIR" > "$FINAL_LOG" 2>&1 < /dev/null &
echo "Resumed $GROUP PID=$PID log=$LOG"
echo "Finalizer PID=$! log=$FINAL_LOG"
