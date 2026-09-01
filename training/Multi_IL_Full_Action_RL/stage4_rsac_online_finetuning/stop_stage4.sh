#!/bin/bash
set -euo pipefail
RUN_DIR="${1:?Usage: bash stop_stage4.sh RUN_DIR}"
SCRIPT_NAME="train_stage4_group.py"
for GROUP in rnn_only_critic multi_il_critic; do
  PID_FILE="$RUN_DIR/$GROUP/pid"
  [[ -f "$PID_FILE" ]] || continue
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" 2>/dev/null; then
    CMD="$(tr '\0' ' ' < "/proc/$PID/cmdline" 2>/dev/null || true)"
    if [[ "$CMD" == *"$SCRIPT_NAME"* && "$CMD" == *"$RUN_DIR"* ]]; then
      kill -TERM "$PID"
      echo "Stopped $GROUP PID=$PID"
    else
      echo "Refusing to stop PID=$PID: command does not belong to this Stage4 run" >&2
    fi
  fi
done
