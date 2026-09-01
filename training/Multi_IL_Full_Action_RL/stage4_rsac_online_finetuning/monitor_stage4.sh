#!/bin/bash
set -euo pipefail
RUN_DIR="${1:?Usage: bash monitor_stage4.sh RUN_DIR}"
for GROUP in rnn_only_critic multi_il_critic; do
  echo "===== $GROUP ====="
  PID="$(cat "$RUN_DIR/$GROUP/pid" 2>/dev/null || true)"
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then echo "PID $PID RUNNING"; else echo "PID ${PID:-missing} NOT RUNNING"; fi
  if [[ -f "$RUN_DIR/$GROUP/training_metrics.csv" ]]; then tail -n 2 "$RUN_DIR/$GROUP/training_metrics.csv"; fi
  if [[ -f "$RUN_DIR/$GROUP/evaluation_metrics.csv" ]]; then echo "Latest evaluation:"; tail -n 1 "$RUN_DIR/$GROUP/evaluation_metrics.csv"; fi
  tail -n 3 "$RUN_DIR/$GROUP/train.log" 2>/dev/null || true
done
