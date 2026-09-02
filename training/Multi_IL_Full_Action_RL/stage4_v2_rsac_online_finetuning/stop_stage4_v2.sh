#!/bin/bash
set -euo pipefail
RUN_DIR="${1:?Usage: bash stop_stage4_v2.sh RUN_DIR}"
for FILE in "$RUN_DIR/rnn_only_critic/pid" "$RUN_DIR/multi_il_critic/pid" "$RUN_DIR/finalizer.pid";do if [[ -f "$FILE" ]];then PID="$(cat "$FILE")";CMD="$(tr '\0' ' ' <"/proc/$PID/cmdline" 2>/dev/null||true)";if [[ "$CMD" == *"$RUN_DIR"* ]];then kill -TERM "$PID" 2>/dev/null||true;echo "TERM $PID";fi;fi;done
