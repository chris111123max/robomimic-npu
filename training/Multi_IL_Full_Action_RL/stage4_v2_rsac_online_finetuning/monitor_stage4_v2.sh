#!/bin/bash
set -euo pipefail
RUN_DIR="${1:?Usage: bash monitor_stage4_v2.sh RUN_DIR}"
for G in rnn_only_critic multi_il_critic;do echo "===== $G =====";tail -n 1 "$RUN_DIR/$G/train.log" 2>/dev/null||true;if [[ -f "$RUN_DIR/$G/evaluation_metrics.csv" ]];then echo "Latest evaluation:";tail -n 1 "$RUN_DIR/$G/evaluation_metrics.csv";fi;if [[ -f "$RUN_DIR/$G/training_metrics.csv" ]];then echo "Latest training metrics:";tail -n 1 "$RUN_DIR/$G/training_metrics.csv";fi;done
