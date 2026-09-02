#!/bin/bash
set -euo pipefail
ROOT="/data/home/3220251075/lerobot_workspace/robomimic";HERE="$ROOT/training/Multi_IL_Full_Action_RL/stage4_v2_rsac_online_finetuning";OLD="/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage4_v2_rsac_online_finetuning/20260902_163222"
if [[ -d "$OLD" ]];then bash "$HERE/stop_stage4_v2.sh" "$OLD";sleep 3;fi
if pgrep -af "$OLD";then echo "Residual old-run processes remain; refusing benchmark.";exit 4;else echo "Old run stopped; no residual worker references to $OLD";fi
bash "$HERE/run_parallelism_benchmark.sh"
LATEST="$(find /data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage4_v2_rsac_online_finetuning -maxdepth 1 -type d -name 'parallel_benchmark_*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
python - "$LATEST/benchmark_summary.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not r['sixteen_env_clearly_faster']:raise SystemExit('16 env/group was not >=10% faster; formal launch cancelled')
print(f"8 env: {r['eight_env']['transitions_per_second']:.3f} transitions/s")
print(f"16 env: {r['sixteen_env']['transitions_per_second']:.3f} transitions/s")
print(f"speedup: {r['speedup_ratio']:.3f}x")
PY
STAGE4_V2_BENCHMARK_PASS=1 bash "$HERE/run_stage4_v2_two_groups.sh" formal
