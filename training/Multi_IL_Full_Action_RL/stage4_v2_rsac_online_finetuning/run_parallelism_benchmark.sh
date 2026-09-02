#!/bin/bash
set -euo pipefail
ROOT="/data/home/3220251075/lerobot_workspace/robomimic";HERE="$ROOT/training/Multi_IL_Full_Action_RL/stage4_v2_rsac_online_finetuning";OUT="/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage4_v2_rsac_online_finetuning/parallel_benchmark_$(date +%Y%m%d_%H%M%S)";mkdir -p "$OUT";python "$HERE/allocate_cpu_affinity.py" --output "$OUT"
for N in 8 16;do
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True python -u "$HERE/benchmark_parallelism.py" --workers "$N" --transitions 4000 --device npu:0 --config "$HERE/stage4_v2_config.json" --affinity "$OUT/cpu_affinity.json" --output "$OUT/${N}_env.json" >"$OUT/${N}_env.log" 2>&1 & BENCH_PID=$!
  (while kill -0 "$BENCH_PID" 2>/dev/null;do date --iso-8601=seconds;npu-smi info || true;sleep 5;done) >"$OUT/npu_smi_${N}_env.log" 2>&1 & MON_PID=$!
  wait "$BENCH_PID";wait "$MON_PID" || true;cat "$OUT/${N}_env.log"
done
python - "$OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);a=json.load(open(p/'8_env.json'));b=json.load(open(p/'16_env.json'));s=b['transitions_per_second']/a['transitions_per_second'];r={'eight_env':a,'sixteen_env':b,'speedup_ratio':s,'sixteen_env_clearly_faster':s>=1.10,'recommended_parallel_envs':16 if s>=1.10 else 8};(p/'benchmark_summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));print('BENCHMARK PASS: use 16 env/group' if s>=1.10 else 'BENCHMARK FAIL: do not start 16-env formal run')
PY
echo "Benchmark directory: $OUT"
