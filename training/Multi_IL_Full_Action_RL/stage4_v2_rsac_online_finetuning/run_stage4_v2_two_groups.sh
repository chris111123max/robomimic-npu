#!/bin/bash
set -euo pipefail
MODE="${1:-formal}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then echo "Usage: bash run_stage4_v2_two_groups.sh [smoke|formal]";exit 2;fi
if [[ "$MODE" == formal && "${STAGE4_V2_BENCHMARK_PASS:-0}" != 1 ]];then echo "Refusing 16-env formal run: run run_optimized_stage4_v2.sh so the 8-vs-16 benchmark gates launch.";exit 3;fi
ROOT="/data/home/3220251075/lerobot_workspace/robomimic"
HERE="$ROOT/training/Multi_IL_Full_Action_RL/stage4_v2_rsac_online_finetuning"
V1="$ROOT/training/Multi_IL_Full_Action_RL/stage4_rsac_online_finetuning"
CONFIG="$HERE/stage4_v2_config.json"
OUT="/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage4_v2_rsac_online_finetuning"
COUNT="$(python - <<'PY'
import torch,torch_npu
print(torch.npu.device_count())
PY
)"
if (( COUNT >= 3 ));then RNN_DEVICE="npu:1";MULTI_DEVICE="npu:2";elif (( COUNT >= 2 ));then RNN_DEVICE="npu:0";MULTI_DEVICE="npu:1";else echo "Stage4-v2 requires two visible NPUs; found $COUNT";exit 1;fi
STAMP="$(date +%Y%m%d_%H%M%S)";[[ "$MODE" == smoke ]]&&STAMP="smoke_$STAMP";RUN="$OUT/$STAMP";mkdir -p "$RUN"
python "$HERE/allocate_cpu_affinity.py" --output "$RUN"
python - "$V1/stage4_config.json" "$CONFIG" "$RUN/config_diff.json" <<'PY'
import json,sys
a,b,out=map(lambda p:json.load(open(p)) if p!=sys.argv[3] else p,sys.argv[1:])
allowed={'stage','output_root','diagnostic_sequences','total_env_steps','actor_freeze_steps','actor_warmup_end','automatic_entropy_tuning','initial_entropy_alpha','target_entropy_active','evaluation_interval','checkpoint_steps','parallel_envs','evaluation_parallel_envs','performance_log_interval','resource_log_interval'}
diff={k:{'v1':a.get(k),'v2':b.get(k)} for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}
illegal=sorted(set(diff)-allowed)
if illegal:raise SystemExit(f'Illegal Stage4-v2 config differences: {illegal}')
json.dump({'allowed_difference_keys':sorted(allowed),'differences':diff},open(out,'w'),indent=2)
print(json.dumps(diff,indent=2))
PY
declare -A DEV=( [rnn_only_critic]="$RNN_DEVICE" [multi_il_critic]="$MULTI_DEVICE" )
for G in rnn_only_critic multi_il_critic;do python -u "$HERE/preflight_stage4_v2.py" --group "$G" --device "${DEV[$G]}" --config "$CONFIG" --output "$RUN/preflight_$G.json";done
python - "$RUN" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);x=[json.load(open(r/f'preflight_{g}.json')) for g in ('rnn_only_critic','multi_il_critic')]
assert len({v['actor_hash'] for v in x})==1 and len({v['critic_hash'] for v in x})==2
print('STAGE4-V2 PREFLIGHT PASSED: identical Actors, distinct Critics, fixed alpha=0.001')
PY
if [[ "$MODE" == smoke ]];then echo "STAGE4-V2 SMOKE COMPLETE; NO ROLLOUT STARTED";echo "Run directory: $RUN";exit 0;fi
declare -A PIDS
for G in rnn_only_critic multi_il_critic;do
  mkdir -p "$RUN/$G";[[ "$G" == rnn_only_critic ]]&&CPU_KEY=rnn_trainer_cpu_id||CPU_KEY=multi_trainer_cpu_id
  TRAINER_CPU="$(python -c "import json;print(json.load(open('$RUN/cpu_affinity.json'))['$CPU_KEY'])")"
  nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:True taskset -c "$TRAINER_CPU" python -u "$V1/train_stage4_group.py" --group "$G" --device "${DEV[$G]}" --config "$CONFIG" --run-dir "$RUN" --cpu-affinity-file "$RUN/cpu_affinity.json" >"$RUN/$G/train.log" 2>&1 </dev/null &PIDS[$G]=$!;echo "${PIDS[$G]}" >"$RUN/$G/pid"
done
python - "$RUN/pids.json" "${PIDS[rnn_only_critic]}" "${PIDS[multi_il_critic]}" "$RNN_DEVICE" "$MULTI_DEVICE" <<'PY'
import json,sys
p,r,m,rd,md=sys.argv[1:];json.dump({'rnn_only_critic':{'pid':int(r),'device':rd},'multi_il_critic':{'pid':int(m),'device':md}},open(p,'w'),indent=2)
PY
nohup python -u "$HERE/finalize_stage4_v2.py" --run-dir "$RUN" >"$RUN/finalizer.log" 2>&1 </dev/null &echo $! >"$RUN/finalizer.pid"
echo "Stage4-v2 run directory: $RUN";echo "RNN PID=${PIDS[rnn_only_critic]} device=$RNN_DEVICE log=$RUN/rnn_only_critic/train.log";echo "Multi PID=${PIDS[multi_il_critic]} device=$MULTI_DEVICE log=$RUN/multi_il_critic/train.log";echo "NO RANDOM GROUP RUN; budget=100000/group; parallel_envs=16/group; evaluation_parallel_envs=16"
