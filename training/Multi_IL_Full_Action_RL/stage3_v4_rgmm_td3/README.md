# Stage3-v4：完整 GMM 采样期望 + recurrent Actor 加速

这是独立于 Stage3-v3 的实验目录；没有修改 Stage2、v3、环境、评估种子、Critic 架构或 50/50 replay 比例。BC-RNN-GMM Actor 从原 checkpoint 严格载入，Stage2 RNN-Q/Multi-Q 分别初始化相同结构的 twin MLP Critic（59+14 输入、[256,256]、ReLU、LayerNorm）。

## 训练目标和机制

- v3 Actor 近似：`-E[Σ_k p_k Q1(s, μ_k)]`；v4：`-E[Σ_k p_k mean_j Q1(s, μ_k + σ_k ε_kj)]`。默认每 mode 采样一次，`K=5, M=1`。GMM mean、std、logits 和 LSTM 均保留计算图。
- v3 target 近似：`r + γ(1-terminal) Σ_k p'_k min(Q1',Q2')(s',μ'_k)`；v4：`r + γ(1-terminal) Σ_k p'_k mean_j min(Q1',Q2')(s',μ'_k+σ'_k ε_kj)`。完整 target 计算在 `no_grad` 内。
- Actor 的 10 个时间步、5 个 mode、每 mode M 个采样合并成一次 batched Q1 forward；target 的 5×M 个动作合并成一次 twin-Q forward。环境执行动作仍是类别采样后再采 Gaussian，不是加权均值。
- 前 10,000 aggregate env steps Actor 不更新、Critic 按 UTD=1 更新。Phase-0 的转移等价性及 20 个固定种子的成功率门槛必须通过；10k 后 gate 锁存。`policy_delay=4`，Actor sequence batch=64，Critic batch=256（离线/在线各128）。BC 权重为0，无 adaptive BC、Q 尺度归一化、SAC alpha 或 online CQL。
- 现有 `BatchedGMMExecutor` 在 episode reset 和 episode timestep 10 的倍数处将 hidden 清零；replay 为每个 episode 保存从0连续增长的 `episode_steps`，且 episode 单独存放。因此 Actor 训练只抽取同一 episode 的 `[0..9]`、`[10..19]` 等完整窗口，`burn_in=0`。Critic 的 11-step context 与 v3 一致。
- Target Actor 是独立深拷贝，冻结参数，在 gate 打开后以 `tau=0.005` Polyak 更新。v3 在 `eval()` 模式下会把 target GMM std 固定为 `1e-4`；v4 仅在 target 副本上关闭 `low_noise_eval`，令 Bellman target 使用其学习到的 std，仍保持 eval 模式与原 soft-update 机制。

重要限制：v3 原有环境执行器在 `actor.eval()` 且 `low_noise_eval=True` 下将环境动作的 Gaussian std 固定为 `1e-4`。v4 按任务要求保留原环境采样行为；Actor RL objective 使用训练模式的 learned std。所以 v4 修复了 Actor/target 的分量均值近似，却**尚未实现与真实环境执行方差严格一致**。这在 `runtime_audit.json` 明确记录，不能将本版本结论解释为完全消除了 rollout/objective 的分布差异。若未来要让环境执行 learned std，须作为独立消融明确改变 rollout 协议。

终止语义保留 v3：成功（配置启用）或非 horizon 的 `raw_done` 作为 terminal，horizon 截断不作为 terminal；Bellman mask 是 `1-terminal`。不在本实验中变更。

## 验证与产物

先运行不依赖 checkpoint 的数学检查，再运行 checkpoint-backed 验证。后者输出真实 GMM head/LSTM 梯度范数、step0 严格迁移、目标 Actor std、采样/均值对照、时间和 mode 向量化等检查。验证脚本只做合成 batch，不启动 MuJoCo。`run_phase0_stage3_v4.py` 验证真实 checkpoint 转移，并在固定种子 20000–20019 上执行20轮 competence 评估；也可从严格兼容的 v3/v4 已完成 pair 复用 Phase-0，避免重复等待。正式训练之前仍建议在新 pair 上跑 `--smoke --num-envs 2 --total-env-steps 12000`，覆盖 gate 两侧并检查 checkpoint/replay round-trip；正式 pair 必须另建。

训练输出：`rnn_q/` 与 `multi_q/` 各有 `console.log`、`train_metrics.jsonl`、`episode_metrics.jsonl`、`evaluations/`、`diagnostics/gmm_step_*.json`、`stage_timing.jsonl`、`throughput_metrics.jsonl`、`checkpoints/`、`summary.json`。GMM 诊断包含 std 全局及每 mode 的均值。`benchmark_stage3_v4.py` 按 gate 前后汇总 aggregate env steps/s、Critic updates/s 和 Actor updates/s。
若有可比 v3 的 `throughput_metrics.jsonl`，可向 benchmark 加 `--v3-throughput-jsonl <路径>`，得到 gate 后实际速度比；未测量前不预设加速倍数。

## 服务器命令（两张 NPU）

以下在仓库根目录、`robosuite_npu` 环境运行。先根据服务器实际路径确认三个 checkpoint 和 Phase-0 参考 pair。参考 pair 可用已有 `stage3v3_reuse_20260912_180309`，但准备脚本会核验 SHA256、Actor hash、数据集、固定种子、成功率和 Phase-0 文件；不兼容时会拒绝复用。

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
conda activate robosuite_npu
SCRIPT_DIR=training/Multi_IL_Full_Action_RL/stage3_v4_rgmm_td3
OUTPUT_ROOT=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v4_rgmm_td3
STAGE2_RUN=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage2_new_critic_pretraining/stage2new_formal_001
BC_RNN_CHECKPOINT="/data/home/3220251075/lerobot_workspace/training_runs/Pure IL/two_arm_transport_bc_rnn_official_ph_low_dim/two_arm_transport_bc_rnn_official_ph_low_dim/20260811151701/models/model_epoch_1000_low_dim_v15_success_0.9.pth"
RNN_Q_CHECKPOINT="$STAGE2_RUN/rnn_q/checkpoints/best.pth"
MULTI_Q_CHECKPOINT="$STAGE2_RUN/multi_q/checkpoints/best.pth"
REFERENCE_PHASE0_PAIR=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v3_rgmm_td3/stage3v3_reuse_20260912_180309

python "$SCRIPT_DIR/validate_stage3_v4_math.py"
python "$SCRIPT_DIR/validate_stage3_v4.py" --bc-rnn-checkpoint "$BC_RNN_CHECKPOINT" --device npu:0

SMOKE_ID=stage3v4_smoke_$(date +%Y%m%d_%H%M%S)
SMOKE_RUN_DIR="$OUTPUT_ROOT/$SMOKE_ID"
python "$SCRIPT_DIR/prepare_stage3_v4_pair.py" --run-id "$SMOKE_ID" --bc-rnn-checkpoint "$BC_RNN_CHECKPOINT" --rnn-q-checkpoint "$RNN_Q_CHECKPOINT" --multi-q-checkpoint "$MULTI_Q_CHECKPOINT" --reuse-phase0-from "$REFERENCE_PHASE0_PAIR"
python -u "$SCRIPT_DIR/train_stage3_v4_vector.py" --group multi_q --device npu:0 --pair-run-dir "$SMOKE_RUN_DIR" --critic-init-checkpoint "$MULTI_Q_CHECKPOINT" --num-envs 2 --total-env-steps 12000 --smoke
cat "$SMOKE_RUN_DIR/multi_q/smoke_validation.json"

PAIR_ID=stage3v4_pair_$(date +%Y%m%d_%H%M%S)
PAIR_RUN_DIR="$OUTPUT_ROOT/$PAIR_ID"
python "$SCRIPT_DIR/prepare_stage3_v4_pair.py" --run-id "$PAIR_ID" --bc-rnn-checkpoint "$BC_RNN_CHECKPOINT" --rnn-q-checkpoint "$RNN_Q_CHECKPOINT" --multi-q-checkpoint "$MULTI_Q_CHECKPOINT" --reuse-phase0-from "$REFERENCE_PHASE0_PAIR"

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
nohup python -u "$SCRIPT_DIR/train_stage3_v4_vector.py" --group rnn_q --device npu:0 --pair-run-dir "$PAIR_RUN_DIR" --critic-init-checkpoint "$RNN_Q_CHECKPOINT" --num-envs 16 --total-env-steps 3000000 > "$PAIR_RUN_DIR/rnn_q/console.log" 2>&1 &
echo $! > "$PAIR_RUN_DIR/rnn_q/train.pid"
nohup python -u "$SCRIPT_DIR/train_stage3_v4_vector.py" --group multi_q --device npu:1 --pair-run-dir "$PAIR_RUN_DIR" --critic-init-checkpoint "$MULTI_Q_CHECKPOINT" --num-envs 16 --total-env-steps 3000000 > "$PAIR_RUN_DIR/multi_q/console.log" 2>&1 &
echo $! > "$PAIR_RUN_DIR/multi_q/train.pid"
```

若不复用 Phase-0，准备时去掉 `--reuse-phase0-from`，随后执行 `python "$SCRIPT_DIR/run_phase0_stage3_v4.py" --pair-run-dir "$PAIR_RUN_DIR" --device npu:0`，成功后再启动训练。两分支各自独立使用 NPU:0 / NPU:1，不共享优化器或 replay。

```bash
tail -f "$PAIR_RUN_DIR/multi_q/console.log"
python "$SCRIPT_DIR/benchmark_stage3_v4.py" --pair-run-dir "$PAIR_RUN_DIR" --group multi_q
```
