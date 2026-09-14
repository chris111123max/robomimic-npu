# Stage3-v4：在线动作分布审计后的 Case A

本目录只改 Stage3-v4，不改 Stage2、Stage3-v3、环境或固定评估协议。当前正式训练目标只有一种：**categorical component-mean Q**。旧的 learned-std sampled-GMM 目标已撤回，旧 v4 sampled 目标 checkpoint 因 `objective_revision` 不同不得静默续训。

## 真实调用链

| 环节 | 真实行为 |
| --- | --- |
| 训练入口 | `train_stage3_v4_vector.py` 创建 v3 `BatchedGMMExecutor`，`actions_for` 产生动作 |
| Actor mode | 执行器临时调用 `actor.eval()`，动作生成后恢复原先 mode |
| RNN hidden | 每个环境各有 hidden slot；episode reset 和 timestep 模 10 为 0 时清零 |
| GMM 分布 | `RNNGMMActorNetwork` 输出 mean、raw scale、logits；`low_noise_eval=True` 且 eval 时 Gaussian σ 被固定为 `1e-4`，而 train 时才用 `softplus(raw_scale)+min_std` |
| 抽样 | `MixtureSameFamily.sample()` 按 categorical 抽一个 mode，再从该分量的 Normal 抽动作；不是 argmax 或加权均值 |
| 动作后处理 | 用 checkpoint 的 action scale/offset 反归一化；本实验 external noise 为 0、clip 为 false |
| 环境与 replay | 同一个 `actions` 对象传给 vector `env.step`，也作为 `online.add(..., action, ...)` 的 action；worker 与 `EnvRobosuite.step` 原样转交动作 |
| 固定评估 | 同一执行器，因此也是 eval + fixed σ=`1e-4` |
| Actor RL update | `actor.train()`；learned σ 存在于分布，但 RL loss 只使用 mean/probs，std head 无直接 RL gradient |
| Target Actor | 独立、冻结、Polyak 更新；保持 eval + `low_noise_eval=True`；Bellman target 只枚举分量均值 |

数学上真实执行分布是 `π_env(a|h)=Σ_k p_k(h) N(a; μ_k(h), (10^-4)^2 I)`，此处 μ/σ 在 checkpoint 归一化动作空间，实际环境动作还须乘 action scale 并加 offset。它并非严格 Dirac 分布，因此分量均值 Q 是低噪声近似而非精确积分。训练目标为：

```text
L_actor = -E_{s,h}[Σ_k p_k(h) Q1(s, denorm(μ_k(h)))]
y = r + γ(1-terminal) Σ_k p'_k(h') min(Q1'(s', denorm(μ'_k)), Q2'(s', denorm(μ'_k)))
```

`component_mean_expected_Q` 与 `sampled_learned_std_expected_Q` 仅在详细指标采集时、`no_grad` 下做反事实对照；后者不能参与梯度或 TD target。std head 和原 BC checkpoint 保留，共享 RNN/encoder 的变化可间接改变其输出，但 std head 本身无直接 RL 梯度。

其余合同不变：Stage2 twin MLP + LayerNorm Critic，0–10k Actor 冻结/Critic UTD=1 更新，10k 且 Phase-0 gate 通过后 Actor 更新；Critic batch 256（离线/在线各 128），`policy_delay=4`，Actor recurrent batch 64，`train_seq_len=10`、`burn_in=0`。后两者成立的证据是执行器每 10 步清零 hidden，replay 的 `episode_steps` 从 0 连续且不跨 episode，`aligned_start` 只返回 0、10、20 等完整窗口并拒绝错位。无 adaptive BC、BC 权重为 0、无 Q-scale normalization、无 online CQL。

## 先审计，再做 12k 冒烟

下面命令在服务器仓库根目录及 `robosuite_npu` 环境中运行。不要直接启动 3M。`REFERENCE_PHASE0_PAIR` 只能指向已经通过且准备脚本认定兼容的 Phase-0 pair；若没有，就不使用复用参数，准备后运行 `run_phase0_stage3_v4.py`。

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
python "$SCRIPT_DIR/audit_stage3_v4_rollout.py" --bc-rnn-checkpoint "$BC_RNN_CHECKPOINT" --device npu:0 --samples 8
python "$SCRIPT_DIR/validate_stage3_v4.py" --bc-rnn-checkpoint "$BC_RNN_CHECKPOINT" --device npu:0
```

审计 JSON 应显示 `rollout_sigma_*≈0.0001`、`executor_reconstruction_max_abs_diff≤1e-5`，并输出实际动作与所选分量均值的差值（环境动作单位）。不应仅凭数学 validator PASS 就判定在线 contract 正确。

```bash
SMOKE_ID=stage3v4_casea_smoke_$(date +%Y%m%d_%H%M%S)
SMOKE_RUN_DIR="$OUTPUT_ROOT/$SMOKE_ID"
python "$SCRIPT_DIR/prepare_stage3_v4_pair.py" --run-id "$SMOKE_ID" --bc-rnn-checkpoint "$BC_RNN_CHECKPOINT" --rnn-q-checkpoint "$RNN_Q_CHECKPOINT" --multi-q-checkpoint "$MULTI_Q_CHECKPOINT" --reuse-phase0-from "$REFERENCE_PHASE0_PAIR"
test -f "$SMOKE_RUN_DIR/shared/config_resolved.json" && test -f "$SMOKE_RUN_DIR/shared/phase0_gate.json" && python -u "$SCRIPT_DIR/train_stage3_v4_vector.py" --group multi_q --device npu:0 --pair-run-dir "$SMOKE_RUN_DIR" --critic-init-checkpoint "$MULTI_Q_CHECKPOINT" --num-envs 2 --total-env-steps 12000 --smoke
cat "$SMOKE_RUN_DIR/multi_q/smoke_validation.json"
```

重点看 `multi_q/console.log`、`train_metrics.jsonl`、`runtime_audit.json`、`stage_timing.jsonl`、`smoke_validation.json`：0–10k Actor 更新数应为 0 且参数 hash 不变；10k 后 gate 打开、Actor 更新数增长、Critic 持续更新；梯度有限、std head 直接梯度为 0；无 NaN；`actor_expected_component_mean_q` 是训练值，`actor_q_sampled_learned_std_diagnostic` 只是对照；吞吐以 gate 前/后实际 steps/s 判断。通过后再决定是否准备新的正式 pair。
