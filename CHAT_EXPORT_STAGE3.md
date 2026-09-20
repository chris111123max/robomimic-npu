# Stage3 强化学习项目聊天记录整理版

> 说明：这是当前会话的 Markdown 整理导出，按项目推进顺序汇总了关键问题、命令、日志结论和实验设置；不是聊天界面的逐字原始转储。

## 1. 项目与环境

- 项目：`robomimic + robosuite + NPU`
- 环境：`robosuite_npu`
- 主要实验目录：

```text
/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/
```

- Stage2 Critic 预训练：

```text
stage2_new_critic_pretraining/stage2new_formal_001
```

- BC-RNN-GMM checkpoint：

```text
/data/home/3220251075/lerobot_workspace/training_runs/Pure IL/two_arm_transport_bc_rnn_official_ph_low_dim/two_arm_transport_bc_rnn_official_ph_low_dim/20260811151701/models/model_epoch_1000_low_dim_v15_success_0.9.pth
```

该 checkpoint 被确认是：

- `RNNGMMActorNetwork`
- action dim：14
- GMM mode 数：5
- RNN：2 层 LSTM，hidden dim 400
- RNN horizon：10
- 参数量：2,078,945
- observation dim：59

## 2. Stage2-new Critic 预训练

Stage2 分为两个 Critic 分支：

- `rnn_q`
- `multi_q`

两者均使用 twin Q、LayerNorm MLP Critic，并生成：

```text
rnn_q/checkpoints/best.pth
multi_q/checkpoints/best.pth
```

Stage2 验证结果曾输出 balanced aggregate、BC-RNN、BC-Transformer 和 BC-GMM 的 Q 误差、ROC-AUC、Q gap 等指标。

## 3. Stage3-new SAC / CQL-lite 阶段

早期 Stage3-new 使用 Standard SAC 风格训练，随后加入了：

- CQL-lite 随机动作 Q 惩罚
- Frozen Stage2 value geometry anchor
- RNN-Q 与 Multi-Q 双分支并行训练
- 16 环境并行 rollout
- progressive handoff

典型问题包括：

- `ObsUtils.OBS_KEYS_TO_MODALITIES` 为 `None`
- `EnvRobosuite` 没有 `action_spec`
- prepare 阶段缓存生成过慢
- 目录删除后 `config_resolved.json` 不存在
- 两个 NPU 的 nohup 路径、PID 和日志目录混乱

标准检查方法：

```bash
ps -fp <PID>
tail -n 50 "$PAIR_RUN_DIR/rnn_q/console.log"
tail -n 50 "$PAIR_RUN_DIR/multi_q/console.log"
npu-smi info
```

训练日志一般位于：

```text
$PAIR_RUN_DIR/rnn_q/console.log
$PAIR_RUN_DIR/multi_q/console.log
$PAIR_RUN_DIR/rnn_q/train_metrics.jsonl
$PAIR_RUN_DIR/multi_q/train_metrics.jsonl
```

## 4. 16 环境并行训练

Stage3-vector 的配置曾设置：

```json
{
  "num_envs": 16,
  "multiprocessing_start_method": "spawn",
  "env_startup_stagger": true
}
```

训练步数的语义是 aggregate environment steps。16 个环境完成一轮时，aggregate env steps 增加约 16。

曾使用的启动方式：

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

nohup python -u train_stage3_v3_vector.py \
  --group multi_q \
  --device npu:1 \
  --pair-run-dir "$PAIR_RUN_DIR" \
  --critic-init-checkpoint "$MULTI_Q_CHECKPOINT" \
  --num-envs 16 \
  --total-env-steps 3000000 \
  > "$PAIR_RUN_DIR/multi_q/console.log" 2>&1 &
```

RNN-Q 使用 `--group rnn_q --device npu:0`。

## 5. 速度诊断

`stage_timing.jsonl` 显示主要耗时通常来自：

- MuJoCo / robosuite 环境步进
- Critic update
- Actor update
- NPU 同步和 Python multiprocessing 通信

典型计时：

```text
vector_env_step_ms: 约 90～103 ms
critic_update_ms: 约 55～70 ms
actor_update_ms: 约 250 ms
```

因此单纯增加 NPU 数量不能完全解决问题，环境仿真和 Actor/Critic 更新都需要优化。

曾讨论过的优化方向：

- 批量准备 replay minibatch
- 减少 `.item()` 和 CPU/NPU 往返
- CPU 核心绑定
- 批量收发环境结果
- 共享内存传输
- TorchAir 图编译

TorchAir 验证最初因 `pkg_resources` 缺失失败，之后通过：

```bash
python -m pip install "setuptools<81"
```

恢复了 `pkg_resources`。

## 6. 旧 Multi-only 成功实验

旧实验目录：

```text
/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v3_rgmm_td3/stage3v3_multi_only_20260910_170025
```

旧实验的重要设置：

```text
policy_delay = 2
utd = 1
offline_fraction = 0.5
online_fraction = 0.5
critic_layer_norm = true
```

旧 Multi-only **确实有 BC 衰减**，不是 10K 后立即纯 RL：

```json
[
  {"step": 10000, "value": 1.0},
  {"step": 100000, "value": 1.0},
  {"step": 300000, "value": 0.2},
  {"step": 500000, "value": 0.0},
  {"step": 3000000, "value": 0.0}
]
```

也就是说：

- 0～10K：Actor 冻结
- 10K～100K：Actor 更新，但 BC 权重仍为 1.0
- 100K～300K：BC 逐渐衰减
- 300K～500K：BC 从 0.2 降到 0
- 500K 以后：纯 RL

旧日志中 25K 的典型数据：

```text
actor_rl_loss       ≈ -0.10
actor_bc_loss_raw   ≈ -41.67
lambda_bc           = 1.0
actor_total_loss    ≈ -41.77
```

BC 损失在标量目标中占约 99.76%，所以 25K 的 0.8 成功率主要由 BC-RNN 能力支撑，RL 只是小幅修正。

## 7. Stage3-v4 当前实验

当前实验目录：

```text
/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v4_rgmm_td3/stage3v4_fixed_shm_20260918_150328
```

当前 V4 的验证结果包含：

```text
no_bc = true
no_q_normalization = true
policy_delay = 1
```

因此当前 V4 的 Actor 训练逻辑是：

- BC-RNN 只用于初始化 Actor
- 0～10K：Actor 冻结
- 10K 以后：纯 RL Actor 更新
- 没有 BC loss
- 没有 BC 衰减 schedule

## 8. V4 的 0K/10K 对比

Multi-Q：

```text
0K success_rate  = 0.6
10K success_rate = 0.9
```

但 10K checkpoint 已显示：

```text
actor_updates = 1
actor hash 已变化
```

Multi-Q 10K 漂移：

```text
parameter_drift_l2              ≈ 0.01395
component_mean_drift_mse       ≈ 5.97e-06
logit_drift_mse                ≈ 0.00687
action_drift_mse               ≈ 3.56e-05
```

RNN-Q：

```text
0K success_rate  = 0.6
10K success_rate = 0.6
```

RNN-Q 10K 也执行了 1 次 Actor 更新，但漂移更小：

```text
parameter_drift_l2              ≈ 0.01392
component_mean_drift_mse       ≈ 4.18e-06
logit_drift_mse                ≈ 0.00126
action_drift_mse               ≈ 9.90e-06
```

所以 Multi-Q 在 10K 的成功率提升来自第一次 RL 更新的影响与随机 GMM 评估波动；RNN-Q 的第一次更新没有改善成功率。

## 9. V4 在 25K 后归零的原因

两个分支在 25K 后都归零，说明这是共享训练机制的问题，而不是某一个 Critic 单独失效。

Multi-Q 日志显示：

```text
17K 左右 parameter_drift_l2 ≈ 5.5
25K 左右 parameter_drift_l2 ≈ 8.0
lambda_bc = 0.0
policy_delay = 1
gmm_std_max 约 5～6
effectively_active_modes 约 1.1
```

可能的退化链条：

```text
纯 RL Actor 更新
→ Actor 离开 BC-RNN 动作分布
→ GMM mode 逐渐塌缩、方差增大
→ 产生分布外动作
→ Critic 对分布外动作的估计被 Actor 利用
→ 任务成功率下降至 0
```

这也解释了为什么冻结只能延迟问题，而不能从根本上解决问题。

## 10. PPO 讨论

考虑过用 Recurrent PPO 替代当前纯 Q Actor 更新。

PPO 的优势：

- clipped policy ratio 限制单次策略变化
- KL early stop 可以控制策略漂移
- GAE 提供相对稳定的优势估计
- 不直接依赖 Critic 对分布外动作的最大化

推荐的独立路线：

```text
Stage3-v5 = Recurrent GMM PPO + GAE + PPO clipping + KL early stop
```

典型设置：

```text
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.1～0.2
ppo_epochs = 3～5
target_kl = 0.01～0.03
max_grad_norm = 0.5～1.0
entropy_coef = 0～0.01
```

PPO 不能直接复用当前 TD3/SAC replay update，需要：

- on-policy rollout buffer
- old log probability
- recurrent hidden state
- GAE returns/advantages
- Value head
- recurrent minibatch update

## 11. 当前最重要结论

旧 Multi-only 的 25K 成功主要来自 BC-RNN，因为当时 BC 权重仍是 1.0；当前 V4 在 10K 后完全去除 BC，导致 Actor 纯 RL 漂移，约 25K 后两个分支都退化。

因此不能把当前 V4 的 10K 成功率直接等价为“RL 已经学会了任务”。后续应使用确定性评估、多次重复评估，并考虑恢复 BC/KL 约束或实现独立的 Recurrent PPO 路线。
