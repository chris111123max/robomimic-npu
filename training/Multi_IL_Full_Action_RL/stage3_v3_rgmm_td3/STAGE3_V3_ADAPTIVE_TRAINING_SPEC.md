# Stage3-v3：自适应 BC + Q 尺度归一化训练说明

这份文档描述当前仓库中实际生效的 Stage3-v3 训练设置，供外部 GPT 或研究者审查。它不是训练结果报告，也不声称这些超参数已经在 TwoArmTransport 上证明最优。

## 1. 实验目标

在保留原始 robomimic BC-RNN-GMM Actor 网络结构的前提下，使用 Stage2 预训练的双 Q Critic 进行离线/在线混合 TD3-style 微调。此次版本相对此前固定 BC 衰减版本有三项主要变化：

1. `policy_delay` 改回 `2`：每 2 次 Critic 更新执行 1 次 Actor 更新。
2. 固定的 `bc_lambda_schedule` 被替换为基于固定种子评估成功率的自适应 BC 权重。
3. Actor 的 RL 项使用 TD3+BC 风格的批次 Q 尺度归一化，降低 Critic 数值尺度变化对 Actor 梯度的影响。

## 2. Actor 网络与权重迁移

Actor 直接从 BC checkpoint 加载，不做 MLP 蒸馏：

- checkpoint 类型：`RNNGMMActorNetwork`
- 观测维度：59
- 动作维度：14
- 循环结构：两层 LSTM，hidden size 400
- RNN horizon：10
- GMM mode 数量：5
- 每个 mode 输出 14 维均值和尺度，并输出 mode logits
- 标准差激活：`softplus`
- `min_std=1e-4`
- `use_tanh=false`
- `open_loop=false`
- Actor 参数量：2,078,945
- checkpoint state-dict 条目：20
- rollout 隐状态在 episode 开始和每 10 个动作后重置

准备脚本会严格加载 Actor，并记录：

- BC checkpoint SHA256
- Actor 参数 hash
- 观测/动作归一化统计量
- 原始 checkpoint 配置

训练启动时再次检查 Actor hash 和 Phase-0 等价性。两条分支使用同一个 Actor 初始权重；唯一的主要分支变量是 Stage2 Critic 初始化 checkpoint：

- `rnn_q`：Stage2 RNN-Q checkpoint
- `multi_q`：Stage2 Multi-Q checkpoint

## 3. Critic 与混合数据

Critic 是 Stage2 的 twin MLP Q 网络：

- 输入：59 维 observation + 14 维 action
- hidden dims：`[256, 256]`
- activation：ReLU
- LayerNorm：开启
- `gamma=0.99`
- `tau=0.005`
- Critic optimizer：AdamW
- Critic learning rate：`3e-4`
- Critic weight decay：`1e-4`
- Critic gradient clipping：`100.0`

每个 Critic 更新使用边界安全的 recurrent sequence：

- batch size：256
- 离线样本：128，比例 0.5
- 在线样本：128，比例 0.5
- UTD：1
- minimum online replay size：1000
- online transition capacity：1,000,000
- online sequence capacity：250,000
- Critic context length：11
- burn-in：10
- train sequence length：10
- burn-in 不反传梯度
- sequence 不跨 episode 边界

Critic 的 target 使用 target Actor 的 GMM component means，并对 twin Q 取最小值后按 GMM 概率加权：

```text
y = r + gamma * (1 - terminal)
    * sum_k p_target(k | history)
    * min(Q1_target(s_next, mu_k), Q2_target(s_next, mu_k))
```

当前版本明确关闭：

- online CQL
- SAC entropy
- SAC alpha tuning
- target policy smoothing
- Q filter
- AWAC
- handoff selector
- hybrid bootstrap
- expert RNN proposal cache
- recurrent Critic

## 4. Actor 更新目标

Actor 使用 GMM component means 计算 Q1 期望，且 Actor 目标只建立 Q1 计算图，不使用 Q2 的 Actor 梯度：

```text
actor_rl = - E_{(s, history)} [ sum_k p_k * Q1(s, mu_k) ]
```

BC 项只对离线 demonstration 样本计算 GMM negative log-likelihood：

```text
actor_bc = mean_offline( -log GMM(a_demo | history) )
```

最终 Actor loss 为：

```text
L_actor = alpha * actor_rl / stop_gradient(mean_abs_Q1_data)
           + lambda_bc * actor_bc
```

其中：

- `alpha=2.5`
- `mean_abs_Q1_data` 是当前 Actor batch 对应 replay/data actions 的 `abs(Q1)` 均值
- Q 尺度分母使用 `detach`，不会通过归一化项反向传播
- 分母最小值 `epsilon=1e-6`
- `lambda_bc` 不再根据 env step 线性衰减，而由自适应反馈状态提供

这样做的目的，是避免 Q 值绝对尺度变大时，Actor RL 梯度相对于 GMM NLL 被无意放大。它只改变 Actor loss 的尺度平衡，不改变 Critic 的 Bellman target。

## 5. 自适应 BC 权重

配置如下：

```json
{
  "initial_weight": 1.0,
  "min_weight": 0.0,
  "max_weight": 1.0,
  "target_success_rate": 0.65,
  "ema_rate": 0.5,
  "kp": 0.1,
  "kd": 0.5,
  "feedback_source": "fixed_seed_evaluation_success_rate"
}
```

反馈只在固定 evaluation seeds 的完整评估结束后更新，不在每个 transition 或每个 episode 上更新。评估成功率为 `R_current`，EMA 成功率为 `R_ema`。

第一次评估只初始化 EMA，不改变 BC 权重：

```text
R_ema = R_current
lambda_bc = 1.0
```

后续评估使用：

```text
R_ema_new = ema_rate * R_current
             + (1 - ema_rate) * R_ema_old

delta = kp * (R_ema_new - target_success_rate)
        + kd * max(0, R_ema_old - R_current)

lambda_bc_new = clip(lambda_bc_old + delta, min_weight, max_weight)
```

当前数值含义：

- 近期表现低于目标 `0.65` 时，比例项通常使 BC 权重下降，给 RL 更多改进空间。
- 当前评估明显低于此前 EMA 时，导数项增大 BC 权重，抑制突然漂移。
- 权重始终限制在 `[0, 1]`。
- 这不是永久冻结 Actor，也不是永久锁定在 BC-RNN；它允许逐步放松约束。

自适应状态会写入每个 checkpoint：

```json
{
  "weight": "当前 lambda_bc",
  "ema_success": "成功率 EMA",
  "last_success": "上次固定种子成功率",
  "feedback_count": "反馈更新次数"
}
```

恢复训练时会恢复该状态；如果 checkpoint 缺少该状态，训练会拒绝恢复，避免 BC 系数悄悄重置为 1。

## 6. Phase-0 与训练门控

Phase-0 包含：

- Actor 严格权重迁移等价性
- component means/scales/logits/probabilities 等价性
- LSTM hidden state 等价性
- 固定 20 个 evaluation episodes
- 至少 10/20 成功才允许开启 Actor 更新

Actor gate 设置：

- warmup：10,000 aggregate environment steps
- competence episodes：20
- minimum successes：10
- gate：latched，一旦打开不会因为后续单次评估下降而关闭

当前新实验可以复用一个兼容的、已通过的 Phase-0 pair。复用只复制 Phase-0 证据文件，不复制旧 Actor/Critic 训练状态：

- `transfer_validation.json`
- `step0_competence.json`
- `phase0_gate.json`

`policy_delay`、训练总步数等训练设置可以改变；但 BC checkpoint、数据集、Actor contract、evaluation seeds、horizon 和 gate contract 必须一致。

## 7. 训练规模与环境

- 总 aggregate environment steps：3,000,000
- 并行环境：16
- `rnn_q` 使用 `npu:0`
- `multi_q` 使用 `npu:1`
- multiprocessing start method：`spawn`
- 环境启动 stagger：开启
- 环境启动间隔：0.5 秒
- episode horizon：700
- 成功时终止：开启
- 训练 external action noise：0
- evaluation external action noise：0
- 环境 action bounds clipping：关闭

注意：3,000,000 是 16 个环境 aggregate 后的环境步数，不是每个环境各自 3,000,000 步。

## 8. 评估与保存点

固定 evaluation seeds：`20000` 到 `20019`，共 20 个。

评估步数：

```text
0, 10000, 25000, 50000, 100000, 150000,
200000, 300000, 500000, 1000000, 2000000, 3000000
```

checkpoint 步数与评估步数相同。每个分支会生成：

- `console.log`
- `train_metrics.jsonl`
- `episode_metrics.jsonl`
- `gate_metrics.jsonl`
- `throughput_metrics.jsonl`
- `stage_timing.jsonl`
- `evaluations/step_*.json`
- `diagnostics/gmm_step_*.json`
- `checkpoints/step_*.pth`
- `checkpoints/latest.pth`
- `checkpoints/last.pth`
- `checkpoints/best_success.pth`

重点监控指标：

- evaluation `success_rate`
- online episode success
- `lambda_bc`
- `bc_ema_success`
- `actor_rl_loss`
- `actor_rl_loss_normalized`
- `actor_q_data_abs_mean`
- `actor_bc_loss_raw`
- `actor_total_loss`
- `critic_loss_q1/q2`
- `q1_mean/q2_mean/qmin_mean`
- `component_mean_drift_mse`
- `logit_drift_mse`
- `parameter_drift_l2`

## 9. 当前实现的研究限制

这次实现是一个受论文启发的工程实验，不是某篇论文的完整复现：

1. 自适应 BC 的反馈使用 TwoArmTransport 的固定种子成功率；论文中的任务、回报归一化和控制器参数不同。
2. Q 归一化借鉴 TD3+BC 的思想，但这里的 BC 项是 recurrent GMM NLL，而不是原论文的确定性动作 MSE。
3. Critic 仍是 Stage2 初始化的 twin MLP，没有加入 Cal-QL、CQL 或 REDQ ensemble。
4. 当前没有把策略信赖域正则加入 Bellman target，因此不等价于完整 PROTO。
5. `policy_delay=2` 只是恢复较高 Actor 更新频率；它本身不能证明能消除后期退化。
6. 真实训练仍需比较多个随机种子，并同时看固定评估成功率、在线成功率、Q 尺度和 Actor 漂移，不能只看单个 checkpoint。

## 10. 给外部 GPT 的审查问题

请重点审查以下问题：

1. 当前自适应 BC 更新的符号是否适合“BC NLL 加到 Actor loss”这一实现？
2. 用 replay/data action 的 `mean(abs(Q1))` 做 RL 项归一化，是否会导致梯度尺度过小或过大？
3. 固定成功率目标 `0.65` 是否应改为 Phase-0 competence、历史最佳值或滑动置信区间？
4. 固定 20 个种子的评估频率是否足够稳定，是否应使用置信区间或连续多个评估点确认下降？
5. 是否需要 Critic 侧的保守估计、价值校准或策略信赖域，而不仅是 Actor 侧 BC 正则？
6. 对 recurrent GMM policy，是否应使用完整分布 KL，而不是只使用 GMM NLL？
7. 如何设计 `policy_delay=2` 与 `policy_delay=4` 的严格 matched-seed 消融实验？

## 11. 关键代码文件

- `stage3_v3_config.json`：本实验配置
- `stage3_v3_agent.py`：Critic 更新、Actor loss、自适应 BC、Q 归一化
- `train_stage3_v3_vector.py`：16 环境训练循环、评估、checkpoint 和恢复
- `prepare_stage3_v3_pair.py`：输入校验、权重迁移、Phase-0 复用
- `run_phase0_stage3_v3.py`：Phase-0 等价性与 competence gate
- `validate_stage3_v3.py`：无环境合成验证

