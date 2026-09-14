# Stage3-v3：Multi-only 旧实验 vs 当前自适应版本

本文档比较两个版本：

- 旧实验：stage3v3_multi_only_20260910_170025
- 当前版本：仓库中现行的 Stage3-v3 配置，加入自适应 BC 和 Q 尺度归一化

旧实验参数来自归档文件 stage3v3_pd2_diagnostics_full.tar.gz 内的 shared/config_resolved.json，不是根据日志猜测。

## 1. 总体区别

| 项目 | 旧 Multi-only | 当前版本 |
|---|---|---|
| 训练分支 | 只运行 multi_q | 配置支持 rnn_q 与 multi_q 配对；也可以只启动 multi_q |
| NPU 使用 | 旧实验只占用一个 NPU | 若双分支启动，rnn_q=npu:0、multi_q=npu:1；只跑 Multi 时仍可只用一个 NPU |
| Actor | 原始 robomimic RNNGMMActorNetwork | 完全相同，仍然是原始网络权重迁移，不蒸馏、不改结构 |
| Critic | Stage2 twin MLP Critic | 完全相同的 Stage2 twin MLP Critic |
| policy_delay | 2 | 2 |
| BC 权重 | 按 env steps 固定线性衰减 | 根据固定种子 evaluation success rate 自适应更新 |
| Q 尺度归一化 | 无 | 有，alpha=2.5，按 replay/data action 的 mean(abs(Q1)) 归一化 Actor RL 项 |
| Phase-0 | 已通过 | 可复用旧实验中兼容的 Phase-0 证据 |
| 评估种子 | 20000–20019，共 20 个 | 相同 |
| 并行环境配置 | 16 个环境的配置已写入旧 pair | 仍为 16 个环境 |
| 总训练步数 | 配置为 3,000,000 | 配置为 3,000,000 |

旧 Multi-only 与当前版本的核心网络、数据混合和 policy_delay 没有变化；主要算法变量是 BC 权重机制和 Actor RL 项的 Q 尺度归一化。当前版本如果只启动 multi_q，它与旧实验的比较仍然有效，但新旧 pair 目录必须分开。

## 2. 网络结构与 LayerNorm

### 2.1 Actor：原始 RNN-GMM

两次实验使用同一个 BC checkpoint 和同一个 Actor contract：

- 类：RNNGMMActorNetwork
- 输入 observation：59 维
- 动作：14 维
- 循环核心：2 层 LSTM
- LSTM hidden size：400
- horizon：10
- GMM mode：5
- 每个 mode：14 维均值 + 14 维尺度
- mode logits：5 维
- std_activation=softplus
- min_std=1e-4
- open_loop=false
- use_tanh=false
- 参数量：2,078,945

### 2.2 LayerNorm 结论

| 位置 | 旧实验 | 当前版本 | 说明 |
|---|---:|---:|---|
| Critic hidden blocks | 有 | 有 | critic_layer_norm=true；Stage2 build_critic 强制要求每个 hidden block 使用 LayerNorm |
| Actor 的 Stage3 新增 LayerNorm | 无 | 无 | Stage3 直接复制原始 Actor，没有额外插入 LayerNorm |
| LSTM 内部 LayerNorm | 未配置 | 未配置 | checkpoint contract 是普通 2-layer LSTM，没有 LayerNorm-LSTM 配置 |
| Observation LayerNorm | 无 | 无 | observation 是 59 维 low-dim；当前版本没有新增 observation LayerNorm |
| Action normalization | 有 | 有 | 这是 BC checkpoint 中的 action scale/offset，不是 LayerNorm |

当前训练的 Critic 确实有 LayerNorm；Actor 没有被 Stage3 额外加 LayerNorm。Actor 的稳定性主要依赖原始 recurrent GMM 结构、BC NLL、梯度裁剪和当前新增的 Q 尺度归一化。

## 3. 共同的 Critic 设置

两个版本的 Critic 设置一致：

    obs_dim               = 59
    action_dim            = 14
    hidden_dims           = [256, 256]
    activation             = ReLU
    critic_layer_norm      = true
    gamma                  = 0.99
    tau                    = 0.005
    critic_lr              = 3e-4
    weight_decay           = 1e-4
    critic_grad_clip       = 100.0

Critic optimizer 是 AdamW，target 是 twin Q target。Target 使用 target Actor 的 GMM component means，并按 mode probability 加权：

    y = r + gamma * (1-terminal)
        * sum_k p_target(k) * min(Q1_target(s_next, mu_k),
                                  Q2_target(s_next, mu_k))

两个版本都没有启用：

- online CQL
- SAC entropy / alpha tuning
- target policy smoothing
- AWAC
- Q filter
- recurrent Critic
- handoff selector
- expert RNN proposal cache

## 4. 共同的数据与更新设置

    batch_size                = 256
    offline_fraction           = 0.5
    online_fraction            = 0.5
    UTD                        = 1
    min_online_replay_size     = 1000
    online_replay_capacity     = 1,000,000
    online_sequence_capacity   = 250,000
    actor_lr                   = 1e-5
    actor_grad_clip            = 10.0
    burn_in                    = 10
    train_seq_len              = 10
    critic_context_length      = 11
    actor_sequence_batch       = 32
    burn_in_gradient           = false
    boundary_policy            = same_episode_only
    horizon                    = 700
    terminate_on_success       = true

policy_delay=2 在两个版本中都表示：每完成 2 次 Critic update，执行 1 次 Actor update；但 Actor 仍然必须等 10k warmup 和 Phase-0 competence gate 通过后才更新。

## 5. 旧 Multi-only 的 BC 机制

旧实验使用固定的 env-step schedule：

    [
      {"step": 10000, "value": 1.0},
      {"step": 100000, "value": 1.0},
      {"step": 300000, "value": 0.2},
      {"step": 500000, "value": 0.0},
      {"step": 3000000, "value": 0.0}
    ]

旧 Actor loss 为：

    L_old = - E[sum_k p_k * Q1(s, mu_k)]
            + lambda_bc(step) * GMM_NLL(a_demo | history)

因此在 500k 之后，BC NLL 不再约束 Actor；Actor 完全由 Critic 的 Q1 梯度驱动。

## 6. 当前版本新增的自适应 BC

当前版本移除了上述固定 step schedule，配置为：

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

评估成功率只在固定 20 个 evaluation episodes 完成后用于更新，不在每个 transition 上更新。

    R_ema_new = ema_rate * R_current
                + (1 - ema_rate) * R_ema_old

    delta = kp * (R_ema_new - target_success_rate)
            + kd * max(0, R_ema_old - R_current)

    lambda_bc_new = clip(lambda_bc_old + delta, 0, 1)

行为解释：

- 表现低于目标时，比例项会逐渐减小 BC 权重，让 RL 有改进空间。
- 当前表现相对历史 EMA 突然下降时，导数项会增加 BC 权重，抑制快速漂移。
- 权重不会超过 [0, 1]。
- 该状态会写入 checkpoint，并在 resume 时恢复。

## 7. 当前版本新增的 Q 尺度归一化

当前 Actor 的 RL 项改为：

    L_rl_raw = - E[sum_k p_k * Q1(s, mu_k)]

    q_scale = stop_gradient(mean(abs(Q1(s, a_replay))))

    L_rl_normalized = 2.5 * L_rl_raw / max(q_scale, 1e-6)

    L_current = L_rl_normalized + lambda_bc * GMM_NLL

a_replay 是当前 Actor batch 中的 replay/data action，不是 Actor 生成的 component mean。归一化分母停止反向传播，因此不会通过分母改变 Critic 或 Actor 的梯度图；它只用于平衡 RL 项和 GMM NLL 项的数值尺度。

该修改不改变 Critic Bellman target、Critic loss、Q 网络结构、Q target 的 twin minimum 或 offline/online 数据比例；它只改变 Actor update 的 RL 项尺度。

## 8. 旧 Multi-only 的实际评估结果

旧归档中的 multi_q/evaluations/ 显示：

| aggregate env steps | 成功数 / 20 | success rate |
|---:|---:|---:|
| 0 | 13/20 | 0.65 |
| 10,000 | 13/20 | 0.65 |
| 25,000 | 16/20 | 0.80 |
| 50,000 | 13/20 | 0.65 |
| 100,000 | 14/20 | 0.70 |

旧实验的 25k 峰值为 0.80，但之后已经出现波动。因此不能把 25k 的峰值直接视为稳定收敛，也不能只用一次 25k 评估证明某个参数一定优于另一个参数。

## 9. 旧版本与当前版本的因果对照

如果当前版本只启动 multi_q，最干净的比较是：

| 变量 | 旧 Multi-only | 当前 Multi-only |
|---|---|---|
| Actor checkpoint | 相同 | 相同 |
| Stage2 Multi-Q checkpoint | 相同 | 相同 |
| expert dataset | 相同 | 相同 |
| Actor architecture | 相同 | 相同 |
| Critic architecture | 相同 | 相同 |
| Critic LayerNorm | 相同，开启 | 相同，开启 |
| replay ratio | 50/50 | 50/50 |
| UTD | 1 | 1 |
| policy delay | 2 | 2 |
| BC | 固定 step 衰减到 0 | 评估反馈自适应 |
| Actor RL loss | 未归一化 Q | Q-scale normalized |

这不是一次单纯的 policy_delay 对照；当前版本同时改变了两个 Actor objective 机制。若要严格判断“自适应 BC”和“Q 归一化”各自的作用，至少需要三组同 seed 消融：

1. 旧 objective：固定 BC schedule + 无 Q 归一化。
2. 只启用自适应 BC：Q 归一化关闭。
3. 自适应 BC + Q 归一化：当前版本。

建议所有组使用同一个 training_seed、同一组环境 seed、同一组 evaluation seed 和同一个 Stage2 Multi-Q checkpoint。

## 10. 需要外部 GPT 重点审查的问题

1. Critic LayerNorm 已经在两个版本中同时开启，因此后期退化不能简单归因于旧版本没有 LayerNorm。
2. Actor 没有新增 LayerNorm；应判断 recurrent GMM Actor 的漂移是否主要来自 Q 梯度和 BC 约束消失，而不是归一化层缺失。
3. 当前自适应 BC 的符号是否适合把 NLL 作为正则项加入 loss。
4. target_success_rate=0.65 是否应由 Phase-0 competence 或历史最佳性能确定。
5. mean(abs(Q1_data)) 是否是 recurrent GMM Actor 的合适 Q scale，是否需要 robust median、EMA 或 twin-Q 统计。
6. 当前 Critic 没有保守 Q、Q ensemble 或 actor-critic alignment；如果 Q 在 OOD action 上高估，仅做 Actor loss 归一化可能不够。
7. 评估应报告多个 checkpoint 和多个随机种子，不应只比较 25k 单点成功率。

## 11. 关键文件与证据

- 旧实验配置：归档中的 shared/config_resolved.json
- 当前配置：stage3_v3_config.json
- Actor/Actor loss：stage3_v3_agent.py
- 训练循环与 checkpoint：train_stage3_v3_vector.py
- Stage2 Critic LayerNorm contract：stage2_new_critic_pretraining/critic_network.py
- 无环境验证：validate_stage3_v3.py
- 旧评估结果：旧 pair 的 multi_q/evaluations/step_*.json

