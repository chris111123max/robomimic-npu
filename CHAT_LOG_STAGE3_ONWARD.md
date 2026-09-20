# Stage3 方向聊天记录（重建版）

> 起点：暂停 Stage3C / Actor distillation，转向 **BC-RNN → Recurrent SAC（Stage3-R）**。
> 本文依据当前工作区可见聊天上下文整理，保留关键请求、命令、报错、修复和结论；不是平台内部消息数据库的逐字导出。

## 1. 从 Stage3C 转向 Stage3-R

### 用户请求

暂停 Stage3C / Actor distillation 方向，尝试 **BC-RNN → Recurrent SAC Actor**。目标是把 BC-RNN 的行为迁移到 recurrent SAC actor，使用现有项目和 Ascend NPU 环境。

### 结论

- Stage3-R 独立放在 `training/Multi_IL_Full_Action_RL/stage3_r_bc_rnn_to_rsac/`。
- 不使用旧 Stage3C checkpoint 作为训练起点。
- 使用现有 BC-RNN checkpoint 和项目内 recurrent SAC 结构。
- Stage3-R 先做 Actor 初始化/蒸馏；Stage2-R（recurrent critic）和 Stage4 不在 smoke 阶段自动启动。

## 2. pomdp-baselines 依赖

用户明确要求不要使用 Git submodule，把 `pomdp-baselines` 作为 `third_party` 源码依赖接入；GitHub 访问优先使用 SSH over port 443：

```bash
git clone git@github.com:twni2016/pomdp-baselines.git \
  third_party/pomdp-baselines
```

依赖按源码方式放入 Stage3-R 所需路径，没有使用 Git submodule。

## 3. Stage3-R 初次 smoke 错误

启动命令：

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
NPU_ID=0 bash \
  training/Multi_IL_Full_Action_RL/stage3_r_bc_rnn_to_rsac/run.sh smoke
```

首先遇到 pomdp-baselines 旧代码的 Python 3.10 兼容问题：

```text
SyntaxError: unknown encoding: future_fstrings
```

随后遇到：

```text
ImportError: cannot import name 'Set' from 'collections'
```

来源是旧式 `collections.Set` 导入。再次修复导入路径后遇到：

```text
ModuleNotFoundError: No module named 'component_targets'
```

原因是 Stage3-R 训练脚本没有把所需辅助模块目录加入导入路径。

## 4. Stage3-R smoke 成功

修复兼容和导入路径后，smoke 通过：

```text
epoch 001/2 A_head_only train_mse=0.17566891 val_mse=0.17656143
epoch 002/2 A_head_only train_mse=0.16479097 val_mse=0.16852753
STAGE3-R SMOKE PASSED; Stage2-R/Stage4 NOT started.
```

这证明 Stage3-R 的数据读取、模型构建、NPU 前向/反向和最小训练流程可以运行。

## 5. Stage3-R 正式训练参数问题

曾使用 `run.sh train` 启动正式训练，启动器返回：

```text
Usage: NPU_ID=0 bash run.sh [smoke|formal]
```

因此合法的正式参数是 `formal`，不是 `train`。

## 6. Stage3-R Actor 正式训练结果

正式训练运行到 300/300 epoch，包含两个阶段：

- `A_head_only`：冻结 recurrent backbone，仅训练动作头。
- `B_lstm_finetune`：进行 LSTM 相关部分微调。

末尾日志示例：

```text
epoch 290/300 B_lstm_finetune train_mse=0.01784884 val_mse=0.03044212
epoch 291/300 B_lstm_finetune train_mse=0.01783545 val_mse=0.03047143
...
epoch 300/300 B_lstm_finetune train_mse=0.01779515 val_mse=0.03067300
```

用户随后检查：

```bash
find "$RUN_DIR/candidate_evaluations" -maxdepth 3 -type f
find "$RUN_DIR/candidate_evaluations" \
  -maxdepth 1 -type f -name 'stage3_r_*.json' | wc -l
```

结果为 0，说明该次正式训练只完成了网络训练，没有自动生成候选环境 rollout 评估文件。要判断哪个 checkpoint 的环境表现最好，需要单独运行 candidate evaluation。

## 7. Stage3-R 评估和并行环境

用户提出评估阶段使用 8 个并行环境。需要区分：

- epoch 是网络优化轮数，不等同于环境 rollout 数量；
- candidate evaluation 是独立阶段；
- 没有 `candidate_evaluations/*.json` 时，不能仅凭训练 loss 声称环境评估完成；
- 是否真正使用 8 个环境取决于 evaluation runner 的实现，不能由训练日志单独推断。

## 8. Stage2-R：Recurrent SAC Critic Pretraining

在 Stage3-R Actor 初始化后开始实现独立的 Stage2-R recurrent critic 预训练，保持：

- Stage3-R epoch150 frozen Actor 不变；
- recurrent critic architecture 不变；
- dataset、gamma、tau、sequence length、validation protocol 不变；
- RNN-only 和 Multi-IL 定义不变；
- 每组目标为 50,000 updates；
- 使用 Ascend NPU。

日志曾显示：

```text
rnn_only_critic update 1000/50000 ...
rnn_only_critic update 6000/50000 ... ranking=1.0
...
rnn_only_critic update 17000/50000 ...
```

随后在 update 17380 失败：

```text
RuntimeError: Non-finite gradient norm at recurrent critic update 17380
```

此前没有明显 TD/Q 发散，例如：

```text
update 16000: val_td_mse=0.000374
update 17000: val_td_mse=0.000419
```

结论：旧 Stage2-R 没有完成，不能直接 resume 失败 checkpoint 来伪装成同一实验。

## 9. Stage2-R-v2 稳定性修正版

用户要求创建与旧 Stage2-R 完全分开的 **Stage2-R-v2**：

1. RNN-only 和 Multi-IL 从头训练。
2. critic learning rate 若为 `3e-4`，改成 `1e-4`；否则使用当前值除以 3，两组完全相同。
3. 全程使用 global gradient norm clipping：

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
grad_norm = torch.nn.utils.clip_grad_norm_(
    critic.parameters(),
    max_norm=1.0,
    norm_type=2.0,
    error_if_nonfinite=True,
    foreach=False,
)
optimizer.step()
```

4. 每个 validation interval 记录 pre-clipping gradient norm 的 mean、median、p95、p99、max 和 `fraction_grad_clipped`。
5. NaN/Inf gradient 不跳过 batch：保存 failure batch、critic/target/optimizer 状态和诊断 JSON，然后停止当前 group。
6. 增加 `debug_replay_nan_batch.py`，支持 CPU/NPU 及 `--detect-anomaly`，仅用于单个 failure batch。
7. 只有 RNN-only 与 Multi-IL 都完成 `50000/50000` 且没有 NaN/Inf，才算 Stage2-R-v2 完成。
8. Random baseline 使用相同 architecture/initialization protocol，但不训练。
9. 后续若继续 Stage4，必须继承 Stage2-R-v2 的 critic LR 和 `max_gradient_norm=1.0`。

## 10. Final Critic Comparison

Stage2-R-v2 完成后增加独立的 Final Critic Comparison，只统计：

- RNN-only recurrent critic；
- Multi-IL recurrent critic；
- 不重新训练；
- 两组使用相同 validation protocol 和固定比较指标。

## 11. Stage4 分支（后来放弃）

之后曾短暂尝试 Stage4 / Stage4-v2 RSAC online finetuning，包括：

- RNN-only 与 Multi-IL 两组；
- 固定 alpha；
- critic initialization washout audit；
- actor objective decomposition；
- 8/16 environment parallelism；
- persistent environment workers、CPU affinity 和 nohup 启动。

该分支后来被放弃，研究主线回到 Stage3（以及为 Stage3 服务的 Stage2-R critic 预训练）。Stage4 的 online finetuning 结果不应混入当前 Stage3 结论。

## 12. 当前主线

```text
BC-RNN checkpoint
        │
        ▼
Stage3-R：BC-RNN → Recurrent SAC Actor 初始化
        │
        ├── Actor checkpoint / candidate evaluation
        │
        ▼
Stage2-R-v2：Recurrent SAC Critic 预训练（RNN-only / Multi-IL）
        │
        ▼
Final Critic Comparison
```

当前实验纪律：

- 不把 Stage3-R 训练 loss 当成环境成功率；
- 没有 candidate evaluation 文件时，不声称 rollout 评估完成；
- Stage2-R 的 NaN 失败不能通过 resume 掩盖；
- Stage2-R-v2 是独立修正版，必须从头训练并记录 clipping 诊断；
- 放弃 Stage4 后，不把 Stage4 online finetuning 结果混入 Stage3 结论。

