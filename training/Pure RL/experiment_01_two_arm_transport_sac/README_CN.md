# TwoArmTransport Pure Online SAC

## 实验目的

这是当前项目的第一个 Pure RL 基线：随机初始化 Actor、随机初始化 Twin-Q、在线 Replay Buffer 和标准 SAC。HDF5 文件只用于读取 TwoArmTransport 的 environment metadata；其中的 observation、action、reward、done 和 demonstration transition 均不会进入 Replay Buffer，也不会用于更新 Actor 或 Critic。

## 环境与状态动作

- 环境：从 `datasets/transport/PH/low_dim_v15.hdf5` metadata 创建的 `TwoArmTransport`。
- observation：按固定顺序拼接 7 个 low-dimensional key，得到 `(59,)` float32，不使用图像、历史帧或数据集归一化。
- action：14 维。SAC policy 输出 `[-1, 1]`；启动时检查 robosuite 的真实 action bounds。预期 bounds 也是 `[-1, 1]`，因此直接传递；如果实际 bounds 不同，Adapter 会打印并采用严格线性映射。
- 成功立即结束，`terminal=1`；700 步上限属于 truncation，Replay Buffer 中保持 `terminal=0`。
- training env 与 evaluation env 独立创建。

## 网络与 SAC

- Actor：RLKit `TanhGaussianPolicy`，`59 -> 1024 -> 1024 -> mean/log_std(14)`。
- Q1/Q2/Target Q1/Target Q2：RLKit `FlattenMlp`，`73 -> 1024 -> 1024 -> 1`。
- Target Q 在训练前使用 `ptu.copy_model_params_from_to` 与对应 Q 完全同步。
- SAC：直接使用 RLKit `SACTrainer`，自动 entropy tuning，`gamma=0.99`，`tau=0.005`。
- Replay Buffer：直接使用 RLKit `EnvReplayBuffer`，容量 1,000,000。
- warmup 仍使用当前随机初始化 stochastic SAC Actor，不使用 BC 或 expert policy。

## 训练制度

- 1 epoch = 10 个完整 training episodes。
- 每局最多 700 steps。
- Replay Buffer 达到 1000 transitions 后开始更新。
- 开始更新后，每新增 1 个 training environment step 做 1 次 gradient update。
- 默认 30 epochs。
- 每个 epoch 后执行 10 个 deterministic evaluation episodes。
- evaluation transition 不进入 Replay Buffer，也不触发更新。

所有参数均位于 `config.json`。修改训练轮数只需修改：

```json
"training": {
    "num_epochs": 30
}
```

## 启动

服务器端短冒烟测试（写入 `/tmp`，不会污染正式训练目录）：

```bash
cd "/data/home/3220251075/lerobot_workspace/robomimic/training/Pure RL/experiment_01_two_arm_transport_sac"
NPU_ID=0 bash run.sh --smoke-test
```

正式训练：

```bash
cd "/data/home/3220251075/lerobot_workspace/robomimic/training/Pure RL/experiment_01_two_arm_transport_sac"
NPU_ID=0 bash run.sh
```

后台运行：

```bash
nohup env NPU_ID=0 bash run.sh > nohup_sac.log 2>&1 &
echo $!
```

## 输出

正式训练写入：

```text
/data/home/3220251075/lerobot_workspace/training_runs/RL/two_arm_transport_sac/<timestamp>/
├── config.json
├── logs/
│   ├── log.txt
│   ├── metrics.jsonl
│   └── tb/
└── models/
    ├── last.pth
    ├── best_success.pth
    └── model_epoch_XX.pth
```

checkpoint 保存网络、优化器、entropy coefficient 和训练计数，但默认不保存 Replay Buffer。因此它可以恢复网络权重与优化器状态，但不是 bitwise-exact 的 replay-buffer resume。
