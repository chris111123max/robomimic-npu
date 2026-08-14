# TwoArmTransport Pure Online SAC（16 环境并行）

## 实验定义

这是 Pure RL SAC 基线。HDF5 数据集只用于读取 `TwoArmTransport` 的环境 metadata；其中的 demonstration transition 不进入 Replay Buffer，也不参与 Actor 或 Critic 更新。

- 观测：按配置顺序拼接 7 个 low-dim key，合计 59 维。
- 动作：14 维，SAC 输出范围 `[-1, 1]`。
- 网络：Actor 和 Twin-Q 均为两层 1024 单元 MLP。
- 优化：每新增一个达到 warmup 条件的 transition，执行一次 SAC gradient update。
- 设备：robosuite / MuJoCo worker 使用 CPU；Actor、Critic 和优化器使用 Ascend NPU。

## 并行结构

- 使用 `multiprocessing` 的 `spawn` 模式启动 16 个独立 robosuite worker。
- 主进程在 NPU 上一次批量计算最多 16 个动作。
- 16 个 CPU worker 同时执行环境 step，结果返回主进程后写入同一个 Replay Buffer。
- 训练和评估顺序使用同一组 16 个 worker，任何时刻都不会同时创建 32 个环境。
- 每个 worker 使用独立 seed；评估 transition 不进入 Replay Buffer。
- `OMP_NUM_THREADS` 等保持为 1，避免 16 个进程各自再创建大量数学库线程。

## 默认训练规模

- `parallel_envs = 16`
- `episodes_per_epoch = 64`，即每个 epoch 进行 4 批并行 episode。
- 每局最多 700 steps，成功时提前结束。
- `num_epochs = 40`
- 每 4 个 epoch 评估一次，每次并行评估 16 局；40 个 epoch 共评估 10 次，最后一次位于 Epoch 40。
- Replay Buffer 达到 1000 条 transition 后开始更新。

若所有 episode 都跑满，训练 transition 总量为：

```text
64 × 700 × 40 = 1,792,000
```

依据原单环境第一个 epoch 的实测采集和 NPU 更新时间，这套设置预计会超过原先的 10 小时预算。并行后的准确耗时应以第一个正式 epoch 的 `Time_Epoch` 为准。

## 启动

先进行真实的 16 环境冒烟测试。冒烟模式会让每个 worker 只运行 5 步：

```bash
cd "/data/home/3220251075/lerobot_workspace/robomimic/training/Pure RL/experiment_01_two_arm_transport_sac"
NPU_ID=0 bash run.sh --smoke-test
```

正式后台训练：

```bash
cd "/data/home/3220251075/lerobot_workspace/robomimic/training/Pure RL/experiment_01_two_arm_transport_sac"
nohup env NPU_ID=0 bash run.sh > formal_training.log 2>&1 < /dev/null &
echo $! > formal_training.pid
```

查看日志：

```bash
tail -f formal_training.log
```

正式输出目录：

```text
/data/home/3220251075/lerobot_workspace/training_runs/RL/two_arm_transport_sac/<timestamp>/
```

每个 epoch 保存 `last.pth`，成功率提高时保存 `best_success.pth`，并按配置保存 `model_epoch_XX.pth`。Checkpoint 不保存 Replay Buffer。
