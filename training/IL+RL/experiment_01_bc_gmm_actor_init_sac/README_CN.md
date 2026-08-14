# 实验 01：BC-GMM Actor 初始化 SAC

本实验用于和 `training/Pure RL/experiment_01_two_arm_transport_sac` 做严格对比。唯一的实验变量是 SAC Actor 的初始化方式：Pure RL 使用随机初始化；本实验先把 Epoch 1850 的 BC-GMM Teacher 蒸馏到与 SAC 完全同构的 `TanhGaussianPolicy`，随后用该 Actor 初始化标准在线 SAC。

## 实验边界

- 环境：`TwoArmTransport`，low-dim PH v15，观测维度 59，动作维度 14。
- NPU：由 `NPU_ID` 选择，代码要求 Ascend NPU 可用。
- 环境并行：固定 16 个独立进程，使用已经验证过的 `forkserver`、逐个创建与 ready/step 超时逻辑。
- Stage 1 可以读取 HDF5 的 `train` / `valid` mask 和对应 observation；绝不读取 HDF5 中的 actions、rewards 或 dones。
- Stage 2 是纯在线 SAC：不加载 Teacher，不读取示范 transition，不加入 BC loss，也不把离线数据放入 replay buffer。
- 不修改 Pure SAC 基线目录。

## Stage 1：冻结 BC-GMM Teacher，蒸馏 SAC Actor

Teacher 必须是文件：

```text
model_epoch_1850_low_dim_v15_success_0.3.pth
```

默认路径写在 `config.json`。如果路径发生移动，启动器只会在工作区 `training_runs` 下接受唯一的同名文件；零个或多个匹配都会立即退出，避免误用 checkpoint。

Teacher 通过 robomimic 官方 `policy_from_checkpoint` / `RolloutPolicy` 加载并设为 eval、冻结参数。BC-GMM 的 `low_noise_eval=true` 只把各 Gaussian scale 压到很小，mixture component 仍由 categorical distribution 采样。因此本实验固定随机种子，只让 Teacher 对每个 train/valid observation 生成一次 target，然后保存为稳定缓存；后续 200 个 epoch 始终使用缓存 target，避免每个 epoch 重新采样标签。

Student 与 SAC Actor 完全同构：

```text
59 -> 1024 -> 1024 -> 14
TanhGaussianPolicy
```

训练损失是 Student 确定性输出 `tanh(mean)` 与缓存 Teacher action 的 MSE。batch size 100、Adam 1e-4、每 epoch 100 次更新，共 200 epochs。验证集固定取 1000 个 observation，即 10 个固定 batch。`log_std` 分支不参与 MSE，代码会在结束时断言其参数未被直接更新。

Teacher 只在训练开始时用固定的 16 个 seed 做一次 rollout benchmark。Student 每 20 epochs 使用完全相同的 16 个 seed 评估一次。`best_actor.pth` 的选择顺序为：

1. 环境成功率更高；
2. 成功率相同时 validation MSE 更低；
3. 前两项相同时 epoch 更早。

成功保持率定义为 `student_success / teacher_success`，目标值为 0.9；它只用于诊断，不会绕过上述 best checkpoint 规则。

## Stage 2：Actor 初始化后的纯在线 SAC

Stage 2 构建顺序如下：

1. 按 Pure SAC 基线正常随机创建 Actor、两个 Q、两个 target Q 和全新的优化器；
2. 只把 Stage 1 checkpoint 的 `actor` state dict 严格加载到 Actor；
3. 逐参数验证加载结果完全一致；
4. Q 网络保持随机，replay buffer 保持为空；
5. 在任何采样和梯度更新之前执行 Epoch 0 的确定性 16-episode 评估；
6. 进入与 Pure SAC 相同的在线训练循环。

正式配置固定为 16 environments、64 train episodes/epoch、40 epochs、每 4 epochs 评估 16 episodes。SAC 网络、学习率、gamma、tau、batch size、learning starts、updates per env step 和 replay buffer 大小均与当前 Pure SAC 配置一致。

## 输出结构

每次启动会创建独立时间戳目录：

```text
training_runs/IL+RL/experiment_01_bc_gmm_actor_init_sac/<timestamp>/
├── config.json
├── stage1_actor_pretraining/
│   ├── teacher_targets/
│   ├── logs/
│   ├── evaluation/
│   └── models/
│       ├── best_actor.pth
│       └── last_actor.pth
└── stage2_rl_training/
    ├── logs/
    ├── evaluation/epoch_000.json
    └── models/
```

Stage 2 checkpoint 会记录 `stage1_actor_checkpoint`，用于追溯实际加载的 Actor。Stage 2 单独启动时必须显式给出 `best_actor.pth`，不会自动用 `last_actor.pth` 替代。

## 与 Pure SAC 的公平性

| 项目 | Pure SAC | Experiment 01 |
|---|---:|---:|
| Actor architecture | 1024 × 1024 | 1024 × 1024 |
| Actor init | Random | BC-GMM distilled |
| Critic init | Random | Random |
| Replay init | Empty | Empty |
| Parallel envs | 16 | 16 |
| Episodes / epoch | 64 | 64 |
| Max steps | 700 | 700 |
| RL epochs | 40 | 40 |
| Batch | 100 | 100 |
| Policy LR | 1e-4 | 1e-4 |
| Q LR | 1e-4 | 1e-4 |
| Gamma | 0.99 | 0.99 |
| Tau | 0.005 | 0.005 |
| Learning starts | 1000 | 1000 |
| Update ratio | 1:1 | 1:1 |
| Eval interval | 4 epochs | 4 epochs |
| Eval episodes | 16 | 16 |
| Demo during RL | No | No |
| BC loss during RL | No | No |
| Teacher during RL | No | No |
| Expert replay | No | No |

除实验名、输出路径、额外的 Epoch 0 评估和 Actor initialization 外，Stage 2 的科学训练配置与 Pure SAC baseline 保持一致。比较曲线时应以 `Training_Env_Steps_Total` 为主要横轴，因为成功后 episode 可能提前终止。重点观察：初始策略是否更好、successful behavior 是否更早出现、相同在线环境步数下成功率是否更高；Epoch 40 最终成功率更高属于额外收益，不是本实验成立的唯一条件。

## 启动命令

先进入实验目录：

```bash
cd "/data/home/3220251075/lerobot_workspace/robomimic/training/IL+RL/experiment_01_bc_gmm_actor_init_sac"
```

完整冒烟测试会保留 16 个环境，但将每阶段缩短为 1 epoch、每个 rollout 最多 5 步：

```bash
NPU_ID=0 bash run.sh all --smoke-test
```

正式连续运行 Stage 1 和 Stage 2：

```bash
NPU_ID=0 bash run.sh all
```

分阶段运行：

```bash
NPU_ID=0 bash run.sh stage1
NPU_ID=0 bash run.sh stage2 "/绝对路径/stage1_actor_pretraining/models/best_actor.pth"
```

后台正式训练：

```bash
nohup env NPU_ID=0 bash run.sh all > ilrl_experiment_01.out 2>&1 &
tail -f ilrl_experiment_01.out
```

如需在 NPU 1 同时运行另一项实验，把 `NPU_ID=0` 改成 `NPU_ID=1`，并确保两个任务的 CPU / 环境进程预算不会相互挤占。

## 冒烟测试通过标准

- 成功加载文件名与 epoch 均正确的 BC-GMM Teacher；
- 只从 HDF5 observation 路径建立 train/valid state；
- Teacher target 为 `[N, 14]` 且缓存重读完全一致；
- 16 个 `forkserver` worker 全部 ready，没有 startup/step timeout；
- Stage 1 至少完成一次有限 MSE 更新、固定验证和 Teacher/Student rollout；
- 生成 `best_actor.pth`，且 Teacher 参数指纹不变、Student 参数发生变化；
- Stage 2 严格加载 Actor，Q 随机且 replay 为零；
- Epoch 0 评估发生在首次在线采样和首次梯度更新之前；
- Stage 2 至少完成一次在线采样、replay 写入和 SAC 更新并保存 checkpoint。
