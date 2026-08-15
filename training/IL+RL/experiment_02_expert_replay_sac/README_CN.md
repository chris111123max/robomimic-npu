# IL+RL Experiment 02：Expert Replay SAC

## 实验定义

本实验只研究 Expert Replay 对 Online SAC sample efficiency 的影响。Actor、Q1、Q2、Target Q、Alpha 和 Online Replay 的初始化均与 Pure SAC 相同；唯一主要变化是每次 SAC gradient update 的 batch 从100条 Online transition改成50条固定 Expert transition加50条 Online transition。

本实验没有 Actor pretraining、BC-GMM Teacher、BC loss、behavior cloning、guided exploration、Q-filter、residual RL、offline Actor optimization或 demonstration warmup。Expert action 只通过标准 `Q(s_expert, a_expert)` 参与 Critic 学习；Actor 始终使用原始 SAC 的 `alpha * log_pi - Q` 目标。

## 数据流

```text
low_dim_v15.hdf5 / mask/train
        ↓
(s, a, stored reward, stored next_obs, stored done)
        ↓
Fixed Expert Buffer（CPU、只读、83,979条）
        ↓ sample 50
                         Mixed Batch = 100
        ↑ sample 50              ↓
Online Replay Buffer        原始 SACTrainer
        ↑                         ↓
16个 TwoArmTransport       标准 Q / Actor / Alpha 更新
        ↑
随机初始化 SAC Actor
```

Expert Buffer 和 Online Replay Buffer 完全独立。Expert transition 不会写进 Online Replay，不计入 `Training_Env_Steps_Total`，也不计入 `learning_starts`。每个达到 warmup 条件的新 Online environment step 仍只产生一次 gradient update；Expert sampling 不额外增加 update 数量。

## 真实 HDF5 检查结果

开发使用的真实文件是仓库根目录 `low_dim_v15.hdf5`；正式服务器配置仍使用 `/data/home/3220251075/lerobot_workspace/datasets/transport/PH/low_dim_v15.hdf5`。

- Top-level groups：`data`、`mask`。
- Total demos：200。
- `mask/train`：180 demos，83,979 transitions。
- `mask/valid`：20 demos，9,773 transitions。
- Train 和 Valid 重叠：0；二者合计覆盖全部200个 demos。
- Train demo length：min 373，max 669，mean 466.55。
- Valid demo length：min 415，max 714，mean 488.65。
- HDF5 原始 action dtype：float64；Expert Buffer 加载为 NumPy float32。
- Train actions：shape `[83979,14]`，min -1.0，max 1.0，mean -0.01826645，std 0.43443727；没有 clip。
- Train rewards：shape `[83979]`，只含0和1，mean 0.01131235，std 0.10575624。
- Train dones：shape `[83979]`，只含0和1，sum 950，frequency 0.01131235。
- Train 中 reward 与 done 逐 transition 完全相等；全部180个 demo 的最后一个 done 都是1。
- Valid actions：shape `[9773,14]`，min -1.0，max 1.0，mean -0.02100602，std 0.43233596。
- Valid rewards/dones：只含0和1，done sum 107；Valid 不进入 Expert Buffer。
- 每个 demo 均原生存在 `next_obs`，因此不会用跨 demo 拼接或人工重建。
- 数据中不存在 NaN 或 Inf。

只按以下固定顺序拼接 observation，得到59维：

```text
robot0_eef_pos          3
robot0_eef_quat         4
robot0_gripper_qpos     2
robot1_eef_pos          3
robot1_eef_quat         4
robot1_gripper_qpos     2
object                 41
```

HDF5 还包含 joint、velocity 等 observation，但本实验明确不使用它们，也不按字母排序。

## Reward 与 terminal 语义

HDF5 `data.env_args` 明确记录：`TwoArmTransport`、robosuite 1.5.1、双 Panda、`single-arm-opposed`、20 Hz、`reward_shaping=false`。Pure SAC 本身就是从同一份 HDF5 metadata 创建 Online environment，因此 Expert 与 Online 使用同一个环境和 sparse reward 定义。

Expert reward 直接使用 HDF5 stored rewards，不重新计算。Expert terminal 直接使用 HDF5 stored dones，不忽略 dones，也不把所有 demo 边界强制改成 terminal。由于 sparse reward 与 done 逐元素完全相等，而 Pure SAC 使用 success 作为 Bellman terminal，这一规则与当前 Online terminal 语义一致。

## Fixed Expert Buffer

- 只加载 `mask/train` 的全部180个 demonstrations。
- Capacity 和 size 都严格等于83,979。
- CPU / NumPy 存储，不把完整数据常驻 NPU。
- 加载后 sealed，RL 阶段调用 `add_sample` 会立即报错。
- Validation split 不进入 replay。
- checkpoint 不复制整个 Expert Buffer，只保存数据集路径、split、数量和采样统计；恢复时可从固定 HDF5 重建。

## 50:50 Mixed Batch

每次 update 固定采样：

```text
Expert Buffer：50
Online Buffer：50
总 batch：100
```

对 `observations`、`actions`、`rewards`、`terminals`、`next_observations` 分别 concatenate，然后用同一个 permutation 统一 shuffle。所有字段都会执行 shape 和 finite 检查。sampler 真实累计 `Expert_Samples_Total` 与 `Online_Samples_Total`，并由计数器计算 `Actual_Expert_Sampling_Ratio`，不是直接回显 config。

## 与 Pure SAC 的公平性

| 项目 | Pure SAC | Experiment 02 |
|---|---:|---:|
| Actor Init | Random | Random |
| Critic Init | Random | Random |
| Online Buffer | Yes | Yes |
| Expert Buffer | No | Yes，Train only |
| Validation Expert Data | No | No |
| Batch Size | 100 | 100 |
| Online Samples/Batch | 100 | 50 |
| Expert Samples/Batch | 0 | 50 |
| Expert Ratio | 0 | 0.5 fixed |
| Parallel Envs | 16 | 16 |
| Episodes/Epoch | 64 | 64 |
| Max Steps | 700 | 700 |
| RL Epochs | 40 | 40 |
| Policy LR | 1e-4 | 1e-4 |
| Q LR | 1e-4 | 1e-4 |
| Gamma | 0.99 | 0.99 |
| Tau | 0.005 | 0.005 |
| Learning Starts | 1000 online | 1000 online |
| Update Ratio | 1:1 | 1:1 |
| Eval Every | 4 epochs | 4 epochs |
| Eval Episodes | 16 | 16 |
| Actor Pretraining | No | No |
| BC Loss | No | No |
| Teacher Exploration | No | No |

启动时会逐 section 比较当前 Pure SAC `config.json`。除实验名称/输出路径、Expert Replay 配置和专属日志外，若任一 Environment、Network、SAC、Training、Evaluation、Logging、Checkpoint 或 Device 配置漂移，程序会 fail fast。

## 输出与日志

正式输出目录：

```text
/data/home/3220251075/lerobot_workspace/training_runs/IL+RL/experiment_02_expert_replay_sac/<timestamp>/
├── config.json
├── logs/log.txt
├── models/
└── expert_buffer_info/hdf5_schema_and_statistics.json
```

保留 Pure SAC 全部训练、Q、Actor、Alpha、时间、env step 和 replay 指标，额外记录 Expert/Online buffer size、batch size、累计 sample 数、真实 ratio、Train demo 数和 Expert transition 数。checkpoint 保留 Pure SAC 全部网络与优化器状态，并加入 Expert Replay metadata。

主要比较横轴必须是 `Training_Env_Steps_Total`。重点观察是否更早出现成功、相同 Online steps 下 Eval Success 是否更高，以及 Q predictions、Q targets 和 QF loss 是否更快形成有意义的估计。

## 服务器冒烟测试

冒烟模式仍真实启动16个 `forkserver` 环境，但每个 episode 最多5步、只运行1个 epoch。它会完整加载真实83,979条 Expert Buffer，CPU 连续构造100个 mixed batches验证5000:5000及真实 ratio=0.5，验证 online size 999/1000 gate，然后完成真实 NPU SAC update。

```bash
cd "/data/home/3220251075/lerobot_workspace/robomimic/training/IL+RL/experiment_02_expert_replay_sac"
NPU_ID=2 bash run.sh --smoke-test
```

## 正式训练

```bash
cd "/data/home/3220251075/lerobot_workspace/robomimic/training/IL+RL/experiment_02_expert_replay_sac"

LOG="ilrl_experiment_02_npu2_$(date +%Y%m%d_%H%M%S).out"
nohup env NPU_ID=2 PYTHONPATH="${PYTHONPATH:-}" bash run.sh > "$LOG" 2>&1 < /dev/null &
echo "PID=$! LOG=$LOG"
```

这里使用逻辑 NPU 2（对应当前容器物理 NPU 6），避免与 NPU 0 的 Pure SAC和 NPU 1 的 Experiment 01 冲突。
