# 实验 03：TwoArmTransport 官方参数 BC-GMM 复现检查

## 目的

实验 01 已经训练了 BC-GMM，但使用了 `batch_size=4096`，并且训练期间关闭 rollout。
实验 03 不做新算法，而是将训练和评估方式恢复为 robomimic 论文 / paper config 风格，
用于判断当前 v1.5.1 数据与运行环境下，官方风格的 BC-GMM 是否能够得到非零成功率。

## 目录结构

```text
experiment_03_two_arm_transport_bc_gmm_official/
├── __init__.py
├── config.json
├── generate_config.py
├── train.py
└── README_CN.md
```

## 核心参数

```text
Task             = TwoArmTransport
Dataset          = Transport / PH / low_dim_v15.hdf5
Policy           = BC-GMM
GMM modes        = 5
Actor MLP        = [1024, 1024]
Learning rate    = 1e-4
Optimizer        = Adam

Batch size       = 100
Epochs           = 2000
Gradient steps   = 100 / epoch
Seed             = 1

Validation       = ON
Rollout          = ON
Rollouts/check   = 50
Evaluate every   = 50 epochs
Horizon          = 700
Save every       = 50 epochs
Best-success ckpt saving = ON
```

数据集路径：

```text
/data/home/3220251075/lerobot_workspace/datasets/transport/PH/low_dim_v15.hdf5
```

输出目录：

```text
/data/home/3220251075/lerobot_workspace/training_runs/two_arm_transport_bc_gmm_official_ph_low_dim
```

## 与实验 01 的关键区别

```text
                         实验01                 实验03
Policy                   BC-GMM                BC-GMM
GMM modes                5                     5
MLP                      1024x1024             1024x1024
LR                       1e-4                  1e-4
Epochs                   2000                  2000
Steps/epoch              100                   100

Batch                    4096                  100
Validation               OFF                   ON
Training rollout         OFF                   ON
Rollouts/check           0                     50
Evaluate every           -                     50 epochs
```

因此实验 03 的目的不是和实验 01 做“单变量算法对照”，而是做官方训练流程的 sanity check。

## 进入环境

```bash
source /data/home/3220251075/lerobot_workspace/miniconda3/etc/profile.d/conda.sh
conda activate robosuite_npu
```

## 重新生成 config

只有在需要重新生成配置时运行：

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic

python training/experiment_03_two_arm_transport_bc_gmm_official/generate_config.py
```

## 正式训练

建议使用 nohup：

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic

nohup python training/experiment_03_two_arm_transport_bc_gmm_official/train.py \
    > /data/home/3220251075/lerobot_workspace/bc_gmm_official_2000.log 2>&1 &
```

查看日志：

```bash
tail -f /data/home/3220251075/lerobot_workspace/bc_gmm_official_2000.log
```

查看 NPU：

```bash
npu-smi info
```

## 注意

本实验训练过程中会真实执行环境 rollout，因此会明显比实验 01 / 02 慢。
每 50 epoch 需要执行 50 个 rollout，这是为了能够看到训练过程中的成功率峰值，
而不是只测试 epoch 2000 的最终 checkpoint。
