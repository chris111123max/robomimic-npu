# 实验 04：TwoArmTransport 官方参数 Vanilla BC

实验 04 与实验 03 平行，用于比较官方训练参数下的：

- Experiment 03: BC-GMM
- Experiment 04: Vanilla deterministic BC

核心训练和评估设置保持一致，唯一核心算法差异是 GMM 是否开启。

## 核心参数

```text
Task            = TwoArmTransport
Dataset         = Transport PH low_dim_v15.hdf5
Actor MLP       = [1024, 1024]
LR              = 1e-4
Optimizer       = Adam

Batch size      = 100
Epochs          = 2000
Steps / epoch   = 100
Seed            = 1

Validation      = ON
Rollout         = ON
Rollout n       = 10
Rollout rate    = every 100 epochs
Horizon         = 700
Training video  = OFF

GMM             = OFF
Gaussian        = OFF
RNN             = OFF
Transformer     = OFF
VAE             = OFF
```

因此实验 04 使用 robomimic 的 Vanilla deterministic BC。

## 与实验 03 的关系

```text
Experiment 03                       Experiment 04
BC-GMM                              Vanilla BC
batch=100                           batch=100
epochs=2000                         epochs=2000
steps/epoch=100                     steps/epoch=100
lr=1e-4                             lr=1e-4
seed=1                              seed=1
validation=ON                       validation=ON
10 rollouts / 100 epochs            10 rollouts / 100 epochs

GMM=True                            GMM=False
```

## NPU 并行运行

如果 Experiment 03 使用默认 NPU 0，可以将 Experiment 04 放到物理 NPU 1：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 \
nohup python training/experiment_04_two_arm_transport_bc_official/train.py \
    > /data/home/3220251075/lerobot_workspace/bc_official_seed1_npu1.log 2>&1 &
```

在这个进程内部，物理 NPU 1 会映射成 `npu:0`，与当前 robomimic 设备选择代码兼容。
