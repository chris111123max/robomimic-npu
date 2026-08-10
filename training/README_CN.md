# TwoArmTransport 训练架构

本目录只面向官方 `TwoArmTransport`：

```text
experiment_01_two_arm_transport_bc
    robomimic 官方 BC-GMM（5-mode GMM policy）

experiment_02_two_arm_transport_bc
    与实验 01 参数一致的纯 BC（确定性 policy）
```

依赖仓库均与 `robosuite-master` 同级：

```text
E:\robosuit\robomimic
E:\robosuit\robosuite-master
```

两个实验均使用 Transport PH low-dimensional v1.5 数据集。配置由 robomimic
`generate_paper_configs.py` 中的官方 BC 配置函数生成；实验 02 只关闭实验 01 的 GMM policy，
其余训练参数保持一致，不在 training 层重新定义网络或损失。

本机 venv 只用于代码与 CPU smoke test。正式训练位于华为 910B 时，应在目标机器安装
与其 CANN 版本匹配的 PyTorch 和 `torch_npu`；不要安装 NVIDIA CUDA wheel。设备适配层会在
CUDA 不可用且 Ascend 可用时自动选择 `npu:0`。
