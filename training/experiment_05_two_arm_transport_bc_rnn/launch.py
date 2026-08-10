#!/usr/bin/env python3
from pathlib import Path
import os
import subprocess
import sys

WORKSPACE = Path("/data/home/3220251075/lerobot_workspace")

# 修改后的版本：自动使用 launch.py 自己所在的目录作为实验目录
# 因此无论实验放在 robomimic/training 还是其他位置，都不再写死路径
EXPERIMENT_DIR = Path(__file__).resolve().parent

CONFIG_PATH = EXPERIMENT_DIR / "config.json"
DATASET_PATH = WORKSPACE / "datasets" / "transport" / "PH" / "low_dim_v15.hdf5"
ROBOMIMIC_TRAIN = WORKSPACE / "robomimic" / "robomimic" / "scripts" / "train.py"

# 启动前检查关键文件
for p, name in [
    (CONFIG_PATH, "config"),
    (DATASET_PATH, "dataset"),
    (ROBOMIMIC_TRAIN, "robomimic train.py"),
]:
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")

# 保留已有 CANN / Ascend PYTHONPATH，不要整体覆盖
env = os.environ.copy()

robomimic_root = str(WORKSPACE / "robomimic")
rlkit_root = str(WORKSPACE / "robomimic" / "rlkit")
old_pythonpath = env.get("PYTHONPATH", "")

env["PYTHONPATH"] = ":".join(
    x for x in [robomimic_root, rlkit_root, old_pythonpath] if x
)

cmd = [
    sys.executable,
    str(ROBOMIMIC_TRAIN),
    "--config",
    str(CONFIG_PATH),
]

print("=" * 84)
print("Experiment 05: Official BC-RNN-GMM | TwoArmTransport PH low_dim")
print("Python :", sys.executable)
print("ExpDir :", EXPERIMENT_DIR)
print("Config :", CONFIG_PATH)
print("Dataset:", DATASET_PATH)
print("Train  :", ROBOMIMIC_TRAIN)
print("NPU    :", env.get("ASCEND_RT_VISIBLE_DEVICES", "<not explicitly set>"))
print("=" * 84)
print("Command:")
print(" ".join(cmd))
print("=" * 84, flush=True)

subprocess.run(cmd, env=env, check=True)