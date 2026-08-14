#!/usr/bin/env python3
from pathlib import Path
import os
import subprocess
import sys

WORKSPACE = Path("/data/home/3220251075/lerobot_workspace")
EXPERIMENT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_DIR / "config.json"
DATASET_PATH = WORKSPACE / "datasets" / "transport" / "PH" / "low_dim_v15.hdf5"
ROBOMIMIC_TRAIN = WORKSPACE / "robomimic" / "robomimic" / "scripts" / "train.py"

for p, name in [
    (CONFIG_PATH, "config"),
    (DATASET_PATH, "dataset"),
    (ROBOMIMIC_TRAIN, "robomimic train.py"),
]:
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")

env = os.environ.copy()

# Preserve CANN / existing PYTHONPATH entries instead of overwriting them.
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
print("Experiment 06: Official-tuned BC-Transformer-GMM | TwoArmTransport PH low_dim")
print("Python :", sys.executable)
print("Config :", CONFIG_PATH)
print("Dataset:", DATASET_PATH)
print("Train  :", ROBOMIMIC_TRAIN)
print("NPU    :", env.get("ASCEND_RT_VISIBLE_DEVICES", "<not explicitly set>"))
print("=" * 84)
print("Command:")
print(" ".join(cmd))
print("=" * 84, flush=True)

subprocess.run(cmd, env=env, check=True)
