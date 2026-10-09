"""Import existing modules; all writes remain inside V8 or its isolated run."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
TRAIN = HERE.parent
ROOT = TRAIN.parents[1]
for name in ("stage3_v5_rgmm_td3","stage3_v6_dual_2q","stage3_v7_pirlnav_schedule","stage3_v3_rgmm_td3","stage3_new_sac"):
    p = str(TRAIN / name)
    if p not in sys.path: sys.path.insert(0,p)
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
V7_RUN = Path("/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v7_pirlnav_schedule/stage3v6_readiness_v2_multi_mean_random_formal_20260929_130945/random2q/multi_q")
V7_CHECKPOINT = V7_RUN / "checkpoints/best_success.pth"
PREPARED = Path("/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_readiness_v2_multi_mean_random_formal_20260929_130945")
