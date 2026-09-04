"""Uniform transition loader for the original Transport PH low-dim demonstrations."""
from __future__ import annotations
import json
from pathlib import Path
import h5py, numpy as np
from stage3_new_replay import FIELDS

KEYS=("robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","robot1_eef_pos","robot1_eef_quat","robot1_gripper_qpos","object")

class ExpertDataset:
    def __init__(self,path,seed=0):
        self.path=str(Path(path).resolve()); self.rng=np.random.default_rng(seed); rows={k:[] for k in FIELDS}
        with h5py.File(self.path,"r") as f:
            if "episodes" in f: raise RuntimeError("Stage1 IL rollout HDF5 is forbidden as Stage3 expert replay")
            if "data" not in f: raise RuntimeError("Expert HDF5 must use the robomimic /data/demo_* layout")
            env_args=json.loads(f["data"].attrs.get("env_args","{}")); env_name=str(env_args.get("env_name",""))
            if "Transport" not in env_name: raise RuntimeError(f"Expert dataset environment is not Transport: {env_name!r}")
            for name in sorted(f["data"]):
                demo=f["data"][name]
                for required in ("obs","next_obs","actions","rewards","dones"):
                    if required not in demo: raise RuntimeError(f"{demo.name}: missing {required}")
                missing_obs=set(KEYS)-set(demo["obs"].keys());missing_next=set(KEYS)-set(demo["next_obs"].keys())
                if missing_obs or missing_next:raise RuntimeError(f"{demo.name}: missing canonical observation keys obs={sorted(missing_obs)} next_obs={sorted(missing_next)}")
                obs=self._flatten(demo["obs"]); nxt=self._flatten(demo["next_obs"]); actions=np.asarray(demo["actions"],np.float32); rewards=np.asarray(demo["rewards"],np.float32).reshape(-1,1); dones=np.asarray(demo["dones"],np.float32).reshape(-1,1)
                length=len(actions)
                if obs.shape!=(length,59) or nxt.shape!=(length,59) or actions.shape!=(length,14) or rewards.shape!=(length,1) or dones.shape!=(length,1): raise RuntimeError(f"{demo.name}: transition shape mismatch")
                # robomimic's standard offline convention uses stored dones as terminal masks.
                for key,value in zip(FIELDS,(obs,actions,rewards,nxt,dones)): rows[key].append(value)
        self.data={k:np.concatenate(v).astype(np.float32,copy=False) for k,v in rows.items()}; self.size=len(self.data["actions"])
        if self.size==0 or not all(np.isfinite(v).all() for v in self.data.values()): raise RuntimeError("Expert dataset is empty or non-finite")
    @staticmethod
    def _flatten(group): return np.concatenate([np.asarray(group[k],np.float32).reshape(len(group[k]),-1) for k in KEYS],axis=1)
    def sample(self,count):
        idx=self.rng.integers(self.size,size=int(count)); return {k:v[idx].copy() for k,v in self.data.items()}
    def audit(self): return {"path":self.path,"transitions":self.size,"obs_dim":59,"action_dim":14,"sampling":"uniform_transition","terminal_source":"stored robomimic dones"}
