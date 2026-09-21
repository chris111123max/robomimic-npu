"""Episode-preserving Stage1 loader; never infers missing terminal fields."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np

POLICIES=("bc_rnn","bc_transformer","bc_gmm")
KEYS=("robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","robot1_eef_pos","robot1_eef_quat","robot1_gripper_qpos","object")

def scalar(group, name):
    value=group.attrs[name] if name in group.attrs else group[name][()]
    return np.asarray(value).reshape(-1)[0].item()

def mc_return(reward, gamma):
    out=np.empty(len(reward),np.float32); running=0.
    for i in range(len(reward)-1,-1,-1): running=float(reward[i])+float(gamma)*running; out[i]=running
    return out

@dataclass(frozen=True)
class Episode:
    policy:str; seed:int; episode_id:int; success:bool
    observations:np.ndarray; next_observations:np.ndarray; actions:np.ndarray
    rewards:np.ndarray; returns:np.ndarray; terminated:np.ndarray; truncated:np.ndarray; dones:np.ndarray
    @property
    def length(self): return len(self.actions)

class EpisodeDataset:
    def __init__(self, policy, path, seeds, gamma=.99):
        self.policy,self.path,self.episodes=policy,str(Path(path).resolve()),[]
        with h5py.File(path,"r") as f:
            if "episodes" not in f: raise RuntimeError(f"{path}: missing /episodes")
            if tuple(json.loads(f.attrs["canonical_observation_keys"])) != KEYS: raise RuntimeError("observation schema mismatch")
            lookup={int(scalar(g,"initial_seed")):g for g in f["episodes"].values()}
            for seed in map(int,seeds):
                if seed not in lookup: raise RuntimeError(f"missing seed {seed}")
                self.episodes.append(self._read(lookup[seed],seed,gamma))
        self.obs_dim,self.action_dim=59,14

    def _read(self,g,seed,gamma):
        for key in ("obs","next_obs","actions","rewards","dones","terminated","truncated","episode_id","episode_success","episode_length"):
            if key not in g and key not in g.attrs: raise RuntimeError(f"{g.name}: missing {key}")
        actions=np.asarray(g["actions"],np.float32); n=len(actions)
        obs=np.concatenate([np.asarray(g["obs"][k],np.float32).reshape(n,-1) for k in KEYS],1)
        nxt=np.concatenate([np.asarray(g["next_obs"][k],np.float32).reshape(n,-1) for k in KEYS],1)
        reward=np.asarray(g["rewards"],np.float32).reshape(-1)
        term=np.asarray(g["terminated"],bool).reshape(-1); trunc=np.asarray(g["truncated"],bool).reshape(-1); dones=np.asarray(g["dones"],bool).reshape(-1)
        if actions.shape!=(n,14) or obs.shape!=(n,59) or nxt.shape!=(n,59): raise RuntimeError(f"{g.name}: shape mismatch")
        if not np.array_equal(dones,term|trunc) or np.any(dones[:-1]) or not dones[-1]: raise RuntimeError(f"{g.name}: boundary mismatch")
        if int(scalar(g,"episode_length"))!=n: raise RuntimeError(f"{g.name}: length mismatch")
        return Episode(self.policy,seed,int(scalar(g,"episode_id")),bool(scalar(g,"episode_success")),obs,nxt,actions,reward,mc_return(reward,gamma),term,trunc,dones)

def load_splits(root,train_seeds,val_seeds,gamma):
    root=Path(root); paths={p:root/p/"transitions.hdf5" for p in POLICIES}
    if set(train_seeds)&set(val_seeds): raise RuntimeError("seed leakage")
    return ({p:EpisodeDataset(p,paths[p],train_seeds,gamma) for p in POLICIES},
            {p:EpisodeDataset(p,paths[p],val_seeds,gamma) for p in POLICIES})

def previous_actions(actions):
    result=np.zeros_like(actions); result[1:]=actions[:-1]; return result
