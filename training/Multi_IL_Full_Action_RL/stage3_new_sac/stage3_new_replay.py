"""Independent offline/online buffers and exact symmetric batch composition."""
from __future__ import annotations
from pathlib import Path
import numpy as np

FIELDS = ("observations", "actions", "rewards", "next_observations", "terminals")

class TransitionBuffer:
    def __init__(self, capacity, obs_dim=59, action_dim=14, seed=0):
        self.capacity, self.obs_dim, self.action_dim = int(capacity), int(obs_dim), int(action_dim)
        self.data = {"observations":np.empty((capacity,obs_dim),np.float32), "actions":np.empty((capacity,action_dim),np.float32),
            "rewards":np.empty((capacity,1),np.float32), "next_observations":np.empty((capacity,obs_dim),np.float32), "terminals":np.empty((capacity,1),np.float32)}
        self.top=self.size=0; self.rng=np.random.default_rng(seed); self.insertions=0
    def add(self, observation, action, reward, next_observation, terminal):
        values=(observation,action,[reward],next_observation,[terminal])
        for key,value in zip(FIELDS,values): self.data[key][self.top]=np.asarray(value,dtype=np.float32)
        self.top=(self.top+1)%self.capacity; self.size=min(self.size+1,self.capacity); self.insertions+=1
    def sample(self, count):
        if self.size==0: raise RuntimeError("Cannot sample an empty online replay")
        idx=self.rng.integers(self.size,size=int(count)); return {k:v[idx].copy() for k,v in self.data.items()}
    def save(self,path):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        stored=self.capacity if self.size==self.capacity else self.size
        np.savez_compressed(path,capacity=self.capacity,top=self.top,size=self.size,insertions=self.insertions,rng_state=np.asarray([self.rng.bit_generator.state],dtype=object),**{k:v[:stored] for k,v in self.data.items()})
    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=True) as p:
            obj=cls(int(p["capacity"]),p["observations"].shape[1],p["actions"].shape[1]); obj.top=int(p["top"]);obj.size=int(p["size"]);obj.insertions=int(p["insertions"])
            for key in FIELDS: obj.data[key][:len(p[key])]=p[key]
            obj.rng.bit_generator.state=p["rng_state"].item()
        return obj

class SymmetricSampler:
    def __init__(self, offline, online, seed=0): self.offline,self.online,self.turn=offline,online,0;self.rng=np.random.default_rng(seed);self.counts={"offline":0,"online":0}
    def sample(self,batch_size):
        base,rem=divmod(int(batch_size),2); offline_n=base+(rem if self.turn==0 else 0); online_n=int(batch_size)-offline_n; self.turn=1-self.turn if rem else self.turn
        left,right=self.offline.sample(offline_n),self.online.sample(online_n); result={k:np.concatenate((left[k],right[k])) for k in FIELDS}; order=self.rng.permutation(batch_size)
        self.counts["offline"]+=offline_n;self.counts["online"]+=online_n
        return {k:v[order] for k,v in result.items()}
    def fractions(self):
        total=sum(self.counts.values()); return {k:(v/total if total else 0.0) for k,v in self.counts.items()}
