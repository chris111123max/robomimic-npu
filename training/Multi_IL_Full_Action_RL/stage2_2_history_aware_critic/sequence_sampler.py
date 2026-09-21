"""Deterministic episode-prefix sampling for Stage2.2.

Training always unrolls from episode step zero. ``learning_mask`` selects the
supervised block, so training and validation use identical full-history state.
The legacy sampler remains only for exact replay of the failed runs.
"""
from __future__ import annotations
import numpy as np
from sequence_dataset import POLICIES, previous_actions

class _SourceAllocation:
    def _init_allocation(self,datasets,seed,balanced):
        self.datasets=datasets;self.rng=np.random.default_rng(seed);self.balanced=bool(balanced);self.cursor=0;self.counts={p:0 for p in POLICIES}

    def allocation(self,count):
        if not self.balanced: return {"bc_rnn":int(count),"bc_transformer":0,"bc_gmm":0}
        base,rem=divmod(int(count),3); out={p:base for p in POLICIES}
        for i in range(rem): out[POLICIES[(self.cursor+i)%3]]+=1
        self.cursor=(self.cursor+rem)%3; return out

    def proportions(self):
        total=sum(self.counts.values()); return {p:self.counts[p]/total if total else 0. for p in POLICIES}

class SequenceSampler(_SourceAllocation):
    """Sample a target block and rebuild every context from episode start."""
    def __init__(self,datasets,burn_in,learning,horizon,seed,balanced):
        del burn_in
        self.learning=int(learning);self.horizon=float(horizon);self._init_allocation(datasets,seed,balanced)
        self.valid={p:[(ei,start) for ei,e in enumerate(datasets[p].episodes) for start in range(max(1,e.length-self.learning+1))] for p in datasets}
        if any(not rows for rows in self.valid.values()):raise RuntimeError("a source has no episode")

    def sample(self,count):
        rows=[]
        for policy,n in self.allocation(count).items():
            ids=self.rng.integers(len(self.valid[policy]),size=n);rows.extend((policy,*self.valid[policy][int(i)]) for i in ids);self.counts[policy]+=n
        self.rng.shuffle(rows);specs=[];max_stop=0
        for policy,ei,start in rows:
            e=self.datasets[policy].episodes[ei];stop=min(e.length,start+self.learning);max_stop=max(max_stop,stop);specs.append((policy,e,start,stop))
        b=len(specs);o=np.zeros((b,max_stop,59),np.float32);pa=np.zeros((b,max_stop,14),np.float32);a=np.zeros((b,max_stop,14),np.float32)
        progress=np.zeros((b,max_stop,1),np.float32);returns=np.zeros((b,max_stop,1),np.float32);valid=np.zeros((b,max_stop,1),bool);learning=np.zeros((b,max_stop,1),bool)
        metadata={k:[] for k in ("policy","seed","episode_id","start","stop","prefix_length","episode_length")}
        for row,(policy,e,start,stop) in enumerate(specs):
            prev=previous_actions(e.actions);o[row,:stop]=e.observations[:stop];pa[row,:stop]=prev[:stop];a[row,:stop]=e.actions[:stop]
            progress[row,:stop,0]=np.arange(stop,dtype=np.float32)/self.horizon;returns[row,:stop,0]=e.returns[:stop];valid[row,:stop]=True;learning[row,start:stop]=True
            for key,value in (("policy",policy),("seed",e.seed),("episode_id",e.episode_id),("start",start),("stop",stop),("prefix_length",start),("episode_length",e.length)):metadata[key].append(value)
        result={"observations":o,"previous_actions":pa,"actions":a,"progress":progress,"returns":returns,"valid_mask":valid,"learning_mask":learning}
        result.update({k:np.asarray(v) for k,v in metadata.items()});return result

class LegacyWindowSampler(_SourceAllocation):
    """Exact old arbitrary-window sampler. Never use for training."""
    def __init__(self,datasets,burn_in,learning,horizon,seed,balanced):
        self.burn=int(burn_in);self.learning=int(learning);self.total=self.burn+self.learning;self.horizon=float(horizon);self._init_allocation(datasets,seed,balanced)
        self.valid={p:[(ei,start) for ei,e in enumerate(datasets[p].episodes) for start in range(e.length-self.total+1)] for p in datasets}
        if any(not rows for rows in self.valid.values()):raise RuntimeError("a source has no legal legacy window")

    def sample(self,count):
        rows=[]
        for policy,n in self.allocation(count).items():
            ids=self.rng.integers(len(self.valid[policy]),size=n);rows.extend((policy,*self.valid[policy][int(i)]) for i in ids);self.counts[policy]+=n
        self.rng.shuffle(rows);fields={k:[] for k in ("observations","previous_actions","actions","progress","returns","policy","seed","episode_id","start","stop")}
        for policy,ei,start in rows:
            e=self.datasets[policy].episodes[ei];stop=start+self.total;prev=previous_actions(e.actions)
            fields["observations"].append(e.observations[start:stop]);fields["previous_actions"].append(prev[start:stop]);fields["actions"].append(e.actions[start:stop]);fields["progress"].append((np.arange(start,stop,dtype=np.float32)/self.horizon)[:,None]);fields["returns"].append(e.returns[start:stop,None])
            for key,value in (("policy",policy),("seed",e.seed),("episode_id",e.episode_id),("start",start),("stop",stop)):fields[key].append(value)
        return {k:np.asarray(v) if k in ("policy","seed","episode_id","start","stop") else np.stack(v) for k,v in fields.items()}
