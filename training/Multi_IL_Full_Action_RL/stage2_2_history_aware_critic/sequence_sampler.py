"""Deterministic horizon-limited sampling for Stage2.2.

Each supervised transition t is evaluated from at most the latest 10 temporal
tokens. The sampled recurrent state always starts from zero. To prevent history
outside the window from leaking through the token schema, the first token in a
window always uses a zero previous-action vector. Loss is applied only to the
final valid transition of each sampled window.

The legacy sampler remains only for exact replay of historical failed runs.
"""
from __future__ import annotations
import numpy as np
from sequence_dataset import POLICIES, previous_actions


class _SourceAllocation:
    def _init_allocation(self,datasets,seed,balanced):
        self.datasets=datasets;self.rng=np.random.default_rng(seed);self.balanced=bool(balanced);self.cursor=0;self.counts={p:0 for p in POLICIES}

    def allocation(self,count):
        if not self.balanced:return {"bc_rnn":int(count),"bc_transformer":0,"bc_gmm":0}
        base,rem=divmod(int(count),3);out={p:base for p in POLICIES}
        for i in range(rem):out[POLICIES[(self.cursor+i)%3]]+=1
        self.cursor=(self.cursor+rem)%3;return out

    def proportions(self):
        total=sum(self.counts.values());return {p:self.counts[p]/total if total else 0. for p in POLICIES}


class SequenceSampler(_SourceAllocation):
    """Sample one target transition with a zero-state context of at most N steps."""
    def __init__(self,datasets,burn_in,context_length,horizon,seed,balanced):
        del burn_in
        self.context_length=int(context_length);self.horizon=float(horizon)
        if self.context_length<=0:raise ValueError("context_length must be positive")
        self._init_allocation(datasets,seed,balanced)
        self.valid={p:[(ei,t) for ei,e in enumerate(datasets[p].episodes) for t in range(e.length)] for p in datasets}
        if any(not rows for rows in self.valid.values()):raise RuntimeError("a source has no transition")

    def sample(self,count):
        rows=[]
        for policy,n in self.allocation(count).items():
            ids=self.rng.integers(len(self.valid[policy]),size=n)
            rows.extend((policy,*self.valid[policy][int(i)]) for i in ids)
            self.counts[policy]+=n
        self.rng.shuffle(rows)
        b=len(rows);T=self.context_length
        o=np.zeros((b,T,59),np.float32);pa=np.zeros((b,T,14),np.float32);a=np.zeros((b,T,14),np.float32)
        progress=np.zeros((b,T,1),np.float32);returns=np.zeros((b,T,1),np.float32)
        valid=np.zeros((b,T,1),bool);learning=np.zeros((b,T,1),bool)
        metadata={k:[] for k in ("policy","seed","episode_id","start","stop","target_step","context_steps","episode_length")}
        for row,(policy,ei,target) in enumerate(rows):
            e=self.datasets[policy].episodes[ei]
            start=max(0,int(target)-T+1);stop=int(target)+1;n=stop-start
            o[row,:n]=e.observations[start:stop];a[row,:n]=e.actions[start:stop]
            if n>1:pa[row,1:n]=e.actions[start:stop-1]
            progress[row,:n,0]=np.arange(start,stop,dtype=np.float32)/self.horizon
            returns[row,:n,0]=e.returns[start:stop]
            valid[row,:n]=True;learning[row,n-1]=True
            for key,value in (("policy",policy),("seed",e.seed),("episode_id",e.episode_id),
                              ("start",start),("stop",stop),("target_step",target),
                              ("context_steps",n),("episode_length",e.length)):
                metadata[key].append(value)
        result={"observations":o,"previous_actions":pa,"actions":a,"progress":progress,
                "returns":returns,"valid_mask":valid,"learning_mask":learning}
        result.update({k:np.asarray(v) for k,v in metadata.items()})
        return result


class LegacyWindowSampler(_SourceAllocation):
    """Exact old arbitrary-window sampler. Never use for current training."""
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
