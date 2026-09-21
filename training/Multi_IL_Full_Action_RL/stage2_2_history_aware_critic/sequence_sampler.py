"""Prevalidated same-episode burn-in + learning windows."""
from __future__ import annotations
import numpy as np
from sequence_dataset import POLICIES, previous_actions

class SequenceSampler:
    def __init__(self,datasets,burn_in,learning,horizon,seed,balanced):
        self.datasets=datasets; self.burn=int(burn_in); self.learning=int(learning); self.total=self.burn+self.learning
        self.horizon=float(horizon); self.rng=np.random.default_rng(seed); self.balanced=bool(balanced); self.cursor=0
        self.valid={p:[(ei,start) for ei,e in enumerate(datasets[p].episodes) for start in range(e.length-self.total+1)] for p in datasets}
        if any(not rows for rows in self.valid.values()): raise RuntimeError("a source has no legal same-episode window")
        self.counts={p:0 for p in POLICIES}

    def allocation(self,count):
        if not self.balanced: return {"bc_rnn":int(count),"bc_transformer":0,"bc_gmm":0}
        base,rem=divmod(int(count),3); out={p:base for p in POLICIES}
        for i in range(rem): out[POLICIES[(self.cursor+i)%3]]+=1
        self.cursor=(self.cursor+rem)%3; return out

    def sample(self,count):
        rows=[]
        for policy,n in self.allocation(count).items():
            ids=self.rng.integers(len(self.valid[policy]),size=n)
            rows.extend((policy,*self.valid[policy][int(i)]) for i in ids); self.counts[policy]+=n
        self.rng.shuffle(rows); fields={k:[] for k in ("observations","previous_actions","actions","progress","returns","success","policy","episode_id","start")}
        for policy,ei,start in rows:
            e=self.datasets[policy].episodes[ei]; stop=start+self.total; prev=previous_actions(e.actions)
            fields["observations"].append(e.observations[start:stop]); fields["previous_actions"].append(prev[start:stop]); fields["actions"].append(e.actions[start:stop])
            fields["progress"].append((np.arange(start,stop,dtype=np.float32)/self.horizon)[:,None]); fields["returns"].append(e.returns[start:stop,None])
            fields["success"].append(e.success); fields["policy"].append(policy); fields["episode_id"].append(e.episode_id); fields["start"].append(start)
        return {k:(np.stack(v) if k not in ("policy",) else np.asarray(v)) for k,v in fields.items()}

    def proportions(self):
        total=sum(self.counts.values()); return {p:self.counts[p]/total if total else 0. for p in POLICIES}
