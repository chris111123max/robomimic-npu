"""Outcome-filtered actor data; inherited critic sampling is untouched."""
import copy
import hashlib
from collections import Counter, deque
from pathlib import Path
import h5py
import numpy as np
import stage3_v8_paths
from stage3_v5_replay import OnlineSequenceReplay as V5OnlineReplay, CANONICAL_KEYS

ACTIVE_POOL = None

def bind_pool(pool):
    global ACTIVE_POOL
    ACTIVE_POOL = pool

def digest_episode(ep, source):
    h = hashlib.sha256(source.encode())
    for k in ("observations", "actions", "episode_steps"):
        h.update(np.ascontiguousarray(ep[k]).tobytes())
    return h.hexdigest()

def eligible(ep, source, ended, full_success, timeout, reason):
    n = len(ep["actions"])
    if not ended or not full_success or n < 10: return False
    steps = np.asarray(ep["episode_steps"])
    if not np.array_equal(steps, np.arange(n)): return False
    obs, acts = np.asarray(ep["observations"]), np.asarray(ep["actions"])
    if obs.shape != (n,59) or acts.shape != (n,14): return False
    if not np.isfinite(obs).all() or not np.isfinite(acts).all(): return False
    done = np.asarray(ep["dones"], bool).reshape(-1)
    if len(done) != n or done[:-1].any() or not done[-1]: return False
    # Stage1 successful collector-stop is encoded as truncated, not timeout.
    if source == "offline":
        if reason not in ("success_collector_stop", "environment_done"): return False
        if timeout and reason != "success_collector_stop": return False
    elif source == "online_success":
        if timeout: return False
        terminated = np.asarray(ep["terminated"],bool).reshape(-1)
        if len(terminated) != n or not terminated[-1]: return False
    else:
        raise ValueError("Unknown good source")
    # Corroborate full Transport task label with audited final object flags.
    nxt = np.asarray(ep["next_observations"])
    if nxt.shape != (n,59) or not np.isfinite(nxt).all(): return False
    if not np.all(nxt[-1,[45,46]] > .5): return False
    return True

class GoodReplay:
    def __init__(self, config, seed):
        self.config = copy.deepcopy(config)
        self.rng = np.random.default_rng(int(seed))
        self.offline = []
        self.online = deque()
        self.online_transitions = 0
        self.recent_hashes = deque()
        self.seen = set()
        self.counts = Counter()
        self.source_audit = {}
        self.samples = Counter()

    def add_episode(self, ep, source, *, ended, full_success, timeout=False,
                    reason="environment_done", label_source="", seed=-1, episode_id=-1):
        if not eligible(ep,source,ended,full_success,timeout,reason):
            self.counts[source+"_rejected"] += 1
            return False
        ident = digest_episode(ep, source)
        if ident in self.seen:
            self.counts["duplicates"] += 1
            return False
        saved = {k:np.asarray(ep[k]).copy() for k in
                 ("observations","actions","episode_steps")}
        saved.update(source=source, seed=int(seed), episode_id=int(episode_id),
                     label_source=label_source, identity=ident)
        n=len(saved["actions"])
        if source == "online_success":
            if n > int(self.config["online_capacity_transitions"]): return False
            self.online.append(saved); self.online_transitions += n
            while (self.online_transitions > int(self.config["online_capacity_transitions"])
                   or len(self.online) > int(self.config["online_capacity_episodes"])):
                old=self.online.popleft(); self.online_transitions-=len(old["actions"])
                self.counts["evicted"]+=1
        else:
            self.offline.append(saved)
        self.recent_hashes.append(ident); self.seen.add(ident)
        while len(self.recent_hashes) > int(self.config["dedup_history"]):
            self.seen.discard(self.recent_hashes.popleft())
        self.counts[source+"_accepted"]+=1
        return True

    def load_offline(self, paths):
        for name,path in sorted(paths.items()):
            if not path: continue
            accepted=0; labels=0; reasons=Counter(); rejected=[]
            with h5py.File(path,"r") as f:
                import json
                keys=tuple(json.loads(f.attrs["canonical_observation_keys"]))
                if keys != CANONICAL_KEYS: raise RuntimeError("Offline canonical keys differ")
                meta=json.loads(f.attrs["environment_metadata"])
                if meta["env_name"] != "TwoArmTransport": raise RuntimeError("Wrong task")
                schema=json.loads(f.attrs["progress_observation_schema"])
                fields=schema["fields"]
                if (fields["payload_in_target_bin"]["flat_index"] != 27
                    or fields["trash_in_trash_bin"]["flat_index"] != 28):
                    raise RuntimeError("Subtask flag schema changed")
                for ep_name,g in f["episodes"].items():
                    if "episode_success" not in g or "termination_reason" not in g.attrs:
                        raise RuntimeError("Offline lacks reliable outcome/boundary provenance")
                    labels_array=np.asarray(g["episode_success"],bool).reshape(-1)
                    n=len(g["actions"])
                    if len(labels_array)!=n or not np.all(labels_array==labels_array[0]):
                        raise RuntimeError("Inconsistent episode success labels")
                    won=bool(labels_array[0]); labels+=int(won)
                    if "success" in g.attrs and bool(g.attrs["success"])!=won:
                        raise RuntimeError("Conflicting offline labels")
                    reason=str(g.attrs["termination_reason"]); reasons[reason]+=1
                    if not won: continue
                    ep={k:np.asarray(g[k]) for k in ("actions","dones","terminated","truncated")}
                    ep["episode_steps"]=np.asarray(g["timestep"],np.int64).reshape(-1)
                    for dst,src in (("observations","obs"),("next_observations","next_obs")):
                        ep[dst]=np.concatenate([np.asarray(g[src][k],np.float32).reshape(n,-1) for k in keys],1)
                    added=self.add_episode(ep,"offline",ended=bool(ep["dones"][-1]),
                        full_success=won,timeout=bool(ep["truncated"][-1]),reason=reason,
                        label_source=name,seed=int(g.attrs["initial_seed"]),
                        episode_id=int(g.attrs["episode_id"]))
                    accepted+=int(added)
                    if not added: rejected.append(ep_name)
            self.source_audit[name]=dict(path=str(path),success_labels=labels,
                accepted=accepted,success_rejected=rejected,reasons=dict(reasons))
        if not self.offline: raise RuntimeError("No verified offline good H10; stop before environments")

    def sample(self, count, online_fraction=None):
        count=int(count)
        if count<=0: raise ValueError("Positive good batch required")
        if not self.offline and not self.online: raise RuntimeError("No verified good experience")
        fraction=self.config["online_fraction"] if online_fraction is None else online_fraction
        if not 0<=float(fraction)<=1: raise ValueError("Invalid good source mixture")
        online_count=int(count*float(fraction)) if self.online else 0
        if not self.offline: online_count=count
        rows=[]
        for source,n in (("offline",count-online_count),("online_success",online_count)):
            episodes=self.offline if source=="offline" else self.online
            for _ in range(n):
                ep=episodes[int(self.rng.integers(len(episodes)))]
                start=10*int(self.rng.integers((len(ep["actions"])-10)//10+1))
                row={k:ep[k][start:start+10].copy() for k in ("observations","actions","episode_steps")}
                row.update(source=source,seed=ep["seed"],episode_id=ep["episode_id"],
                           label_source=ep["label_source"],identity=ep["identity"])
                rows.append(row); self.samples[source]+=1
                self.samples["label:"+ep["label_source"]]+=1
        order=self.rng.permutation(count); rows=[rows[i] for i in order]
        out={k:np.stack([row[k] for row in rows]) for k in ("observations","actions","episode_steps")}
        out["mask"]=np.ones((count,10),np.float32)
        out["weights"]=np.stack([np.full(10,self.config["offline_weight"] if row["source"]=="offline"
                                        else self.config["online_weight"],np.float32) for row in rows])
        for k in ("source","seed","episode_id","identity","label_source"):
            out[k]=np.asarray([row[k] for row in rows])
        return out

    def state_dict(self):
        return copy.deepcopy(dict(config=self.config,rng=self.rng.bit_generator.state,
            offline=self.offline,online=list(self.online),online_transitions=self.online_transitions,
            recent_hashes=list(self.recent_hashes),counts=dict(self.counts),
            source_audit=self.source_audit,samples=dict(self.samples)))

    def load_state_dict(self, state):
        if state["config"]!=self.config: raise RuntimeError("Good pool configuration mismatch")
        self.rng.bit_generator.state=copy.deepcopy(state["rng"])
        self.offline=copy.deepcopy(state["offline"]); self.online=deque(copy.deepcopy(state["online"]))
        self.online_transitions=int(state["online_transitions"])
        self.recent_hashes=deque(state["recent_hashes"]); self.seen=set(self.recent_hashes)
        self.counts=Counter(state["counts"]); self.source_audit=copy.deepcopy(state["source_audit"])
        self.samples=Counter(state["samples"])

    def metrics(self):
        online_seeds=Counter(ep["seed"] for ep in self.online)
        return dict(good_offline_episodes=len(self.offline),good_online_episodes=len(self.online),
            good_online_transitions=self.online_transitions,good_online_distinct_seeds=len(online_seeds),
            good_online_largest_seed_fraction=max(online_seeds.values(),default=0)/max(1,len(self.online)),
            good_offline_samples=self.samples["offline"],good_online_samples=self.samples["online_success"],
            good_online_total_accepted=self.counts["online_success_accepted"])

class OnlineSequenceReplay(V5OnlineReplay):
    def __init__(self, capacity_transitions, seed=0, good_pool=None):
        super().__init__(capacity_transitions,seed)
        self.good_pool=good_pool if good_pool is not None else ACTIVE_POOL

    def finish(self, env_id, success=False, seed=-1, episode_id=-1):
        ep=self.current.get(int(env_id))
        if ep and self.good_pool is not None:
            arr={k:np.asarray(v) for k,v in ep.items()}
            self.good_pool.add_episode(arr,"online_success",ended=True,full_success=bool(success),
                timeout=bool(arr["truncated"][-1]),reason="environment_done",
                label_source="v8_env.is_success.task",seed=seed,episode_id=episode_id)
        # Exact original insert and RNG behavior, including its diagnostic reservoir.
        return super().finish(env_id,success=success)
