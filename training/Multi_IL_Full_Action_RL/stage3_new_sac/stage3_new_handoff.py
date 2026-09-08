"""Frozen robomimic BC-RNN proposals and target-Q handoff primitives."""
from __future__ import annotations
import hashlib,json,random
from collections import OrderedDict
from pathlib import Path
import h5py,numpy as np,torch
from stage3_new_evaluation import KEYS,flatten,mujoco_fatal_error_type,reset_seed,success

def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

class FrozenRNNProposer:
    def __init__(self,checkpoint,device):
        import robomimic.utils.file_utils as FileUtils
        self.checkpoint=str(Path(checkpoint).resolve());self.rollout,self.payload=FileUtils.policy_from_checkpoint(ckpt_path=self.checkpoint,device=device,verbose=False)
        nets=self.rollout.policy.nets;nets.eval();nets.requires_grad_(False);self.nets=nets;self.calls=0;self.episodes=0
    def start_episode(self):self.rollout.start_episode();self.calls=0;self.episodes+=1
    def action(self,observation):
        obs=OrderedDict((key,np.asarray(observation[key]).copy()) for key in KEYS);action=np.asarray(self.rollout(ob=obs),np.float32).reshape(-1);self.calls+=1
        if action.shape!=(14,) or not np.isfinite(action).all():raise RuntimeError(f"Invalid BC-RNN proposal {action.shape}")
        return action
    def state_dict(self):
        algo=self.rollout.policy
        return {"hidden":_cpu_copy(getattr(algo,"_rnn_hidden_state",None)),"counter":int(getattr(algo,"_rnn_counter",0)),"open_loop_obs":_cpu_copy(getattr(algo,"_open_loop_obs",None)),"calls":self.calls}
    def load_state_dict(self,state):
        algo=self.rollout.policy;algo._rnn_hidden_state=_device_copy(state["hidden"],algo.device);algo._rnn_counter=int(state["counter"]);self.calls=int(state.get("calls",algo._rnn_counter))
        if state.get("open_loop_obs") is not None:algo._open_loop_obs=_device_copy(state["open_loop_obs"],algo.device)

def _cpu_copy(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,tuple):return tuple(_cpu_copy(x) for x in value)
    if isinstance(value,dict):return {k:_cpu_copy(v) for k,v in value.items()}
    return value
def _device_copy(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,tuple):return tuple(_device_copy(x,device) for x in value)
    if isinstance(value,dict):return {k:_device_copy(v,device) for k,v in value.items()}
    return value

def _rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.random.get_rng_state()}

def _set_rng_state(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.random.set_rng_state(state["torch"])

def select_actions(target,states,rl_actions,rnn_actions):
    with torch.no_grad():
        q1rl,q2rl=target(states,rl_actions);q1rnn,q2rnn=target(states,rnn_actions);qrl=torch.minimum(q1rl,q2rl).reshape(-1);qrnn=torch.minimum(q1rnn,q2rnn).reshape(-1);rl_wins=qrl>qrnn;chosen=torch.where(rl_wins[:,None],rl_actions,rnn_actions)
    return chosen,qrl,qrnn,rl_wins

def hybrid_bootstrap(target,states,rl_actions,rnn_actions,rl_logp,alpha):
    chosen,qrl,qrnn,rl_wins=select_actions(target,states,rl_actions,rnn_actions)
    # TD bootstrap is a fixed critic target and must not backpropagate through
    # the sampled next-policy log probability into the actor.
    with torch.no_grad():value=torch.where(rl_wins,qrl-alpha.detach()*rl_logp.reshape(-1),qrnn)
    return {"value":value[:,None],"q_rl":qrl,"q_rnn":qrnn,"rl_wins":rl_wins,"chosen":chosen}

def handoff_imitation(actor,states,rnn_actions,mask):
    mask=mask.reshape(-1).bool()
    if not bool(mask.any()):return states.sum()*0.0,0
    mean=actor(states[mask],deterministic=True)[0];return torch.nn.functional.mse_loss(mean,rnn_actions[mask]),int(mask.sum().item())

def build_expert_cache(dataset_path,checkpoint,output,device="cpu"):
    proposer=FrozenRNNProposer(checkpoint,device);actions=[];next_actions=[];episodes=[]
    with h5py.File(dataset_path,"r") as f:
        if "data" not in f or "episodes" in f:raise RuntimeError("Expert cache requires ordered robomimic /data/demo_* trajectories")
        for name in sorted(f["data"]):
            demo=f["data"][name];length=len(demo["actions"])
            if length<=0 or "obs" not in demo or "next_obs" not in demo:raise RuntimeError(f"Incomplete expert trajectory {demo.name}")
            proposer.start_episode();episode_actions=[];first_rng=_rng_state()
            for t in range(length):episode_actions.append(proposer.action({key:demo["obs"][key][t] for key in KEYS}))
            final_next=proposer.action({key:demo["next_obs"][key][length-1] for key in KEYS});episode_actions=np.stack(episode_actions);episode_next=np.concatenate((episode_actions[1:],final_next[None]),axis=0);actions.append(episode_actions);next_actions.append(episode_next)
            # A loaded rollout policy may consume RNG internally. Reproduce the
            # original first-action RNG state so this check isolates recurrent
            # reset semantics, then restore the uninterrupted cache RNG stream.
            continued_rng=_rng_state();_set_rng_state(first_rng);proposer.start_episode();first_again=proposer.action({key:demo["obs"][key][0] for key in KEYS});_set_rng_state(continued_rng)
            if not np.allclose(first_again,episode_actions[0],rtol=0,atol=1e-6):raise RuntimeError(f"BC-RNN hidden reset integrity failed for {name}")
            episodes.append({"name":name,"length":length,"offset":sum(x["length"] for x in episodes),"first_action_reset_max_abs_diff":float(np.max(np.abs(first_again-episode_actions[0])))})
    action=np.concatenate(actions).astype(np.float32);nxt=np.concatenate(next_actions).astype(np.float32);output=Path(output);np.savez_compressed(output,rnn_actions=action,rnn_next_actions=nxt)
    manifest={"source_dataset_path":str(Path(dataset_path).resolve()),"bc_rnn_checkpoint_path":str(Path(checkpoint).resolve()),"bc_rnn_checkpoint_sha256":sha256(checkpoint),"cache_path":str(output.resolve()),"cache_sha256":sha256(output),"transition_count":len(action),"episode_count":len(episodes),"episode_boundaries":episodes,"recurrent_protocol":"start_episode once per demo; obs called sequentially; final next_obs called once"}
    with open(output.with_suffix(".manifest.json"),"x",encoding="utf-8") as f:json.dump(manifest,f,indent=2,sort_keys=True);f.write("\n")
    return manifest

def load_expert_cache(path,expected_size):
    with np.load(path) as p:result={"rnn_actions":p["rnn_actions"].astype(np.float32),"rnn_next_actions":p["rnn_next_actions"].astype(np.float32)}
    if result["rnn_actions"].shape!=(expected_size,14) or result["rnn_next_actions"].shape!=(expected_size,14):raise RuntimeError("Expert BC-RNN proposal cache shape mismatch")
    return result

def evaluate_handoff(actor,target,proposer,env,seeds,horizon,device,retries=1,pure_rnn=False):
    rows=[];all_qrl=[];all_qrnn=[];all_margins=[];rnn_total=rl_total=ties=0
    for seed in seeds:
        fatal=None
        for attempt in range(int(retries)+1):
            try:
                obs=reset_seed(env,seed);proposer.start_episode();ret=0.;rnn_count=rl_count=0;eq_count=0;episode_qrl=[];episode_qrnn=[]
                for step in range(int(horizon)):
                    rnn=proposer.action(obs);state=torch.as_tensor(flatten(obs)[None],dtype=torch.float32,device=device);rnn_t=torch.as_tensor(rnn[None],dtype=torch.float32,device=device)
                    if pure_rnn:action=rnn
                    else:
                        with torch.no_grad():rl=actor(state,deterministic=True)[0];chosen,qrl,qrnn,rl_wins=select_actions(target,state,rl,rnn_t);action=chosen[0].cpu().numpy();episode_qrl.append(float(qrl.item()));episode_qrnn.append(float(qrnn.item()));is_rl=bool(rl_wins.item());rl_count+=int(is_rl);rnn_count+=int(not is_rl);eq_count+=int(float(qrl.item())==float(qrnn.item()))
                    obs,reward,done,_=env.step(action);ret+=float(reward);won=success(env)
                    if done or won:break
                length=step+1;row={"seed":int(seed),"success":bool(won),"return":ret,"length":length,"sim_error":False,"attempts":attempt+1}
                if not pure_rnn:
                    margin=np.asarray(episode_qrl)-np.asarray(episode_qrnn);row.update({"rnn_selected_fraction":rnn_count/length,"rl_selected_fraction":rl_count/length,"mean_q_rnn":float(np.mean(episode_qrnn)),"mean_q_rl":float(np.mean(episode_qrl)),"mean_q_margin":float(np.mean(margin))});rnn_total+=rnn_count;rl_total+=rl_count;ties+=eq_count;all_qrl.extend(episode_qrl);all_qrnn.extend(episode_qrnn);all_margins.extend(margin.tolist())
                rows.append(row);fatal=None;break
            except mujoco_fatal_error_type() as error:fatal=error
        if fatal is not None:rows.append({"seed":int(seed),"success":None,"return":None,"length":None,"sim_error":True,"attempts":int(retries)+1,"exception":str(fatal)})
    valid=[r for r in rows if not r["sim_error"]];result={"episodes":rows,"valid_episodes":len(valid),"sim_error_episodes":len(rows)-len(valid),"success_count":sum(bool(r["success"]) for r in valid),"success_rate":float(np.mean([r["success"] for r in valid])) if valid else None,"mean_return":float(np.mean([r["return"] for r in valid])) if valid else None,"mean_length":float(np.mean([r["length"] for r in valid])) if valid else None}
    if not pure_rnn:
        margin=np.asarray(all_margins);total=rnn_total+rl_total;result.update({"rnn_selected_count":rnn_total,"rl_selected_count":rl_total,"rnn_selected_fraction":rnn_total/total if total else None,"rl_selected_fraction":rl_total/total if total else None,"q_rnn_mean":float(np.mean(all_qrnn)) if all_qrnn else None,"q_rl_mean":float(np.mean(all_qrl)) if all_qrl else None,"q_margin_rl_minus_rnn_mean":float(margin.mean()) if len(margin) else None,"q_margin_median":float(np.median(margin)) if len(margin) else None,"q_margin_p10":float(np.percentile(margin,10)) if len(margin) else None,"q_margin_p90":float(np.percentile(margin,90)) if len(margin) else None,"tie_count":ties})
    return result
