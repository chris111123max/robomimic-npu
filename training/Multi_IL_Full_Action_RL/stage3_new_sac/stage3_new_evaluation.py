"""TwoArmTransport environment construction and deterministic policy evaluation."""
from __future__ import annotations
import random
import numpy as np, torch

KEYS=("robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","robot1_eef_pos","robot1_eef_quat","robot1_gripper_qpos","object")

def seed_all(seed):
    random.seed(int(seed));np.random.seed(int(seed));torch.manual_seed(int(seed));npu=getattr(torch,"npu",None)
    if npu is not None and npu.is_available(): npu.manual_seed_all(int(seed))
def seed_env(env,seed):
    current,seen=env,set()
    while current is not None and id(current) not in seen:
        seen.add(id(current)); fn=getattr(current,"seed",None)
        if callable(fn):
            try: fn(int(seed))
            except (TypeError,AttributeError,NotImplementedError): pass
        current=getattr(current,"env",None)
def flatten(observation):
    missing=[k for k in KEYS if k not in observation]
    if missing: raise RuntimeError(f"Environment observation misses canonical keys: {missing}")
    value=np.concatenate([np.asarray(observation[k]).reshape(-1) for k in KEYS]).astype(np.float32)
    if value.shape!=(59,) or not np.isfinite(value).all(): raise RuntimeError(f"Invalid flattened observation {value.shape}")
    return value
def success(env):
    result=env.is_success(); return bool(result.get("task",any(result.values()))) if isinstance(result,dict) else bool(result)
def build_env(expert_dataset):
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    # Stage3-new consumes only the fixed seven-key low-dimensional contract and
    # does not load a robomimic algorithm config. Initialize the global modality
    # registry explicitly before EnvRobosuite.get_observation is first called.
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim":list(KEYS)})
    meta=FileUtils.get_env_metadata_from_dataset(dataset_path=str(expert_dataset));
    if "Transport" not in str(meta.get("env_name","")): raise RuntimeError(f"Expected Transport metadata, got {meta.get('env_name')!r}")
    return EnvUtils.create_env_from_metadata(meta,render=False,render_offscreen=False,use_image_obs=False)
def close_env(env):
    current=getattr(env,"env",env); fn=getattr(current,"close",None)
    if callable(fn): fn()
def reset_seed(env,seed): seed_all(seed);seed_env(env,seed);obs=env.reset();seed_all(seed);return obs
def evaluate(actor,env,seeds,horizon,device):
    rows=[]
    for seed in seeds:
        obs=reset_seed(env,seed); total=0.0;steps=0;raw_done=False;won=success(env)
        for step in range(int(horizon)):
            state=torch.as_tensor(flatten(obs)[None],device=device)
            with torch.no_grad(): action=actor(state,deterministic=True)[0][0].cpu().numpy()
            obs,reward,raw_done,_=env.step(action);total+=float(reward);steps=step+1;won=success(env)
            if raw_done or won or steps>=horizon: break
        time_limit=bool(steps>=horizon and not won)
        rows.append({"seed":int(seed),"return":total,"length":steps,"success":won,"terminated":bool(won or (raw_done and not time_limit)),"truncated":time_limit})
    return {"episodes":rows,"success_rate":float(np.mean([r["success"] for r in rows])),"mean_return":float(np.mean([r["return"] for r in rows])),"mean_length":float(np.mean([r["length"] for r in rows]))}
