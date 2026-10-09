import os,sys,json,hashlib,importlib.util
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'critic_gradient_rootcause'
RL=HERE.parents[2]
for d in (RL/'stage3_v5_rgmm_td3',RL/'stage3_new_sac',RL/'stage3_v3_rgmm_td3',RL/'stage3_v6_dual_2q'):
 sys.path.insert(0,str(d))
sys.path.insert(0,str(HERE))
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def load_tools():
 spec=importlib.util.spec_from_file_location('prior_gradient_test_helpers',OLD/'run_round1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);bt,core=m.load_tools();bt.HERE=HERE;bt.acc.qg.HERE=HERE;return bt,core
def canonical(x,depth=0):
 if depth>5:return None
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if x is None or isinstance(x,(bool,int,float,str)):return x
 if isinstance(x,dict):return {str(k):canonical(v,depth+1) for k,v in sorted(x.items(),key=lambda z:str(z[0])) if not callable(v)}
 if isinstance(x,(list,tuple)):return [canonical(v,depth+1) for v in x]
 if 'controller' in type(x).__module__ or 'interpolator' in type(x).__module__:
  return {'class':type(x).__name__,'state':{k:canonical(v,depth+1) for k,v in sorted(vars(x).items()) if k not in ('sim','robot','model') and not callable(v)}}
 return None
def diagnostic_snapshot(env):
 raw=env.env;sim=raw.sim;state=env.get_state();physics={}
 for k in ('time','qpos','qvel','act','ctrl','qacc_warmstart','mocap_pos','mocap_quat','userdata','qfrc_applied','xfrc_applied','eq_active'):
  try:physics[k]=canonical(getattr(sim.data,k))
  except AttributeError:physics[k]='NOT_EXPOSED'
 controls=[]
 for r in raw.robots:
  controls.append(canonical(getattr(r,'composite_controller',getattr(r,'controller',None))))
 return {'sim_states':canonical(state['states']),'physics':physics,'controllers':controls,'timestep':int(raw.timestep),'cur_time':float(raw.cur_time),'obs_cache':canonical(getattr(raw,'_obs_cache',{})),'numpy_rng':canonical(np.random.get_state()),'model_xml_sha256':hashlib.sha256(state.get('model','').encode()).hexdigest()}
def snapshot_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
