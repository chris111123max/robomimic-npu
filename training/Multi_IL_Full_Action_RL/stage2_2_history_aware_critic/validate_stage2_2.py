#!/usr/bin/env python3
"""Synthetic end-to-end smoke; never reads or writes formal datasets/runs."""
import json,subprocess,sys,tempfile
from pathlib import Path
import h5py,numpy as np
HERE=Path(__file__).resolve().parent
KEYS=("robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","robot1_eef_pos","robot1_eef_quat","robot1_gripper_qpos","object")
WIDTHS=(3,4,2,3,4,2,41)
def fixture(root):
 for pi,policy in enumerate(("bc_rnn","bc_transformer","bc_gmm")):
  path=root/policy/"transitions.hdf5";path.parent.mkdir(parents=True);
  with h5py.File(path,"w") as f:
   f.attrs["canonical_observation_keys"]=json.dumps(KEYS);eps=f.create_group("episodes")
   for seed in range(10000,10100):
    n=50;success=(seed+pi)%2==0;g=eps.create_group(str(seed));g.attrs["initial_seed"]=seed;g.attrs["episode_id"]=seed-10000;g.attrs["episode_success"]=success;g.attrs["episode_length"]=n
    actions=np.sin(np.arange(n)[:,None]/10+np.arange(14)[None,:]/7).astype(np.float32);g.create_dataset("actions",data=actions);reward=np.zeros(n,np.float32);reward[-1]=float(success);g.create_dataset("rewards",data=reward);done=np.zeros(n,bool);done[-1]=1;g.create_dataset("dones",data=done);g.create_dataset("terminated",data=np.zeros(n,bool));g.create_dataset("truncated",data=done)
    for name in ("obs","next_obs"):
     parent=g.create_group(name)
     for key,width in zip(KEYS,WIDTHS):parent.create_dataset(key,data=np.full((n,width),(seed-10000)/100+pi*.1+(name=="next_obs")*.001,np.float32))
def main():
 subprocess.run([sys.executable,str(HERE/"validate_temporal_alignment.py")],check=True)
 with tempfile.TemporaryDirectory() as d:
  root=Path(d);data=root/"data";out=root/"out";fixture(data)
  subprocess.run([sys.executable,str(HERE/"train_stage2_2.py"),"--dataset-root",str(data),"--output-root",str(out),"--device","cpu","--mode","both","--max-updates","3","--run-id","smoke"],check=True)
  run=out/"smoke"
  for group in ("rnn_q","multi_q"):
   assert (run/group/"checkpoints"/"best.pth").is_file();assert (run/group/"final_validation.json").is_file();assert (run/group/"performance.json").is_file()
  ratios=json.loads((run/"multi_q"/"sampling_audit.json").read_text())["ratios"];assert max(ratios.values())-min(ratios.values())<1e-12
 print(json.dumps({"status":"PASS","scope":"synthetic CPU temporal + rnn_q/multi_q smoke"}))
if __name__=="__main__":main()
