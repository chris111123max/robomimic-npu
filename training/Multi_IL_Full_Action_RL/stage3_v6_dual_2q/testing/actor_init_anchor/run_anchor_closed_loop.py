"""Four fixed success seeds, production executor, retained in-memory Actor clones."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
for f in (ROOT/"stage3_v3_rgmm_td3",ROOT/"stage3_new_sac"): sys.path.insert(0,str(f))
from stage3_v3_actor import BatchedGMMExecutor
from stage3_new_evaluation import build_env,close_env,reset_seed,success,seed_all
SEEDS=(20008,20002,20005,20007)

def run_closed_loop(run,out,original,clones,scale,offset,device):
 path=out/"closed_loop.json"; tracepath=out/"closed_loop_steps.jsonl"
 if path.exists() or tracepath.exists(): raise FileExistsError(path)
 actors={"ORIGINAL_140K":original,**clones}
 envs=[]; results={}; reference={}; scale_np=scale.detach().cpu().numpy()
 try:
  for branch,actor in actors.items():
   for i in range(4):
    print(json.dumps({"event":"env_start","branch":branch,"index":i}),flush=True)
    envs.append(build_env(Path("/data/home/3220251075/lerobot_workspace/datasets/transport/PH/low_dim_v15.hdf5")))
    print(json.dumps({"event":"env_ready","branch":branch,"index":i}),flush=True)
   actor.eval(); executor=BatchedGMMExecutor(actor,scale,offset,4,horizon=10)
   observations=[reset_seed(env,seed) for env,seed in zip(envs,SEEDS)]
   seed_all(SEEDS[-1])
   active=[True]*4; lengths=[0]*4; won=[False]*4; totals=[0.]*4; errors=[None]*4; traces=[[] for _ in range(4)]
   print(json.dumps({"event":"closed_loop_start","branch":branch}),flush=True)
   for t in range(700):
    # Always sample all four slots: identical categorical/Gaussian RNG consumption
    # across branches even after one environment terminates. Inactive actions discarded.
    actions=executor.actions_for(list(range(4)),observations,0.0,None,None)
    for i,env in enumerate(envs):
     if not active[i]: continue
     obs=observations[i]; action=actions[i]
     row={"branch":branch,"seed":SEEDS[i],"timestep":t,"action":action.tolist(),
          "eef0":np.asarray(obs["robot0_eef_pos"]).reshape(-1).tolist(),"eef1":np.asarray(obs["robot1_eef_pos"]).reshape(-1).tolist()}
     if branch!="ORIGINAL_140K" and t<len(reference[SEEDS[i]]):
      b=reference[SEEDS[i]][t]
      row["normalized_action_deviation"]=float(np.linalg.norm((action-np.asarray(b["action"]))/scale_np))
      row["arm0_eef_distance"]=float(np.linalg.norm(np.asarray(row["eef0"])-np.asarray(b["eef0"])))
      row["arm1_eef_distance"]=float(np.linalg.norm(np.asarray(row["eef1"])-np.asarray(b["eef1"])))
     try:
      nxt,reward,done,_=env.step(action)
      lengths[i]=t+1; totals[i]+=float(reward); won[i]=bool(success(env)); observations[i]=nxt
      row.update(reward=float(reward),success=won[i],sim_error=False)
      active[i]=not (won[i] or done or lengths[i]>=700)
     except Exception as error:
      # Keep real simulator failures visible, no retries or parameter changes.
      errors[i]=repr(error); active[i]=False; row.update(sim_error=True,error=repr(error))
     traces[i].append(row)
    if t%100==0: print(json.dumps({"event":"closed_loop_progress","branch":branch,"step":t,"active":sum(active)}),flush=True)
    if not any(active): break
   episodes=[]
   for i,seed in enumerate(SEEDS):
    if branch=="ORIGINAL_140K": reference[seed]=traces[i]
    action_div=[r["timestep"] for r in traces[i] if r.get("normalized_action_deviation",0)>1e-3]
    state_div=[r["timestep"] for r in traces[i] if max(r.get("arm0_eef_distance",0),r.get("arm1_eef_distance",0))>1e-3]
    episodes.append({"seed":seed,"success":won[i],"length":lengths[i],"return":totals[i],"sim_error":errors[i],
                     "first_action_divergence_gt_1e_3":min(action_div) if action_div else None,
                     "first_eef_divergence_gt_1mm":min(state_div) if state_div else None,
                     "selected_steps":[r for r in traces[i] if r["timestep"] in (0,1,2,3,5,9,10,20,50,100,200,300,400,500,600)]})
   results[branch]={"episodes":episodes,"success_count":sum(won),"count":4,"mean_length":float(np.mean(lengths)),"sim_error_count":sum(e is not None for e in errors)}
   with tracepath.open("a") as f:
    for rows in traces:
     for r in rows: f.write(json.dumps(r,allow_nan=False)+"\n")
   print(json.dumps({"event":"closed_loop_done","branch":branch,"success":sum(won),"mean_length":float(np.mean(lengths)),"errors":errors}),flush=True)
   for i,env in enumerate(envs):
    close_env(env); print(json.dumps({"event":"env_closed","branch":branch,"index":i}),flush=True)
   envs.clear()
  path.write_text(json.dumps({"seeds":SEEDS,"horizon":700,"executor":"production BatchedGMMExecutor, low-noise GMM categorical sampling, no external noise, horizon reset 10",
                             "rng":"reset identically after 4 seed resets; all 4 slots sampled every timestep so early termination does not change consumption",
                             "branches":results,"trace_path":str(tracepath)},indent=2,allow_nan=False)+"\n")
 finally:
  for i,env in enumerate(envs):
   close_env(env); print(json.dumps({"event":"env_closed","index":i}),flush=True)

if __name__=='__main__':
 import argparse,copy
 sys.path.insert(0,str(Path(__file__).resolve().parent))
 import run_anchor as A
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='npu:0');args=ap.parse_args()
 out=args.output.resolve();device=A.T.resolve_device(args.device)
 data=json.loads((out/'offline_results.json').read_text())
 actor,rollout,_=A.T.load_exact_actor(args.run/'shared/bc_rnn_gmm_source.pth',device)
 ready=torch.load(args.run/'mean2q/multi_q/checkpoints/critic_ready.pth',map_location='cpu',weights_only=False)
 actor.load_state_dict(ready['actor'],strict=True)
 clones={}
 for b in data['closed_loop_selection']['branches'][1:]:
  ck=torch.load(out/f'{b}_testing_actor_1000.pth',map_location='cpu',weights_only=False)
  assert ck['testing_only'] and ck['source_env_steps']==140000 and ck['actor_virtual_updates']==1000
  clones[b]=copy.deepcopy(actor);clones[b].load_state_dict(ck['actor'],strict=True)
 scale=torch.as_tensor(rollout.action_normalization_stats['actions']['scale'],dtype=torch.float32,device=device).reshape(14)
 offset=torch.as_tensor(rollout.action_normalization_stats['actions']['offset'],dtype=torch.float32,device=device).reshape(14)
 run_closed_loop(args.run,out,actor,clones,scale,offset,device)
 print('COMPLETE; FORMAL TRAINING REMAINS STOPPED',flush=True)
