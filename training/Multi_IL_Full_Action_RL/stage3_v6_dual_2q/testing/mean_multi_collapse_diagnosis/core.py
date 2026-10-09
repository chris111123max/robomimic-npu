"""Testing-only mean2q causal tools; all simulation uses four concurrent workers."""
import sys,json,copy,time
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
TEST=HERE.parent
RL=HERE.parents[2]
for d in (RL/'stage3_v5_rgmm_td3',RL/'stage3_v6_dual_2q',RL/'stage3_v3_rgmm_td3',RL/'stage3_new_sac',TEST/'actor_optimizer_diagnostics',TEST/'actor_collapse_diagnosis'):
 sys.path.insert(0,str(d))
from run_optimizer_step_probe import production_actor_loss,actor_outputs,policy_drift,resolve_device
from run_adam_dynamics_probe import build_batch_bank
from run_offline import a3_forward
from stage3_v5_actor import load_exact_actor,module_hash,obs_to_flat
from stage3_v6_agent import strict_stage2_load
from stage3_v5_vector_env import StaggeredVectorEnv
from stage3_v3_actor import BatchedGMMExecutor
from stage3_v5_schedule import CriticHandoff,HandoffState,TrainingState
RUN=Path('/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_readiness_v2_multi_mean_random_formal_20260929_130945')
SEEDS=(20008,20002,20005,20007)
DATASET='/data/home/3220251075/lerobot_workspace/datasets/transport/PH/low_dim_v15.hdf5'
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists(): raise FileExistsError(p)
 p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def setup():
 device=resolve_device('npu:0')
 ready=torch.load(RUN/'mean2q/multi_q/checkpoints/critic_ready.pth',map_location='cpu',weights_only=False)
 actor,roll,_=load_exact_actor(RUN/'shared/bc_rnn_gmm_source.pth',device)
 actor.load_state_dict(ready['actor'],strict=True)
 source=json.loads((RUN/'shared/stage2_source_manifest.json').read_text())
 critic,_=strict_stage2_load(source['multi_q']['checkpoint'],device)
 critic.load_state_dict(ready['q1_q2'],strict=True);critic.eval().requires_grad_(False)
 scale=torch.as_tensor(roll.action_normalization_stats['actions']['scale'],device=device,dtype=torch.float32).reshape(14)
 offset=torch.as_tensor(roll.action_normalization_stats['actions']['offset'],device=device,dtype=torch.float32).reshape(14)
 assert ready['env_steps']==140000 and ready['actor_updates']==0 and ready['critic_target_mode']=='mean2q'
 return device,ready,actor,critic,scale,offset

def schedule(cfg,updates):
 rows=[json.loads(l) for l in (RUN/'mean2q/multi_q/train_metrics.jsonl').read_text().splitlines()]
 points=sorted(set([(0,140000)]+[(int(r['actor_updates']),int(r['env_steps'])) for r in rows if r.get('actor_updates',0)>0]))
 monot=[]
 for u,e in points:
  if not monot or u>monot[-1][0] and e>=monot[-1][1]: monot.append((u,e))
 envs=np.rint(np.interp(np.arange(1,updates+1),[x[0] for x in monot],[x[1] for x in monot])).astype(int)
 scheduler=CriticHandoff(cfg,HandoffState(state=TrainingState.ACTOR_WARMUP,critic_ready=True,critic_ready_step=140000,actor_warmup_steps=140000,joint_rl_start_step=280000))
 lrs=[scheduler.schedule(int(e),cfg['critic_lr'])['actor_lr'] for e in envs]
 return envs,lrs

def qprobe(actor,reference,critic,data,device,scale,offset):
 raw=a3_forward(reference,actor,critic,data,device,scale.reshape(1,1,1,14),offset.reshape(1,1,1,14),700)
 d1=raw['q1_expected_current']-raw['q1_expected_init'];d2=raw['q2_expected_current']-raw['q2_expected_init']
 ai=raw['init_action_env'];ac=raw['current_action_env'];ar=raw['replay_action_env']
 q={}
 for label,mask in [('all',np.ones(len(d1),bool)),('success',data['a3_success'].astype(bool)),('failure',~data['a3_success'].astype(bool))]:
  q[label]={'count':int(mask.sum()),'q1_gain':float(d1[mask].mean()),'q2_gain':float(d2[mask].mean()),'qmean_gain':float(((d1+d2)/2)[mask].mean()),'both_positive_fraction':float(np.mean((d1[mask]>0)&(d2[mask]>0))), 'q1_positive_fraction':float(np.mean(d1[mask]>0)), 'q1_current':float(raw['q1_expected_current'][mask].mean()),'q2_current':float(raw['q2_expected_current'][mask].mean()),'twin_disagreement_current':float(np.abs(raw['q1_expected_current'][mask]-raw['q2_expected_current'][mask]).mean()),'twin_disagreement_reference':float(np.abs(raw['q1_expected_init'][mask]-raw['q2_expected_init'][mask]).mean()),'argmax_action_drift':float(np.linalg.norm((ac-ai)/scale.cpu().numpy(),axis=-1)[mask].mean()),'distance_to_replay_change':float((np.linalg.norm((ac-ar)/scale.cpu().numpy(),axis=-1)-np.linalg.norm((ai-ar)/scale.cpu().numpy(),axis=-1))[mask].mean())}
 return q,raw

def evaluate_four(branches,scale,offset,out):
 # One pool of exactly four spawned simulator processes, reused sequentially.
 vec=None;results={};contract={'parallel_envs_started':0,'parallel_envs_initialized':0,'parallel_envs_used':0,'parallel_envs_closed':0,'seeds':list(SEEDS)}
 try:
  vec=StaggeredVectorEnv(DATASET,4,20000,delay=.5,timeout=120,startup_parallelism=4,shared_memory=False)
  contract.update(parallel_envs_started=len(vec.processes),parallel_envs_initialized=len(vec.initial_observations),worker_pids=[p.pid for p in vec.processes])
  assert vec.alive_worker_count()==4 and len(vec.processes)==4
  for name,actors,alpha in branches:
   actors=[a.eval() for a in actors]
   executors=[BatchedGMMExecutor(a,scale,offset,4,horizon=10) for a in actors]
   initial=vec.reset_many(dict(enumerate(SEEDS)));obs=[initial[i] for i in range(4)]
   torch.manual_seed(20007);torch.npu.manual_seed_all(20007)
   active=[True]*4;traces=[[] for _ in range(4)];counts=[0]*4;won=[False]*4;totals=[0.]*4
   # Common RNG per policy: blending does not alter Actor recurrent state contract.
   for t in range(700):
    samples=[]
    rng=torch.get_rng_state();nrng=torch.npu.get_rng_state()
    for ex in executors:
     torch.set_rng_state(rng);torch.npu.set_rng_state(nrng)
     samples.append(ex.actions_for(list(range(4)),obs,0.,None,None))
    actions=samples[0] if len(samples)==1 else [(1-alpha)*a+alpha*b for a,b in zip(samples[0],samples[1])]
    ids=[i for i in range(4) if active[i]]
    before={i:obs_to_flat(obs[i]).copy() for i in ids}
    messages=vec.step([actions[i] for i in ids],ids)
    for i,m in messages:
     if m[0]!='OK': raise RuntimeError((name,i,m))
     _,nextobs,r,done,success,info=m
     traces[i].append({'seed':SEEDS[i],'timestep':t,'observation_flat':before[i].tolist(),'action':np.asarray(actions[i]).tolist(),'reward':r,'success':success,'terminated':bool(done or success),'truncated':bool(t==699 and not(done or success))})
     counts[i]+=1;totals[i]+=r;won[i]=bool(success);obs[i]=nextobs
     active[i]=not(done or success or t==699)
    if t%100==0: print(json.dumps({'event':'rollout','branch':name,'timestep':t,'active':sum(active)}),flush=True)
    if not any(active): break
   assert all(c>0 for c in counts)
   contract['parallel_envs_used']=4
   episodes=[]
   for i in range(4):
    ret=0.
    for row in reversed(traces[i]): ret=row['reward']+.99*ret;row['finite_mc_return']=ret
    episodes.append({'seed':SEEDS[i],'success':won[i],'length':counts[i],'return':totals[i],'sim_error':False,'start_mc_return':traces[i][0]['finite_mc_return']})
   results[name]={'episodes':episodes,'success_count':sum(won),'mean_length':float(np.mean(counts)),'sim_errors':0,'alpha':alpha}
   with (out/f'{name}_trajectories.jsonl').open('x') as f:
    for rs in traces:
     for r in rs: f.write(json.dumps(r,allow_nan=False)+'\n')
   print(json.dumps({'event':'evaluation_done','branch':name,**results[name]}),flush=True)
 finally:
  if vec is not None:
   vec.close();contract['parallel_envs_closed']=sum(not p.is_alive() and p.exitcode==0 for p in vec.processes)
   contract['worker_exitcodes']=[p.exitcode for p in vec.processes]
 contract['test_valid']=all(contract[k]==4 for k in ('parallel_envs_started','parallel_envs_initialized','parallel_envs_used','parallel_envs_closed'))
 dump(out/'closed_loop.json',{'contract':contract,'branches':results})
 assert contract['test_valid'],contract
 return results,contract
