"""Observational fixed-data and four-seed diagnostics; restore learner RNG."""
import json,sys
from pathlib import Path
import numpy as np
import torch
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'actor_init_anchor'))
import run_anchor as A
from run_anchor_closed_loop import BatchedGMMExecutor,build_env,close_env,reset_seed,success,seed_all
REG=json.loads((H/'preregistration.json').read_text());SEEDS=tuple(REG['seeds']);CACHE={}

def write(p,d):
 p=Path(p);assert not p.exists(),p;p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def evaluate_one(agent,out,step):
 reference_path=H.parent/'actor_init_anchor/results_20261003/closed_loop_steps.jsonl'
 refs={seed:[] for seed in SEEDS}
 for line in reference_path.read_text().splitlines():
  r=json.loads(line)
  if r['branch']=='ORIGINAL_140K':refs[r['seed']].append(r)
 envs=[];actor=agent.actor;mode=actor.training;traces=[[] for _ in SEEDS]
 try:
  for i in range(4):
   print(json.dumps({'event':'diagnostic_env_start','step':step,'index':i}),flush=True)
   envs.append(build_env(Path(agent.config['expert_dataset'])))
  observations=[reset_seed(env,s) for env,s in zip(envs,SEEDS)];seed_all(SEEDS[-1])
  scale=agent.action_scale.reshape(14);offset=agent.action_offset.reshape(14);scale_np=scale.cpu().numpy()
  ex=BatchedGMMExecutor(actor,scale,offset,4,horizon=10);active=[True]*4;won=[False]*4;lengths=[0]*4;returns=[0.]*4;errors=[None]*4
  actor.eval()
  for t in range(700):
   actions=ex.actions_for(list(range(4)),observations,0.,None,None)
   for i,env in enumerate(envs):
    if not active[i]:continue
    obs=observations[i];action=actions[i];r={'step':step,'seed':SEEDS[i],'timestep':t,'action':action.tolist(),'eef0':np.asarray(obs['robot0_eef_pos']).reshape(-1).tolist(),'eef1':np.asarray(obs['robot1_eef_pos']).reshape(-1).tolist()}
    if t<len(refs[SEEDS[i]]):
     old=refs[SEEDS[i]][t];r.update(normalized_action_deviation=float(np.linalg.norm((action-np.asarray(old['action']))/scale_np)),arm0_eef_distance=float(np.linalg.norm(np.asarray(r['eef0'])-old['eef0'])),arm1_eef_distance=float(np.linalg.norm(np.asarray(r['eef1'])-old['eef1'])))
    try:
     nxt,reward,done,_=env.step(action);lengths[i]=t+1;returns[i]+=float(reward);won[i]=bool(success(env));observations[i]=nxt
     r.update(success=won[i],reward=float(reward),sim_error=False);active[i]=not(won[i] or done or lengths[i]>=700)
    except Exception as er:
     errors[i]=repr(er);active[i]=False;r.update(sim_error=True,error=repr(er))
    traces[i].append(r)
   if t%100==0:print(json.dumps({'event':'diagnostic_eval_progress','env_step':step,'t':t,'active':sum(active)}),flush=True)
   if not any(active):break
  episodes=[]
  for i,s in enumerate(SEEDS):
   episodes.append({'seed':s,'success':won[i],'length':lengths[i],'return':returns[i],'sim_error':errors[i],'first_action_divergence_gt_1e_3':next((r['timestep'] for r in traces[i] if r.get('normalized_action_deviation',0)>1e-3),None),'first_eef_divergence_gt_1mm':next((r['timestep'] for r in traces[i] if max(r.get('arm0_eef_distance',0),r.get('arm1_eef_distance',0))>1e-3),None),'selected_steps':[r for r in traces[i] if r['timestep'] in (0,1,2,3,5,9,10,20,50,100,200,300,400,500,600)]})
  tracefile=out/f'trajectory_{step:07d}.jsonl';assert not tracefile.exists()
  with tracefile.open('w') as f:
   for ts in traces:
    for r in ts:f.write(json.dumps(r,allow_nan=False)+'\n')
  return {'success_count':sum(won),'count':4,'mean_length':float(np.mean(lengths)),'sim_error_count':sum(x is not None for x in errors),'episodes':episodes,'trajectory':str(tracefile),'reference_trajectory':str(reference_path)}
 finally:
  for i,env in enumerate(envs):close_env(env);print(json.dumps({'event':'diagnostic_env_closed','step':step,'index':i}),flush=True)
  actor.train(mode)

def testing_milestone(agent,config,step,group_dir,torch_arg):
 out=Path(group_dir)/'preservation_diagnostics';out.mkdir(exist_ok=True)
 target=out/f'step_{step:07d}.json'
 if target.exists():return
 # Full learner RNG preservation, including diagnostic simulator initialization.
 import train_stage3_v6_vector as P
 rng=P.rng_state(torch)
 mode=agent.actor.training
 try:
  assert A.T.module_hash(agent.preserve_reference)==agent.reference_hash
  if 'probe' not in CACHE:
   prev=H.parent/'actor_temporal_supervision/results_20261003'
   cfg,data,original,critic,scale,offset,obs,contexts,ck,meta=A.setup(Path(REG['target_run']),prev,agent.device)
   critic.eval().requires_grad_(False)
   baseline=dict(np.load(H.parent/'actor_init_anchor/results_20261003/reference_outputs.npz'))
   initial={n:p.detach().cpu().clone() for n,p in agent.preserve_reference.named_parameters()}
   CACHE['probe']=(critic,scale,offset,obs,contexts,baseline,initial)
  critic,scale,offset,obs,contexts,baseline,initial=CACHE['probe']
  outputs=A.T.evaluate(agent.actor,obs,critic,contexts,scale,offset)
  metrics=A.extra_metrics(outputs,baseline,agent.actor,initial)
  if step==140000:
   assert A.T.module_hash(agent.actor)==agent.reference_hash
   prior=json.loads((H.parent/'actor_init_anchor/results_20261003/closed_loop.json').read_text())
   closed={**prior['branches']['ORIGINAL_140K'],'reused_prior_140K_identical_actor':True,'source':str(H.parent/'actor_init_anchor/results_20261003/closed_loop.json')}
  else:closed=evaluate_one(agent,out,step)
  result={'testing_only':True,'branch':agent.branch,'env_steps':step,'actor_updates':agent.actor_updates,'critic_updates':agent.critic_updates,'fixed_140K_critic_probe':metrics,'closed_loop':closed,'reference_hash_unchanged':A.T.module_hash(agent.preserve_reference)==agent.reference_hash}
  np.savez_compressed(out/f'outputs_{step:07d}.npz',**outputs)
  write(target,result)
  print('TEST_MILESTONE '+json.dumps({'branch':agent.branch,'env_steps':step,'actor_updates':agent.actor_updates,'success':closed['success_count'],'execution_drift':metrics['execution_drift'],'fixed_score_gain':metrics['RL_improvement'],'sim_errors':closed['sim_error_count']}),flush=True)
  if closed['sim_error_count']:raise RuntimeError('Diagnostic simulator error; do not continue stage')
 finally:
  agent.actor.train(mode);agent.preserve_reference.eval();P.restore_rng(rng,torch)
