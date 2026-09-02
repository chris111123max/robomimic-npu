#!/usr/bin/env python3
"""Stage4-v2: 16-process synchronous collector with learning-relative schedule."""
import argparse,csv,json,os,sys,time
from pathlib import Path
for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):os.environ[name]="1"
import numpy as np,torch
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent;V1=PROJECT/"stage4_rsac_online_finetuning"
for path in (V1,PROJECT/"stage1_rollout_collection",PROJECT/"stage3_actor_initialization"):
 if str(path) not in sys.path:sys.path.insert(0,str(path))
from stage4_core import OnlineSequenceReplay,Stage4SAC,append_csv,atomic_json,cpu_tree,initialize_models,model_audit,phase_at,read_json,seed_all,select_device,state_hash
from train_stage4_group import SHAPES,add_finished_episode,batched_actor_actions,build_environment,evaluate,new_episode,save_checkpoint,unflatten
from stage4_parallel_env import Stage4ParallelEnvPool
from collect_multi_il_rollouts import progress_observation_schema
from evaluate_stage3_actor import close_env,extract_transport_progress

def args():
 p=argparse.ArgumentParser();p.add_argument("--group",choices=("rnn_only_critic","multi_il_critic"),required=True);p.add_argument("--device",required=True);p.add_argument("--config",required=True);p.add_argument("--run-dir",required=True);p.add_argument("--cpu-affinity-file",required=True);p.add_argument("--smoke-test",action="store_true");return p.parse_args()
def phase(config,learning_start,env_step):
 if learning_start is None:return {"phase":"replay_warmup","actor_lr":0.0,"critic_lr":float(config["critic_phase1_lr"]),"actor_updates":False}
 return phase_at(max(0,int(env_step)-int(learning_start)),config)
def context(pool,contexts,hidden,counters,next_episode,learning_start,completed_updates,actor_updates,global_episodes):
 workers=sorted(contexts)
 return {"collector_mode":"sync","parallel_envs":pool.num_envs,"active_workers":workers,"env_states":pool.get_states(workers),"contexts":contexts,"actor_hidden":cpu_tree(hidden),"actor_counters":counters.tolist(),"next_episode":next_episode,"learning_start_env_step":learning_start,"completed_updates":completed_updates,"actor_updates":actor_updates,"global_completed_episodes":global_episodes,"resume_boundary_requirement":"quiescent synchronous vector-step boundary; no env.step in flight"}
def main():
 a=args();torch.set_num_threads(1)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:
  if torch.get_num_interop_threads()!=1:raise
 config=read_json(a.config);config.update(group=a.group,device=a.device)
 if a.smoke_test:config.update(total_env_steps=96,minimum_replay_size=16,actor_freeze_steps=32,actor_warmup_end=512,evaluation_interval=32,evaluation_steps=[0,16,48,96],evaluation_horizon=1,evaluation_seed_end=10081,replay_capacity=1000,batch_size=2,checkpoint_steps=[0,16,48,96])
 if config["collector_mode"]!="sync" or int(config["parallel_envs"])!=16:raise RuntimeError("Stage4-v2 requires sync collector and 16 env/group")
 device=select_device(a.device);seed_all(config["seed"]);group=Path(a.run_dir).resolve()/a.group;group.mkdir(parents=True,exist_ok=True)
 for name in ("checkpoints","evaluations","debug_nan","replay"): (group/name).mkdir(exist_ok=True)
 actor,payload,critic,target,source=initialize_models(a.group,config,device);engine=Stage4SAC(actor,critic,target,config,device);replay=OnlineSequenceReplay(config);audit=model_audit(a.group,actor,critic,target,source,config);atomic_json(group/"initialization_audit.json",audit)
 affinity=read_json(a.cpu_affinity_file);cpu_ids=affinity["rnn_cpu_ids" if a.group=="rnn_only_critic" else "multi_cpu_ids"][:16];config.update(cpu_affinity_file=str(Path(a.cpu_affinity_file).resolve()),cpu_affinity_ids=cpu_ids,num_envs=16,critic_only_learning_steps=5000 if not a.smoke_test else 32,actor_warmup_learning_start=5000 if not a.smoke_test else 32,actor_warmup_learning_end=20000 if not a.smoke_test else 512,fixed_alpha=.001)
 atomic_json(group/"config.json",config);atomic_json(group/"learning_start_audit.json",{"minimum_replay_size":config["minimum_replay_size"],"sequence_length":config["sequence_length"],"replay_commit_mode":"complete_episode_commit","replay_size_definition":"sum of transitions in committed complete episodes","first_sample_condition":"replay.transitions >= minimum_replay_size AND replay contains at least one valid recurrent sequence start","first_update_condition":"first_sample_condition and initialized finite Actor/Critic/optimizers","partial_episode_transitions_sampleable":False})
 schema_env=build_environment(payload);schema=progress_observation_schema(schema_env,payload["observation_keys"],SHAPES);close_env(schema_env)
 pool=Stage4ParallelEnvPool(payload["teacher_checkpoint"],payload["observation_keys"],SHAPES,16,config["evaluation_horizon"],config["terminate_on_success"],config["seed"],config["parallel_start_method"],config["parallel_startup_timeout_seconds"],config["parallel_step_timeout_seconds"],cpu_ids)
 contexts={};hidden=(torch.zeros(2,16,400,device=device),torch.zeros(2,16,400,device=device));counters=np.zeros(16,np.int64);env_step=0;next_episode=1;completed_updates=0;actor_updates=0;global_episodes=0;learning_start=None;first_critic=None;first_actor=None;critic_only_updates=0;critic_only_checked=False;evaluations=[];best=(-1.,-1.,float("-inf"));vector_batch_sizes=[];started=time.perf_counter()
 evaluation_steps=set(map(int,config["evaluation_steps"]));checkpoint_steps=set(map(int,config["checkpoint_steps"]))
 try:
  report=evaluate(actor,payload,config,device,0,group/"evaluations"/"step_00000000.json",pool,schema,contexts);evaluations.append(report);append_csv(group/"evaluation_metrics.csv",{k:v for k,v in report.items() if k!="episodes"});initial=context(pool,contexts,hidden,counters,next_episode,learning_start,completed_updates,actor_updates,global_episodes);save_checkpoint(group/"checkpoints"/"step_00000000.pth",engine,replay,config,0,next_episode,payload,initial,report);save_checkpoint(group/"checkpoints"/"best_success.pth",engine,replay,config,0,next_episode,payload,initial,report);best=(report["success_rate"],report["mean_progress"],0)
  while env_step<int(config["total_env_steps"]):
   empty=[worker for worker in range(16) if worker not in contexts]
   if empty:
    seeds=[]
    for worker in empty:seed=int(config["seed"])+next_episode;seeds.append(seed);contexts[worker]=new_episode(next_episode,None);contexts[worker]["environment_seed"]=seed;next_episode+=1;counters[worker]=0;hidden[0][:,worker].zero_();hidden[1][:,worker].zero_()
    observations=pool.reset(empty,seeds)
    for worker in empty:contexts[worker]["observation"]=observations[worker]
   milestones=set(evaluation_steps)|set(checkpoint_steps)|{int(config["total_env_steps"])}
   if learning_start is not None:milestones|={learning_start+int(config["actor_freeze_steps"]),learning_start+int(config["actor_warmup_end"])}
   future=sorted(step for step in milestones if env_step<step<=int(config["total_env_steps"]));boundary=future[0];workers=sorted(contexts)[:min(len(contexts),boundary-env_step)]
   observations={worker:contexts[worker]["observation"] for worker in workers};vector_batch_sizes.append(len(workers));before_step=env_step;actions=batched_actor_actions(actor,observations,hidden,counters,workers,device,False)
   if not np.isfinite(actions).all():raise RuntimeError("Non-finite training action")
   results=pool.step(workers,actions);finished=[]
   for worker,action in zip(workers,actions):
    c=contexts[worker];state=c["observation"];next_state,reward,done,info=results[worker];progress=extract_transport_progress(unflatten(next_state,payload["observation_keys"]),schema);c["states"].append(state);c["actions"].append(action);c["rewards"].append(float(reward));c["dones"].append(float(done));c["terminated"].append(float(info["terminated"]));c["truncated"].append(float(info["truncated"]));c["next_states"].append(next_state);c["observation"]=next_state;c["return"]+=float(reward);c["length"]+=1;c["success"]|=bool(info["success"]);c["trash_ever"]|=bool(progress["trash_in_trash_bin"]);c["payload_ever"]|=bool(progress["payload_in_target_bin"]);env_step+=1
    if done:finished.append((worker,c));del contexts[worker];counters[worker]=0;hidden[0][:,worker].zero_();hidden[1][:,worker].zero_()
   if env_step-before_step!=len(workers):raise RuntimeError("Synchronous vector step did not count one transition per worker")
   if env_step==int(config["total_env_steps"]):
    for worker in sorted(contexts):
     c=contexts[worker]
     if c["states"]:c["dones"][-1]=1.;c["terminated"][-1]=0.;c["truncated"][-1]=1.;finished.append((worker,c))
    contexts.clear()
   for _,c in finished:add_finished_episode(replay,c)
   learner_ready=replay.transitions>=int(config["minimum_replay_size"]) and bool(np.any(replay.buffer._valid_starts))
   if learner_ready and learning_start is None:learning_start=env_step;atomic_json(group/"learning_start.json",{"learning_start_env_step":learning_start,"replay_size":replay.transitions,"episodes":replay.episodes})
   learning_step=0 if learning_start is None else env_step-learning_start;current_phase=phase(config,learning_start,env_step);target_updates=0 if learning_start is None else int(np.floor(replay.transitions*float(config["updates_per_env_step"])));new_updates=max(0,target_updates-completed_updates);rows=[]
   if learner_ready:
    for _ in range(new_updates):
     row=engine.update(replay.sample(config["batch_size"],device),env_step,group/"debug_nan",phase_step=learning_step);rows.append(row);completed_updates+=1
     if first_critic is None:first_critic={"first_critic_update_env_step":env_step,"first_critic_update_learning_step":learning_step};atomic_json(group/"first_critic_update.json",first_critic)
     if row["actor_loss"] is not None:
      actor_updates+=1
      if first_actor is None:first_actor={"first_actor_update_env_step":env_step,"first_actor_update_learning_step":learning_step,"first_actor_update_lr":row["actor_lr"]};atomic_json(group/"first_actor_update.json",first_actor)
     if current_phase["phase"]=="critic_only":critic_only_updates+=1
   if current_phase["phase"]=="replay_warmup" and rows:raise RuntimeError("Optimizer update occurred during replay_warmup")
   if current_phase["phase"]=="critic_only" and any(row["actor_loss"] is not None for row in rows):raise RuntimeError("Actor updated during critic_only")
   if learning_start is not None and learning_step>=int(config["actor_freeze_steps"]) and not critic_only_checked:
    if critic_only_updates<=0:raise RuntimeError("Critic-only learning interval ended without Critic updates")
    critic_only_checked=True;atomic_json(group/"critic_only_summary.json",{"learning_start_env_step":learning_start,"critic_only_end_env_step":env_step,"critic_only_learning_transitions":learning_step,"critic_update_count":critic_only_updates,"actor_update_count":0})
   for worker,c in finished:
    global_episodes+=1;row=rows[-1] if rows else {};log={"group":a.group,"env_step":env_step,"learning_start_env_step":learning_start,"learning_step":learning_step,"global_completed_episode":global_episodes,"worker_id":worker,"episode_id":c["episode"],"environment_seed":c["environment_seed"],"phase":current_phase["phase"],"success":int(c["success"]),"episode_return":c["return"],"episode_length":c["length"],"replay_size":replay.transitions,"critic_updates":completed_updates,"actor_updates":actor_updates,"actor_lr":current_phase["actor_lr"],"critic_lr":current_phase["critic_lr"],"alpha":float(engine.algo.alpha_entropy)};append_csv(group/"training_metrics.csv",log);print(" ".join(f"{k}={v}" for k,v in log.items()),flush=True)
   if env_step in evaluation_steps:
    report=evaluate(actor,payload,config,device,env_step,group/"evaluations"/f"step_{env_step:08d}.json",pool,schema,contexts);evaluations.append(report);append_csv(group/"evaluation_metrics.csv",{k:v for k,v in report.items() if k!="episodes"});key=(report["success_rate"],report["mean_progress"],-env_step);saved=context(pool,contexts,hidden,counters,next_episode,learning_start,completed_updates,actor_updates,global_episodes)
    if key>best:save_checkpoint(group/"checkpoints"/"best_success.pth",engine,replay,config,env_step,next_episode,payload,saved,report);best=key
   if env_step in checkpoint_steps:saved=context(pool,contexts,hidden,counters,next_episode,learning_start,completed_updates,actor_updates,global_episodes);save_checkpoint(group/"checkpoints"/f"step_{env_step:08d}.pth",engine,replay,config,env_step,next_episode,payload,saved,evaluations[-1] if evaluations[-1]["env_step"]==env_step else None)
  if first_actor and first_actor["first_actor_update_learning_step"]<int(config["actor_freeze_steps"]):raise RuntimeError("Actor first update preceded learning-relative warmup")
  if a.smoke_test:
   if learning_start is None or first_critic is None or first_actor is None or critic_only_updates<=0:raise RuntimeError("Learning-start smoke did not exercise all required phases")
   if first_critic["first_critic_update_learning_step"]!=0:raise RuntimeError("Critic did not start at learning_step=0")
   if first_actor["first_actor_update_lr"]>=.1*float(config["actor_target_lr"]):raise RuntimeError("First Actor LR is not near the scaled smoke warmup origin")
   atomic_json(group/"sync_collector_smoke.json",{"status":"PASS","collector_mode":"sync","workers_ready":pool.live_worker_count(),"wait_all_barrier":True,"async_ready_queue_used":False,"dynamic_inference_used":False,"vector_batch_sizes":vector_batch_sizes,"full_batch_size":16,"env_step":env_step,"env_steps_per_second":env_step/(time.perf_counter()-started),"learning_start_env_step":learning_start,"first_critic_update":first_critic,"first_actor_update":first_actor,"critic_only_updates":critic_only_updates,"critic_only_actor_updates":0,"replay_sequence_boundary":"per-env complete episodes only","hidden_slots":16,"fixed_alpha":float(engine.algo.alpha_entropy)})
  summary={"status":"COMPLETE","collector_mode":"sync","num_envs":16,"minimum_replay_size":config["minimum_replay_size"],"replay_commit_mode":"complete_episode_commit","learning_start_env_step":learning_start,**(first_critic or {}),**(first_actor or {}),"critic_only_learning_steps":config["actor_freeze_steps"],"critic_only_update_count":critic_only_updates,"critic_only_actor_update_count":0,"absolute_20k_learning_step":None if learning_start is None else max(0,20000-learning_start),"absolute_40k_learning_step":None if learning_start is None else max(0,40000-learning_start),"total_absolute_env_steps":env_step,"gradient_updates":completed_updates,"actor_updates":actor_updates,"fixed_alpha":float(engine.algo.alpha_entropy),"automatic_entropy_tuning":False};atomic_json(group/"schedule_summary.json",summary);saved=context(pool,contexts,hidden,counters,next_episode,learning_start,completed_updates,actor_updates,global_episodes);save_checkpoint(group/"checkpoints"/"last.pth",engine,replay,config,env_step,next_episode,payload,saved,evaluations[-1]);atomic_json(group/"status.json",summary);print(json.dumps(summary,indent=2),flush=True)
 except BaseException as error:atomic_json(group/"status.json",{"status":"FAILED","group":a.group,"env_step":env_step,"learning_start_env_step":learning_start,"error":repr(error)});raise
 finally:pool.close()
if __name__=="__main__":main()
