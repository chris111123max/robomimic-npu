"""Persistent process-per-environment robosuite pool for Stage4 collection."""
from __future__ import annotations
import multiprocessing as mp
import os
import random
import traceback
import numpy as np


def _success(env):
    value=env.is_success()
    if isinstance(value,dict):return bool(value.get("task",any(bool(item) for item in value.values())))
    return bool(value)


def _flatten(observation,keys,shapes):
    values=[]
    for key in keys:
        value=np.asarray(observation[key]);expected=tuple(shapes[key])
        if value.shape!=expected:raise RuntimeError(f"Observation {key} shape {value.shape} != {expected}")
        values.append(value.reshape(-1))
    result=np.concatenate(values).astype(np.float32)
    if result.shape!=(59,):raise RuntimeError(f"Flattened observation shape {result.shape} != (59,)")
    return result


def _seed_env(env,seed):
    random.seed(seed);np.random.seed(seed);seen=set();current=env
    while current is not None and id(current) not in seen:
        seen.add(id(current));method=getattr(current,"seed",None)
        if callable(method):
            try:method(int(seed))
            except (TypeError,AttributeError,NotImplementedError):pass
        current=getattr(current,"env",None)


THREAD_ENV=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS")


def _worker(connection,teacher_checkpoint,keys,shapes,horizon,terminate_on_success,seed,cpu_id):
    env=None
    try:
        for name in THREAD_ENV:os.environ[name]="1"
        if cpu_id is not None:
            if not hasattr(os,"sched_setaffinity"):raise RuntimeError("CPU affinity requested but os.sched_setaffinity is unavailable")
            os.sched_setaffinity(0,{int(cpu_id)})
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.obs_utils as ObsUtils
        checkpoint=FileUtils.maybe_dict_from_checkpoint(ckpt_path=teacher_checkpoint);config,_=FileUtils.config_from_checkpoint(ckpt_dict=checkpoint);ObsUtils.initialize_obs_utils_with_config(config)
        env,_=FileUtils.env_from_checkpoint(ckpt_dict=checkpoint,render=False,render_offscreen=False,verbose=False);_seed_env(env,seed);steps=0
        connection.send(("ready",{"pid":os.getpid(),"cpu_id":cpu_id,"affinity":sorted(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else None}))
        while True:
            command,payload=connection.recv()
            if command=="reset":
                _seed_env(env,int(payload));observation=env.reset();steps=0;connection.send(("result",_flatten(observation,keys,shapes)))
            elif command=="reset_to":
                observation=env.reset_to(payload["state"]);steps=int(payload["steps"]);connection.send(("result",_flatten(observation,keys,shapes)))
            elif command=="step":
                observation,reward,raw_done,_=env.step(np.asarray(payload,np.float32));steps+=1;success=_success(env);terminated=bool(raw_done);truncated=bool(not raw_done and ((terminate_on_success and success) or steps>=horizon));done=terminated or truncated
                connection.send(("result",(_flatten(observation,keys,shapes),float(reward),done,{"success":success,"terminated":terminated,"truncated":truncated})))
            elif command=="get_state":connection.send(("result",env.get_state()))
            elif command=="close":connection.send(("closed",None));break
            else:raise RuntimeError(f"Unknown Stage4 environment command: {command}")
    except (EOFError,KeyboardInterrupt):pass
    except BaseException:
        try:connection.send(("error",traceback.format_exc()))
        except BaseException:pass
    finally:
        if env is not None:
            close=getattr(getattr(env,"env",env),"close",None)
            if callable(close):close()
        connection.close()


class Stage4ParallelEnvPool:
    def __init__(self,teacher_checkpoint,keys,shapes,num_envs,horizon,terminate_on_success,seed,start_method="forkserver",startup_timeout=300,step_timeout=120,cpu_ids=None):
        self.num_envs=int(num_envs);self.step_timeout=float(step_timeout);self.connections=[];self.processes=[]
        self.cpu_ids=None if cpu_ids is None else list(map(int,cpu_ids))
        if self.cpu_ids is not None and len(self.cpu_ids)!=self.num_envs:raise RuntimeError(f"Need {self.num_envs} CPU IDs, got {len(self.cpu_ids)}")
        if start_method not in mp.get_all_start_methods():raise RuntimeError(f"Unavailable multiprocessing method {start_method}")
        context=mp.get_context(start_method)
        try:
            for worker in range(self.num_envs):
                cpu_id=None if self.cpu_ids is None else self.cpu_ids[worker]
                parent,child=context.Pipe();process=context.Process(target=_worker,args=(child,teacher_checkpoint,keys,shapes,int(horizon),bool(terminate_on_success),int(seed)+worker,cpu_id),name=f"stage4-env-{worker:02d}",daemon=True);process.start();child.close();self.connections.append(parent);self.processes.append(process);ready=self._receive(worker,"ready",startup_timeout);print(f"Stage4 environment worker {worker+1:02d}/{self.num_envs:02d} ready (pid={process.pid}, cpu={ready['cpu_id']})",flush=True)
        except BaseException:self.close(force=True);raise

    def _receive(self,worker,expected="result",timeout=None):
        connection=self.connections[worker];process=self.processes[worker];timeout=self.step_timeout if timeout is None else timeout
        if not connection.poll(timeout):raise TimeoutError(f"Stage4 env worker {worker} timeout; alive={process.is_alive()} exit={process.exitcode}")
        status,payload=connection.recv()
        if status=="error":raise RuntimeError(f"Stage4 env worker {worker} failed:\n{payload}")
        if status!=expected:raise RuntimeError(f"Stage4 env worker {worker}: {status} != {expected}")
        return payload

    def reset(self,workers,seeds):
        for worker,seed in zip(workers,seeds):self.connections[worker].send(("reset",int(seed)))
        return {worker:self._receive(worker) for worker in workers}

    def reset_to(self,states,steps):
        workers=sorted(states)
        for worker in workers:self.connections[worker].send(("reset_to",{"state":states[worker],"steps":steps[worker]}))
        return {worker:self._receive(worker) for worker in workers}

    def step(self,workers,actions):
        for worker,action in zip(workers,actions):self.connections[worker].send(("step",action))
        return {worker:self._receive(worker) for worker in workers}

    def get_states(self,workers):
        for worker in workers:self.connections[worker].send(("get_state",None))
        return {worker:self._receive(worker) for worker in workers}

    @property
    def worker_pids(self):return [process.pid for process in self.processes if process.is_alive()]

    def live_worker_count(self):return sum(process.is_alive() for process in self.processes)

    def worker_cpu_seconds(self):
        ticks=float(os.sysconf("SC_CLK_TCK"));total=0.0
        for pid in self.worker_pids:
            try:
                fields=open(f"/proc/{pid}/stat").read().split();total+=(int(fields[13])+int(fields[14]))/ticks
            except (FileNotFoundError,ProcessLookupError):pass
        return total

    def close(self,force=False):
        for connection,process in zip(getattr(self,"connections",[]),getattr(self,"processes",[])):
            if process.is_alive() and not force:
                try:connection.send(("close",None))
                except (OSError,EOFError):pass
        for worker,process in enumerate(getattr(self,"processes",[])):
            if process.is_alive() and not force:
                try:self._receive(worker,"closed",5)
                except BaseException:pass
            process.join(5)
            if process.is_alive():process.terminate();process.join(5)
        for connection in getattr(self,"connections",[]):connection.close()
        self.connections=[];self.processes=[]
