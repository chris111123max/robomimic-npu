"""Spawn-based staggered MuJoCo workers for the progressive experiment only."""
from __future__ import annotations
import multiprocessing as mp,os,time,traceback
import numpy as np

def _worker(conn,env_id,dataset,initial_seed):
    os.environ.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
    env=None
    try:
        from stage3_new_evaluation import build_env,close_env,mujoco_fatal_error_type,reset_seed,success
        env=build_env(dataset);obs=reset_seed(env,initial_seed);low,high=env.action_spec
        conn.send(("READY",obs,np.asarray(low,np.float32),np.asarray(high,np.float32)))
        while True:
            command,payload=conn.recv()
            if command=="step":
                try:
                    obs,reward,done,info=env.step(payload);conn.send(("OK",obs,float(reward),bool(done),bool(success(env)),info))
                except mujoco_fatal_error_type() as error:conn.send(("FATAL",str(error)))
            elif command=="reset":conn.send(("OK",reset_seed(env,int(payload))))
            elif command=="rebuild":
                close_env(env);env=build_env(dataset);conn.send(("OK",reset_seed(env,int(payload))))
            elif command=="close":break
            else:raise RuntimeError(f"Unknown worker command {command}")
    except BaseException as error:
        try:conn.send(("ERROR",env_id,repr(error),traceback.format_exc()))
        except BaseException:pass
    finally:
        if env is not None:
            try:close_env(env)
            except BaseException:pass
        conn.close()

class StaggeredVectorEnv:
    def __init__(self,dataset,num_envs,seed_base,delay=.5,timeout=120.,start_method="spawn"):
        self.num_envs=int(num_envs);self.ctx=mp.get_context(start_method);self.processes=[];self.connections=[];self.initial_observations=[];self.vector_steps=0
        try:
            for env_id in range(self.num_envs):
                seed=int(seed_base)+env_id;print(f"[ENV STARTUP] {env_id+1:02d}/{self.num_envs} STARTING seed={seed}",flush=True);parent,child=self.ctx.Pipe();process=self.ctx.Process(target=_worker,args=(child,env_id,dataset,seed),name=f"stage3-env-{env_id:02d}");process.start();child.close()
                if not parent.poll(float(timeout)):raise TimeoutError(f"env_id={env_id} seed={seed} startup exceeded {timeout}s")
                message=parent.recv()
                if message[0]!="READY":raise RuntimeError(f"env_id={env_id} seed={seed} startup failed: {message}")
                _,obs,low,high=message;self.processes.append(process);self.connections.append(parent);self.initial_observations.append(obs)
                if env_id==0:self.action_low,self.action_high=low,high
                elif not np.array_equal(low,self.action_low) or not np.array_equal(high,self.action_high):raise RuntimeError("Vector environments have different action bounds")
                print(f"[ENV STARTUP] {env_id+1:02d}/{self.num_envs} READY",flush=True)
                if env_id+1<self.num_envs:time.sleep(float(delay))
            print(f"[ENV STARTUP] all {self.num_envs} environments ready",flush=True)
        except BaseException:self.close(force=True);raise
    def step(self,actions,env_ids=None):
        ids=list(range(self.num_envs)) if env_ids is None else list(env_ids)
        for env_id,action in zip(ids,actions):self.connections[env_id].send(("step",action))
        results=[]
        for env_id in ids:results.append((env_id,self.connections[env_id].recv()))
        self.vector_steps+=1;return results
    def reset(self,env_id,seed,rebuild=False):
        self.connections[env_id].send(("rebuild" if rebuild else "reset",int(seed)));message=self.connections[env_id].recv()
        if message[0]!="OK":raise RuntimeError(f"env_id={env_id} reset failed: {message}")
        return message[1]
    def close(self,force=False):
        for conn,process in zip(self.connections,self.processes):
            if process.is_alive() and not force:
                try:conn.send(("close",None))
                except BaseException:pass
        for process in self.processes:process.join(timeout=5)
        for process in self.processes:
            if process.is_alive():process.terminate();process.join(timeout=5)
        for conn in self.connections:
            try:conn.close()
            except BaseException:pass
