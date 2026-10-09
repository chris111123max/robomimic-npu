"""Bounded diagnostics for the same V8 loop; all output/checkpoints are test-only."""
import copy,json,math,random,time
from pathlib import Path
import numpy as np
import torch
import stage3_v8_paths as paths
from stage3_v5_vector_env import StaggeredVectorEnv
from stage3_v5_actor import BatchedGMMExecutor,obs_to_flat,module_hash,flat_to_obs
from stage3_v5_history_critic import encode_replay_contexts,component_mean_q
from stage3_v5_diagnostics import isolated_training_rng
from stage3_v8_checkpoint import restore_checkpoint,source_payload
from stage3_v8_gmm_loss import good_loss
from stage3_v8_dependencies import audit_loaded_dependencies

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+"\n")

def vector_type(output):
    class TrackedVector(StaggeredVectorEnv):
        def __init__(self,*args,**kwargs):
            self.used=set()
            super().__init__(*args,**kwargs)
            assert self.num_envs==4 and len(self.initial_observations)==4
            self.record()
        def step_async(self,actions,env_ids=None):
            ids=list(range(self.num_envs)) if env_ids is None else list(env_ids)
            self.used.update(ids)
            return super().step_async(actions,ids)
        def record(self):
            write(Path(output)/"environment_audit.json",dict(
                initialized=len(self.initial_observations),used=len(self.used),
                closed=sum(not p.is_alive() for p in self.processes) if self._closed else 0,
                worker_pids=[p.pid for p in self.processes],
                worker_exitcodes=[p.exitcode for p in self.processes],
                live_workers=sum(p.is_alive() for p in self.processes),
                true_parallel_workers=len(self.processes),startup_parallelism=self.startup_parallelism))
        def close(self,*args,**kwargs):
            super().close(*args,**kwargs);self.record()
    return TrackedVector

def baseline(vector,actor,scale,offset,config,observations,contexts,executor,output,step):
    """Four pre-update baseline episodes reuse the existing four simulator workers."""
    assert step==340000 and vector.num_envs==4
    seeds=[20002,20003,20004,20005]
    states=vector.reset_many(dict(enumerate(seeds)))
    obs=[states[i] for i in range(4)];policy=BatchedGMMExecutor(actor,scale,offset,4,horizon=10)
    active=list(range(4));rows={i:dict(seed=seeds[i],steps=0,return_value=0.,success=False) for i in active}
    started=time.monotonic()
    with isolated_training_rng():
        torch.manual_seed(int(config["training_seed"])+7000000+int(step))
        torch.npu.manual_seed(int(config["training_seed"])+7000000+int(step))
        for t in range(int(config["horizon"])):
            if not active: break
            actions=policy.actions_for(active,[obs[i] for i in active])
            finished=[]
            for i,msg in vector.step(actions,active):
                if msg[0]!="OK": raise RuntimeError("Baseline simulator error: "+repr(msg))
                _,next_obs,reward,done,info,won=msg
                obs[i]=next_obs;row=rows[i];row["steps"]+=1;row["return_value"]+=float(reward)
                flags=obs_to_flat(next_obs)
                row.update(payload=bool(flags[45]>.5),trash=bool(flags[46]>.5),success=bool(won))
                if won or done or row["steps"]>=int(config["horizon"]): finished.append(i)
            active=[i for i in active if i not in finished]
            if (t+1)%100==0: print("[STAGE3-V8 BASELINE]",t+1,"active",len(active),flush=True)
    report=dict(stage="stage3-v8",env_steps=step,scope="four known-success seeds; integration baseline only",
        rows=list(rows.values()),success_count=sum(r["success"] for r in rows.values()),
        episode_count=4,actor_hash=module_hash(actor),time_sec=time.monotonic()-started,
        formal_acceptance=False)
    write(Path(output)/"baseline_340k.json",report)
    resets=vector.reset_many({i:contexts[i]["seed"] for i in range(4)})
    observations[:]=[resets[i] for i in range(4)];executor.reset_indices(range(4))
    print("[STAGE3-V8 BASELINE COMPLETE]",report["success_count"],"/4",flush=True)

def difference(a,b):
    if isinstance(a,torch.Tensor):
        if a.numel()==0:return 0.
        return float((a.detach().cpu()-b.detach().cpu()).abs().max())
    if isinstance(a,np.ndarray):return float(np.max(np.abs(a-b))) if a.size else 0.
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a)-set(b),set(b)-set(a))
        return max((difference(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,(tuple,list)):
        assert len(a)==len(b);return max((difference(x,y) for x,y in zip(a,b)),default=0.)
    if a is None or isinstance(a,(str,bool)):assert a==b;return 0.
    return abs(float(a)-float(b))

def q_loss(agent,batch):
    device=agent.device
    obs=torch.as_tensor(batch["observations"],dtype=torch.float32,device=device)
    actions=torch.as_tensor(batch["actions"],dtype=torch.float32,device=device)
    steps=torch.as_tensor(batch["episode_steps"],dtype=torch.long,device=device)
    d=agent.actor.forward_train(flat_to_obs(obs),rnn_init_state=None,return_state=False)
    b=d.component_distribution.base_dist
    final=torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=d.mixture_distribution.logits[:,-1]),
        torch.distributions.Independent(torch.distributions.Normal(b.loc[:,-1],b.scale[:,-1]),1))
    with torch.no_grad(): contexts=encode_replay_contexts(agent.critic,obs,actions,steps,agent.config["horizon"])
    expected,*_=component_mean_q(agent.critic,(contexts[0][:,-1],contexts[1][:,-1]),
        final,agent.action_scale,agent.action_offset,twin_min=False)
    return -expected.mean()

def gradient_probe(agent):
    """One paired cloned NPU update tests gradient addition, clipping and actual Adam."""
    a=copy.deepcopy(agent);a.actor.train();a.critic.requires_grad_(False)
    from stage3_v7_schedule import CriticHandoffV7,HandoffStateV7
    original=source_payload(paths.V7_CHECKPOINT)
    schedule=CriticHandoffV7(a.config,HandoffStateV7.restore(original["training_state"])).schedule(350000,a.config["critic_lr"])
    a.set_learning_rates(schedule["actor_lr"],schedule["critic_lr"])
    a.critic_updates=(a.critic_updates//4)*4
    data=np.load(paths.HERE/"testing/calibration_inputs.npz")
    qb={k:data["q_"+k] for k in ("observations","actions","episode_steps")}
    gb={k[len("good_"):]:data[k] for k in data.files if k.startswith("good_")}
    named=list(a.actor.named_parameters());params=[p for _,p in named]
    q=q_loss(a,qb);qg=torch.autograd.grad(q,params,allow_unused=True)
    g=good_loss(a.actor,gb,a.action_scale,a.action_offset,a.v8_loss)
    gg=torch.autograd.grad(g,params,allow_unused=True)
    expected=copy.deepcopy(a);start=copy.deepcopy(a.actor.state_dict())
    coefficient=float(a.v8_loss["lambda_good"])
    for p,u,v in zip(expected.actor.parameters(),qg,gg):
        if u is None: assert v is None;p.grad=None
        else:p.grad=u.detach().clone()+(0 if v is None else coefficient*v.detach())
    torch.nn.utils.clip_grad_norm_(expected.actor.parameters(),a.config["actor_max_grad_norm"])
    expected.actor_optimizer.step()
    a.critic.requires_grad_(True)
    started=time.monotonic();metrics=a.actor_update(qb,350000,True,good_batch=gb);torch.npu.synchronize()
    errors=dict(gradient=difference([p.grad for p in a.actor.parameters()],[p.grad for p in expected.actor.parameters()]),
        parameters=difference(a.actor.state_dict(),expected.actor.state_dict()),
        optimizer=difference(a.actor_optimizer.state_dict(),expected.actor_optimizer.state_dict()))
    assert max(errors.values())<=1e-5,errors
    change=difference(start,a.actor.state_dict());assert change>0
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in a.actor.parameters())
    return dict(status="PASS",scope="paired cloned NPU diagnostic at350K LR; not collected training steps",
        errors=errors,tolerance=1e-5,nonzero_actor_lr=schedule["actor_lr"],
        actual_parameter_change=change,actor_update_time_sec=time.monotonic()-started,
        pre_clip_gradient_metrics=a.last_v8_metrics,actual_actor_metrics=metrics)

def roundtrip_and_gradient_check(path,agent,saved,output,offline):
    print("[STAGE3-V8 ROUNDTRIP] restoring actual NPU checkpoint",flush=True)
    with isolated_training_rng(offline=offline):
        clone=copy.deepcopy(agent)
        expected_good=copy.deepcopy(agent.good_replay).sample(8)
        draws=(random.random(),float(np.random.random()),torch.rand(4),torch.rand(4,device=agent.device))
        restored,replay=restore_checkpoint(path,clone,clone.config)
        actual_good=clone.good_replay.sample(8)
        assert all(np.array_equal(expected_good[k],actual_good[k]) for k in expected_good)
        actual=(random.random(),float(np.random.random()),torch.rand(4),torch.rand(4,device=agent.device))
        assert difference(draws,actual)==0
        errors={}
        for key,obj in (("actor",clone.actor),("target_actor",clone.target_actor),
                        ("q1_q2",clone.critic),("target_q1_q2",clone.target_critic),
                        ("actor_optimizer",clone.actor_optimizer),("critic_optimizer",clone.critic_optimizer)):
            errors[key]=difference(saved[key],obj.state_dict())
        assert max(errors.values())==0,errors
        assert restored["training_state"]==saved["training_state"]
        offline.load_state_dict(saved["offline_sampler_state"])
        assert difference(offline.state_dict(),saved["offline_sampler_state"])==0
        assert difference(clone.pipeline.metrics(),saved["pipeline_state"])==0
        assert difference(clone.rollout_executor.metrics(),saved["rollout_snapshot_state"])==0
        assert difference(clone.target_selector_state_dict(),saved["target_selector_state"])==0
        assert replay.fixed_critic_diagnostic_set is not None
        report=dict(status="PASS",checkpoint=str(path),stage=restored["stage"],
            checkpoint_purpose=restored["checkpoint_purpose"],state_max_errors=errors,
            global_python_numpy_torch_npu_rng_identical=True,good_pool_rng_identical=True,
            frozen_readiness_restored=True,partial_episodes_discarded=not replay.current,
            pipeline_and_offline_sampler_preserved=True,rollout_snapshot_state_preserved=True,
            loaded_dependencies=audit_loaded_dependencies())
        write(Path(output)/"npu_roundtrip.json",report)
        write(Path(output)/"npu_gradient_probe.json",gradient_probe(clone))
    print("[STAGE3-V8 ROUNDTRIP] PASS",flush=True)

def finish(output,preflight,budget):
    output=Path(output)
    summary=json.loads((output/"summary.json").read_text())
    audit=json.loads((output/"environment_audit.json").read_text())
    roundtrip=json.loads((output/"npu_roundtrip.json").read_text())
    gradient=json.loads((output/"npu_gradient_probe.json").read_text())
    baseline_report=json.loads((output/"baseline_340k.json").read_text())
    checks=dict(actual_steps=summary["env_steps"]-preflight["env_steps"]==budget,
        critic_updated=summary["updates"]>84750,actor_updated=summary["actor_updates"]>12,
        device=summary["device"]=="npu:0",target=summary["critic_target_mode"]=="random2q",
        checkpoint_stage=roundtrip["stage"]=="stage3-v8",
        test_checkpoint_isolated=roundtrip["checkpoint_purpose"]=="INTEGRATION_TEST_ONLY",
        environments_4_4_4=all(audit[k]==4 for k in ("initialized","used","closed")),
        workers_closed=audit["live_workers"]==0 and all(v==0 for v in audit["worker_exitcodes"]),
        restore=roundtrip["status"]=="PASS",gradient=gradient["status"]=="PASS",
        baseline_completed=baseline_report["episode_count"]==4)
    write(output/"integration_validation.json",dict(status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,summary=summary,environment_audit=audit,baseline=baseline_report,
        formal_v7="STOPPED",formal_v8="NOT_STARTED"))
    if not all(checks.values()):raise RuntimeError("V8 integration failed: "+repr(checks))
    print("[STAGE3-V8 INTEGRATION COMPLETE] PASS",flush=True)
