"""Strict V7 340K fork and independent V8 checkpoint/RNG bundle."""
import copy
import hashlib
import os
from pathlib import Path
import torch
from types import SimpleNamespace
import stage3_v8_paths
import train_stage3_v7_vector as v7
from stage3_v8_good_replay import OnlineSequenceReplay, bind_pool
from stage3_v5_actor import module_hash
from stage3_v7_schedule import HandoffStateV7, fixed_set_digest

VERSION="stage3-v8-qse-gmm-v1"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""): h.update(chunk)
    return h.hexdigest()

def source_payload(path,config=None):
    path=Path(path).resolve()
    payload=torch.load(path,map_location="cpu",weights_only=False)
    stage=payload.get("stage")
    if stage=="stage3-v7":
        if payload.get("env_steps")!=340000 or payload.get("actor_updates")!=12:
            raise RuntimeError("Only exact V7 340K/12 fork is permitted")
        if payload.get("readiness_version")!=7:
            raise RuntimeError("Wrong V7 readiness version")
        if payload["training_state"]["state"]!="ACTOR_WARMUP":
            raise RuntimeError("Wrong 340K handoff phase")
        if any(float(g["lr"])!=0 for g in payload["actor_optimizer"]["param_groups"]):
            raise RuntimeError("340K Actor optimizer LR was not zero")
        required_replay=path.with_suffix(".sequences.npy")
    elif stage=="stage3-v8":
        if (payload.get("algorithm_version")!=VERSION or "v8_state" not in payload
            or payload.get("checkpoint_purpose") not in ("CPU_TEST_ONLY","INTEGRATION_TEST_ONLY","FORMAL_TRAINING")):
            raise RuntimeError("Invalid V8 checkpoint schema")
        required_replay=Path(payload["online_sequence_replay"])
    else:
        raise RuntimeError("V8 accepts V7 340K or V8 only")
    for key in ("actor","target_actor","q1_q2","target_q1_q2","actor_optimizer",
                "critic_optimizer","target_selector_state","stage2_critic_reference_state",
                "rng_state","training_state","offline_sampler_state","pipeline_state"):
        if not payload.get(key): raise RuntimeError("Missing checkpoint field: "+key)
    if payload.get("group")!="multi_q" or payload.get("critic_target_mode")!="random2q":
        raise RuntimeError("Only random2q/multi_q")
    actual_replay=Path(payload["online_sequence_replay"]).resolve()
    if not required_replay.is_file() or actual_replay!=required_replay.resolve():
        raise RuntimeError("Missing/mismatched exact companion replay")
    if stage=="stage3-v8" and payload.get("v8_dependency_lock_sha256")!=sha(stage3_v8_paths.HERE/"stage3_v8_dependency_lock.json"):
        raise RuntimeError("V8 checkpoint dependency lock differs; explicit compatibility review required")
    if stage=="stage3-v8" and sha(actual_replay)!=payload["replay_sha256"]:
        raise RuntimeError("V8 replay hash mismatch")
    if config is not None:
        for k in ("critic_source_sha256","bc_rnn_checkpoint_sha256","policy_delay","utd",
                  "recurrent_replay","critic_readiness_v7","v7_schedule","tau",
                  "actor_source_contract","objective_revision","rl_policy_expectation",
                  "adaptive_bc_enabled","bc_weight","actor_q_scale_normalization",
                  "boundary_aligned_sequence_sampling"):
            if payload["config"].get(k)!=config.get(k):
                raise RuntimeError("Preserved V7 contract mismatch: "+k)
    return payload

def restore_checkpoint(path,agent,config,torch_module=torch,OnlineSequenceReplay=OnlineSequenceReplay):
    payload=source_payload(path,config)
    if payload.get("checkpoint_purpose")=="INTEGRATION_TEST_ONLY" and not config.get("v8_integration_smoke"):
        raise RuntimeError("Integration checkpoint cannot seed formal training")
    if payload.get("checkpoint_purpose")=="CPU_TEST_ONLY" and agent.device.type!="cpu":
        raise RuntimeError("CPU validation checkpoint cannot seed formal training")
    if payload["stage"]=="stage3-v7":
        if any(not torch.equal(v.cpu(),agent.actor.state_dict()[k].cpu()) for k,v in payload["actor"].items()):
            raise RuntimeError("V7 340K actor differs from loaded Stage1 BC")
        agent.v8_fork_provenance=dict(checkpoint=str(Path(path).resolve()),sha256=sha(path),
            replay_sha256=sha(payload["online_sequence_replay"]),env_steps=340000,actor_updates=12,
            bc_actor_hash=module_hash(agent.actor),twelve_updates_lr_zero=True)
    else:
        if payload["v8_state"]["loss_config"]!=agent.v8_loss:
            raise RuntimeError("V8 loss/calibration changed across resume")
        agent.good_replay.load_state_dict(payload["v8_state"]["good_replay"])
        agent.v8_fork_provenance=copy.deepcopy(payload["v8_state"]["fork"])
    for obj,key in ((agent.actor,"actor"),(agent.target_actor,"target_actor"),
                    (agent.critic,"q1_q2"),(agent.target_critic,"target_q1_q2")):
        obj.load_state_dict(payload[key],strict=True)
    agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
    agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])
    agent.load_target_selector_state_dict(payload["target_selector_state"])
    agent.initial_critic_state={k:v.cpu().clone() for k,v in payload["stage2_critic_reference_state"].items()}
    agent.critic_updates=int(payload["updates"]); agent.actor_updates=int(payload["actor_updates"])
    agent.actor_enabled_critic_updates=int(payload["actor_enabled_critic_updates"])
    agent.actor_gate_open=bool(payload["actor_gate_open"]);agent.gate_open_step=payload["gate_open_step"]
    norm=payload["action_normalization_stats"]
    if not torch.equal(torch.tensor(norm["scale"]),agent.action_scale.cpu().reshape(-1)):
        raise RuntimeError("Action scale mismatch")
    if not torch.equal(torch.tensor(norm["offset"]),agent.action_offset.cpu().reshape(-1)):
        raise RuntimeError("Action offset mismatch")
    bind_pool(agent.good_replay)
    online=OnlineSequenceReplay.load(payload["online_sequence_replay"])
    online.good_pool=agent.good_replay
    # Same reset/discard semantics as V7; never promote partial episodes.
    online.current={}
    state=HandoffStateV7.restore(payload["training_state"])
    if state.fixed_diagnostic_sha256 and (
        online.fixed_critic_diagnostic_set is None or
        fixed_set_digest(online.fixed_critic_diagnostic_set)!=state.fixed_diagnostic_sha256):
        raise RuntimeError("Frozen readiness set differs")
    rng=payload["rng_state"]
    import random,numpy as np
    random.setstate(rng["python"]);np.random.set_state(rng["numpy"])
    torch_module.set_rng_state(rng["torch"].cpu())
    # CPU tests retain original NPU state in returned payload without initializing NPU.
    if agent.device.type=="npu" and "npu" in rng:
        torch_module.npu.set_rng_state(rng["npu"].cpu())
    agent._v8_preserved_npu_rng=rng.get("npu")
    return payload,online

def ensure_v8_destination(path):
    path=Path(path).resolve()
    if "stage3_v8_qse_gmm" not in path.parts:
        raise RuntimeError("Checkpoint must be isolated under V8")
    if path.exists() or path.with_suffix(".sequences.npy").exists():
        raise FileExistsError("Never overwrite existing V8 checkpoint; choose new immutable name")
    return path

def save_checkpoint(path,agent,config,group,env_steps,generations,episodes,successes,
                    online,torch_module=torch,handoff=None,update_credit=0.0,offline=None):
    path=ensure_v8_destination(path);path.parent.mkdir(parents=True,exist_ok=True)
    replay=path.with_suffix(".sequences.npy"); online.save(replay)
    rng_api=(SimpleNamespace(get_rng_state=torch_module.get_rng_state)
             if agent.device.type=="cpu" else torch_module)
    payload=v7.checkpoint_payload(agent,config,group,env_steps,generations,episodes,
        successes,online,rng_api,handoff,update_credit,offline)
    payload.update(stage="stage3-v8",algorithm_version=VERSION,
        v8_dependency_lock_sha256=sha(stage3_v8_paths.HERE/"stage3_v8_dependency_lock.json"),
        checkpoint_purpose=("CPU_TEST_ONLY" if agent.device.type=="cpu" else
            "INTEGRATION_TEST_ONLY" if config.get("v8_integration_smoke") else "FORMAL_TRAINING"),
        online_sequence_replay=str(replay),replay_sha256=sha(replay),
        v8_state=dict(good_replay=agent.good_replay.state_dict(),
            loss_config=copy.deepcopy(agent.v8_loss),lambda_good=agent.v8_loss["lambda_good"],
            fork=copy.deepcopy(agent.v8_fork_provenance)))
    if agent.device.type=="cpu" and agent._v8_preserved_npu_rng is not None:
        payload["rng_state"]["npu"]=agent._v8_preserved_npu_rng.cpu()
    temp=path.with_name("."+path.name+".tmp")
    torch_module.save(payload,temp);os.replace(temp,path)
    # latest is a text pointer, never an overwritten binary checkpoint.
    pointer=path.parent/"LATEST.json";v7.write_json(pointer,{"checkpoint":str(path),"env_steps":env_steps})
    return payload
