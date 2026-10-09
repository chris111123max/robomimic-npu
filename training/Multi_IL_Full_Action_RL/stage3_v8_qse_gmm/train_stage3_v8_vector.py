#!/usr/bin/env python3
"""Isolated pinned V7 loop; formal launch and bounded V8 NPU integration are separate."""
import argparse,ast,copy,json,sys,time
from pathlib import Path
import stage3_v8_paths as paths
from stage3_v8_dependencies import validate_dependencies
import train_stage3_v7_vector as v7
from stage3_v8_checkpoint import source_payload,restore_checkpoint,save_checkpoint,sha
from stage3_v8_good_replay import GoodReplay

def configuration(path=None):
    p=Path(path) if path else paths.HERE/"stage3_v8_config_calibrated.json"
    config=json.loads(p.read_text())
    if (config.get("algorithm_version")!="stage3-v8-qse-gmm-v1"
        or config.get("calibration_required",True)
        or not isinstance(config.get("lambda_good"),(int,float))
        or config.get("std_mode")!="fixed" or len(config.get("supervision_std") or [])!=14
        or config.get("online_sampling_policy")!="episode_mass_capped_v1"):
        raise RuntimeError("Calibrated fixed-std V8 config required")
    return config

class ExtensionPoints(ast.NodeTransformer):
    def __init__(self): self.counts=dict(agent_import=0,replay_import=0,output_root=0,finish_metadata=0)
    def visit_ImportFrom(self,node):
        if node.module=="stage3_v6_agent":
            self.counts["agent_import"]+=1;node.module="stage3_v8_agent"
        elif node.module=="stage3_v5_replay":
            online=[n for n in node.names if n.name=="OnlineSequenceReplay"]
            if online:
                self.counts["replay_import"]+=1
                node.names=[n for n in node.names if n.name!="OnlineSequenceReplay"]
                return [node,ast.ImportFrom(module="stage3_v8_good_replay",names=online,level=0)]
        return node
    def visit_Constant(self,node):
        if not isinstance(node.value,str): return node
        if node.value=="stage3_v7_pirlnav_schedule":
            self.counts["output_root"]+=1;node.value="stage3_v8_qse_gmm"
        elif node.value=="stage3-v7": node.value="stage3-v8"
        elif node.value=="stage3-v7-pirlnav-schedule": node.value="stage3-v8-qse-gmm"
        else:
            if node.value.startswith("V7 "): node.value="V8 "+node.value[3:]
            for before,after in (("[STAGE3-V7]","[STAGE3-V8]"),("[V7 RESTORE]","[V8 RESTORE]"),
                                 ("[V7 FRESH]","[V8 FRESH]"),("Stage3-v7","Stage3-v8"),
                                 ("Stage3-V7 non-finite","Stage3-V8 non-finite")):
                node.value=node.value.replace(before,after)
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if (isinstance(node.func,ast.Attribute) and node.func.attr=="finish"
            and isinstance(node.func.value,ast.Name) and node.func.value.id=="online"):
            self.counts["finish_metadata"]+=1
            for key in ("seed","episode_id"):
                value=ast.Subscript(value=ast.Name(id="context",ctx=ast.Load()),
                                    slice=ast.Constant(value=key),ctx=ast.Load())
                node.keywords.append(ast.keyword(arg=key,value=value))
        return node

class IntegrationPoints(ast.NodeTransformer):
    """Execution-only differences: four workers, isolated path and integration run label."""
    def __init__(self): self.counts=dict(environment_guard=0,run_label=0)
    def visit_Compare(self,node):
        if (isinstance(node.left,ast.Name) and node.left.id=="num_envs"
            and len(node.ops)==1 and isinstance(node.ops[0],ast.NotEq)
            and isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value==16):
            self.counts["environment_guard"]+=1;node.comparators[0].value=4
        return self.generic_visit(node)
    def visit_Constant(self,node):
        if node.value=="FORMAL":
            self.counts["run_label"]+=1;node.value="V8_INTEGRATION_SMOKE"
        return node

def build_main(loss_config,integration=None):
    dependency_report=validate_dependencies()
    source=Path(v7.__file__)
    if sha(source)!=loss_config["v7_trainer_source_sha256"]:
        raise RuntimeError("Audited V7 trainer source changed; re-audit extension points")
    original=ast.parse(source.read_text())
    main=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=="main"]
    if len(main)!=1: raise RuntimeError("No unique V7 main")
    transform=ExtensionPoints();body=transform.visit(main[0])
    if transform.counts!={k:1 for k in transform.counts}:
        raise RuntimeError("V7 extension contract changed: "+str(transform.counts))
    if integration:
        x=IntegrationPoints();body=x.visit(body)
        if x.counts!={k:1 for k in x.counts}: raise RuntimeError("Integration extension contract changed")
        assignments=[n for n in body.body if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=="group_dir" for t in n.targets)]
        if len(assignments)!=1: raise RuntimeError("No unique group directory assignment")
        assignments[0].value=ast.Name(id="_v8_output_dir",ctx=ast.Load())
        class BaselineInsert(ast.NodeTransformer):
            count=0
            def visit_Expr(self,node):
                if any(isinstance(v,ast.Constant) and isinstance(v.value,str)
                       and v.value.startswith("[STAGE3-V8] target=") for v in ast.walk(node)):
                    self.count+=1
                    call=ast.parse("_v8_baseline(vector,actor,scale,offset,config,observations,contexts,executor,group_dir,env_steps)").body[0]
                    return [call,node]
                return self.generic_visit(node)
        insert=BaselineInsert();body=insert.visit(body)
        if insert.count!=1: raise RuntimeError("No unique training-start insertion")
    tree=ast.Module(body=[body],type_ignores=[]);ast.fix_missing_locations(tree)
    namespace=dict(v7.__dict__)
    def read_json(path):
        value=v7.read_json(path)
        if Path(path).name=="config_resolved.json":
            value=dict(value,v8_loss=copy.deepcopy(loss_config),v8_load_offline_good=True)
            if integration:
                value.update(v8_integration_smoke=True,train_metrics_interval_updates=1)
                value["parallel_env"]=dict(value["parallel_env"],startup_parallelism=4)
        return value
    def write_json(path,value):
        value=dict(value)
        if value.get("stage")=="stage3-v7": value["stage"]="stage3-v8"
        if Path(path).name=="runtime_audit.json":
            value.update(algorithm_version=loss_config["algorithm_version"],v8_loss=loss_config,
                actor_objective="V7 final-H10 Q1 component expectation + lambda_good*all-H10 verified-success GMM NLL",
                adapter_extension_counts=transform.counts,dependencies=dependency_report,
                base_objective_revision=value.get("objective_revision"),
                v8_integration_smoke=bool(integration))
        return v7.write_json(path,value)
    def log_jsonl(path,value):
        return v7.log_jsonl(path,dict(value,stage="stage3-v8",algorithm_version=loss_config["algorithm_version"]))
    def immutable_save(path,*args,**kwargs):
        p=Path(path)
        if p.exists() or p.with_suffix(".sequences.npy").exists():
            p=p.with_name(p.stem+"_"+str(time.time_ns())+p.suffix)
        payload=save_checkpoint(p,*args,**kwargs)
        if integration and p.stem=="last":
            from stage3_v8_integration import roundtrip_and_gradient_check
            roundtrip_and_gradient_check(p,args[0],payload,integration["output"],args[-1])
        return payload
    namespace.update(read_json=read_json,write_json=write_json,log_jsonl=log_jsonl,
                     save_checkpoint=immutable_save,restore_checkpoint=restore_checkpoint)
    if integration:
        from stage3_v8_integration import baseline,vector_type
        namespace["StaggeredVectorEnv"]=vector_type(integration["output"])
        namespace.update(_v8_baseline=baseline,_v8_output_dir=Path(integration["output"]))
    exec(compile(tree,str(paths.HERE/"<v7_loop_adapter>"),"exec"),namespace)
    return namespace["main"],transform.counts,ast.unparse(tree)

def preflight(resume,loss_config,allow_integration=False):
    dependency_report=validate_dependencies()
    cp=source_payload(resume)
    if cp.get("checkpoint_purpose")=="CPU_TEST_ONLY":
        raise RuntimeError("Synthetic CPU test checkpoint is excluded from training preflight")
    if cp.get("checkpoint_purpose")=="INTEGRATION_TEST_ONLY" and not allow_integration:
        raise RuntimeError("Integration test checkpoint is excluded from formal training")
    from stage3_v7_schedule import validate_v7_config
    validate_v7_config(cp["config"])
    if cp["stage"]=="stage3-v7" and sha(resume)!=loss_config["calibration_checkpoint_sha256"]:
        raise RuntimeError("Fork differs from calibration")
    pool=GoodReplay(loss_config,int(cp["config"]["training_seed"])+loss_config["good_rng_seed_offset"])
    if cp["stage"]=="stage3-v7": pool.load_offline(cp["config"]["offline_sources"])
    else: pool.load_state_dict(cp["v8_state"]["good_replay"])
    _,counts,_=build_main(loss_config)
    return dict(stage="PREPARED",formal_training="STOPPED",checkpoint=str(Path(resume).resolve()),
        env_steps=cp["env_steps"],actor_updates=cp["actor_updates"],
        good_pool=pool.metrics(),extension_points=counts,dependencies=dependency_report)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--resume",default=str(paths.V7_CHECKPOINT))
    parser.add_argument("--v6-run-dir",default=str(paths.PREPARED))
    parser.add_argument("--config")
    parser.add_argument("--total-env-steps",type=int,default=3000000)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--execute",action="store_true",help="Explicit subsequent authorization for formal training")
    mode.add_argument("--integration-smoke",action="store_true",help="Isolated bounded NPU integration; never formal training")
    parser.add_argument("--additional-env-steps",type=int,default=64)
    parser.add_argument("--smoke-output")
    args=parser.parse_args()
    config=configuration(args.config);report=preflight(args.resume,config,args.integration_smoke)
    print(json.dumps(report,indent=2),flush=True)
    if not args.execute and not args.integration_smoke: return
    cp=source_payload(args.resume)
    integration=None
    if args.integration_smoke:
        if cp["stage"]!="stage3-v7" or report["env_steps"]!=340000:
            raise RuntimeError("Integration smoke forks only the exact V7 340K")
        if not 32<=args.additional_env_steps<=256 or args.additional_env_steps%4:
            raise ValueError("Integration budget must be 32..256 additional aggregate steps, divisible by4")
        total=report["env_steps"]+args.additional_env_steps
        if cp["training_state"]["next_evaluation_step"]<=total:
            raise RuntimeError("Bounded smoke must not cross the next formal evaluation")
        output=Path(args.smoke_output).resolve() if args.smoke_output else paths.HERE/"testing/integration_smoke"/str(time.time_ns())
        if not output.is_relative_to(paths.HERE/"testing/integration_smoke"):
            raise RuntimeError("Integration output must be below V8 testing/integration_smoke")
        if output.exists(): raise FileExistsError("Integration output must be new")
        integration=dict(output=str(output),initial_steps=report["env_steps"])
        num_envs=4
    else:
        total=args.total_env_steps;num_envs=16
        if total<=report["env_steps"]: raise ValueError("Budget must exceed restored aggregate steps")
        output=Path(cp["config"]["output_root"]).parent/"stage3_v8_qse_gmm"/Path(args.v6_run_dir).name/"random2q/multi_q"
        if (output/"runtime_audit.json").exists() and cp["stage"]!="stage3-v8":
            raise RuntimeError("V8 run exists; resume its own formal checkpoint")
    manifest=v7.read_json(Path(args.v6_run_dir)/"shared/stage2_source_manifest.json")
    old=list(sys.argv)
    sys.argv=[__file__,"--group","multi_q","--target-mode","random2q","--device","npu:0",
        "--num-envs",str(num_envs),"--quad-run-dir",args.v6_run_dir,
        "--critic-init-checkpoint",manifest["multi_q"]["checkpoint"],
        "--resume",args.resume,"--total-env-steps",str(total)]
    try:
        build_main(config,integration)[0]()
        if integration:
            from stage3_v8_integration import finish
            finish(output,report,args.additional_env_steps)
    finally: sys.argv=old
if __name__=="__main__":main()
