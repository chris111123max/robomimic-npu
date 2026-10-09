#!/usr/bin/env python3
"""Small isolated adapter for V7's nested loop, with fail-closed source contract.

V7 exposes no trainer class or collector/checkpoint dependency injection.
Clone only its main function AST in memory; redirect four explicit extension
points. The original source, helper functions, replay samplers and loop stay
unchanged. No production module globals or files are patched.
"""
import argparse,ast,copy,json,os,sys,time
from pathlib import Path
import stage3_v8_paths as paths
import train_stage3_v7_vector as v7
from stage3_v8_checkpoint import source_payload,restore_checkpoint,save_checkpoint,sha
from stage3_v8_good_replay import GoodReplay

def configuration(path=None):
    p=Path(path) if path else paths.HERE/"stage3_v8_config_calibrated.json"
    config=json.loads(p.read_text())
    if (config.get("algorithm_version")!="stage3-v8-qse-gmm-v1"
        or config.get("calibration_required",True)
        or not isinstance(config.get("lambda_good"),(int,float))
        or config.get("std_mode")!="fixed" or len(config.get("supervision_std") or [])!=14):
        raise RuntimeError("Calibrated fixed-std V8 config required")
    return config

class ExtensionPoints(ast.NodeTransformer):
    def __init__(self): self.counts=dict(agent_import=0,replay_import=0,output_root=0,finish_metadata=0)
    def visit_ImportFrom(self,node):
        if node.module=="stage3_v6_agent":
            self.counts["agent_import"]+=1
            node.module="stage3_v8_agent"
        elif node.module=="stage3_v5_replay":
            online=[n for n in node.names if n.name=="OnlineSequenceReplay"]
            if online:
                self.counts["replay_import"]+=1
                node.names=[n for n in node.names if n.name!="OnlineSequenceReplay"]
                return [node,ast.ImportFrom(module="stage3_v8_good_replay",names=online,level=0)]
        return node
    def visit_Constant(self,node):
        if node.value=="stage3_v7_pirlnav_schedule":
            self.counts["output_root"]+=1
            node.value="stage3_v8_qse_gmm"
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

def build_main(loss_config):
    source=Path(v7.__file__)
    if sha(source)!=loss_config["v7_trainer_source_sha256"]:
        raise RuntimeError("Audited V7 trainer source changed; re-audit extension points")
    original=ast.parse(source.read_text())
    main=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=="main"]
    if len(main)!=1: raise RuntimeError("No unique V7 main")
    transform=ExtensionPoints()
    tree=ast.Module(body=[transform.visit(main[0])],type_ignores=[])
    if transform.counts!={k:1 for k in transform.counts}:
        raise RuntimeError("V7 extension contract changed: "+str(transform.counts))
    ast.fix_missing_locations(tree)
    namespace=dict(v7.__dict__)
    def read_json(path):
        value=v7.read_json(path)
        if Path(path).name=="config_resolved.json":
            value=dict(value,v8_loss=copy.deepcopy(loss_config),v8_load_offline_good=True)
        return value
    def write_json(path,value):
        value=dict(value)
        if value.get("stage")=="stage3-v7": value["stage"]="stage3-v8"
        if Path(path).name=="runtime_audit.json":
            value.update(algorithm_version=loss_config["algorithm_version"],v8_loss=loss_config,
                actor_objective="V7 final-H10 Q1 component expectation + lambda_good*all-H10 verified-success GMM NLL",
                adapter_extension_counts=transform.counts)
        return v7.write_json(path,value)
    def immutable_save(path,*args,**kwargs):
        p=Path(path)
        if p.exists() or p.with_suffix(".sequences.npy").exists():
            p=p.with_name(p.stem+"_"+str(time.time_ns())+p.suffix)
        return save_checkpoint(p,*args,**kwargs)
    namespace.update(read_json=read_json,write_json=write_json,
                     save_checkpoint=immutable_save,restore_checkpoint=restore_checkpoint)
    exec(compile(tree,str(paths.HERE/"<v7_loop_adapter>"),"exec"),namespace)
    return namespace["main"],transform.counts,ast.unparse(tree)

def preflight(resume,loss_config):
    cp=source_payload(resume)
    if cp.get("checkpoint_purpose")=="CPU_TEST_ONLY":
        raise RuntimeError("Synthetic CPU test checkpoint is excluded from training preflight")
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
        good_pool=pool.metrics(),extension_points=counts)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--resume",default=str(paths.V7_CHECKPOINT))
    parser.add_argument("--v6-run-dir",default=str(paths.PREPARED))
    parser.add_argument("--config")
    parser.add_argument("--total-env-steps",type=int,default=3000000)
    parser.add_argument("--execute",action="store_true",help="Start formal training only when subsequently authorized")
    args=parser.parse_args()
    config=configuration(args.config)
    report=preflight(args.resume,config)
    print(json.dumps(report,indent=2),flush=True)
    if not args.execute: return
    if args.total_env_steps<=report["env_steps"]: raise ValueError("Budget must exceed restored aggregate steps")
    output=Path(source_payload(args.resume)["config"]["output_root"]).parent/"stage3_v8_qse_gmm"/Path(args.v6_run_dir).name/"random2q/multi_q"
    if (output/"runtime_audit.json").exists() and source_payload(args.resume)["stage"]!="stage3-v8":
        raise RuntimeError("V8 run exists; resume its own checkpoint or explicitly create another prepared run")
    manifest=v7.read_json(Path(args.v6_run_dir)/"shared/stage2_source_manifest.json")
    old=list(sys.argv)
    sys.argv=[__file__,"--group","multi_q","--target-mode","random2q","--device","npu:0",
        "--num-envs","16","--quad-run-dir",args.v6_run_dir,
        "--critic-init-checkpoint",manifest["multi_q"]["checkpoint"],
        "--resume",args.resume,"--total-env-steps",str(args.total_env_steps)]
    try: build_main(config)[0]()
    finally: sys.argv=old
if __name__=="__main__":main()
