#!/usr/bin/env python3
"""Stage3C-v2 audit, training, spawn-16 screening, selection, comparisons."""
from __future__ import annotations
import argparse, json, multiprocessing as mp, shutil, subprocess, sys
from datetime import datetime
from pathlib import Path
import h5py, numpy as np, torch

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]; EVAL=HERE.parent/"stage3_actor_initialization"/"evaluate_stage3_actor.py"; TRAIN=HERE/"train_stage3c_v2.py"
def read(path):
    with open(path,encoding="utf-8") as f:return json.load(f)
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:json.dump(value,f,indent=2,ensure_ascii=False)
def run(cmd):
    print("Command:"," ".join(map(str,cmd)),flush=True);subprocess.run(list(map(str,cmd)),cwd=ROOT,check=True)
def worker(command,log_path):
    with open(log_path,"w",encoding="utf-8") as stream: subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
def shards():
    result=[]; cursor=10080
    for i in range(16):
        count=2 if i<4 else 1;result.append((i,cursor,count));cursor+=count
    return result
def checkpoint_candidates(directory):
    fixed=sorted(directory.glob("bc_gmm_component_epoch_*.pth"),key=lambda p:int(torch.load(p,map_location="cpu")["epoch"]));best=directory/"bc_gmm_component_best_val_mse.pth";best_epoch=int(torch.load(best,map_location="cpu")["epoch"])
    by_epoch={int(torch.load(p,map_location="cpu")["epoch"]):p for p in fixed};aliases={}
    if best_epoch in by_epoch: aliases["best_val_mse"]=by_epoch[best_epoch].name
    else: by_epoch[best_epoch]=best
    return [by_epoch[e] for e in sorted(by_epoch)],aliases,best_epoch
def aggregate(paths,checkpoint,output):
    reports=[read(p) for p in paths];episodes=sorted([x for r in reports for x in r["episodes"]],key=lambda x:x["initial_seed"])
    if [x["initial_seed"] for x in episodes]!=list(range(10080,10100)):raise RuntimeError("missing or duplicate evaluation seeds")
    n=len(episodes); count=lambda k:sum(int(x[k]) for x in episodes); both=lambda prefix:sum(int(x[f"{prefix}_trash_in_bin"] and x[f"{prefix}_payload_in_bin"]) for x in episodes)
    result={"checkpoint":str(checkpoint),"evaluation_complete":True,"num_workers":16,"multiprocessing_start_method":"spawn","episodes":episodes,"success_count":sum(int(x["success"]) for x in episodes)}; result["success_rate"]=result["success_count"]/n
    result.update({"trash_ever_count":count("ever_trash_in_bin"),"trash_final_count":count("final_trash_in_bin"),"payload_ever_count":count("ever_payload_in_bin"),"payload_final_count":count("final_payload_in_bin"),"both_ever_count":both("ever"),"both_final_count":both("final"),"mean_progress":float(np.mean([x["partial_progress_score"] for x in episodes])),"progress_histogram":{str(i):sum(x["partial_progress_score"]==i for x in episodes) for i in range(4)},"mean_return":float(np.mean([x["episode_return"] for x in episodes])),"mean_episode_length":float(np.mean([x["episode_length"] for x in episodes]))})
    for key in ("trash_ever","trash_final","payload_ever","payload_final","both_ever","both_final"):result[key+"_rate"]=result[key+"_count"]/n
    write(output,result);return result
def evaluate(checkpoint,outdir,device):
    name=checkpoint.stem; workers=outdir/"workers"/name;workers.mkdir(parents=True,exist_ok=False);ctx=mp.get_context("spawn");processes=[];paths=[]
    for wid,start,count in shards():
        report=workers/f"worker_{wid:02d}.json";log=workers/f"worker_{wid:02d}.log";cmd=[sys.executable,"-u",str(EVAL),"--checkpoint",str(checkpoint),"--seed-start",str(start),"--num-seeds",str(count),"--device",device,"--deterministic","--output",str(report)];p=ctx.Process(target=worker,args=(cmd,log));p.start();processes.append((wid,p));paths.append(report)
    failures=[]
    for wid,p in processes:p.join(); failures.append((wid,p.exitcode)) if p.exitcode else None
    if failures:raise RuntimeError(f"evaluation worker failures: {failures}")
    return aggregate(paths,checkpoint,outdir/f"{name}.json")
def teacher_metrics(dataset):
    with h5py.File(dataset,"r") as h:
        schema=json.loads(h.attrs["progress_observation_schema"]);fields=schema["fields"];rows=[]
        for g in h["episodes"].values():
            seed=int(g.attrs["initial_seed"])
            if not 10080<=seed<=10099:continue
            obj=g["next_obs/object"][:];ti=int(fields["trash_in_trash_bin"]["flat_index"]);pi=int(fields["payload_in_target_bin"]["flat_index"]);trash=obj[:,ti].astype(bool);payload=obj[:,pi].astype(bool);success=bool(g.attrs["success"]);score=3 if success else 2 if trash.any() and payload.any() else 1 if trash.any() or payload.any() else 0;rows.append((success,trash.any(),payload.any(),score))
    return {"success":sum(x[0] for x in rows)/20,"trash_ever":sum(x[1] for x in rows)/20,"payload_ever":sum(x[2] for x in rows)/20,"both_ever":sum(x[1] and x[2] for x in rows)/20,"mean_progress":float(np.mean([x[3] for x in rows]))}
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default=str(HERE/"stage3c_v2_config.json"));p.add_argument("--device",default="npu:0");p.add_argument("--smoke-test",action="store_true");p.add_argument("--run-id");a=p.parse_args();cfg=read(a.config);run_dir=Path(cfg["output_root"])/(a.run_id or (("smoke_" if a.smoke_test else "")+datetime.now().strftime("%Y%m%d_%H%M%S")));run_dir.mkdir(parents=True,exist_ok=False);(run_dir/"logs").mkdir();print("Run directory:",run_dir)
    if a.smoke_test:
        run([sys.executable,"-u",TRAIN,"--config",Path(a.config).resolve(),"--run-dir",run_dir,"--device",a.device,"--smoke-test"])
        print("STAGE3C-V2 SMOKE PASSED; Stage4 was NOT started.");return
    smoke_dir=run_dir/"preflight_smoke"
    run([sys.executable,"-u",TRAIN,"--config",Path(a.config).resolve(),"--run-dir",smoke_dir,"--device",a.device,"--smoke-test"])
    if read(smoke_dir/"smoke_test.json").get("status")!="PASS":raise RuntimeError("Stage3C-v2 preflight smoke failed")
    print("Preflight smoke PASS; starting mandatory full 300-epoch formal training.",flush=True)
    run([sys.executable,"-u",TRAIN,"--config",Path(a.config).resolve(),"--run-dir",run_dir,"--device",a.device])
    summary=read(run_dir/"training_summary.json");
    if summary["epochs_completed"]!=300:raise RuntimeError("formal Stage3C-v2 did not complete 300 epochs")
    candidates,aliases,best_epoch=checkpoint_candidates(run_dir/"checkpoints");out=run_dir/"candidate_evaluations";out.mkdir();rows=[]
    for checkpoint in candidates:
        report=evaluate(checkpoint,out,a.device);payload=torch.load(checkpoint,map_location="cpu");rows.append({"checkpoint":str(checkpoint),"epoch":int(payload["epoch"]),"val_mse":float(payload["validation"]["val_mse"]),**{k:v for k,v in report.items() if k not in ("episodes","checkpoint")}})
    ranked=sorted(rows,key=lambda x:(-x["success_rate"],-x["both_ever_rate"],-x["payload_ever_rate"],-x["mean_progress"],-x["trash_ever_rate"],x["val_mse"]));selected=ranked[0];final=run_dir/"checkpoints"/"stage3c_v2_bc_gmm_shared_actor_best.pth";shutil.copy2(selected["checkpoint"],final)
    selection={"ranking_priority":["success_rate","both_completion","payload_ever","mean_progress","trash_ever","validation_mse"],"best_val_mse_epoch":best_epoch,"best_val_mse_aliases":aliases,"unique_candidate_count":len(rows),"candidates_ranked":ranked,"selected_checkpoint":selected["checkpoint"],"selected_epoch":selected["epoch"],"source_candidate":Path(selected["checkpoint"]).name,"success_rate":selected["success_rate"],"trash_rate":selected["trash_ever_rate"],"payload_rate":selected["payload_ever_rate"],"both_rate":selected["both_ever_rate"],"mean_progress":selected["mean_progress"],"validation_mse":selected["val_mse"],"final_checkpoint":str(final),"stage4_started":False};write(run_dir/"stage3c_v2_actor_selection.json",selection)
    v1={"success":0.0,"trash_ever":0.2,"payload_ever":0.0,"both_ever":0.0,"mean_progress":0.2,"source":"known heldout20 result supplied for Stage3C-v1"};v2={"success":selected["success_rate"],"trash_ever":selected["trash_ever_rate"],"payload_ever":selected["payload_ever_rate"],"both_ever":selected["both_ever_rate"],"mean_progress":selected["mean_progress"]};comparison={"original_bc_gmm":teacher_metrics(cfg["dataset"]),"stage3c_v1":v1,"stage3c_v2":v2};write(run_dir/"stage3c_v1_vs_v2_comparison.json",comparison);summary.update({"status":"STAGE3C_V2_COMPLETE","selection":selection,"comparison":comparison,"stage4_started":False});write(run_dir/"training_summary.json",summary);print("STAGE3C-V2 COMPLETE",final,"Stage4 was NOT started.")
if __name__=="__main__":main()
