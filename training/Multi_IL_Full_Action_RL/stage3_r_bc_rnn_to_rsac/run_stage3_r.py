#!/usr/bin/env python3
"""Smoke, fixed 300 epochs, serial candidate screening with 16 env workers."""
from __future__ import annotations
import argparse,json,multiprocessing as mp,shutil,subprocess,sys
from datetime import datetime
from pathlib import Path
import h5py,numpy as np,torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];TRAIN=HERE/"train_stage3_r.py";EVAL=HERE/"evaluate_stage3_r.py"
def read(p):
    with open(p,encoding="utf-8") as f:return json.load(f)
def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:json.dump(v,f,indent=2,ensure_ascii=False)
def run(cmd):print("Command:"," ".join(map(str,cmd)),flush=True);subprocess.run(list(map(str,cmd)),cwd=ROOT,check=True)
def worker(cmd,log):
    with open(log,"w",encoding="utf-8") as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def shards(num_workers):
    if not 1 <= num_workers <= 20:raise ValueError("num_evaluation_workers must be in [1, 20]")
    base,remainder=divmod(20,num_workers);out=[];cursor=10080
    for i in range(num_workers):
        n=base+(1 if i<remainder else 0);out.append((i,cursor,n));cursor+=n
    return out
def candidates(directory):
    fixed=sorted(directory.glob("stage3_r_epoch_*.pth"),key=lambda p:int(torch.load(p,map_location="cpu")["epoch"]));best=directory/"stage3_r_best_val_mse.pth";by={int(torch.load(p,map_location="cpu")["epoch"]):p for p in fixed};be=int(torch.load(best,map_location="cpu")["epoch"]);aliases={}
    if be in by:aliases["best_val_mse"]=by[be].name
    else:by[be]=best
    return [by[k] for k in sorted(by)],aliases,be
def aggregate(paths,checkpoint,out,num_workers):
    episodes=sorted([e for p in paths for e in read(p)["episodes"]],key=lambda x:x["initial_seed"])
    if [e["initial_seed"] for e in episodes]!=list(range(10080,10100)):raise RuntimeError("missing/duplicate heldout seeds")
    n=20;count=lambda k:sum(int(e[k]) for e in episodes);both=lambda pre:sum(int(e[f"{pre}_trash_in_bin"] and e[f"{pre}_payload_in_bin"]) for e in episodes)
    r={"checkpoint":str(checkpoint),"evaluation_complete":True,"num_workers":num_workers,"episodes":episodes,"success_count":count("success"),"success_rate":count("success")/n,"trash_ever_count":count("ever_trash_in_bin"),"trash_final_count":count("final_trash_in_bin"),"payload_ever_count":count("ever_payload_in_bin"),"payload_final_count":count("final_payload_in_bin"),"both_ever_count":both("ever"),"both_final_count":both("final"),"mean_progress":float(np.mean([e["partial_progress_score"] for e in episodes])),"progress_histogram":{str(i):sum(e["partial_progress_score"]==i for e in episodes) for i in range(4)},"mean_return":float(np.mean([e["episode_return"] for e in episodes])),"mean_episode_length":float(np.mean([e["episode_length"] for e in episodes]))}
    for k in ("trash_ever","trash_final","payload_ever","payload_final","both_ever","both_final"):r[k+"_rate"]=r[k+"_count"]/n
    write(out,r);return r
def evaluate(checkpoint,outdir,device,num_workers):
    wd=outdir/"workers"/checkpoint.stem;wd.mkdir(parents=True,exist_ok=False);ctx=mp.get_context("spawn");ps=[];paths=[]
    for i,start,n in shards(num_workers):
        report=wd/f"worker_{i:02d}.json";log=wd/f"worker_{i:02d}.log";cmd=[sys.executable,"-u",str(EVAL),"--checkpoint",str(checkpoint),"--seed-start",str(start),"--num-seeds",str(n),"--device",device,"--output",str(report)];p=ctx.Process(target=worker,args=(cmd,log));p.start();ps.append((i,p));paths.append(report)
    failures=[]
    for i,p in ps:p.join();failures.append((i,p.exitcode)) if p.exitcode else None
    if failures:raise RuntimeError(f"evaluation worker failures: {failures}")
    return aggregate(paths,checkpoint,outdir/f"{checkpoint.stem}.json",num_workers)

def completed_report(path):
    if not path.exists():return None
    report=read(path);seeds=[e["initial_seed"] for e in report.get("episodes",[])]
    if report.get("evaluation_complete") is not True or sorted(seeds)!=list(range(10080,10100)):
        raise RuntimeError(f"Invalid completed candidate report: {path}")
    return report
def teacher(dataset):
    with h5py.File(dataset,"r") as h:
        schema=json.loads(h.attrs["progress_observation_schema"]);fields=schema["fields"];rows=[]
        for g in h["episodes"].values():
            seed=int(g.attrs["initial_seed"])
            if not 10080<=seed<=10099:continue
            obj=g["next_obs/object"][:];t=obj[:,int(fields["trash_in_trash_bin"]["flat_index"])].astype(bool);p=obj[:,int(fields["payload_in_target_bin"]["flat_index"])].astype(bool);s=bool(g.attrs["success"]);score=3 if s else 2 if t.any() and p.any() else 1 if t.any() or p.any() else 0;rows.append((s,t.any(),p.any(),score))
    return {"episodes":len(rows),"success_rate":sum(x[0] for x in rows)/len(rows),"trash_ever_rate":sum(x[1] for x in rows)/len(rows),"payload_ever_rate":sum(x[2] for x in rows)/len(rows),"both_ever_rate":sum(x[1] and x[2] for x in rows)/len(rows),"mean_progress":float(np.mean([x[3] for x in rows]))}
def main():
    p=argparse.ArgumentParser();p.add_argument("--config",default=str(HERE/"stage3_r_config.json"));p.add_argument("--device",default="npu:0");p.add_argument("--smoke-test",action="store_true");p.add_argument("--run-id");p.add_argument("--resume-run-dir");a=p.parse_args();cfg=read(a.config)
    if a.resume_run_dir:
        if a.smoke_test or a.run_id:raise ValueError("resume cannot be combined with smoke-test or run-id")
        run_dir=Path(a.resume_run_dir).resolve()
        if not run_dir.is_dir():raise FileNotFoundError(f"resume run directory not found: {run_dir}")
    else:
        run_dir=Path(cfg["output_root"])/(a.run_id or (("smoke_" if a.smoke_test else "")+datetime.now().strftime("%Y%m%d_%H%M%S")));run_dir.mkdir(parents=True,exist_ok=False);(run_dir/"logs").mkdir()
    print("Run directory:",run_dir,"(resume)" if a.resume_run_dir else "")
    if a.smoke_test:run([sys.executable,"-u",str(TRAIN),"--config",str(Path(a.config).resolve()),"--run-dir",str(run_dir),"--device",a.device,"--smoke-test"]);print("STAGE3-R SMOKE PASSED; Stage2-R/Stage4 NOT started.");return
    if not a.resume_run_dir:
        smoke=run_dir/"preflight_smoke";run([sys.executable,"-u",str(TRAIN),"--config",str(Path(a.config).resolve()),"--run-dir",str(smoke),"--device",a.device,"--smoke-test"])
        if read(smoke/"smoke_test.json")["status"]!="PASS":raise RuntimeError("preflight failed")
        run([sys.executable,"-u",str(TRAIN),"--config",str(Path(a.config).resolve()),"--run-dir",str(run_dir),"--device",a.device])
    summary=read(run_dir/"training_summary.json")
    if summary["epochs_completed"]!=300:raise RuntimeError("did not complete 300 epochs")
    num_workers=int(cfg.get("num_evaluation_workers",8));cs,aliases,best_epoch=candidates(run_dir/"checkpoints");out=run_dir/"candidate_evaluations";out.mkdir(exist_ok=True);rows=[]
    for ck in cs:
        report_path=out/f"{ck.stem}.json";report=completed_report(report_path)
        if report is None:
            partial=out/"workers"/ck.stem
            if partial.exists():shutil.rmtree(partial)
            report=evaluate(ck,out,a.device,num_workers)
        else:print("Reusing completed candidate evaluation:",report_path,flush=True)
        payload=torch.load(ck,map_location="cpu");rows.append({"checkpoint":str(ck),"epoch":int(payload["epoch"]),"val_mse":float(payload["validation"]["val_mse"]),**{k:v for k,v in report.items() if k not in ("checkpoint","episodes")}})
    ranked=sorted(rows,key=lambda x:(-x["success_rate"],-x["both_final_rate"],-x["payload_ever_rate"],-x["mean_progress"],-x["trash_ever_rate"],x["val_mse"]));selected=ranked[0];final=run_dir/"checkpoints"/"stage3_r_bc_rnn_initialized_rsac_actor_best.pth";shutil.copy2(selected["checkpoint"],final);teach=teacher(cfg["dataset"])
    selection={"ranking_priority":["success_rate","both_final","payload_ever","mean_progress","trash_ever","validation_mse"],"best_val_epoch":best_epoch,"aliases":aliases,"unique_candidate_count":len(rows),"candidates_ranked":ranked,"selected_epoch":selected["epoch"],"selected_source":selected["checkpoint"],"final_checkpoint":str(final),"original_bc_rnn_heldout20":teach,"behavior_retention":{"success_rate_ratio":None if teach["success_rate"]==0 else selected["success_rate"]/teach["success_rate"],"mean_progress_ratio":None if teach["mean_progress"]==0 else selected["mean_progress"]/teach["mean_progress"]},"stage2_r_started":False,"stage4_started":False};write(run_dir/"stage3_r_actor_selection.json",selection);summary.update({"status":"STAGE3_R_COMPLETE","selection":selection,"stage2_r_started":False,"stage4_started":False});write(run_dir/"training_summary.json",summary);print("STAGE3-R COMPLETE",final,"Stage2-R/Stage4 NOT started.")
if __name__=="__main__":main()
