#!/usr/bin/env python3
import argparse,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def alive(pid,run):
 try:return str(run) in Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0",b" ").decode()
 except OSError:return False
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",required=True);a=p.parse_args();run=Path(a.run_dir);items=json.load(open(run/"pids.json"))
 while any(alive(v["pid"],run) for v in items.values()):time.sleep(30)
 if not all(json.load(open(run/g/"status.json"))["status"]=="COMPLETE" for g in items):raise SystemExit("Stage4-v2 group incomplete; analysis not run")
 schedules={group:json.load(open(run/group/"schedule_summary.json")) for group in items}
 with open(run/"schedule_comparison.json","w",encoding="utf-8") as handle:json.dump({"collector_mode":"sync","absolute_evaluation_steps":[0,5000,10000,20000,30000,40000,50000,60000,80000,100000],"groups":schedules},handle,indent=2);handle.write("\n")
 subprocess.run([sys.executable,"-u",str(HERE/"analyze_stage4_v2.py"),"--run-dir",str(run),"--config",str(HERE/"stage4_v2_config.json"),"--device",items["rnn_only_critic"]["device"]],check=True)
if __name__=="__main__":main()
