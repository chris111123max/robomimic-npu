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
 subprocess.run([sys.executable,"-u",str(HERE/"analyze_stage4_v2.py"),"--run-dir",str(run),"--config",str(HERE/"stage4_v2_config.json"),"--device",items["rnn_only_critic"]["device"]],check=True)
if __name__=="__main__":main()
