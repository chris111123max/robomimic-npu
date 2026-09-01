#!/usr/bin/env python3
"""Wait for the three owned Stage4 PIDs, then summarize successful runs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def alive(pid,run):
    try:
        os.kill(pid,0)
        command=Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")
        return "train_stage4_group.py" in command and str(run) in command
    except OSError:return False
def main():
    p=argparse.ArgumentParser();p.add_argument("--run-dir",required=True);a=p.parse_args();run=Path(a.run_dir)
    pids=json.loads((run/"pids.json").read_text());
    while any(alive(int(pid),run) for pid in pids.values()):time.sleep(30)
    statuses=[]
    for group in pids:
        path=run/group/"status.json";statuses.append(path.exists() and json.loads(path.read_text())["status"]=="COMPLETE")
    if all(statuses):subprocess.run([sys.executable,"-u",str(HERE/"summarize_stage4.py"),"--run-dir",str(run)],check=True)
    else:print("Stage4 ended with incomplete/failed group; comparison was not generated",file=sys.stderr);sys.exit(1)
if __name__=="__main__":main()
