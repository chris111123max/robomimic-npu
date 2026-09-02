#!/usr/bin/env python3
"""Select disjoint physical-core CPU pools from the container's allowed CPUs."""
import argparse,json,os,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 text=subprocess.check_output(["lscpu","-e=CPU,CORE,SOCKET,NODE"],text=True);(out/"cpu_topology.txt").write_text(text)
 rows=[]
 for line in text.splitlines()[1:]:
  fields=line.split()
  if len(fields)>=4:rows.append(dict(cpu=int(fields[0]),core=int(fields[1]),socket=int(fields[2]) if fields[2]!="-" else -1,node=int(fields[3]) if fields[3]!="-" else -1))
 allowed=set(os.sched_getaffinity(0));rows=[r for r in rows if r["cpu"] in allowed]
 unique={}
 for r in rows:unique.setdefault((r["socket"],r["core"]),r)
 cores=list(unique.values());cores.sort(key=lambda r:(r["node"],r["socket"],r["core"],r["cpu"]))
 # Interleave NUMA nodes, then alternate selected physical cores between groups.
 buckets={}
 for r in cores:buckets.setdefault((r["node"],r["socket"]),[]).append(r)
 ordered=[]
 while any(buckets.values()):
  for key in sorted(buckets):
   if buckets[key]:ordered.append(buckets[key].pop(0))
 if len(ordered)<34:raise SystemExit(f"Need at least 34 distinct allowed physical cores; found {len(ordered)}")
 rnn=ordered[0:32:2];multi=ordered[1:32:2];trainers=ordered[32:34]
 cpu_max=Path("/sys/fs/cgroup/cpu.max").read_text().strip() if Path("/sys/fs/cgroup/cpu.max").exists() else "unavailable"
 quota=None
 if cpu_max!="unavailable" and cpu_max.split()[0]!="max":quota=float(cpu_max.split()[0])/float(cpu_max.split()[1])
 payload={"cgroup_cpu_max":cpu_max,"cpu_quota_equivalents":quota,"allowed_logical_cpus":sorted(allowed),"selection_policy":"one logical CPU per physical core; NUMA-interleaved; disjoint groups","rnn_cpu_ids":[r["cpu"] for r in rnn],"multi_cpu_ids":[r["cpu"] for r in multi],"rnn_trainer_cpu_id":trainers[0]["cpu"],"multi_trainer_cpu_id":trainers[1]["cpu"],"rnn_mapping":rnn,"multi_mapping":multi,"trainer_mapping":trainers}
 if quota is not None and quota<34:raise SystemExit(f"CPU quota {quota} is below required 34 CPU equivalents")
 (out/"cpu_affinity.json").write_text(json.dumps(payload,indent=2)+"\n");print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
