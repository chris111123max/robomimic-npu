#!/usr/bin/env python3
"""Read-only audit of Stage3-new expert terminal and bootstrap semantics."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import h5py,numpy as np

KEYS=("robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","robot1_eef_pos","robot1_eef_quat","robot1_gripper_qpos","object")
ALIASES={"reward":("rewards","reward"),"done":("dones","done"),"terminated":("terminated","terminations"),"truncated":("truncated","truncations"),"success":("success",),"episode_ends":("episode_ends",)}

def read_json(path):
    with open(path,encoding="utf-8") as f:return json.load(f)
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n")
def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);fields=sorted({k for row in rows for k in row}) if rows else ["episode_id"]
    with open(path,"w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def selected(group,names):
    return next((name for name in names if name in group),None)
def vector(group,index):
    return np.concatenate([np.asarray(group[key][index]).reshape(-1) for key in KEYS])
def scalar(dataset,index):
    if dataset is None:return None
    value=np.asarray(dataset[index]).reshape(-1);return None if not len(value) else float(value[0])
def audit_dataset(expert_path):
    episodes=[];positives=[];schema={"expert_data":str(Path(expert_path).resolve()),"stage3_loader_required_fields":["obs","next_obs","actions","rewards","dones"],"stage3_terminal_field":"dones","stage3_bootstrap_mask":"1 - dones","episodes":{}};warnings=[];previous=None
    counts={"num_episodes":0,"num_transitions":0,"reward_positive_count":0,"reward_positive_episode_count":0,"done_true_count":0,"terminated_true_count":None,"truncated_true_count":None,"final_transition_done_rate":None,"final_transition_terminated_rate":None,"final_transition_truncated_rate":None,"cross_episode_next_obs_match_count":0}
    final={"episode_final_transition_count":0,"bootstrap_mask_0_count":0,"bootstrap_mask_1_count":0,"success_ending":{"count":0,"bootstrap_mask_0_count":0,"bootstrap_mask_1_count":0},"failure_or_horizon_ending":{"count":0,"bootstrap_mask_0_count":0,"bootstrap_mask_1_count":0}}
    positive_mask={"positive_reward_count":0,"positive_reward_bootstrap_mask_0_count":0,"positive_reward_bootstrap_mask_1_count":0,"positive_reward_bootstrap_rate":None}
    terminated_values=[];truncated_values=[];final_done=[];final_terminated=[];final_truncated=[]
    with h5py.File(expert_path,"r") as f:
        if "data" not in f:raise RuntimeError("Expert HDF5 has no /data group")
        names=sorted(f["data"])
        for episode_index,name in enumerate(names):
            demo=f["data"][name];keys=sorted(demo.keys());chosen={kind:selected(demo,candidates) for kind,candidates in ALIASES.items()};schema["episodes"][name]={"top_level_keys":keys,"selected_fields":chosen,"obs_keys":sorted(demo["obs"].keys()) if "obs" in demo else None,"next_obs_keys":sorted(demo["next_obs"].keys()) if "next_obs" in demo else None}
            for required in ("obs","next_obs","actions","rewards","dones"):
                if required not in demo:raise RuntimeError(f"{demo.name}: Stage3 loader would fail because {required} is absent")
            n=len(demo["actions"]);rewards=np.asarray(demo["rewards"]).reshape(n,-1)[:,0];dones=np.asarray(demo["dones"]).reshape(n,-1)[:,0];terminated=np.asarray(demo[chosen["terminated"]]).reshape(n,-1)[:,0] if chosen["terminated"] else None;truncated=np.asarray(demo[chosen["truncated"]]).reshape(n,-1)[:,0] if chosen["truncated"] else None;success=np.asarray(demo[chosen["success"]]).reshape(n,-1)[:,0] if chosen["success"] else None
            if np.any((dones<0)|(dones>1)):warnings.append(f"AUDIT WARNING: {name} contains dones outside [0,1]")
            positive=np.flatnonzero(rewards>0);success_episode=bool(len(positive) or (success is not None and np.any(success>0)))
            cross=False
            if previous is not None:
                current_first=vector(demo["obs"],0);cross=bool(np.allclose(previous["next"],current_first,rtol=1e-6,atol=1e-7))
                if cross:counts["cross_episode_next_obs_match_count"]+=1;warnings.append(f"AUDIT WARNING: {previous['name']} final next_obs matches {name} first obs")
            previous={"name":name,"next":vector(demo["next_obs"],-1)}
            last=n-1;mask=1.0-float(dones[last]);category="success_ending" if success_episode else "failure_or_horizon_ending";final["episode_final_transition_count"]+=1;final[category]["count"]+=1
            bucket="bootstrap_mask_0_count" if np.isclose(mask,0) else "bootstrap_mask_1_count";final[bucket]+=1;final[category][bucket]+=1
            row={"episode_id":name,"episode_index":episode_index,"episode_length":n,"final_reward":float(rewards[last]),"final_done":float(dones[last]),"final_terminated":scalar(demo[chosen["terminated"]],last) if chosen["terminated"] else None,"final_truncated":scalar(demo[chosen["truncated"]],last) if chosen["truncated"] else None,"reward_positive":bool(len(positive)),"reward_positive_timesteps":";".join(map(str,positive.tolist())),"success_transition_is_final":bool(len(positive) and np.all(positive==last)),"final_bootstrap_mask":mask,"cross_from_previous_episode":cross};episodes.append(row)
            for step in positive:
                pmask=1.0-float(dones[step]);positives.append({"episode_id":name,"timestep":int(step),"reward":float(rewards[step]),"done":float(dones[step]),"bootstrap_mask":pmask,"is_episode_final":bool(step==last)})
                positive_mask["positive_reward_count"]+=1;positive_mask["positive_reward_bootstrap_mask_0_count" if np.isclose(pmask,0) else "positive_reward_bootstrap_mask_1_count"]+=1
            counts["num_episodes"]+=1;counts["num_transitions"]+=n;counts["reward_positive_count"]+=len(positive);counts["reward_positive_episode_count"]+=int(bool(len(positive)));counts["done_true_count"]+=int(np.count_nonzero(dones));final_done.append(float(dones[last]))
            if terminated is not None:terminated_values.extend(terminated.tolist());final_terminated.append(float(terminated[last]))
            if truncated is not None:truncated_values.extend(truncated.tolist());final_truncated.append(float(truncated[last]))
    counts["terminated_true_count"]=int(np.count_nonzero(terminated_values)) if terminated_values else None;counts["truncated_true_count"]=int(np.count_nonzero(truncated_values)) if truncated_values else None;counts["final_transition_done_rate"]=float(np.mean(final_done)) if final_done else None;counts["final_transition_terminated_rate"]=float(np.mean(final_terminated)) if final_terminated else None;counts["final_transition_truncated_rate"]=float(np.mean(final_truncated)) if final_truncated else None
    if positive_mask["positive_reward_count"]:positive_mask["positive_reward_bootstrap_rate"]=positive_mask["positive_reward_bootstrap_mask_1_count"]/positive_mask["positive_reward_count"]
    return schema,counts,episodes,positives,{"stage3_bootstrap_mask":"1 - stored dones","positive_rewards":positive_mask,"episode_final_transitions":final,"warnings":warnings}
def main():
    p=argparse.ArgumentParser();p.add_argument("--pair-run-dir",required=True);p.add_argument("--expert-data");p.add_argument("--output-dir");a=p.parse_args();pair=Path(a.pair_run_dir).resolve();config=read_json(pair/"shared"/"config_resolved.json");expert=Path(a.expert_data or config.get("expert_dataset") or "")
    if not expert.is_file():raise FileNotFoundError(f"Expert dataset does not exist: {expert}")
    out=Path(a.output_dir) if a.output_dir else pair/"audits"/"terminal_semantics";schema,summary,episodes,positives,masks=audit_dataset(expert);write_json(out/"terminal_schema.json",schema);write_json(out/"terminal_summary.json",summary);write_csv(out/"episode_terminal_samples.csv",episodes);write_csv(out/"positive_reward_terminal_samples.csv",positives);write_json(out/"bootstrap_mask_summary.json",masks)
    sampler={"batch_size":int(config["batch_size"]),"offline_count":int(config["batch_size"])//2,"online_count":int(config["batch_size"])-int(config["batch_size"])//2,"implementation":"SymmetricSampler.sample; even batch is exact 50/50","configured_offline_fraction":config["offline_fraction"],"configured_online_fraction":config["online_fraction"]};write_json(pair/"audits"/"sampler_audit.json",sampler)
    (out/"README.txt").write_text("Read-only audit. Stage3 loader uses /data/demo_*/rewards, dones, explicit obs and next_obs. SAC bootstrap mask is 1-dones. Missing terminated/truncated fields are reported as null, never inferred.\n",encoding="utf-8");print(json.dumps({"output_dir":str(out.resolve()),"terminal_summary":summary,"bootstrap_mask_summary":masks},indent=2))
if __name__=="__main__":main()
