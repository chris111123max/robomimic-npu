#!/usr/bin/env python3
"""Read-only unified validation of fixed Stage2-R-v2 critic checkpoints."""
from __future__ import annotations

import argparse,csv,json,sys
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent;V1=PROJECT/"stage2_r_recurrent_critic_pretraining";S3R=PROJECT/"stage3_r_bc_rnn_to_rsac";VENDOR=PROJECT/"third_party"/"pomdp_baselines"
for path in (HERE,V1,S3R,VENDOR):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from stage3_r_actor import load_actor  # noqa: E402
from stage2_r_critic import architecture,make_pair,set_device  # noqa: E402
from stage2_r_data import POLICIES,attach_actor_actions,load_episodes,validation_sequences  # noqa: E402
from stage2_r_evaluation import evaluate  # noqa: E402

DEFAULT_RUN="/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage2_r_v2_recurrent_critic_pretraining/20260901_191832"
DEFAULT_ACTOR="/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_r_bc_rnn_to_rsac_actor/20260901_162152/checkpoints/stage3_r_epoch_150.pth"
Q_DEFINITION="min(Q1,Q2) per valid timestep; trajectory score is the mean of those timestep values"
PAIR_DEFINITION="Within each common initial seed, construct each policy-source pair with unequal binary episode success; ranking is correct iff the successful trajectory's mean min-Q exceeds the failed trajectory's mean min-Q"


def read(path):
    with open(path,encoding="utf-8") as handle:return json.load(handle)
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,"w",encoding="utf-8") as handle:json.dump(value,handle,indent=2,ensure_ascii=False)
def device_of(name):
    if name.startswith("npu"):
        import torch_npu  # noqa
        if not torch.npu.is_available():raise RuntimeError("Ascend NPU unavailable")
        torch.npu.set_device(name)
    return torch.device(name)
def csv_rows(path):
    with open(path,newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))
def metric(row,name):return float(row[name])


def audit_checkpoints(run):
    rnn_path=run/"rnn_only_critic"/"checkpoints"/"best.pth";multi_path=run/"multi_il_critic"/"checkpoints"/"update_14000.pth"
    rnn=torch.load(rnn_path,map_location="cpu");multi=torch.load(multi_path,map_location="cpu")
    rnn_update=int(rnn["update"]);multi_update=int(multi["update"])
    rnn_history=csv_rows(run/"rnn_only_critic"/"validation_metrics.csv");multi_history=csv_rows(run/"multi_il_critic"/"validation_metrics.csv")
    rnn_best_row=min(rnn_history,key=lambda row:metric(row,"val_td_mse"));rnn_history_best=int(rnn_best_row["update"])
    if rnn_update!=41000 or rnn_history_best!=rnn_update:
        raise RuntimeError(f"RNN-only best checkpoint audit failed: metadata={rnn_update}, validation_history_best={rnn_history_best}, expected=41000")
    if abs(float(rnn["validation"]["val_td_mse"])-metric(rnn_best_row,"val_td_mse"))>1e-12:
        raise RuntimeError("RNN-only best checkpoint validation metadata differs from validation history")
    if multi_update!=14000:
        raise RuntimeError(f"Multi-IL checkpoint audit failed: metadata={multi_update}, expected=14000")
    multi_row=next((row for row in multi_history if int(row["update"])==multi_update),None)
    if multi_row is None:raise RuntimeError("Multi-IL validation history has no update 14000 row")
    if abs(float(multi["validation"]["val_td_mse"])-metric(multi_row,"val_td_mse"))>1e-12:
        raise RuntimeError("Multi-IL checkpoint validation metadata differs from validation history")
    return {"rnn_only":(rnn_path,rnn,rnn_history,rnn_best_row),"multi_il":(multi_path,multi,multi_history,multi_row)}


def earliest_perfect(rows):
    matches=[int(row["update"]) for row in rows if abs(metric(row,"pairwise_ranking_accuracy")-1.0)<=1e-12]
    return None if not matches else min(matches)


def load_critic(payload,config,device):
    critic,target=make_pair(config,device);critic.load_state_dict(payload["critic_state_dict"]);target.load_state_dict(payload["target_critic_state_dict"])
    critic.eval();target.eval();critic.requires_grad_(False);target.requires_grad_(False);return critic,target


def q_distribution(result):
    successful=np.asarray([row["trajectory_score"] for row in result["trajectory_scores"] if row["success"]],dtype=np.float64)
    failed=np.asarray([row["trajectory_score"] for row in result["trajectory_scores"] if not row["success"]],dtype=np.float64)
    return {"success_q_mean":float(successful.mean()),"success_q_std":float(successful.std()),
            "failure_q_mean":float(failed.mean()),"failure_q_std":float(failed.std()),
            "q_gap":float(successful.mean()-failed.mean()),"q_mean":result["q_mean"],"q_std":result["q_std"],
            "q_min":result["q_min"],"q_max":result["q_max"]}


def pair_table(rnn,multi):
    def index(result):return {(row["source"],int(row["seed"])):row for row in result["trajectory_scores"]}
    ri,mi=index(rnn),index(multi);rows=[]
    if [(item["seed"],item["success_source"],item["failure_source"]) for item in rnn["pairwise_details"]]!=[(item["seed"],item["success_source"],item["failure_source"]) for item in multi["pairwise_details"]]:raise RuntimeError("Critics produced different validation pair identities")
    for number,pair in enumerate(rnn["pairwise_details"]):
        seed=int(pair["seed"]);success=pair["success_source"];failure=pair["failure_source"]
        rs=ri[(success,seed)]["trajectory_score"];rf=ri[(failure,seed)]["trajectory_score"]
        ms=mi[(success,seed)]["trajectory_score"];mf=mi[(failure,seed)]["trajectory_score"]
        rows.append({"pair_id":f"pair_{number:03d}","seed":seed,"success_source":success,"failure_source":failure,
                     "success_q_rnn":rs,"failure_q_rnn":rf,"gap_rnn":rs-rf,"ranking_correct_rnn":bool(rs>rf),
                     "success_q_multi":ms,"failure_q_multi":mf,"gap_multi":ms-mf,"ranking_correct_multi":bool(ms>mf)})
    return rows


def compact(checkpoint,payload,result):
    return {"checkpoint":str(checkpoint),"update":int(payload["update"]),"val_td_mse":result["val_td_mse"],
            "ranking":result["pairwise_ranking_accuracy"],"success_q":result["success_q"],"failure_q":result["failure_q"],
            "q_gap":result["q_gap"],"q_mean":result["q_mean"],"q_std":result["q_std"],"q_min":result["q_min"],"q_max":result["q_max"]}


def markdown(summary):
    rnn=summary["rnn_only"];multi=summary["multi_il"]
    lines=["# Stage2-R-v2 Final Critic Comparison","","| Metric | RNN-only | Multi-IL |","|---|---:|---:|",
           f"| Checkpoint update | {rnn['update']} | {multi['update']} |",
           f"| Val TD MSE | {rnn['val_td_mse']:.10g} | {multi['val_td_mse']:.10g} |",
           f"| Ranking accuracy | {rnn['ranking']:.6g} | {multi['ranking']:.6g} |",
           f"| Success Q | {rnn['success_q']:.10g} | {multi['success_q']:.10g} |",
           f"| Failure Q | {rnn['failure_q']:.10g} | {multi['failure_q']:.10g} |",
           f"| Q-gap | {rnn['q_gap']:.10g} | {multi['q_gap']:.10g} |",
           f"| Q mean | {rnn['q_mean']:.10g} | {multi['q_mean']:.10g} |",
           f"| Q std | {rnn['q_std']:.10g} | {multi['q_std']:.10g} |","",
           f"TD MSE reduction: {summary['comparison']['td_mse_reduction_percent']:.4f}%","",
           f"Q-gap difference: {summary['comparison']['q_gap_difference']:.10g}","",
           f"Q-gap ratio: {summary['comparison']['q_gap_ratio']}","",
           f"Earliest perfect ranking — RNN-only: {summary['ranking_speed']['rnn_only_earliest_perfect_update']}; Multi-IL: {summary['ranking_speed']['multi_il_earliest_perfect_update']}","",
           "NO TRAINING PERFORMED","","STAGE4 NOT STARTED"]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run-dir",default=DEFAULT_RUN);parser.add_argument("--actor-checkpoint",default=DEFAULT_ACTOR);parser.add_argument("--device",default="npu:0");args=parser.parse_args()
    run=Path(args.run_dir).resolve();audited=audit_checkpoints(run);config=read(run/"resolved_config.json")
    if str(Path(config["actor_checkpoint"]).resolve())!=str(Path(args.actor_checkpoint).resolve()):raise RuntimeError("Resolved Stage2-R-v2 Actor differs from requested epoch150 Actor")
    if config["validation_seed_start"]!=10080 or config["validation_seed_end"]!=10099:raise RuntimeError("Formal validation seed split differs from 10080..10099")
    for name in ("rnn_only","multi_il"):
        payload=audited[name][1]
        if str(Path(payload["frozen_actor_checkpoint"]).resolve())!=str(Path(args.actor_checkpoint).resolve()):raise RuntimeError(f"{name} checkpoint frozen Actor metadata mismatch")
        if payload["architecture"]!=architecture(config):raise RuntimeError(f"{name} checkpoint architecture differs from resolved Stage2-R-v2 architecture")
    device=device_of(args.device);set_device(device);actor,actor_payload=load_actor(args.actor_checkpoint,device);actor.requires_grad_(False)
    if int(actor_payload["epoch"])!=150:raise RuntimeError("Frozen Actor checkpoint is not Stage3-R epoch150")
    seeds=list(range(int(config["validation_seed_start"]),int(config["validation_seed_end"])+1));episodes=[]
    for source in POLICIES:
        selected=load_episodes(source,config["datasets"][source],seeds);attach_actor_actions(selected,actor,device);episodes.extend(selected)
    records=validation_sequences(episodes,int(config["sequence_length"]))
    results={}
    for name in ("rnn_only","multi_il"):
        checkpoint,payload,history,_=audited[name];critic,target=load_critic(payload,config,device);results[name]=evaluate(critic,target,records,config,device)
    pairs=pair_table(results["rnn_only"],results["multi_il"]);output=run/"final_comparison";output.mkdir(parents=True,exist_ok=True)
    with open(output/"per_pair_results.csv","w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    distributions={"q_statistic_definition":Q_DEFINITION,"rnn_only":q_distribution(results["rnn_only"]),"multi_il":q_distribution(results["multi_il"])};write(output/"q_statistics.json",distributions)
    rnn=compact(audited["rnn_only"][0],audited["rnn_only"][1],results["rnn_only"]);multi=compact(audited["multi_il"][0],audited["multi_il"][1],results["multi_il"])
    reduction=(rnn["val_td_mse"]-multi["val_td_mse"])/rnn["val_td_mse"]*100.0;difference=multi["q_gap"]-rnn["q_gap"];ratio=None if rnn["q_gap"]==0 else multi["q_gap"]/rnn["q_gap"]
    rnn_perfect=earliest_perfect(audited["rnn_only"][2]);multi_perfect=earliest_perfect(audited["multi_il"][2]);speedup=None if rnn_perfect is None or multi_perfect is None or multi_perfect==0 else rnn_perfect/multi_perfect
    summary={"analysis":"Stage2-R-v2 Final Critic Comparison","actor_checkpoint":str(Path(args.actor_checkpoint).resolve()),"rnn_only":rnn,"multi_il":multi,
             "comparison":{"td_mse_reduction_percent":reduction,"q_gap_difference":difference,"q_gap_ratio":ratio,"multi_lower_td_mse":multi["val_td_mse"]<rnn["val_td_mse"],"multi_higher_q_gap":multi["q_gap"]>rnn["q_gap"]},
             "ranking_speed":{"rnn_only_earliest_perfect_update":rnn_perfect,"multi_il_earliest_perfect_update":multi_perfect,"ranking_speedup_ratio":speedup},
             "validation":{"seeds":seeds,"num_pairs":len(pairs),"sequence_length":int(config["sequence_length"]),"fixed_validation_sequence_count":len(records),"q_statistic_definition":Q_DEFINITION,"pair_construction_definition":PAIR_DEFINITION,"current_q_action":"Stage1 behavior action","next_action":"deterministic frozen Stage3-R epoch150 tanh(mu)","terminal_mask":"terminated OR truncated => no bootstrap"},
             "no_training_performed":True,"stage4_started":False}
    json_path=output/"stage2_r_v2_critic_comparison.json";md_path=output/"stage2_r_v2_critic_comparison.md";write(json_path,summary);md_path.write_text(markdown(summary),encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False));print("Summary JSON:",json_path);print("Summary Markdown:",md_path);print("NO TRAINING PERFORMED");print("STAGE4 NOT STARTED")
if __name__=="__main__":main()
