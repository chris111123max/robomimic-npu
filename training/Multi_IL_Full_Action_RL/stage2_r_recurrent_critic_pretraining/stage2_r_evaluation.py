"""Fixed held-out sequence evaluation using the Stage2.1 ranking protocol."""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
import torch

from stage2_r_critic import aligned, predictions, tensor_batch
from stage2_r_data import POLICIES


def collate(records):
    return {key: np.stack([record[key] for record in records], axis=1)
            for key in ("obs", "obs2", "act", "rew", "term", "mask")}


@torch.no_grad()
def evaluate(critic, target, records, config, device):
    critic.eval(); target.eval(); trajectory_values = defaultdict(list); trajectory_meta = {}
    squared = absolute = 0.0; elements = nonfinite = 0
    values = []; targets = []; hidden_norms = []; starts = []; ends = []
    per_timestep_squared = np.zeros(config["sequence_length"], np.float64)
    per_timestep_count = np.zeros(config["sequence_length"], np.int64)
    size = int(config["validation_sequence_batch_size"])
    for offset in range(0, len(records), size):
        items = records[offset:offset + size]; batch = tensor_batch(collate(items), device)
        q1, q2, bellman = predictions(critic, target, batch, config["gamma"])
        q = torch.minimum(q1, q2); mask = batch["mask"].bool()
        errors = torch.cat((q1 - bellman, q2 - bellman), dim=-1)
        expanded = mask.expand_as(errors); chosen_errors = errors[expanded]
        squared += float(chosen_errors.square().sum().item()); absolute += float(chosen_errors.abs().sum().item()); elements += chosen_errors.numel()
        nonfinite += int((~torch.isfinite(chosen_errors)).sum().item())
        selected_q = q[mask]; selected_target = bellman[mask]
        values.extend(selected_q.cpu().numpy().astype(np.float64).tolist())
        targets.extend(selected_target.cpu().numpy().astype(np.float64).tolist())
        observs, previous_actions, previous_rewards, _ = aligned(batch)
        hidden = critic.get_hidden_states(previous_actions, previous_rewards, observs)[:-1]
        hidden_norms.extend(hidden.norm(dim=-1, keepdim=True)[mask].cpu().numpy().tolist())
        for column, item in enumerate(items):
            valid = int(item["valid"]); sequence_q = q[:valid, column, 0].cpu().numpy().astype(np.float64)
            trajectory_values[(item["source"], item["seed"])].extend(sequence_q.tolist())
            trajectory_meta[(item["source"], item["seed"])] = bool(item["success"])
            starts.append(float(sequence_q[0])); ends.append(float(sequence_q[-1]))
            timestep_error = errors[:valid, column].square().mean(dim=-1).cpu().numpy()
            per_timestep_squared[:valid] += timestep_error; per_timestep_count[:valid] += 1
    rows=[]
    for (source,seed), sequence in sorted(trajectory_values.items()):
        rows.append({"source":source,"seed":seed,"success":trajectory_meta[(source,seed)],
                     "trajectory_score":float(np.mean(sequence)),"length":len(sequence)})
    successful=[row["trajectory_score"] for row in rows if row["success"]]
    failed=[row["trajectory_score"] for row in rows if not row["success"]]
    by_seed={}
    for row in rows:by_seed.setdefault(row["seed"],{})[row["source"]]=row
    correct=ties=total=0;details=[]
    for seed, policies in sorted(by_seed.items()):
        for left,right in itertools.combinations(POLICIES,2):
            if left not in policies or right not in policies:continue
            a,b=policies[left],policies[right]
            if a["success"]==b["success"]:continue
            success,failure=(a,b) if a["success"] else (b,a);margin=success["trajectory_score"]-failure["trajectory_score"]
            total+=1;correct+=int(margin>0);ties+=int(margin==0)
            details.append({"seed":seed,"success_source":success["source"],"failure_source":failure["source"],"margin":margin,"correct":bool(margin>0)})
    q=np.asarray(values);target_values=np.asarray(targets);hn=np.asarray(hidden_norms)
    return {
        "val_td_mse":squared/elements,"val_td_mae":absolute/elements,
        "pairwise_ranking_accuracy":None if not total else correct/total,
        "pairwise_ranking_correct":correct,"pairwise_ranking_ties":ties,"pairwise_ranking_pairs":total,
        "success_q":None if not successful else float(np.mean(successful)),
        "failure_q":None if not failed else float(np.mean(failed)),
        "q_gap":None if not successful or not failed else float(np.mean(successful)-np.mean(failed)),
        "q_mean":float(q.mean()),"q_std":float(q.std()),"q_min":float(q.min()),"q_max":float(q.max()),
        "target_q_mean":float(target_values.mean()),"target_q_std":float(target_values.std()),
        "target_q_min":float(target_values.min()),"target_q_max":float(target_values.max()),
        "hidden_norm_mean":float(hn.mean()),"hidden_norm_std":float(hn.std()),
        "sequence_start_q":float(np.mean(starts)),"sequence_end_q":float(np.mean(ends)),
        "per_timestep_td_mse":[None if count==0 else float(value/count) for value,count in zip(per_timestep_squared,per_timestep_count)],
        "nan_inf_count":nonfinite+int((~np.isfinite(q)).sum())+int((~np.isfinite(target_values)).sum()),
        "validation_sequences":len(records),"effective_timesteps":int(elements//2),
        "trajectory_scores":rows,"pairwise_details":details
    }
