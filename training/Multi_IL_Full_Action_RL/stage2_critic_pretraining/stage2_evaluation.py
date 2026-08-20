"""Shared validation and same-seed evaluation for all three critics."""

from __future__ import annotations

import itertools

import numpy as np
import torch


POLICIES = ("bc_rnn", "bc_transformer", "bc_gmm")


def tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate_critic(critic, target, datasets, gamma, device, batch_size=4096):
    critic.eval()
    target.eval()
    td_squared_sum = td_absolute_sum = 0.0
    td_elements = 0
    predicted_values = []
    trajectory_rows = []
    quartile_values = [[] for _ in range(4)]
    nonfinite = 0

    for source in POLICIES:
        for episode in datasets[source].episodes:
            episode_q = []
            for start in range(0, episode.length, int(batch_size)):
                stop = min(episode.length, start + int(batch_size))
                state = tensor(episode.state[start:stop], device)
                action = tensor(episode.action[start:stop], device)
                reward = tensor(episode.reward[start:stop], device)
                next_state = tensor(episode.next_state[start:stop], device)
                next_action = tensor(episode.next_target_action[start:stop], device)
                mask = tensor(episode.bootstrap_mask[start:stop], device)
                target_q1, target_q2 = target(next_state, next_action)
                bellman = reward + float(gamma) * mask * torch.minimum(target_q1, target_q2)
                q1, q2 = critic(state, action)
                q_min = torch.minimum(q1, q2)
                errors = torch.cat((q1 - bellman, q2 - bellman), dim=0)
                td_squared_sum += float(torch.square(errors).sum().item())
                td_absolute_sum += float(torch.abs(errors).sum().item())
                td_elements += int(errors.numel())
                nonfinite += int((~torch.isfinite(errors)).sum().item())
                values = q_min.squeeze(-1).detach().cpu().numpy().astype(np.float64)
                episode_q.extend(values.tolist())
                predicted_values.extend(values.tolist())
            trajectory_score = float(np.mean(episode_q))
            trajectory_rows.append({
                "source": source,
                "seed": episode.seed,
                "success": episode.success,
                "trajectory_score": trajectory_score,
                "length": episode.length,
            })
            if episode.success:
                for timestep, value in enumerate(episode_q):
                    quartile = min(3, int(4 * timestep / max(episode.length, 1)))
                    quartile_values[quartile].append(value)

    values = np.asarray(predicted_values, dtype=np.float64)
    successful = [row["trajectory_score"] for row in trajectory_rows if row["success"]]
    failed = [row["trajectory_score"] for row in trajectory_rows if not row["success"]]
    by_seed = {}
    for row in trajectory_rows:
        by_seed.setdefault(row["seed"], {})[row["source"]] = row
    correct = total_pairs = ties = 0
    pair_details = []
    for seed, policies in sorted(by_seed.items()):
        for left, right in itertools.combinations(POLICIES, 2):
            if left not in policies or right not in policies:
                continue
            first, second = policies[left], policies[right]
            if first["success"] == second["success"]:
                continue
            success_row, failure_row = (first, second) if first["success"] else (second, first)
            margin = success_row["trajectory_score"] - failure_row["trajectory_score"]
            total_pairs += 1
            correct += int(margin > 0.0)
            ties += int(margin == 0.0)
            pair_details.append({
                "seed": seed,
                "success_source": success_row["source"],
                "failure_source": failure_row["source"],
                "success_score": success_row["trajectory_score"],
                "failure_score": failure_row["trajectory_score"],
                "margin": margin,
                "correct": bool(margin > 0.0),
            })

    return {
        "num_transitions": len(predicted_values),
        "td_mse": td_squared_sum / td_elements,
        "td_mae": td_absolute_sum / td_elements,
        "predicted_q_mean": float(values.mean()),
        "predicted_q_std": float(values.std()),
        "predicted_q_min": float(values.min()),
        "predicted_q_max": float(values.max()),
        "success_trajectory_q_mean": None if not successful else float(np.mean(successful)),
        "failure_trajectory_q_mean": None if not failed else float(np.mean(failed)),
        "pairwise_ranking_accuracy": None if not total_pairs else correct / total_pairs,
        "pairwise_ranking_correct": correct,
        "pairwise_ranking_ties": ties,
        "pairwise_ranking_pairs": total_pairs,
        "pairwise_details": pair_details,
        "q_by_trajectory_quartile": {
            label: None if not items else float(np.mean(items))
            for label, items in zip(("0-25%", "25-50%", "50-75%", "75-100%"), quartile_values)
        },
        "trajectory_scores": trajectory_rows,
        "nan_inf_count": nonfinite + int((~np.isfinite(values)).sum()),
    }

