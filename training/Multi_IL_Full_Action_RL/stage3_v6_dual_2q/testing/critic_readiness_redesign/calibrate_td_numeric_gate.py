#!/usr/bin/env python3
"""TEST-ONLY calibration from existing V6 checkpoints and metric logs."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = Path("/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_multi_mean_random_formal_20260928")
EPS = 1e-6
TD_INTERVAL_INCREASE = 0.35
TD_TWO_INTERVAL_FACTOR = 2.0
MEAN_SHIFT_Z_LIMIT = 0.5
STD_RATIO_LIMIT = 1.5


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quantiles(values):
    return {str(int(p * 100)): float(np.quantile(values, p)) for p in (.5, .75, .9, .95)}


def td_rule(values, rule):
    if len(values) < 3:
        return False
    old, mid, now = map(float, values[-3:])
    if not all(math.isfinite(x) and x >= 0 for x in (old, mid, now)):
        return True
    a = mid / max(old, 1e-12)
    b = now / max(mid, 1e-12)
    total = now / max(old, 1e-12)
    if rule == "A_single_35pct":
        return b >= 1.35
    if rule == "B_recent_best_50pct":
        return now >= 1.5 * min(values[-3:-1])
    if rule == "C_monotonic_50pct_two_step":
        return a > 1 and b > 1 and total >= 1.5
    if rule == "D_two_35pct":
        return a >= 1.35 and b >= 1.35
    if rule == "E_two_35pct_and_2x":
        return a >= 1.35 and b >= 1.35 and total >= 2.0
    raise ValueError(rule)


def pair_scale(current, reference):
    mu, old_mu = current["mean"], reference["mean"]
    sd, old_sd = current["std"], reference["std"]
    if min(sd, old_sd) <= EPS:
        return {"mean_shift_z": float("inf"), "std_ratio": float("inf"),
                "old_relative_mean": float("inf"), "symmetric_mean_shift_z": float("inf"),
                "delta_mean_abs": abs(mu - old_mu), "delta_std_abs": abs(sd - old_sd)}
    return {"mean_shift_z": abs(mu - old_mu) / max(old_sd, EPS),
            "std_ratio": max(sd / old_sd, old_sd / sd),
            "old_relative_mean": abs(mu - old_mu) / max(abs(old_mu), EPS),
            "symmetric_mean_shift_z": abs(mu - old_mu) / max((sd + old_sd) / 2, EPS),
            "delta_mean_abs": abs(mu - old_mu), "delta_std_abs": abs(sd - old_sd)}


def numeric_scale(current, previous, anchor):
    detail = {}
    for source in ("qmean", "td_target"):
        for baseline, reference in (("previous", previous), ("step0", anchor)):
            detail[source + "_vs_" + baseline] = pair_scale(current[source], reference[source])
    z = max(v["mean_shift_z"] for v in detail.values())
    ratio = max(v["std_ratio"] for v in detail.values())
    return {"mean_shift_z": z, "std_ratio": ratio,
            "scale_safe": bool(z <= MEAN_SHIFT_Z_LIMIT and ratio <= STD_RATIO_LIMIT),
            "detail": detail}


def transformed_qmean(base, factor=1.0, offset_sigma=0.0):
    out = {name: dict(value) for name, value in base.items()}
    q = out["qmean"]
    q["mean"] = factor * q["mean"] + offset_sigma * base["qmean"]["std"]
    q["std"] = abs(factor) * q["std"]
    return out


def first_two_consecutive(rows, passes):
    for i in range(1, len(rows)):
        if passes[i - 1] and passes[i]:
            return rows[i]["env_steps"]
    return None


def main():
    rules = ("A_single_35pct", "B_recent_best_50pct", "C_monotonic_50pct_two_step",
             "D_two_35pct", "E_two_35pct_and_2x")
    output = {"constants": {"td_interval_increase": TD_INTERVAL_INCREASE,
                            "td_two_interval_factor": TD_TWO_INTERVAL_FACTOR,
                            "mean_shift_z_limit": MEAN_SHIFT_Z_LIMIT,
                            "std_ratio_limit": STD_RATIO_LIMIT, "epsilon": EPS,
                            "consecutive_readiness_passes": 2},
              "modes": {}, "candidate_rules": {}, "synthetic_td": {},
              "environment_steps_executed": 0, "optimizer_steps_executed": 0}
    for mode in ("mean2q", "random2q"):
        checkpoints = json.loads((HERE / f"calibration_checkpoint_metrics_{mode}.json").read_text())
        rows = checkpoints["rows"]
        assert len(rows) == 4 and [r["env_steps"] for r in rows] == [0,100000,200000,300000]
        assert checkpoints["environment_steps_executed"] == checkpoints["optimizer_steps_executed"] == 0
        enriched = []
        for i, row in enumerate(rows):
            td = row["td"]
            previous = rows[i-1] if i else None
            changes = None if previous is None else {
                key: td[key] / previous["td"][key] - 1
                for key in ("q1_mae", "q2_mae", "members_mae", "member_max_mae", "qmean_mae")}
            scale = None if previous is None else numeric_scale(row["distribution"], previous["distribution"], rows[0]["distribution"])
            enriched.append({"step": row["env_steps"], "td": td, "td_relative_change": changes,
                             "distribution": row["distribution"], "scale": scale,
                             "qmean_spearman": row["ranking"]["qmean"]["spearman"],
                             "finite": row["finite"]})
        training = [r for r in read_jsonl(RUN/mode/"multi_q"/"train_metrics.jsonl") if r["env_steps"] >= 100000]
        losses = np.asarray([r["critic_loss_q1"] + r["critic_loss_q2"] for r in training])
        relative = np.abs(np.diff(losses)) / np.maximum(losses[:-1], 1e-12)
        train_summary = {"samples_from_100k": len(training),
                         "median_env_step_spacing": float(np.median(np.diff([r["env_steps"] for r in training]))),
                         "loss_sum_quantiles": quantiles(losses),
                         "adjacent_absolute_relative_change_quantiles": quantiles(relative),
                         "loss_sum_max": float(losses.max()),
                         "loss_finite_all_samples": bool(np.isfinite(losses).all()),
                         "logged_grad_norm_finite_all_samples": bool(all(math.isfinite(r["critic_grad_norm"]) for r in training))}
        readiness = read_jsonl(RUN/mode/"multi_q"/"readiness_metrics.jsonl")
        assert len(readiness) == 21
        old_td = [r["td_mae"] for r in readiness]
        proxy = []
        for i, row in enumerate(readiness):
            td_healthy_proxy = i >= 2 and not td_rule(old_td[:i+1], "D_two_35pct")
            data = row["env_steps"] >= 100000 and row["completed_episodes"] >= 150 and row["success_episodes"] >= 30 and row["failure_episodes"] >= 30
            rank = row["spearman_q_return"] >= 0.70
            numeric = row["finite"] and row["q_scale_stable"]
            proxy.append({"step": row["env_steps"], "data_proxy": data, "rank_qmin_proxy": rank,
                          "td_old_qmin_proxy": td_healthy_proxy, "numeric_old_qmin_proxy": numeric,
                          "all_pass_proxy": data and rank and td_healthy_proxy and numeric})
        proxy_passes = [r["all_pass_proxy"] for r in proxy]
        output["modes"][mode] = {"fixed_set_sha256": checkpoints["frozen_sequence_sha256"],
                                "fixed_episode_count": checkpoints["fixed_episodes"],
                                "fixed_sequence_count": checkpoints["fixed_sequences"],
                                "checkpoint_rows": enriched, "train_metrics": train_summary,
                                "readiness_resolution_steps": 10000,
                                "readiness_count": len(readiness),
                                "proxy_rows": proxy,
                                "proxy_first_two_pass_step": first_two_consecutive(readiness, proxy_passes)}
        base = rows[-1]["distribution"]
        stress = {}
        for factor in (1.0,1.25,1.5,2.0,5.0,10.0,0.5,0.25,0.1):
            metric = numeric_scale(transformed_qmean(base, factor), base, base)
            stress[str(factor)] = {"mean_shift_z": metric["mean_shift_z"],
                                   "std_ratio": metric["std_ratio"], "safe": metric["scale_safe"]}
        for offset in (0.25,0.5,1.0):
            metric = numeric_scale(transformed_qmean(base, 1.0, offset), base, base)
            stress["plus_"+str(offset)+"_sigma"] = {"mean_shift_z": metric["mean_shift_z"],
                                                      "std_ratio": metric["std_ratio"], "safe": metric["scale_safe"]}
        output["modes"][mode]["synthetic_q_scale"] = stress
        near_zero_ref = {"mean": 0.0, "std": base["qmean"]["std"]}
        near_zero_now = {"mean": 0.01 * base["qmean"]["std"],
                         "std": base["qmean"]["std"]}
        output["modes"][mode]["near_zero_mean_comparison"] = pair_scale(
            near_zero_now, near_zero_ref)
    for rule in rules:
        counts = {}
        for mode in ("mean2q", "random2q"):
            objective = [r["td"]["member_max_mae"] for r in output["modes"][mode]["checkpoint_rows"]]
            old = [json.loads(s)["td_mae"] for s in (RUN/mode/"multi_q"/"readiness_metrics.jsonl").read_text().splitlines() if s.strip()]
            counts[mode] = {"objective_checkpoint_flags": [bool(td_rule(objective[:i+1],rule)) for i in range(len(objective))],
                            "old_qmin_21_proxy_flags": [bool(td_rule(old[:i+1],rule)) for i in range(len(old))]}
            counts[mode]["objective_false_fail_count"] = sum(counts[mode]["objective_checkpoint_flags"])
            counts[mode]["old_qmin_proxy_false_fail_count"] = sum(counts[mode]["old_qmin_21_proxy_flags"])
        output["candidate_rules"][rule] = counts
    for increase in (.10,.25,.50,1.0):
        scenarios = {}
        for intervals in (1,2,3):
            series = [1.0,1.0] + [(1.0+increase)**i for i in range(1,intervals+1)]
            scenarios[str(intervals)] = {rule: bool(any(td_rule(series[:j+1],rule) for j in range(2,len(series)))) for rule in rules}
        output["synthetic_td"][str(increase)] = scenarios
    path = HERE / "td_numeric_calibration.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("OUTPUT",path)
    for mode in output["modes"]:
        d=output["modes"][mode]
        print(mode,"checkpoint scale",[(r["step"],None if r["scale"] is None else (round(r["scale"]["mean_shift_z"],4),round(r["scale"]["std_ratio"],4))) for r in d["checkpoint_rows"]])
        print(mode,"proxy two pass",d["proxy_first_two_pass_step"])
    print("candidate counts",{k:{m:(v[m]["objective_false_fail_count"],v[m]["old_qmin_proxy_false_fail_count"]) for m in v} for k,v in output["candidate_rules"].items()})

if __name__ == "__main__":
    main()
