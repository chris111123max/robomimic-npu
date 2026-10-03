#!/usr/bin/env python3
"""TEST-ONLY counterfactual gate replay from existing readiness JSONL."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = Path("/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/stage3v6_multi_mean_random_formal_20260928")
DESIGNS = ("CURRENT", "NO_STRICT_PLATEAU", "NO_TWIN_P95_HARD",
           "NO_PLATEAU_P95_SOFT", "FINAL_MINIMAL_LOG_PROXY")


def passes(row, design):
    flags = dict(row["individual_gate_flags"])
    if design in ("NO_STRICT_PLATEAU", "NO_PLATEAU_P95_SOFT", "FINAL_MINIMAL_LOG_PROXY"):
        flags.pop("td_not_plateau")
    if design in ("NO_TWIN_P95_HARD", "NO_PLATEAU_P95_SOFT"):
        flags["twin_q_disagreement_high"] = row["twin_q_disagreement_median"] > 0.1
    if design == "FINAL_MINIMAL_LOG_PROXY":
        r = row
        data_ready = (r["env_steps"] >= 100000 and r["completed_episodes"] >= 150
                      and r["success_episodes"] >= 30 and r["failure_episodes"] >= 30)
        rank_ready = r["spearman_q_return"] >= 0.7
        td_healthy_proxy = not r["td_error_worsening"]
        numeric_safe = bool(r["finite"] and r["q_scale_stable"])
        flags = {"DATA_READY": not data_ready, "RANK_READY_QMIN_PROXY": not rank_ready,
                 "TD_HEALTHY_OLD_QMIN_PROXY": not td_healthy_proxy,
                 "NUMERIC_SAFE": not numeric_safe}
    return not any(flags.values()), [key for key, fail in flags.items() if fail]


def first_consecutive(rows, positive, count):
    for i in range(count - 1, len(rows)):
        if all(positive[i-count+1:i+1]):
            return rows[i]["env_steps"]
    return None


def main():
    output = {"source_run": str(RUN), "readiness_checks_per_mode": 21,
              "note": "FINAL_MINIMAL_LOG_PROXY uses historical Qmin Spearman and Qmin TD worsening; future Qmean ranking and objective-aligned TD lack 21-point logs.",
              "modes": {}}
    for mode in ("mean2q", "random2q"):
        path = RUN / mode / "multi_q" / "readiness_metrics.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(rows) == 21 and [r["env_steps"] for r in rows] == list(range(100000, 300001, 10000))
        mode_output = {"source": str(path), "designs": {}}
        for design in DESIGNS:
            evaluations = [passes(r, design) for r in rows]
            positive = [v[0] for v in evaluations]
            mode_output["designs"][design] = {
                "pass_count": sum(positive),
                "pass_steps": [r["env_steps"] for r, ok in zip(rows, positive) if ok],
                "opening_step_by_consecutive_passes": {str(n): first_consecutive(rows, positive, n) for n in (1, 2, 3)},
                "checks": [{"env_steps": r["env_steps"], "pass": ok, "failed_conditions": failed}
                           for r, (ok, failed) in zip(rows, evaluations)],
            }
        output["modes"][mode] = mode_output
    path = HERE / "critic_readiness_replay.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({mode: {design: val["opening_step_by_consecutive_passes"]
                             for design, val in data["designs"].items()}
                      for mode, data in output["modes"].items()}, indent=2))
    print("OUTPUT", path)

if __name__ == "__main__":
    main()
