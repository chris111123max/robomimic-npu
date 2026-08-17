#!/usr/bin/env python3
"""Analyze paired success sets, overlap, oracle gain, and rescue behavior."""

import argparse
import statistics
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from utils.result_utils import POLICY_ORDER, atomic_csv, atomic_json, collect_episode_records, read_json


DISPLAY = {
    "bc": "BC", "bc_gmm": "BC-GMM", "bc_gmm_rnn": "BC-GMM-RNN",
    "bc_gmm_transformer": "BC-GMM-Transformer",
}


def mean(values):
    return statistics.fmean(values) if values else None


def std(values):
    return statistics.pstdev(values) if values else None


def analyze(run_dir):
    run_dir = Path(run_dir)
    seed_manifest = read_json(run_dir / "seed_manifest.json")
    seeds = seed_manifest["environment_seeds"]
    records = collect_episode_records(run_dir)
    by_pair = {(row.get("policy_name"), int(row.get("initial_state_id", -1))): row for row in records}
    runtime_errors = [row for row in records if row.get("status") == "error"]
    missing_pairs = []
    success_rows = []
    paired_complete_ids = []
    for state_id, seed in enumerate(seeds):
        row = {"initial_state_id": state_id, "seed": seed}
        complete = True
        values = []
        for policy_name in POLICY_ORDER:
            result = by_pair.get((policy_name, state_id))
            if result is None:
                row[policy_name] = "MISSING"
                missing_pairs.append({"policy_name": policy_name, "initial_state_id": state_id})
                complete = False
            elif result.get("status") == "error":
                row[policy_name] = "ERROR"
                complete = False
            else:
                value = int(bool(result["success"]))
                row[policy_name] = value
                values.append(value)
        row["any_success"] = max(values) if complete else ""
        if complete:
            paired_complete_ids.append(state_id)
        success_rows.append(row)

    analysis_dir = run_dir / "analysis"
    atomic_csv(
        analysis_dir / "success_matrix.csv",
        ["initial_state_id", "seed", *POLICY_ORDER, "any_success"], success_rows,
    )

    summaries = []
    rates = {}
    for policy_name in POLICY_ORDER:
        valid = [by_pair[(policy_name, state_id)] for state_id in range(len(seeds))
                 if (policy_name, state_id) in by_pair and by_pair[(policy_name, state_id)].get("status") == "complete"]
        successful = [row for row in valid if bool(row["success"])]
        returns = [float(row["episode_return"]) for row in valid]
        lengths = [int(row["episode_length"]) for row in valid]
        success_steps = [int(row["success_step"]) for row in successful]
        rate = len(successful) / len(valid) if valid else None
        rates[policy_name] = rate
        summaries.append({
            "policy_name": policy_name, "num_valid_episodes": len(valid), "num_success": len(successful),
            "success_rate": rate, "mean_episode_return": mean(returns), "std_episode_return": std(returns),
            "mean_episode_length": mean(lengths), "std_episode_length": std(lengths),
            "mean_success_step": mean(success_steps),
            "median_success_step": statistics.median(success_steps) if success_steps else None,
        })
    atomic_csv(analysis_dir / "policy_summary.csv", list(summaries[0]), summaries)

    paired_count = len(paired_complete_ids)
    paired_success = {
        policy: {state_id: bool(by_pair[(policy, state_id)]["success"]) for state_id in paired_complete_ids}
        for policy in POLICY_ORDER
    }
    paired_rates = {
        policy: sum(values.values()) / paired_count if paired_count else None
        for policy, values in paired_success.items()
    }
    best_policy = max(POLICY_ORDER, key=lambda policy: (-1 if paired_rates[policy] is None else paired_rates[policy], -POLICY_ORDER.index(policy)))
    best_rate = paired_rates[best_policy]
    oracle_count = sum(any(paired_success[policy][sid] for policy in POLICY_ORDER) for sid in paired_complete_ids)
    oracle_rate = oracle_count / paired_count if paired_count else None
    all_fail_count = paired_count - oracle_count
    best_fail_count = sum(not paired_success[best_policy][sid] for sid in paired_complete_ids)
    unique_rescue = {}
    conditional_rescue = {}
    for policy in POLICY_ORDER:
        count = sum(paired_success[policy][sid] and not paired_success[best_policy][sid] for sid in paired_complete_ids)
        unique_rescue[policy] = {"count": count, "rate": count / paired_count if paired_count else None}
        conditional_rescue[policy] = count / best_fail_count if best_fail_count else None

    rescue_rows = []
    pairwise = {}
    for policy_i in POLICY_ORDER:
        pairwise[policy_i] = {}
        for policy_j in POLICY_ORDER:
            both_success = sum(paired_success[policy_i][sid] and paired_success[policy_j][sid] for sid in paired_complete_ids)
            both_fail = sum(not paired_success[policy_i][sid] and not paired_success[policy_j][sid] for sid in paired_complete_ids)
            i_success_j_fail = sum(paired_success[policy_i][sid] and not paired_success[policy_j][sid] for sid in paired_complete_ids)
            i_fail_j_success = sum(not paired_success[policy_i][sid] and paired_success[policy_j][sid] for sid in paired_complete_ids)
            data = {
                "policy_i": policy_i, "policy_j": policy_j, "num_paired": paired_count,
                "both_success": both_success, "both_fail": both_fail,
                "i_success_j_fail": i_success_j_fail,
                "i_success_j_fail_rate": i_success_j_fail / paired_count if paired_count else None,
                "i_fail_j_success": i_fail_j_success,
                "i_fail_j_success_rate": i_fail_j_success / paired_count if paired_count else None,
                "agreement_count": both_success + both_fail,
                "agreement_rate": (both_success + both_fail) / paired_count if paired_count else None,
            }
            rescue_rows.append(data)
            pairwise[policy_i][policy_j] = {key: value for key, value in data.items() if key not in ("policy_i", "policy_j")}
    atomic_csv(analysis_dir / "rescue_matrix.csv", list(rescue_rows[0]), rescue_rows)

    expected = len(seeds) * len(POLICY_ORDER)
    summary = {
        "analysis_complete": paired_count == len(seeds) and not runtime_errors and not missing_pairs,
        "num_initial_conditions": len(seeds), "num_policies": len(POLICY_ORDER),
        "expected_rollouts": expected, "num_fully_paired_valid_initial_conditions": paired_count,
        "policy_success_rates": rates, "paired_policy_success_rates": paired_rates,
        "best_single_policy": best_policy, "best_single_success_rate": best_rate,
        "oracle_success_count": oracle_count, "oracle_success_rate": oracle_rate,
        "oracle_gap": None if oracle_rate is None or best_rate is None else oracle_rate - best_rate,
        "all_fail_count": all_fail_count, "all_fail_rate": all_fail_count / paired_count if paired_count else None,
        "unique_rescue": unique_rescue, "conditional_rescue": conditional_rescue,
        "pairwise": pairwise, "runtime_errors": len(runtime_errors), "missing_pairs": len(missing_pairs),
    }
    atomic_json(analysis_dir / "complementarity_summary.json", summary)

    print("=" * 60)
    print("Heterogeneous IL Policy Complementarity")
    print("=" * 60)
    print(f"Valid initial conditions: {paired_count}")
    print()
    for policy in POLICY_ORDER:
        count = sum(paired_success[policy].values())
        rate = paired_rates[policy]
        text = "N/A" if rate is None else f"{count} / {paired_count} = {rate:.3f}"
        print(f"{DISPLAY[policy] + ' Success Rate':36s}: {text}")
    print()
    print(f"Best Single Policy              : {best_policy}")
    print(f"Best Single Success Rate        : {'N/A' if best_rate is None else f'{best_rate:.3f}'}")
    print(f"Episode Oracle Success          : {'N/A' if oracle_rate is None else f'{oracle_count} / {paired_count} = {oracle_rate:.3f}'}")
    oracle_gap_text = "N/A" if summary["oracle_gap"] is None else "+{:.3f}".format(summary["oracle_gap"])
    print(f"Oracle Gap                      : {oracle_gap_text}")
    print(f"All Policies Fail               : {'N/A' if not paired_count else f'{all_fail_count} / {paired_count} = {all_fail_count / paired_count:.3f}'}")
    print("\nRescue of Best Policy Failures:")
    for policy in POLICY_ORDER:
        rescue = unique_rescue[policy]
        conditional = conditional_rescue[policy]
        print(f"{DISPLAY[policy]:32s}: {rescue['count']} total; conditional={'N/A' if conditional is None else f'{conditional:.3f}'}")
    print(f"\nRuntime Errors                  : {len(runtime_errors)}")
    print(f"Missing Pairs                   : {len(missing_pairs)}")
    print(f"Analysis Complete               : {summary['analysis_complete']}")
    print("=" * 60)
    if not summary["analysis_complete"]:
        raise RuntimeError("Analysis is incomplete; ERROR/MISSING cells were not interpreted as failures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    analyze(args.run_dir)


if __name__ == "__main__":
    main()
