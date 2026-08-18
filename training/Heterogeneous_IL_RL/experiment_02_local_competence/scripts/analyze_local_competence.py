#!/usr/bin/env python3
"""Aggregate preregistered local-competence and rescue-window statistics."""

import argparse
import statistics
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from utils.result_utils import atomic_csv, atomic_json, bool_value, read_csv, read_json


def mean(values):
    return statistics.fmean(values) if values else None


def analyze(config_path, run_dir):
    config, run_dir = read_json(config_path), Path(run_dir)
    cases = read_csv(run_dir / "recheck/case_summary.csv")
    source = read_csv(run_dir / "source_trajectories/source_trajectory_summary.csv")
    reconstruction = read_json(run_dir / "reconstruction/reconstruction_summary.json")
    trials = read_csv(run_dir / "branch_results/raw_trials.csv")
    errors = read_csv(run_dir / "branch_results/errors.csv") if (run_dir / "branch_results/errors.csv").exists() else []
    if errors:
        raise RuntimeError("Cannot analyze branch runtime errors as task failures")
    grouped = {}
    for row in trials:
        key = (int(row["initial_state_id"]), row["direction"], int(row["branch_step"]))
        grouped.setdefault(key, []).append(row)
    states = []
    for (sid, direction, step), rows in sorted(grouped.items()):
        source_rows = [r for r in rows if r["role"] == "source_continuation"]
        target_rows = [r for r in rows if r["role"] == "target_takeover"]
        if not source_rows or not target_rows:
            continue
        sr = sum(int(r["success"]) for r in source_rows) / len(source_rows)
        tr = sum(int(r["success"]) for r in target_rows) / len(target_rows)
        strong = tr >= float(config["strong_rescue_target_min"]) and sr <= float(config["strong_rescue_source_max"])
        states.append({"initial_state_id": sid, "direction": direction, "branch_step": step,
                       "source_policy": source_rows[0]["source_policy"], "target_policy": source_rows[0]["target_policy"],
                       "source_trials": len(source_rows), "target_trials": len(target_rows),
                       "source_success_rate": sr, "target_success_rate": tr,
                       "local_competence_gap": tr - sr, "strong_local_rescue": int(strong),
                       "stable_cross_success_flag": source_rows[0]["stable_cross_success_flag"]})
    if not states:
        raise RuntimeError("No complete paired branch states to analyze")
    atomic_csv(run_dir / "analysis/local_competence_states.csv", list(states[0]), states)

    episodes = []
    for (sid, direction) in sorted(set((r["initial_state_id"], r["direction"]) for r in states)):
        rows = [r for r in states if r["initial_state_id"] == sid and r["direction"] == direction]
        rescued = sorted(int(r["branch_step"]) for r in rows if r["strong_local_rescue"])
        episodes.append({"initial_state_id": sid, "direction": direction,
                         "num_valid_branch_points": len(rows), "episode_has_local_rescue": int(bool(rescued)),
                         "earliest_rescue_step": rescued[0] if rescued else "",
                         "latest_rescue_step": rescued[-1] if rescued else "",
                         "num_rescuable_branch_points": len(rescued)})
    atomic_csv(run_dir / "analysis/episode_rescue_summary.csv", list(episodes[0]), episodes)

    direction_rows = []
    for direction in ("RNN_FAIL_TRANSFORMER_SUCCESS", "TRANSFORMER_FAIL_RNN_SUCCESS"):
        srows = [r for r in states if r["direction"] == direction]
        erows = [r for r in episodes if r["direction"] == direction]
        rescued = sum(r["episode_has_local_rescue"] for r in erows)
        direction_rows.append({"direction": direction, "valid_episodes": len(erows),
            "episodes_with_local_rescue": rescued,
            "episode_rescue_rate": rescued / len(erows) if erows else None,
            "num_valid_branch_states": len(srows),
            "strong_rescue_branch_state_rate": sum(r["strong_local_rescue"] for r in srows) / len(srows) if srows else None,
            "mean_local_competence_gap": mean([r["local_competence_gap"] for r in srows])})
    atomic_csv(run_dir / "analysis/direction_summary.csv", list(direction_rows[0]), direction_rows)

    step_rows = []
    for direction in ("RNN_FAIL_TRANSFORMER_SUCCESS", "TRANSFORMER_FAIL_RNN_SUCCESS"):
        direction_episode_count = len([r for r in episodes if r["direction"] == direction])
        for step in config["branch_steps"]:
            rows = [r for r in states if r["direction"] == direction and r["branch_step"] == int(step)]
            step_rows.append({"direction": direction, "branch_step": int(step), "num_valid_states": len(rows),
                "source_mean_success_rate": mean([r["source_success_rate"] for r in rows]),
                "target_mean_success_rate": mean([r["target_success_rate"] for r in rows]),
                "mean_local_gap": mean([r["local_competence_gap"] for r in rows]),
                "strong_rescue_state_rate": (sum(r["strong_local_rescue"] for r in rows) / len(rows)) if rows else None,
                "episode_coverage": len({r["initial_state_id"] for r in rows}) / direction_episode_count if direction_episode_count else None})
    atomic_csv(run_dir / "analysis/branch_step_summary.csv", list(step_rows[0]), step_rows)

    stable = sum(bool_value(row["stable_cross_success"]) for row in cases)
    built = sum(row["status"] == "complete" for row in source)
    unresolved = sum(row["status"] == "source_failure_not_reproduced" for row in source)
    summary = {
        "source_experiment": read_json(run_dir / "source_run_manifest.json")["source_run"],
        "cross_success_candidates": {
            "rnn_fail_transformer_success": sum(r["direction"] == "RNN_FAIL_TRANSFORMER_SUCCESS" for r in cases),
            "transformer_fail_rnn_success": sum(r["direction"] == "TRANSFORMER_FAIL_RNN_SUCCESS" for r in cases),
            "total": len(cases)},
        "stable_cross_success": stable, "unstable_cross_success": len(cases) - stable,
        "source_trajectories_built": built, "unresolved_source_failures": unresolved,
        "reconstruction": reconstruction,
        "directions": {row["direction"]: row for row in direction_rows},
        "branch_steps": step_rows,
        "rescue_windows": episodes,
        "runtime_errors": len(errors),
        "strong_rescue_definition": {"target_min": config["strong_rescue_target_min"],
                                     "source_max": config["strong_rescue_source_max"]},
        "claim_automatically_declared": False,
    }
    atomic_json(run_dir / "analysis/local_competence_summary.json", summary)
    make_plots(run_dir, states, step_rows)

    print("================ Experiment 02 Local Competence ================")
    print(f"Source Experiment:\n{summary['source_experiment']}")
    print("\nCross-success candidates:")
    print(f"RNN fail / T success        : {summary['cross_success_candidates']['rnn_fail_transformer_success']}")
    print(f"T fail / RNN success        : {summary['cross_success_candidates']['transformer_fail_rnn_success']}")
    print(f"Total                       : {len(cases)}")
    print(f"\nStable cross-success         : {stable} (unstable={len(cases)-stable})")
    print(f"Source trajectories built    : {built} (unresolved={unresolved})")
    print(f"Reconstruction               : passed={reconstruction['passed']} failed={reconstruction['failed']} "
          f"unavailable-exact-RNG={reconstruction['unavailable_exact_rng']}")
    for row in direction_rows:
        print(f"\n{row['direction']}:")
        print(f"valid episodes               : {row['valid_episodes']}")
        print(f"episodes with local rescue   : {row['episodes_with_local_rescue']}")
        print(f"episode rescue rate          : {row['episode_rescue_rate']}")
        print(f"strong rescue state rate     : {row['strong_rescue_branch_state_rate']}")
        print(f"mean local gap               : {row['mean_local_competence_gap']}")
    print("\nRescue by branch step:")
    for row in step_rows:
        print(f"{row['direction']} step={row['branch_step']} gap={row['mean_local_gap']} "
              f"strong_rate={row['strong_rescue_state_rate']}")
    print(f"\nRuntime errors               : {len(errors)}")
    print("===============================================================")


def make_plots(run_dir, states, step_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = Path(run_dir) / "analysis/plots"
    fig, ax = plt.subplots(figsize=(8, 5))
    for direction in sorted({r["direction"] for r in step_rows}):
        rows = [r for r in step_rows if r["direction"] == direction and r["strong_rescue_state_rate"] is not None]
        ax.plot([r["branch_step"] for r in rows], [r["strong_rescue_state_rate"] for r in rows], marker="o", label=direction)
    ax.set(xlabel="Branch decision step t", ylabel="Strong rescue state rate", ylim=(0, 1)); ax.legend(); fig.tight_layout()
    fig.savefig(plot_dir / "rescue_rate_vs_step.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for direction in sorted({r["direction"] for r in states}):
        ax.hist([r["local_competence_gap"] for r in states if r["direction"] == direction],
                bins=13, alpha=0.55, label=direction)
    ax.set(xlabel="Target success rate - source success rate", ylabel="Branch-state count"); ax.legend(); fig.tight_layout()
    fig.savefig(plot_dir / "local_gap_distribution.png", dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True); args = parser.parse_args(); analyze(args.config, args.run_dir)


if __name__ == "__main__":
    main()
