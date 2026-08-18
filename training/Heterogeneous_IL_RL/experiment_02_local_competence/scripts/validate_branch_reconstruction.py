#!/usr/bin/env python3
"""Same-policy reconstruction gate before any cross-policy takeover."""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from utils.env_utils import close_environment, create_environment, initialize_observation_utils, restore_branch_state
from utils.exp00_reader import RNN, TRANSFORMER
from utils.history_utils import action_matches, reconstruct_before_action
from utils.policy_loader import load_policy, release_policy, select_device
from utils.result_utils import atomic_csv, atomic_json, read_csv, read_json, stable_seed
from utils.rng_utils import seed_policy_rng
from utils.state_utils import branch_file_valid, load_branch_state, load_trajectory


FIELDS = ["initial_state_id", "direction", "source_policy", "branch_step", "state_hash_match",
          "history_length", "expected_history_length", "rng_exact_restored", "rng_limitation",
          "action_reconstruction_status", "action_max_abs_error", "action_atol",
          "statistical_consistency", "status", "error_message"]


def _key(row):
    return int(row["initial_state_id"]), int(row["branch_step"])


def aggregate(config_path, run_dir, branch_steps, force=False):
    config, run_dir = read_json(config_path), Path(run_dir)
    summary_rows = read_csv(run_dir / "source_trajectories/source_trajectory_summary.csv")
    complete = [row for row in summary_rows if row["status"] == "complete"]
    output = run_dir / "reconstruction/reconstruction_checks.csv"
    candidates = [] if force or not output.exists() else read_csv(output)
    for path in sorted((run_dir / "reconstruction/workers").glob("worker_*_checks.csv")):
        candidates.extend(read_csv(path))
    existing = sorted({_key(row): row for row in candidates}.values(), key=_key)
    expected = set()
    for item in complete:
        sid = int(item["initial_state_id"])
        for step in branch_steps:
            branch_path = (
                run_dir / "branch_states" / item["direction"].lower()
                / f"state_{sid:06d}_step_{step:03d}.npz"
            )
            if branch_path.exists():
                expected.add((sid, int(step)))
    present = {_key(row) for row in existing}
    missing = sorted(expected - present)
    if missing:
        raise RuntimeError(
            f"Reconstruction aggregation is incomplete: {len(present & expected)}/{len(expected)} checks; "
            f"first missing={missing[0]}"
        )
    existing = [row for row in existing if _key(row) in expected]
    atomic_csv(output, FIELDS, existing)
    failures = [row for row in existing if row["status"] != "pass"]
    exact = sum(str(row["action_reconstruction_status"]) == "pass" for row in existing)
    unavailable = sum(str(row["action_reconstruction_status"]) == "unavailable_exact_rng" for row in existing)
    summary = {"num_checks": len(existing), "passed": len(existing) - len(failures),
               "failed": len(failures), "exact_action_passed": exact,
               "unavailable_exact_rng": unavailable,
               "branch_evaluation_allowed": bool(existing) and not failures}
    selected_counts = {}
    for direction in ("RNN_FAIL_TRANSFORMER_SUCCESS", "TRANSFORMER_FAIL_RNN_SUCCESS"):
        selected_counts[direction] = len({int(r["initial_state_id"]) for r in complete if r["direction"] == direction})
    if all(count >= int(config["reconstruction"]["minimum_cases_per_direction"])
           for count in selected_counts.values()):
        for direction, count in selected_counts.items():
            for step in config["reconstruction"]["required_steps"]:
                checked = len({int(r["initial_state_id"]) for r in existing
                               if r["direction"] == direction and int(r["branch_step"]) == int(step)
                               and r["status"] == "pass"})
                if checked < int(config["reconstruction"]["minimum_cases_per_direction"]):
                    summary["branch_evaluation_allowed"] = False
                    summary.setdefault("coverage_errors", []).append(
                        f"{direction} step={step}: {checked} checks, minimum is {config['reconstruction']['minimum_cases_per_direction']}")
    atomic_json(run_dir / "reconstruction/reconstruction_summary.json", summary)
    print(
        f"[reconstruct] aggregate complete checks={len(existing)}, passed={summary['passed']}, "
        f"failed={summary['failed']}",
        flush=True,
    )
    if not summary["branch_evaluation_allowed"]:
        raise RuntimeError("Reconstruction gate did not pass; cross-policy branching is forbidden")


def reconstruct(config_path, run_dir, branch_steps, force=False,
                worker_id=0, shard_index=0, shard_count=1):
    config, run_dir = read_json(config_path), Path(run_dir)
    summary_rows = read_csv(run_dir / "source_trajectories/source_trajectory_summary.csv")
    complete = sorted(
        [row for row in summary_rows if row["status"] == "complete"],
        key=lambda row: int(row["initial_state_id"]),
    )
    if not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("shard-index must satisfy 0 <= shard-index < shard-count")
    assigned = complete[int(shard_index)::int(shard_count)]
    output = run_dir / "reconstruction/reconstruction_checks.csv"
    worker_path = run_dir / "reconstruction/workers" / f"worker_{int(worker_id)}_checks.csv"
    baseline = [] if force or not output.exists() else read_csv(output)
    local_rows = [] if force or not worker_path.exists() else read_csv(worker_path)
    done = {_key(row) for row in baseline + local_rows if row["status"] == "pass"}
    if force:
        atomic_csv(worker_path, FIELDS, [])
    expected = sum(
        (run_dir / "branch_states" / item["direction"].lower()
         / f"state_{int(item['initial_state_id']):06d}_step_{step:03d}.npz").exists()
        and (int(item["initial_state_id"]), int(step)) not in done
        for item in assigned for step in branch_steps
    )
    print(
        f"[reconstruct worker={worker_id}] shard={shard_index}/{shard_count} "
        f"cases={len(assigned)} pending_checks={expected}",
        flush=True,
    )
    if expected == 0:
        return
    initialize_observation_utils(config["policies"][RNN]["checkpoint_path"])
    env, _ = create_environment(config["dataset_path"]); device = select_device()
    try:
        for policy_name in (RNN, TRANSFORMER):
            policy, _, _ = load_policy(policy_name, config["policies"][policy_name]["checkpoint_path"], device)
            try:
                for item in [row for row in assigned if row["source_policy"] == policy_name]:
                    sid = int(item["initial_state_id"]); direction_dir = item["direction"].lower()
                    trajectory = load_trajectory(run_dir / item["trajectory_path"])
                    for step in branch_steps:
                        if (sid, step) in done:
                            continue
                        branch_path = run_dir / "branch_states" / direction_dir / f"state_{sid:06d}_step_{step:03d}.npz"
                        if not branch_path.exists():
                            continue  # source environment_done before this decision point
                        row = {"initial_state_id": sid, "direction": item["direction"],
                               "source_policy": policy_name, "branch_step": step,
                               "state_hash_match": 0, "history_length": "",
                               "expected_history_length": step, "rng_exact_restored": 0,
                               "rng_limitation": "", "action_reconstruction_status": "",
                               "action_max_abs_error": "", "action_atol": config["action_reconstruction_atol"],
                               "statistical_consistency": 0, "status": "error", "error_message": ""}
                        try:
                            if not branch_file_valid(branch_path, step):
                                raise RuntimeError("Branch state file failed its persisted hash validation")
                            branch = load_branch_state(branch_path)
                            _, restored_hash = restore_branch_state(
                                env, branch,
                                observation_atol=float(config.get("observation_reconstruction_atol", 1e-6)),
                            )
                            row["state_hash_match"] = int(restored_hash == branch["metadata"]["state_hash"])
                            history = reconstruct_before_action(policy, trajectory["observations"], step,
                                                                rng_state=branch["rng_state"])
                            row["history_length"] = history["history_length"]
                            row["rng_exact_restored"] = int(history["exact_rng_restored"])
                            row["rng_limitation"] = history["rng_limitation"] or ""
                            reconstructed_action = np.asarray(policy(ob=branch["observation"]))
                            match, error = action_matches(branch["original_action"], reconstructed_action,
                                                          config["action_reconstruction_atol"])
                            row["action_max_abs_error"] = error
                            if history["exact_rng_restored"]:
                                row["action_reconstruction_status"] = "pass" if match else "fail"
                                row["statistical_consistency"] = int(match)
                                if not match:
                                    raise RuntimeError(f"Exact-RNG reconstructed action differs by {error}")
                            else:
                                # Exact accelerator RNG is not available. Rebuild history afresh and
                                # check three independently seeded continuation actions are finite,
                                # correctly shaped, and inside the environment action bounds.
                                valid = []
                                for trial in range(3):
                                    trial_seed = stable_seed(config["seed_base"], "recon_stat", sid, step, trial)
                                    paired_actions = []
                                    for _ in range(2):
                                        reconstruct_before_action(policy, trajectory["observations"], step,
                                                                  trial_seed=trial_seed)
                                        paired_actions.append(np.asarray(policy(ob=branch["observation"])))
                                    action = paired_actions[0]
                                    action_low, action_high = env.env.action_spec
                                    valid.append(action.shape == (int(config["action_dimension"]),) and
                                                 np.all(np.isfinite(action)) and
                                                 np.all(action >= np.asarray(action_low) - 1e-6) and
                                                 np.all(action <= np.asarray(action_high) + 1e-6) and
                                                 np.allclose(paired_actions[0], paired_actions[1], rtol=0.0,
                                                             atol=float(config["action_reconstruction_atol"])))
                                row["statistical_consistency"] = int(all(valid))
                                row["action_reconstruction_status"] = "unavailable_exact_rng"
                                if not all(valid):
                                    raise RuntimeError("Statistical continuation produced invalid actions")
                            row["status"] = "pass"
                            print(f"[reconstruct worker={worker_id}] case {sid:06d} step {step} "
                                  f"state hash=OK history length={step} "
                                  f"action reconstruction={row['action_reconstruction_status']}", flush=True)
                        except Exception as exc:
                            row["error_message"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                        local_rows = [r for r in local_rows if _key(r) != (sid, int(step))] + [row]
                        atomic_csv(worker_path, FIELDS, sorted(local_rows, key=_key))
                        if row["status"] != "pass" and config["reconstruction"].get("fail_fast", True):
                            raise RuntimeError(row["error_message"])
            finally:
                release_policy(policy)
    finally:
        close_environment(env)
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True); parser.add_argument("--branch-steps", required=True)
    parser.add_argument("--worker-id", type=int); parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1); parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(); steps = [int(x) for x in args.branch_steps.split(",") if x]
    if args.aggregate_only:
        aggregate(args.config, args.run_dir, steps, args.force)
    else:
        worker_id = 0 if args.worker_id is None else args.worker_id
        reconstruct(args.config, args.run_dir, steps, args.force,
                    worker_id, args.shard_index, args.shard_count)
        if args.worker_id is None:
            aggregate(args.config, args.run_dir, steps, args.force)


if __name__ == "__main__":
    main()
