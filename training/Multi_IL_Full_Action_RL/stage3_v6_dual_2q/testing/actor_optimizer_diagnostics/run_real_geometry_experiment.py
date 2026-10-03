#!/usr/bin/env python3
"""Sequential one-NPU real continuation for optimizer-geometry causality."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
BRANCHES = ("PRODUCTION_ADAM", "SGD_STEP_MATCHED_CONTROL")
MILESTONES = (130000, 140000, 150000, 160000)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def prepare_pair(source_run, output_root, branch):
    pair = output_root / branch
    shared = pair / "shared"
    shared.mkdir(parents=True, exist_ok=False)

    source_shared = source_run / "shared"
    for name in ("config_resolved.json", "stage2_source_manifest.json"):
        shutil.copyfile(source_shared / name, shared / name)

    fairness = read_json(source_shared / "quad_fairness.json")
    fairness = dict(fairness)
    fairness["npu_mapping"] = dict(fairness["npu_mapping"])
    fairness["npu_mapping"]["random2q/multi_q"] = "npu:0"
    fairness["testing_only_device_remap"] = {
        "random2q/multi_q": "npu:0",
        "reason": "sequential single-NPU optimizer-geometry experiment",
    }
    write_json(shared / "quad_fairness.json", fairness)

    actor_init_source = source_shared / "actor_init.pth"
    (shared / "actor_init.pth").symlink_to(actor_init_source)

    return pair


def milestone_record(group, step):
    path = group / "geometry_diagnostics" / f"step_{int(step):07d}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json(path)
    closed = payload["closed_loop"]
    probe = payload["fixed_policy_probe"]
    return {
        "env_steps": int(step),
        "actor_updates": int(payload["actor_updates"]),
        "critic_updates": int(payload["critic_updates"]),
        "success_count": int(closed["success_count"]),
        "success_total": int(closed["count"]),
        "mean_length": float(closed["mean_length"]),
        "sim_error_count": int(closed["sim_error_count"]),
        "parameter_drift_l2": float(probe["parameter_drift"]["total"]["l2"]),
        "sampled_action_drift": float(
            probe["policy_drift"]["sampled_action_l2_mean"]
        ),
        "weighted_action_drift": float(
            probe["policy_drift"]["weighted_action_l2_mean"]
        ),
        "top1_mode_change_fraction": float(
            probe["policy_drift"]["top1_mode_change_fraction"]
        ),
        "categorical_kl_mean": float(
            probe["policy_drift"]["categorical_kl_mean"]
        ),
        "path": str(path),
    }


def summarize_geometry_updates(group):
    path = group / "actor_geometry_updates.jsonl"
    if not path.is_file():
        return {"path": str(path), "rows": 0}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {"path": str(path), "rows": len(rows)}
    if not rows:
        return result
    result["first"] = rows[0]
    result["last"] = rows[-1]
    steps = [
        float(row["actual_parameter_step_l2"])
        for row in rows
        if row.get("actual_parameter_step_l2") is not None
    ]
    if steps:
        import numpy as np
        arr = np.asarray(steps, dtype=np.float64)
        result["actual_parameter_step_l2"] = {
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    ratios = [
        float(row["step_match_ratio"])
        for row in rows
        if row.get("step_match_ratio") is not None
    ]
    if ratios:
        import numpy as np
        arr = np.asarray(ratios, dtype=np.float64)
        result["step_match_ratio"] = {
            "median": float(np.median(arr)),
            "max_abs_error_from_1": float(np.max(np.abs(arr - 1.0))),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results_real_geometry_20261003",
    )
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    if args.device != "npu:0":
        raise RuntimeError("This experiment is intentionally fixed to one logical NPU: npu:0")

    source_run = args.source_run.resolve()
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing geometry experiment: {output_root}"
        )
    output_root.mkdir(parents=True)

    source_checkpoint = (
        source_run
        / "random2q"
        / "multi_q"
        / "checkpoints"
        / "critic_ready.pth"
    )
    source = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if int(source["env_steps"]) != 130000:
        raise RuntimeError(f"Expected random2q critic_ready=130000, got {source['env_steps']}")
    if int(source["actor_updates"]) != 0:
        raise RuntimeError("Source Actor already updated")
    if source["actor_optimizer"].get("state"):
        raise RuntimeError("Source Actor Adam state is not empty")
    if source.get("critic_target_mode") != "random2q" or source.get("group") != "multi_q":
        raise RuntimeError("Wrong source checkpoint")

    production_paths = [
        HERE.parents[1] / "train_stage3_v6_vector.py",
        HERE.parents[1] / "stage3_v6_agent.py",
        HERE.parents[2] / "stage3_v5_rgmm_td3" / "stage3_v5_agent.py",
        HERE.parents[1] / "stage3_v6_config.json",
    ]
    production_before = {
        str(path.resolve()): sha256(path)
        for path in production_paths
    }

    source_contract = {
        "testing_only": True,
        "source_run": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "source_env_steps": int(source["env_steps"]),
        "source_critic_updates": int(source["updates"]),
        "source_actor_updates": int(source["actor_updates"]),
        "source_actor_optimizer_state_entries": len(source["actor_optimizer"].get("state", {})),
        "source_online_replay": source["online_sequence_replay"],
        "branches": list(BRANCHES),
        "target_mode": "random2q",
        "group": "multi_q",
        "device": args.device,
        "target_env_steps": 160000,
        "milestones": list(MILESTONES),
        "production_source_sha256_before": production_before,
        "formal_training_resumed": False,
    }
    write_json(output_root / "experiment_contract.json", source_contract)

    status = {
        "status": "RUNNING",
        "testing_only": True,
        "formal_training_resumed": False,
        "completed": [],
    }
    write_json(output_root / "experiment_status.json", status)

    branch_results = {}
    manifest = read_json(source_run / "shared" / "stage2_source_manifest.json")
    critic_init = manifest["multi_q"]["checkpoint"]

    for branch in BRANCHES:
        pair = prepare_pair(source_run, output_root, branch)
        group = pair / "random2q" / "multi_q"
        group.mkdir(parents=True, exist_ok=True)
        ready_marker = pair / "startup_ready.json"
        logfile = pair / "real_continuation.log"

        env = dict(
            os.environ,
            OPT_GEOMETRY_BRANCH=branch,
            OPT_GEOMETRY_GROUP_OUT=str(group),
            OPT_GEOMETRY_SOURCE_CHECKPOINT=str(source_checkpoint),
            PYTHONUNBUFFERED="1",
        )
        cmd = [
            sys.executable,
            "-u",
            str(HERE / "train_geometry_testing.py"),
            "--group",
            "multi_q",
            "--target-mode",
            "random2q",
            "--device",
            args.device,
            "--quad-run-dir",
            str(pair),
            "--critic-init-checkpoint",
            str(critic_init),
            "--num-envs",
            "16",
            "--total-env-steps",
            "160000",
            "--resume",
            str(source_checkpoint),
            "--startup-ready-file",
            str(ready_marker),
        ]

        status.update(
            active_branch=branch,
            active_log=str(logfile),
            active_command=cmd,
        )
        write_json(output_root / "experiment_status.json", status)
        print(
            "GEOMETRY_BRANCH_START "
            + json.dumps(
                {
                    "branch": branch,
                    "source": str(source_checkpoint),
                    "target": 160000,
                    "log": str(logfile),
                }
            ),
            flush=True,
        )

        started = time.time()
        with logfile.open("x", encoding="utf-8") as handle:
            process = subprocess.Popen(
                cmd,
                cwd=pair,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            status["active_pid"] = int(process.pid)
            write_json(output_root / "experiment_status.json", status)
            returncode = process.wait()

        log_text = logfile.read_text()
        if (
            returncode != 0
            or "GEOMETRY_TESTING_TRAINER_EXITED_ALL_VECTOR_ENVS_CLOSED" not in log_text
        ):
            status.update(
                status="FAILED",
                active_pid=None,
                failure={
                    "branch": branch,
                    "returncode": int(returncode),
                    "log": str(logfile),
                },
            )
            write_json(output_root / "experiment_status.json", status)
            raise RuntimeError(status["failure"])

        milestones = [
            milestone_record(group, step)
            for step in MILESTONES
        ]
        if any(row["sim_error_count"] for row in milestones):
            raise RuntimeError(f"Simulator diagnostic error in branch {branch}")

        final_checkpoint = group / "checkpoints" / "step_0160000.pth"
        if not final_checkpoint.is_file():
            raise FileNotFoundError(final_checkpoint)
        final = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
        if not final.get("testing_only"):
            raise RuntimeError("Testing checkpoint missing testing_only marker")
        if final.get("optimizer_geometry_branch") != branch:
            raise RuntimeError("Testing checkpoint branch marker mismatch")
        if int(final["env_steps"]) != 160000:
            raise RuntimeError("Final testing checkpoint is not 160K")

        branch_result = {
            "branch": branch,
            "duration_seconds": float(time.time() - started),
            "final_checkpoint": str(final_checkpoint),
            "final_actor_updates": int(final["actor_updates"]),
            "final_critic_updates": int(final["updates"]),
            "milestones": milestones,
            "geometry_updates": summarize_geometry_updates(group),
            "trainer_log": str(logfile),
            "software_and_numeric_safety_pass": True,
        }
        branch_results[branch] = branch_result
        status["completed"].append(branch_result)
        status.update(active_branch=None, active_pid=None)
        write_json(output_root / "experiment_status.json", status)
        print(
            "GEOMETRY_BRANCH_PASS "
            + json.dumps(
                {
                    "branch": branch,
                    "success_160K": milestones[-1]["success_count"],
                    "actor_updates": branch_result["final_actor_updates"],
                    "parameter_drift_l2": milestones[-1]["parameter_drift_l2"],
                    "sampled_action_drift": milestones[-1]["sampled_action_drift"],
                    "weighted_action_drift": milestones[-1]["weighted_action_drift"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    adam = branch_results["PRODUCTION_ADAM"]
    control = branch_results["SGD_STEP_MATCHED_CONTROL"]
    comparison = []
    for index, step in enumerate(MILESTONES):
        a = adam["milestones"][index]
        c = control["milestones"][index]
        comparison.append(
            {
                "env_steps": int(step),
                "production_adam_success": int(a["success_count"]),
                "step_matched_control_success": int(c["success_count"]),
                "production_adam_parameter_drift_l2": a["parameter_drift_l2"],
                "step_matched_control_parameter_drift_l2": c["parameter_drift_l2"],
                "production_adam_sampled_action_drift": a["sampled_action_drift"],
                "step_matched_control_sampled_action_drift": c["sampled_action_drift"],
                "production_adam_weighted_action_drift": a["weighted_action_drift"],
                "step_matched_control_weighted_action_drift": c["weighted_action_drift"],
            }
        )

    production_after = {
        str(path.resolve()): sha256(path)
        for path in production_paths
    }
    if production_after != production_before:
        raise RuntimeError("Production source changed during experiment")

    summary = {
        "testing_only": True,
        "source_contract": source_contract,
        "branches": branch_results,
        "comparison": comparison,
        "production_source_sha256_after": production_after,
        "production_source_unchanged": True,
        "formal_training_resumed": False,
        "interpretation_rule": {
            "adam_collapses_control_survives": "ADAM_PRECONDITIONING_GEOMETRY_CAUSAL",
            "both_collapse_similarly": "GLOBAL_PARAMETER_DISPLACEMENT_CAUSAL_NOT_ADAM_GEOMETRY",
            "neither_collapses": "REAL_CONTINUATION_DOES_NOT_REPRODUCE_FROZEN_MECHANISM",
            "otherwise": "INCONCLUSIVE_OR_MIXED",
        },
    }
    write_json(output_root / "analysis_summary.json", summary)
    status.update(status="COMPLETE", active_branch=None, active_pid=None)
    write_json(output_root / "experiment_status.json", status)

    print(
        "REAL_GEOMETRY_EXPERIMENT_COMPLETE "
        + json.dumps(comparison, sort_keys=True),
        flush=True,
    )
    print("FORMAL TRAINING REMAINS STOPPED", flush=True)


if __name__ == "__main__":
    main()
