#!/usr/bin/env python3
"""Sequential real online component-mean preservation necessity experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BRANCHES = ("FULL_PRODUCTION", "FULL_MEAN_PRESERVATION")
MILESTONES = (130000, 140000, 150000, 160000)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


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
        "reason": "sequential one-NPU component-mean preservation necessity test",
    }
    write_json(shared / "quad_fairness.json", fairness)
    (shared / "actor_init.pth").symlink_to(source_shared / "actor_init.pth")
    return pair


def milestone_record(group, step):
    path = group / "module_diagnostics" / f"step_{int(step):07d}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json(path)
    probe = payload["fixed_policy_probe"]
    policy = probe["policy_drift"]
    closed = payload["closed_loop"]
    return {
        "env_steps": int(step),
        "actor_updates": int(payload["actor_updates"]),
        "critic_updates": int(payload["critic_updates"]),
        "success_count": int(closed["success_count"]),
        "success_total": int(closed["count"]),
        "mean_length": float(closed["mean_length"]),
        "sim_error_count": int(closed["sim_error_count"]),
        "parameter_drift_l2": float(probe["parameter_drift"]["total"]["l2"]),
        "sampled_action_drift": float(policy["sampled_action_l2_mean"]),
        "weighted_action_drift": float(policy["weighted_action_l2_mean"]),
        "hidden_l2": float(policy["hidden_l2_mean"]),
        "component_mean_rms": float(policy["component_mean_rms"]),
        "logits_rms": float(policy["logits_rms"]),
        "categorical_kl_mean": float(policy["categorical_kl_mean"]),
        "top1_mode_change_fraction": float(
            policy["top1_mode_change_fraction"]
        ),
        "path": str(path),
    }


def preservation_summary(group):
    path = group / "mean_preservation_updates.jsonl"
    if not path.is_file():
        return {
            "path": str(path),
            "rows": 0,
            "present": False,
        }
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        return {"path": str(path), "rows": 0, "present": True}

    def arr(key):
        return np.asarray(
            [
                float(row[key])
                for row in rows
                if row.get(key) is not None
                and math.isfinite(float(row[key]))
            ],
            dtype=np.float64,
        )

    severity = arr("controller_severity")
    lam = arr("lambda")
    ratio = arr("actual_anchor_RL_grad_ratio")
    mean_before = arr("mean_rms_before")
    mean_after = arr("mean_rms_after")
    q_gain = arr("same_critic_optimizer_step_Q1_gain")

    result = {
        "path": str(path),
        "present": True,
        "rows": len(rows),
        "controller": {
            "active_fraction": float(np.mean(severity > 0)) if len(severity) else None,
            "full_strength_fraction": float(np.mean(severity >= 1.0)) if len(severity) else None,
            "severity_median": float(np.median(severity)) if len(severity) else None,
            "lambda_median": float(np.median(lam)) if len(lam) else None,
            "lambda_p95": float(np.percentile(lam, 95)) if len(lam) else None,
            "anchor_RL_grad_ratio_median": (
                float(np.median(ratio)) if len(ratio) else None
            ),
            "anchor_RL_grad_ratio_p95": (
                float(np.percentile(ratio, 95)) if len(ratio) else None
            ),
        },
        "training_batch_component_mean_rms": {
            "before_median": (
                float(np.median(mean_before)) if len(mean_before) else None
            ),
            "before_p95": (
                float(np.percentile(mean_before, 95)) if len(mean_before) else None
            ),
            "before_max": float(mean_before.max()) if len(mean_before) else None,
            "sampled_after_count": int(len(mean_after)),
            "sampled_after_median": (
                float(np.median(mean_after)) if len(mean_after) else None
            ),
            "sampled_after_p95": (
                float(np.percentile(mean_after, 95)) if len(mean_after) else None
            ),
            "sampled_after_max": (
                float(mean_after.max()) if len(mean_after) else None
            ),
        },
        "same_critic_optimizer_step_Q1_gain_sampled": {
            "count": int(len(q_gain)),
            "sum": float(q_gain.sum()) if len(q_gain) else None,
            "mean": float(q_gain.mean()) if len(q_gain) else None,
            "median": float(np.median(q_gain)) if len(q_gain) else None,
            "positive_fraction": (
                float(np.mean(q_gain > 0)) if len(q_gain) else None
            ),
        },
        "first": rows[0],
        "last": rows[-1],
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results_real_mean_preservation_20261003",
    )
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    if args.device != "npu:0":
        raise RuntimeError("Experiment is intentionally fixed to npu:0")

    source_run = args.source_run.resolve()
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
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
        raise RuntimeError(f"Expected critic_ready=130000, got {source['env_steps']}")
    if int(source["actor_updates"]) != 0:
        raise RuntimeError("Source Actor already updated")
    if source["actor_optimizer"].get("state"):
        raise RuntimeError("Source Actor Adam state is not empty")
    if (
        source.get("critic_target_mode") != "random2q"
        or source.get("group") != "multi_q"
    ):
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

    contract = {
        "testing_only": True,
        "source_run": str(source_run),
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "source_env_steps": int(source["env_steps"]),
        "source_critic_updates": int(source["updates"]),
        "source_actor_updates": int(source["actor_updates"]),
        "source_actor_optimizer_state_entries": len(
            source["actor_optimizer"].get("state", {})
        ),
        "branches": list(BRANCHES),
        "branch_semantics": {
            "FULL_PRODUCTION": "exact production Actor update",
            "FULL_MEAN_PRESERVATION": (
                "production Actor Q1 objective + adaptive component-mean-only "
                "output preservation; logits/std/hidden are not anchored"
            ),
        },
        "mean_preservation_budget": {
            "safe_rms": 0.004,
            "hard_rms": 0.006,
            "max_anchor_RL_grad_ratio": 4.0,
            "rationale": (
                "Prior online module experiment: RNN-only was 4/4 at 140K "
                "with fixed-probe component mean RMS 0.0050, while mean-only "
                "had first failure at 0.0058; bracket 0.004..0.006 chosen "
                "before this necessity run."
            ),
        },
        "target_mode": "random2q",
        "group": "multi_q",
        "device": args.device,
        "target_env_steps": 160000,
        "milestones": list(MILESTONES),
        "production_source_sha256_before": production_before,
        "formal_training_resumed": False,
    }
    write_json(output_root / "experiment_contract.json", contract)

    status = {
        "status": "RUNNING",
        "testing_only": True,
        "formal_training_resumed": False,
        "completed": [],
    }
    write_json(output_root / "experiment_status.json", status)

    manifest = read_json(source_run / "shared" / "stage2_source_manifest.json")
    critic_init = manifest["multi_q"]["checkpoint"]
    branch_results = {}

    for branch in BRANCHES:
        pair = prepare_pair(source_run, output_root, branch)
        group = pair / "random2q" / "multi_q"
        group.mkdir(parents=True, exist_ok=True)
        logfile = pair / "real_continuation.log"
        ready_marker = pair / "startup_ready.json"

        env = dict(
            os.environ,
            MEAN_PRESERVATION_BRANCH=branch,
            MEAN_PRESERVATION_GROUP_OUT=str(group),
            MEAN_PRESERVATION_SOURCE_CHECKPOINT=str(source_checkpoint),
            PYTHONUNBUFFERED="1",
        )
        cmd = [
            sys.executable,
            "-u",
            str(HERE / "train_mean_preservation_testing.py"),
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
            active_command=cmd,
            active_log=str(logfile),
        )
        write_json(output_root / "experiment_status.json", status)

        print(
            "MEAN_PRESERVATION_BRANCH_START "
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
            or "MEAN_PRESERVATION_TESTING_TRAINER_EXITED_ALL_VECTOR_ENVS_CLOSED"
            not in log_text
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

        milestones = [milestone_record(group, step) for step in MILESTONES]
        if any(row["sim_error_count"] for row in milestones):
            raise RuntimeError(f"Simulator diagnostic error in {branch}")
        if milestones[0]["success_count"] != 4:
            raise RuntimeError(f"{branch} 130K baseline is not 4/4")

        final_checkpoint = group / "checkpoints" / "step_0160000.pth"
        if not final_checkpoint.is_file():
            raise FileNotFoundError(final_checkpoint)
        final = torch.load(
            final_checkpoint, map_location="cpu", weights_only=False
        )
        if not final.get("testing_only"):
            raise RuntimeError("Testing checkpoint missing testing_only marker")
        if final.get("mean_preservation_branch") != branch:
            raise RuntimeError("Checkpoint branch mismatch")
        if int(final["env_steps"]) != 160000:
            raise RuntimeError("Final checkpoint is not 160K")

        preservation = preservation_summary(group)
        if branch == "FULL_MEAN_PRESERVATION":
            if not preservation["present"]:
                raise RuntimeError("Preservation branch produced no controller log")
            if preservation["rows"] != int(final["actor_updates"]):
                raise RuntimeError(
                    "Preservation log row count does not equal Actor updates"
                )
        else:
            if preservation["present"]:
                raise RuntimeError(
                    "FULL_PRODUCTION unexpectedly produced preservation updates"
                )

        result = {
            "branch": branch,
            "duration_seconds": float(time.time() - started),
            "final_checkpoint": str(final_checkpoint),
            "final_actor_updates": int(final["actor_updates"]),
            "final_critic_updates": int(final["updates"]),
            "milestones": milestones,
            "preservation_updates": preservation,
            "trainer_log": str(logfile),
            "software_and_numeric_safety_pass": True,
        }
        branch_results[branch] = result
        status["completed"].append(result)
        status.update(active_branch=None, active_pid=None)
        write_json(output_root / "experiment_status.json", status)

        print(
            "MEAN_PRESERVATION_BRANCH_PASS "
            + json.dumps(
                {
                    "branch": branch,
                    "success_160K": milestones[-1]["success_count"],
                    "actor_updates": result["final_actor_updates"],
                    "parameter_drift_l2": milestones[-1]["parameter_drift_l2"],
                    "component_mean_rms": milestones[-1]["component_mean_rms"],
                    "sampled_action_drift": milestones[-1]["sampled_action_drift"],
                    "weighted_action_drift": milestones[-1]["weighted_action_drift"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    comparison = []
    for index, step in enumerate(MILESTONES):
        row = {"env_steps": int(step)}
        for branch in BRANCHES:
            item = branch_results[branch]["milestones"][index]
            prefix = branch.lower()
            for key in (
                "success_count",
                "actor_updates",
                "critic_updates",
                "parameter_drift_l2",
                "sampled_action_drift",
                "weighted_action_drift",
                "hidden_l2",
                "component_mean_rms",
                "logits_rms",
                "categorical_kl_mean",
                "top1_mode_change_fraction",
            ):
                row[f"{prefix}_{key}"] = item[key]
        comparison.append(row)

    production_after = {
        str(path.resolve()): sha256(path)
        for path in production_paths
    }
    if production_after != production_before:
        raise RuntimeError("Production source changed during experiment")

    full_final = branch_results["FULL_PRODUCTION"]["milestones"][-1]
    protected_final = branch_results["FULL_MEAN_PRESERVATION"]["milestones"][-1]
    controller = branch_results["FULL_MEAN_PRESERVATION"][
        "preservation_updates"
    ]

    summary = {
        "testing_only": True,
        "source_contract": contract,
        "branches": branch_results,
        "comparison": comparison,
        "final_comparison": {
            "full_success": full_final["success_count"],
            "protected_success": protected_final["success_count"],
            "full_component_mean_rms": full_final["component_mean_rms"],
            "protected_component_mean_rms": protected_final["component_mean_rms"],
            "full_weighted_action_drift": full_final["weighted_action_drift"],
            "protected_weighted_action_drift": protected_final["weighted_action_drift"],
            "full_parameter_drift_l2": full_final["parameter_drift_l2"],
            "protected_parameter_drift_l2": protected_final["parameter_drift_l2"],
            "controller_active_fraction": controller["controller"]["active_fraction"],
            "sampled_same_critic_Q1_gain": controller[
                "same_critic_optimizer_step_Q1_gain_sampled"
            ],
        },
        "interpretation_rule": {
            "full_collapses_protected_survives_mean_held_and_q_gain_positive": (
                "COMPONENT_MEAN_DRIFT_NECESSARY_STRONG_EVIDENCE"
            ),
            "full_and_protected_both_collapse_while_mean_held": (
                "COMPONENT_MEAN_DRIFT_NOT_NECESSARY"
            ),
            "protected_survives_but_q_gain_near_zero": (
                "POLICY_PRESERVATION_BY_EFFECTIVE_LEARNING_SUPPRESSION"
            ),
            "protected_mean_not_materially_reduced": (
                "PRESERVATION_INSUFFICIENT_INCONCLUSIVE"
            ),
            "otherwise": "INCONCLUSIVE_OR_MIXED",
        },
        "production_source_sha256_after": production_after,
        "production_source_unchanged": True,
        "formal_training_resumed": False,
    }
    write_json(output_root / "analysis_summary.json", summary)

    status.update(status="COMPLETE", active_branch=None, active_pid=None)
    write_json(output_root / "experiment_status.json", status)

    print(
        "REAL_MEAN_PRESERVATION_EXPERIMENT_COMPLETE "
        + json.dumps(comparison, sort_keys=True),
        flush=True,
    )
    print("FORMAL TRAINING REMAINS STOPPED", flush=True)


if __name__ == "__main__":
    main()
