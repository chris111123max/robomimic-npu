#!/usr/bin/env python3
"""Sequential one-NPU real online Actor-module sufficiency experiment."""
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
BRANCHES = ("FULL_PRODUCTION", "MEAN_HEAD_ONLY", "RNN_ONLY")
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
        "reason": "sequential one-NPU real Actor-module experiment",
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
    closed = payload["closed_loop"]
    return {
        "env_steps": int(step),
        "actor_updates": int(payload["actor_updates"]),
        "critic_updates": int(payload["critic_updates"]),
        "success_count": int(closed["success_count"]),
        "success_total": int(closed["count"]),
        "mean_length": float(closed["mean_length"]),
        "sim_error_count": int(closed["sim_error_count"]),
        "parameter_drift": probe["parameter_drift"],
        "parameter_drift_l2": float(probe["parameter_drift"]["total"]["l2"]),
        "sampled_action_drift": float(
            probe["policy_drift"]["sampled_action_l2_mean"]
        ),
        "weighted_action_drift": float(
            probe["policy_drift"]["weighted_action_l2_mean"]
        ),
        "hidden_l2": float(probe["policy_drift"]["hidden_l2_mean"]),
        "component_mean_rms": float(
            probe["policy_drift"]["component_mean_rms"]
        ),
        "logits_rms": float(probe["policy_drift"]["logits_rms"]),
        "categorical_kl_mean": float(
            probe["policy_drift"]["categorical_kl_mean"]
        ),
        "top1_mode_change_fraction": float(
            probe["policy_drift"]["top1_mode_change_fraction"]
        ),
        "path": str(path),
    }


def module_update_summary(group):
    path = group / "actor_module_updates.jsonl"
    if not path.is_file():
        return {"path": str(path), "rows": 0}
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    result = {"path": str(path), "rows": len(rows)}
    if not rows:
        return result

    import numpy as np

    result["first"] = rows[0]
    result["last"] = rows[-1]
    steps = np.asarray(
        [float(row["actual_parameter_step_l2"]) for row in rows],
        dtype=np.float64,
    )
    forbidden = np.asarray(
        [float(row["forbidden_parameter_step_l2"]) for row in rows],
        dtype=np.float64,
    )
    retained = np.asarray(
        [
            float(row["retained_gradient_fraction_l2"])
            for row in rows
            if row.get("retained_gradient_fraction_l2") is not None
        ],
        dtype=np.float64,
    )
    result["actual_parameter_step_l2"] = {
        "median": float(np.median(steps)),
        "p10": float(np.percentile(steps, 10)),
        "p90": float(np.percentile(steps, 90)),
        "max": float(steps.max()),
    }
    result["forbidden_parameter_step_l2"] = {
        "max": float(forbidden.max()),
        "median": float(np.median(forbidden)),
    }
    if len(retained):
        result["retained_gradient_fraction_l2"] = {
            "median": float(np.median(retained)),
            "p10": float(np.percentile(retained, 10)),
            "p90": float(np.percentile(retained, 90)),
        }
    return result


def allowed_groups(branch):
    if branch == "FULL_PRODUCTION":
        return None
    if branch == "MEAN_HEAD_ONLY":
        return {"gmm_mean"}
    if branch == "RNN_ONLY":
        return {"rnn"}
    raise ValueError(branch)


def validate_final_parameter_mask(branch, milestone):
    allowed = allowed_groups(branch)
    if allowed is None:
        return {"status": "NOT_APPLICABLE_FULL_PRODUCTION"}
    violations = {}
    for group, stats in milestone["parameter_drift"].items():
        if group == "total" or group in allowed:
            continue
        value = float(stats["l2"])
        if value > 1e-10:
            violations[group] = value
    if violations:
        raise RuntimeError(
            f"{branch} forbidden cumulative Actor parameter drift: {violations}"
        )
    return {
        "status": "PASS",
        "allowed": sorted(allowed),
        "forbidden_group_tolerance_l2": 1e-10,
        "violations": {},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results_real_module_online_20261003",
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
        raise RuntimeError("Source Actor already has RL updates")
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
        str(path.resolve()): sha256(path) for path in production_paths
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
            "FULL_PRODUCTION": "exact production Actor update / Adam",
            "MEAN_HEAD_ONLY": "production Adam with grad=None outside gmm_mean",
            "RNN_ONLY": "production Adam with grad=None outside rnn",
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
            ACTOR_MODULE_BRANCH=branch,
            ACTOR_MODULE_GROUP_OUT=str(group),
            ACTOR_MODULE_SOURCE_CHECKPOINT=str(source_checkpoint),
            PYTHONUNBUFFERED="1",
        )
        cmd = [
            sys.executable,
            "-u",
            str(HERE / "train_module_online_testing.py"),
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
            "MODULE_BRANCH_START "
            + json.dumps({
                "branch": branch,
                "source": str(source_checkpoint),
                "target": 160000,
                "log": str(logfile),
            }),
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

        text = logfile.read_text()
        if (
            returncode != 0
            or "MODULE_TESTING_TRAINER_EXITED_ALL_VECTOR_ENVS_CLOSED" not in text
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
        final = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
        if not final.get("testing_only"):
            raise RuntimeError("Final testing checkpoint missing marker")
        if final.get("actor_module_branch") != branch:
            raise RuntimeError("Final checkpoint module branch mismatch")
        if int(final["env_steps"]) != 160000:
            raise RuntimeError("Final testing checkpoint is not 160K")

        update_summary = module_update_summary(group)
        if branch != "FULL_PRODUCTION":
            forbidden_step_max = update_summary[
                "forbidden_parameter_step_l2"
            ]["max"]
            if forbidden_step_max > 1e-10:
                raise RuntimeError(
                    f"{branch} forbidden per-step parameter motion: "
                    f"{forbidden_step_max}"
                )

        mask_integrity = validate_final_parameter_mask(
            branch, milestones[-1]
        )
        result = {
            "branch": branch,
            "duration_seconds": float(time.time() - started),
            "final_checkpoint": str(final_checkpoint),
            "final_actor_updates": int(final["actor_updates"]),
            "final_critic_updates": int(final["updates"]),
            "milestones": milestones,
            "module_updates": update_summary,
            "mask_integrity": mask_integrity,
            "trainer_log": str(logfile),
            "software_and_numeric_safety_pass": True,
        }
        branch_results[branch] = result
        status["completed"].append(result)
        status.update(active_branch=None, active_pid=None)
        write_json(output_root / "experiment_status.json", status)

        print(
            "MODULE_BRANCH_PASS "
            + json.dumps({
                "branch": branch,
                "success_160K": milestones[-1]["success_count"],
                "actor_updates": result["final_actor_updates"],
                "parameter_drift_l2": milestones[-1]["parameter_drift_l2"],
                "sampled_action_drift": milestones[-1]["sampled_action_drift"],
                "weighted_action_drift": milestones[-1]["weighted_action_drift"],
            }, sort_keys=True),
            flush=True,
        )

    comparison = []
    for index, step in enumerate(MILESTONES):
        row = {"env_steps": int(step)}
        for branch in BRANCHES:
            item = branch_results[branch]["milestones"][index]
            prefix = branch.lower()
            row[f"{prefix}_success"] = int(item["success_count"])
            row[f"{prefix}_actor_updates"] = int(item["actor_updates"])
            row[f"{prefix}_parameter_drift_l2"] = item["parameter_drift_l2"]
            row[f"{prefix}_sampled_action_drift"] = item["sampled_action_drift"]
            row[f"{prefix}_weighted_action_drift"] = item["weighted_action_drift"]
        comparison.append(row)

    production_after = {
        str(path.resolve()): sha256(path) for path in production_paths
    }
    if production_after != production_before:
        raise RuntimeError("Production source changed during module experiment")

    final_success = {
        branch: branch_results[branch]["milestones"][-1]["success_count"]
        for branch in BRANCHES
    }
    summary = {
        "testing_only": True,
        "source_contract": contract,
        "branches": branch_results,
        "comparison": comparison,
        "final_success_counts": final_success,
        "production_source_sha256_after": production_after,
        "production_source_unchanged": True,
        "formal_training_resumed": False,
        "interpretation_rule": {
            "mean_collapses_rnn_survives": "GMM_MEAN_HEAD_UPDATE_SUFFICIENT_STRONG_EVIDENCE",
            "mean_and_rnn_both_collapse": "MULTIPLE_ACTOR_SUBSPACES_SUFFICIENT",
            "mean_survives_rnn_collapses": "RNN_UPDATE_SUFFICIENT_NOT_MEAN",
            "both_masked_survive_full_collapses": "FULL_CROSS_MODULE_UPDATE_REQUIRED",
            "otherwise": "INCONCLUSIVE_OR_MIXED",
        },
    }
    write_json(output_root / "analysis_summary.json", summary)
    status.update(status="COMPLETE", active_branch=None, active_pid=None)
    write_json(output_root / "experiment_status.json", status)

    print(
        "REAL_MODULE_EXPERIMENT_COMPLETE "
        + json.dumps(comparison, sort_keys=True),
        flush=True,
    )
    print("FORMAL TRAINING REMAINS STOPPED", flush=True)


if __name__ == "__main__":
    main()
