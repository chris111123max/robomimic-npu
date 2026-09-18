#!/usr/bin/env python3
"""Summarize measured Stage3-v5 throughput and collector/learner timing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import os

PHASES = ("CRITIC_ONLY", "ACTOR_WARMUP", "JOINT_RL", "CRITIC_NOT_READY")
THROUGHPUT_KEYS = ("aggregate_env_steps_per_sec", "critic_updates_per_sec",
                   "actor_updates_per_sec")
TIMING_KEYS = ("actor_inference_ms", "vector_env_step_ms", "critic_replay_ms",
               "critic_update_ms", "actor_replay_ms", "actor_update_ms",
               "polyak_update_ms", "round_replay_prepare_ms", "round_wall_ms",
               "collector_learner_overlap_ms",
               "round_critic_updates", "round_actor_updates",
               "prefetch_requested_critic_batches",
               "prefetch_prepared_critic_batches",
               "prefetch_prepared_actor_batches")


def _rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _mean(rows, key):
    values = [float(row[key]) for row in rows if key in row]
    return sum(values) / len(values) if values else None


def summarize(path):
    rows = _rows(path)
    if not rows:
        raise RuntimeError(f"No throughput rows in {path}")
    result = {}
    for phase in PHASES:
        selected = [row for row in rows if row.get("phase") == phase]
        result[phase] = {"samples": len(selected),
                         **{key: _mean(selected, key) for key in THROUGHPUT_KEYS}}
    result["all"] = {"samples": len(rows),
                     **{key: _mean(rows, key) for key in THROUGHPUT_KEYS}}
    return result


def summarize_timing(path):
    rows = _rows(path)
    return {"samples": len(rows),
            **{key: _mean(rows, key) for key in TIMING_KEYS}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-run-dir", required=True)
    parser.add_argument("--group", choices=("rnn_q", "multi_q"), required=True)
    parser.add_argument("--v3-throughput-jsonl", help="optional matched baseline log")
    parser.add_argument("--run", action="store_true", help="execute actual A/B/C/D runs")
    parser.add_argument("--acceptance", action="store_true", help="ordered tests, A/B/C/D, scaling, then small smoke")
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--critic-init-checkpoint")
    parser.add_argument("--steps", type=int, default=4096, help="measured transitions, excludes warmup")
    parser.add_argument("--warmup-steps", type=int, default=1024)
    parser.add_argument("--smoke-steps", type=int, default=4096)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.run or args.acceptance:
        execute_benchmarks(args)
        return
    group_dir = Path(args.pair_run_dir) / args.group
    report = {"stage": "stage3-v5", "group": args.group,
              "throughput": summarize(group_dir / "throughput_metrics.jsonl"),
              "stage_timing": summarize_timing(group_dir / "stage_timing.jsonl"),
              "measurement_contract": {
                  "env_steps": "aggregate transitions across 16 environments",
                  "phase_source": "CriticHandoff state, never an env-step threshold",
                  "prefetch": "requested and prepared optimizer batches per sampled round",
              }}
    if args.v3_throughput_jsonl:
        baseline = summarize(args.v3_throughput_jsonl)
        report["baseline"] = baseline
        for phase in PHASES:
            current = report["throughput"][phase]["aggregate_env_steps_per_sec"]
            previous = baseline.get(phase, {}).get("aggregate_env_steps_per_sec")
            report.setdefault("speedup_vs_baseline", {})[phase] = (
                current / previous if current is not None and previous else None)
    output = group_dir / "throughput_benchmark.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **report}, indent=2))


def execute_benchmarks(args):
    here = Path(__file__).resolve().parent
    pair = Path(args.pair_run_dir).resolve()
    sources = json.loads((pair / "shared" / "stage2_source_manifest.json").read_text())
    checkpoint = args.critic_init_checkpoint or sources[args.group]["checkpoint"]
    output = pair / args.group / "benchmarks" / f"acceptance_{time.time_ns()}"
    output.mkdir(parents=True)
    report = {"stage": "stage3-v5", "group": args.group, "device": args.device,
              "before_steps_per_sec_user_reported": 22, "measurements": [], "checks": [],
              "timing_mode": "device-synchronized detailed profile; compare C/D in the same mode",
              "warmup_transitions": args.warmup_steps, "measured_transitions": args.steps}

    def save():
        (output / "benchmark_report.json").write_text(json.dumps(report, indent=2)+"\n")

    def run(name, command):
        print(f"[ACCEPTANCE] {name}: {' '.join(command)}", flush=True)
        measurement = None
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace",
                                       env=dict(os.environ, PYTHONPATH=str(here)+os.pathsep+os.environ.get("PYTHONPATH", "")))
            try:
                for line in process.stdout:
                    log.write(line); log.flush(); print(line, end="", flush=True)
                    if line.startswith('{"measurements":'):
                        measurement = json.loads(line)["measurements"]
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate(); process.wait()
        report["checks"].append({"name": name, "returncode": code})
        save()
        if code:
            report["status"] = "FAILED"; save()
            raise SystemExit(f"Acceptance stopped at {name}; evidence: {output}")
        return measurement

    if args.steps <= 0 or args.warmup_steps < 1024 or args.repeat < 1:
        raise SystemExit("Require positive measured steps/repeats and >=1024 warmup transitions")
    prefix = [sys.executable]
    if args.acceptance:
        run("01_py_compile", prefix+["-m", "compileall", "-q", str(here)])
        run("02_replay_tests", prefix+["-m", "unittest", str(here / "test_stage3_v5_replay.py")])
        run("03_schedule_tests", prefix+["-m", "unittest", str(here / "test_stage3_v5_schedule.py")])
        run("04_validator", prefix+[str(here / "validate_stage3_v5.py")])
        for name, suite in (("05_rng_isolation", "TestRNGIsolation"),
                            ("06_rollout_snapshot", "TestRolloutSnapshot"),
                            ("07_bounded_lag", "TestBoundedLag")):
            run(name, prefix+["-m", "unittest", f"test_stage3_v5_runtime.{suite}"])
        run("07b_target_prefetch_math", prefix+["-m", "unittest", "test_stage3_v5_runtime.TestDiagnosticMath"])
        run("07c_trainer_pipeline", prefix+["-m", "unittest", "test_stage3_v5_runtime.TestTrainerPipeline"])
        run("07d_case_a_math", prefix+[str(here / "validate_stage3_v5_math.py")])
    common = [str(here / "train_stage3_v5_vector.py"), "--group", args.group,
              "--device", args.device, "--pair-run-dir", str(pair),
              "--critic-init-checkpoint", str(checkpoint)]
    plans = [(mode, 16) for mode in ("A", "B", "C", "D")]
    if args.scaling or args.acceptance:
        plans += [(mode, count) for count in (2, 4, 8) for mode in ("A", "D")]
    for mode, count in plans:
        for repeat in range(args.repeat):
            measurement = run(f"benchmark_{mode}_{count}_{repeat}", prefix+common+[
                "--benchmark-mode", mode, "--num-envs", str(count),
                "--benchmark-warmup-steps", str(args.warmup_steps),
                "--total-env-steps", str(args.warmup_steps+args.steps)])
            if measurement is None:
                raise SystemExit("Benchmark did not produce measurements; inspect its log")
            report["measurements"].append(measurement); save()
    def speed(mode, count):
        values = [r["aggregate_steps_per_sec"] for r in report["measurements"]
                  if r["benchmark_mode"] == mode and r["num_envs"] == count]
        return sum(values)/len(values) if values else None
    report["async_speedup_C_vs_D"] = speed("D",16)/speed("C",16)
    report["scaling_findings"] = []
    for mode in ("A", "D"):
        if speed(mode,8) is not None and speed(mode,16) <= speed(mode,8):
            rows = [r for r in report["measurements"] if r["benchmark_mode"] == mode and r["num_envs"] == 16]
            profile = rows[0]["profile"]
            contributors = {k: profile.get(k, {}).get("total_ms", 0) for k in
                ("actor_inference_ms", "env_wait_ms", "env_dispatch_ms", "reset_many_ms",
                 "batch_prefetch_ms", "critic_total_ms", "readiness_ms")}
            report["scaling_findings"].append({"mode":mode, "16_not_faster_than_8":True,
                "measured_contributors_ms":contributors, "cpu":rows[0]["cpu"],
                "interpretation":"Rank timings and allocated-core utilization; CPU/MuJoCo/scheduler contention cannot be distinguished solely by wall timers."})
    if args.acceptance:
        measurement = run("13_small_smoke", prefix+common+["--smoke", "--num-envs", "2",
                                                           "--total-env-steps", str(args.smoke_steps)])
        validation = json.loads((Path(measurement["output_dir"])/"smoke_validation.json").read_text())
        report["smoke"] = validation
        if validation["status"] != "PASS":
            report["status"] = "FAILED"; save(); raise SystemExit("Smoke failed")
    report["status"] = "PASS"; save()
    print(json.dumps({"output": str(output / "benchmark_report.json"), **report}, indent=2))


if __name__ == "__main__":
    main()
