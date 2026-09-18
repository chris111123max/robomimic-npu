#!/usr/bin/env python3
"""Validate the V5 FSM, transition-credit contract, and runtime test coverage."""
import argparse
import copy
import json
from pathlib import Path
from stage3_v5_schedule import CriticHandoff, HandoffState, TrainingState
from stage3_v5_pipeline import TransitionCredit


def passing(step):
    return {"env_steps":step, "completed_episodes":150, "success_episodes":30,
            "failure_episodes":30, "spearman_q_return":.7, "success_failure_auc":.8,
            "delta_q":.1, "twin_q_disagreement_median":.1, "twin_q_disagreement_p95":.25,
            "td_plateau":True, "td_error_worsening":False, "q_scale_stable":True,
            "ood_stress_pass":True, "finite":True}


def validate(config):
    checks = {}
    checks["utd"] = config["utd"] == .25
    checks["policy_delay"] = config["policy_delay"] == 4
    checks["formal_num_envs"] = config["parallel_env"]["num_envs"] == 16
    checks["startup_batch_size"] = config["parallel_env"]["startup_parallelism"] == 4
    handoff = CriticHandoff(copy.deepcopy(config))
    checks["critic_only"] = handoff.state.state == TrainingState.CRITIC_ONLY
    checks["readiness_starts_at_100k"] = not handoff.readiness_due(99999) and handoff.readiness_due(100000)
    checks["no_eval_critic_only"] = not handoff.evaluation_due(100000)
    checks["actor_frozen"] = handoff.schedule(100000, .0003)["actor_lr"] == 0
    for step in (100000,110000):
        handoff.submit_readiness(passing(step))
    checks["two_passes_not_ready"] = handoff.state.state == TrainingState.CRITIC_ONLY
    handoff.submit_readiness(passing(120000))
    checks["three_passes_ready"] = handoff.state.state == TrainingState.ACTOR_WARMUP
    start, warmup = handoff.state.critic_ready_step, handoff.state.actor_warmup_steps
    end = start+warmup
    checks["warmup_length_formula"] = warmup == min(300000,max(100000,start))
    checks["no_eval_warmup"] = not handoff.evaluation_due(end-1)
    checks["initial_lr"] = handoff.schedule(start,.0003)["actor_lr"] == 0
    middle = handoff.schedule(start+warmup//2,.0003)
    checks["linear_midpoint_lrs"] = abs(middle["actor_lr"]-1e-6)<1e-12 and abs(middle["critic_lr"]-.0001875)<1e-12
    schedule = handoff.schedule(end,.0003)
    checks["joint_rl"] = handoff.state.state == TrainingState.JOINT_RL
    checks["target_lrs"] = abs(schedule["actor_lr"]-2e-6)<1e-12 and abs(schedule["critic_lr"]-7.5e-5)<1e-12
    checks["immediate_eval"] = handoff.evaluation_due(end)
    checks["eval_every_100k"] = not handoff.evaluation_due(end+99999) and handoff.evaluation_due(end+100000)
    restored = HandoffState.restore(handoff.state.serialize())
    checks["fsm_roundtrip"] = restored.serialize() == handoff.state.serialize()
    credit = TransitionCredit(config["utd"], config["parallel_env"]["max_collector_lag_transitions"])
    credit.collect(256)
    checks["bounded_lag_throttles"] = credit.must_throttle(16) and credit.updates_due == 64
    for _ in range(64): credit.consume()
    checks["learner_catches_up"] = credit.lag == 0 and not credit.must_throttle(16)
    here = Path(__file__).resolve().parent
    checks["rng_snapshot_lag_regressions_present"] = (here/"test_stage3_v5_runtime.py").is_file()
    return {"status":"PASS" if all(checks.values()) else "FAIL", "checks":checks,
            "runtime_regressions":"Run TestRNGIsolation, TestRolloutSnapshot, TestBoundedLag after this validator",
            "readiness_timing":"Checks start at 100K; three diagnostic history points plus three consecutive overall passes normally imply earliest 140K, later if coverage is insufficient"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("stage3_v5_config.json")))
    args = parser.parse_args()
    report = validate(json.loads(Path(args.config).read_text(encoding="utf-8")))
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__": main()
