#!/usr/bin/env python3
"""Static/CPU validation for the Stage3-v6 four-run contract."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

from launch_stage3_v6_4npu import RUNS  # noqa: E402
from prepare_stage3_v6_quad import NPU_MAPPING, validate_config  # noqa: E402
from stage3_v6_agent import RecurrentGMMTD3V6  # noqa: E402
from stage3_v6_target import TARGET_MODES, TargetSelector2Q  # noqa: E402


EXPECTED_RUNS = (
    ("mean2q", "multi_q", "npu:0"),
    ("mean2q", "rnn_q", "npu:1"),
    ("random2q", "multi_q", "npu:2"),
    ("random2q", "rnn_q", "npu:3"),
)


def main():
    config = json.loads((HERE / "stage3_v6_config.json").read_text())
    validate_config(config)

    checks = {}
    checks["target_modes"] = TARGET_MODES == ("mean2q", "random2q")
    checks["fixed_four_npu_runs"] = RUNS == EXPECTED_RUNS
    checks["prepare_mapping_matches_launcher"] = NPU_MAPPING == {
        f"{mode}/{group}": device for mode, group, device in EXPECTED_RUNS
    }

    launcher_source = (HERE / "launch_stage3_v6_4npu.py").read_text()
    checks["launcher_starts_multi_before_rnn"] = (
        "wave1 = [" in launcher_source
        and "wait_wave_ready(wave1, 1)" in launcher_source
        and "wave2 = [" in launcher_source
        and "wait_wave_ready(wave2, 2)" in launcher_source
    )
    checks["launcher_requires_multi_ready_barrier"] = (
        "both Multi were READY before either RNN was launched" in launcher_source
        and "startup-ready" in launcher_source
    )

    bellman_source = inspect.getsource(RecurrentGMMTD3V6.bellman_target)
    checks["v6_bellman_has_no_torch_minimum"] = "torch.minimum" not in bellman_source
    checks["v6_bellman_uses_two_member_expectations"] = (
        "q1_expected" in bellman_source
        and "q2_expected" in bellman_source
        and "_combine_target_members" in bellman_source
    )

    mean = TargetSelector2Q("mean2q", 10)
    q1 = np.asarray([1.0, 3.0])
    q2 = np.asarray([3.0, 5.0])
    checks["mean2q_exact"] = np.array_equal(
        mean.combine(q1, q2), np.asarray([2.0, 4.0])
    )

    np.random.seed(12345)
    expected_global = np.random.random(8)
    np.random.seed(12345)
    random_selector = TargetSelector2Q("random2q", 99)
    picks = [random_selector.begin_update() for _ in range(1000)]
    observed_global = np.random.random(8)
    checks["selector_rng_isolated"] = np.array_equal(
        expected_global, observed_global
    )
    checks["one_pick_per_random_update"] = (
        random_selector.q1_updates + random_selector.q2_updates == 1000
        and set(picks) <= {0, 1}
    )

    trainer_source = (HERE / "train_stage3_v6_vector.py").read_text()
    checks["trainer_uses_v6_agent"] = (
        "from stage3_v6_agent import RecurrentGMMTD3" in trainer_source
    )
    checks["trainer_has_no_hard_min_target_formula"] = (
        "sum_k p'_k*min(Q1',Q2')" not in trainer_source
    )
    checks["trainer_enforces_prepared_npu_mapping"] = (
        "expected_device = fairness[\"npu_mapping\"].get(run_key)"
        in trainer_source
    )
    checks["trainer_supports_startup_ready_marker"] = (
        "--startup-ready-file" in trainer_source
        and "all_vector_envs_initialized" in trainer_source
        and "write_startup_ready" in trainer_source
    )
    checks["checkpoint_saves_selector_state"] = (
        "\"target_selector_state\": agent.target_selector_state_dict()"
        in trainer_source
    )
    checks["resume_restores_selector_state"] = (
        "agent.load_target_selector_state_dict(payload[\"target_selector_state\"])"
        in trainer_source
    )
    checks["actor_objective_unchanged"] = (
        "Q1(h10_final" in trainer_source
    )

    status = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps({"status": status, "checks": checks}, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()