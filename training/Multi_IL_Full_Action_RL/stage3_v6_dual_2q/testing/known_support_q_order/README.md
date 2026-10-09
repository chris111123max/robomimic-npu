# Known-support Q order test (testing only)

**Single question:** On histories/actions labeled relatively supported by the existing
95% AND 99% offline coverage audit, does the frozen Ready twin-Q order an
original BC action and a frozen 625-update Actor action consistently with the
return from a single changed action followed by frozen BC continuation?

This does **not** inspect unseen actions or require training.

## Run on server (Codex executes only)

Use the same working directory and existing robosuite_npu environment.
Only **npu:0** is allowed. Exactly **four real parallel simulator workers**;
one pool reused sequentially, no additional jobs.

    cd /data/home/3220251075/lerobot_workspace/robomimic
    python training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/known_support_q_order/check.py prepare
    python training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/testing/known_support_q_order/check.py run

prepare is **simulation-free**. It selects one supported action pair
per each of four pre-existing BC rollout seeds (20008, 20002, 20005, 20007),
using prior 95% and 99% support tags plus frozen Ready Q1/Q2. It chooses the
largest absolute predicted Q1 difference *with Q1/Q2 agreeing in sign*, not
by observing counterfactual returns. No fallback to unsupported samples.

run verifies input SHA256/model hashes, replays the original BC action
prefix and compares two branches:

- BC component-mean action at the chosen step, then frozen BC continuation.
- Actor625 component-mean action at that **one** step, then the same frozen BC continuation.

The test uses one fixed common future RNG stream, and strictly compares
exposed simulator/controller state, BC hidden state, torch RNG state, and
saved history at each fork. It checks 4/4/4 workers and no model updates.
These are **screening outcomes**, not expected-return estimates over many rollouts.

## Outputs

- output/preregistration.json
- output/run_started.json
- output/results.json
- output/FINAL_REPORT.md

The commands do **not** overwrite artifacts. On a missing prerequisite,
selection failure, or strict pairing mismatch, stop and report the error;
do not adjust thresholds/seeds or change production code.

LOCAL_PAIR_MISORDER_OBSERVED means disagreement with a nonzero *realized*
return contrast, not proof of systematic expected-Q error.
LOCAL_PAIR_ORDER_AGREED_ON_IDENTIFIABLE_PAIRS means no disagreement in
tested distinguishable pairs; **not proof** all Q action rankings are correct
or that insufficient state coverage causes the Actor collapse.
INCONCLUSIVE_NO_DISTINGUISHABLE_RETURNS means no informative outcome contrast.

Prerequisites: existing, server-local audit outputs from
critic_plateau_action_coverage and actor_critic_improvement_consistency;
original READY_trajectories.jsonl, actor_625.pth, and Ready checkpoint.
No previous artifact is recreated or modified.
