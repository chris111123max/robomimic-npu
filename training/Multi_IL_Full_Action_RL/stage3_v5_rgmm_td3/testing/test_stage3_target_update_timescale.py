#!/usr/bin/env python3
"""Short Stage3-v5 target-Critic update-timescale diagnostic.

Three branches start from the same Multi Stage3 step0 checkpoint and consume the
same precomputed canonical replay batches in the same order:

  baseline : tau=0.005, update target Critic every Critic step.
  slow     : tau=0.0005, update target Critic every Critic step.
  delayed10: tau=0.005, update target Critic every 10 Critic steps.

Actor learning is disabled. No environment is created and no training
checkpoint is written. The only intended difference is target-Critic update
speed/frequency.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
STAGE3 = HERE.parent
for directory in (HERE, STAGE3):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from stage3_v5_replay import final_transition  # noqa: E402
from test_stage3_frozen_target_causal import (  # noqa: E402
    DEFAULT_STAGE2,
    build_agent,
    cleanup_device,
    evaluate_probe,
    load_canonical,
    make_probe_refs,
    make_reference_schedule,
    module_digest,
    probe_reference_returns,
    resolve_device,
    stack_sequence_batch,
    sync,
)


MILESTONES = (0, 250, 500, 1000)
POLICIES = {
    "baseline": {"tau": 0.005, "period": 1},
    "slow_tau": {"tau": 0.0005, "period": 1},
    "delayed10": {"tau": 0.005, "period": 10},
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-run-dir", required=True)
    parser.add_argument("--stage2-checkpoint", default=DEFAULT_STAGE2)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probe-size", type=int, default=4096)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--diagnostic-replay")
    parser.add_argument("--output")
    return parser.parse_args()


@torch.no_grad()
def update_target_critic(agent, tau):
    tau = float(tau)
    for source, target in zip(
        agent.critic.parameters(), agent.target_critic.parameters()
    ):
        target.mul_(1.0 - tau).add_(source, alpha=tau)


def run_branch(
    name,
    policy,
    stage2_path,
    step0_payload,
    episodes,
    schedule,
    probe_refs,
    probe_returns,
    probe_labels,
    probe_episode_ids,
    device,
    probe_batch_size,
):
    config = step0_payload["config"]
    length = int(config["recurrent_replay"]["critic_context_length"])
    horizon = int(config["horizon"])
    agent = build_agent(stage2_path, step0_payload, device)

    initial = {
        "online_hash": module_digest(agent.critic),
        "target_hash": module_digest(agent.target_critic),
        "target_actor_hash": module_digest(agent.target_actor),
    }
    trajectory = []
    failure = None
    completed = 0
    target_updates = 0

    def record(update):
        sync(device)
        online = evaluate_probe(
            agent.critic,
            episodes,
            probe_refs,
            probe_returns,
            probe_labels,
            probe_episode_ids,
            device,
            horizon,
            length,
            probe_batch_size,
        )
        target = evaluate_probe(
            agent.target_critic,
            episodes,
            probe_refs,
            probe_returns,
            probe_labels,
            probe_episode_ids,
            device,
            horizon,
            length,
            probe_batch_size,
        )
        trajectory.append(
            {
                "update": int(update),
                "online": online,
                "target": target,
            }
        )
        print(
            f"[{name}] update={update} "
            f"online={online['spearman_q_return']:.6f} "
            f"target={target['spearman_q_return']:.6f} "
            f"qmean={online['qmin_mean']:.6f}",
            flush=True,
        )

    record(0)
    milestone_set = set(value for value in MILESTONES if value <= len(schedule))

    for index in range(len(schedule)):
        sequence = stack_sequence_batch(
            episodes, schedule[index], length
        )
        final = final_transition(sequence)
        try:
            agent.critic_update(final, sequence, collect_metrics=False)
        except FloatingPointError as exc:
            failure = {
                "failed_update": int(index + 1),
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            print(
                f"[{name}] NONFINITE at update={index + 1}: {exc}",
                flush=True,
            )
            break

        completed = index + 1
        if completed % int(policy["period"]) == 0:
            update_target_critic(agent, policy["tau"])
            target_updates += 1

        if completed in milestone_set:
            record(completed)

    result = {
        "name": name,
        "tau": float(policy["tau"]),
        "target_update_period": int(policy["period"]),
        "updates_requested": int(len(schedule)),
        "updates_completed": int(completed),
        "target_updates_completed": int(target_updates),
        "failure": failure,
        "completed": bool(
            failure is None and completed == len(schedule)
        ),
        "initial": initial,
        "final": {
            "online_hash": module_digest(agent.critic),
            "target_hash": module_digest(agent.target_critic),
            "target_actor_hash": module_digest(agent.target_actor),
        },
        "trajectory": trajectory,
    }
    del agent
    cleanup_device(device)
    return result


def by_update(branch):
    return {
        int(row["update"]): row for row in branch["trajectory"]
    }


def main():
    args = arguments()
    if args.updates != 1000:
        raise ValueError("Keep --updates 1000 for this short diagnostic")
    if args.batch_size != 256:
        raise ValueError("Keep production Critic --batch-size 256")
    if args.probe_size < 100:
        raise ValueError("--probe-size must be >= 100")

    run_dir = Path(args.stage3_run_dir).resolve()
    stage2_path = Path(args.stage2_checkpoint).resolve()
    step0_path = run_dir / "multi_q" / "checkpoints" / "step0_transfer.pth"
    replay_path = (
        Path(args.diagnostic_replay).resolve()
        if args.diagnostic_replay
        else run_dir / "multi_q" / "checkpoints"
        / "step_0200000.sequences.npy"
    )
    for path in (stage2_path, step0_path, replay_path):
        if not path.exists():
            raise FileNotFoundError(path)

    step0_payload = torch.load(step0_path, map_location="cpu")
    if (
        step0_payload.get("stage") != "stage3-v5"
        or step0_payload.get("group") != "multi_q"
        or int(step0_payload.get("env_steps", -1)) != 0
        or int(step0_payload.get("updates", -1)) != 0
        or int(step0_payload.get("actor_updates", -1)) != 0
    ):
        raise RuntimeError("Expected untouched Multi Stage3-v5 step0 checkpoint")

    config = step0_payload["config"]
    if float(config["tau"]) != 0.005:
        raise RuntimeError("Production reference tau is no longer 0.005")
    length = int(config["recurrent_replay"]["critic_context_length"])
    if length != 10:
        raise RuntimeError("Expected critic_context_length=10")

    fixed, episodes = load_canonical(replay_path)
    schedule = make_reference_schedule(
        episodes,
        args.updates,
        args.batch_size,
        length,
        args.seed,
    )
    probe_refs = make_probe_refs(
        episodes,
        args.probe_size,
        length,
        args.seed + 1000003,
    )
    probe_returns, probe_labels, probe_episode_ids = probe_reference_returns(
        episodes,
        probe_refs,
        length,
        float(config["gamma"]),
    )
    device = resolve_device(args.device)

    branches = {}
    for name, policy in POLICIES.items():
        branches[name] = run_branch(
            name,
            policy,
            stage2_path,
            step0_payload,
            episodes,
            schedule,
            probe_refs,
            probe_returns,
            probe_labels,
            probe_episode_ids,
            device,
            args.probe_batch_size,
        )

    # Isolation validity: all branches must start identically and target Actor
    # must remain unchanged. Target Critic is expected to change in all three.
    initial_online = {
        value["initial"]["online_hash"] for value in branches.values()
    }
    initial_target = {
        value["initial"]["target_hash"] for value in branches.values()
    }
    initial_actor = {
        value["initial"]["target_actor_hash"] for value in branches.values()
    }
    validity = {
        "same_initial_online_hash": len(initial_online) == 1,
        "same_initial_target_hash": len(initial_target) == 1,
        "same_initial_target_actor_hash": len(initial_actor) == 1,
        "same_batch_schedule": True,
        "target_actor_unchanged_all": all(
            branch["initial"]["target_actor_hash"]
            == branch["final"]["target_actor_hash"]
            for branch in branches.values()
        ),
        "target_critic_changed_all": all(
            branch["initial"]["target_hash"] != branch["final"]["target_hash"]
            for branch in branches.values()
        ),
    }
    valid = bool(all(validity.values()))

    common = sorted(
        set.intersection(
            *(set(by_update(branch)) for branch in branches.values())
        )
    )
    comparison = []
    for update in common:
        row = {"update": int(update)}
        for name, branch in branches.items():
            item = by_update(branch)[update]
            row[f"{name}_online_spearman"] = float(
                item["online"]["spearman_q_return"]
            )
            row[f"{name}_target_spearman"] = float(
                item["target"]["spearman_q_return"]
            )
            row[f"{name}_qmin_mean"] = float(item["online"]["qmin_mean"])
            row[f"{name}_qmin_std"] = float(item["online"]["qmin_std"])
        comparison.append(row)

    output = {
        "status": (
            "PASS"
            if valid and all(x["completed"] for x in branches.values())
            else "PARTIAL"
            if valid
            else "INVALID"
        ),
        "experiment": "stage3_v5_target_update_timescale",
        "environment_steps_performed": 0,
        "actor_updates_performed": 0,
        "training_checkpoints_written": 0,
        "stage3_step0_checkpoint": str(step0_path),
        "canonical_diagnostic_replay": str(replay_path),
        "canonical_seed": fixed.get("seed"),
        "canonical_episode_count": int(len(episodes)),
        "same_precomputed_batch_schedule": True,
        "updates": int(args.updates),
        "batch_size": int(args.batch_size),
        "probe_size": int(len(probe_refs)),
        "probe_seed": int(args.seed + 1000003),
        "schedule_seed": int(args.seed),
        "gamma": float(config["gamma"]),
        "production_tau": float(config["tau"]),
        "policies": POLICIES,
        "validity": validity,
        "branches": branches,
        "comparison": comparison,
    }

    out_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "testing" / "stage2_vs_stage3_readiness"
        / "multi_target_update_timescale.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    print(
        "\nupdate\tbaseline\tslow_tau\tdelayed10\t"
        "slow-baseline\tdelayed-baseline"
    )
    for row in comparison:
        baseline = row["baseline_online_spearman"]
        slow = row["slow_tau_online_spearman"]
        delayed = row["delayed10_online_spearman"]
        print(
            f"{row['update']}\t{baseline:.6f}\t{slow:.6f}\t"
            f"{delayed:.6f}\t{slow-baseline:+.6f}\t"
            f"{delayed-baseline:+.6f}"
        )

    print("\n[VALIDITY]")
    print(json.dumps(validity, indent=2, sort_keys=True))
    print(f"[STATUS] {output['status']}")
    print(f"[SAVED] {out_path}", flush=True)

    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
