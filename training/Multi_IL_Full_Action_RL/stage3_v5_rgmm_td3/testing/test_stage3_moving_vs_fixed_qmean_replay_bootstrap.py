#!/usr/bin/env python3
"""Paired fixed versus moving qmean replay-bootstrap Critic diagnostic.

Both arms reconstruct the Stage3-v5 Multi step0 agent, restore its Critic Adam
state, consume the same precomputed canonical replay batches, and use the same
replay-next-action twin-mean Bellman target. Only target-Critic Polyak feedback
differs. The Actor and target Actor are frozen and unused by the target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import types
from pathlib import Path

import torch

import test_stage3_bootstrap_vs_oracle_mc_causal as prior


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-run-dir", required=True)
    parser.add_argument("--stage2-checkpoint", default=prior.DEFAULT_STAGE2)
    parser.add_argument("--diagnostic-replay")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probe-size", type=int, default=8192)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output")
    return parser.parse_args()


def load_inputs(args):
    if args.updates != 2000 or args.batch_size != 256:
        raise ValueError("Keep 2000 updates and Critic batch size 256")
    if args.probe_size < 100 or args.probe_batch_size < 1:
        raise ValueError("Invalid probe size or probe batch size")

    run_dir = Path(args.stage3_run_dir).resolve()
    stage2_path = Path(args.stage2_checkpoint).resolve()
    step0_path = run_dir / "multi_q/checkpoints/step0_transfer.pth"
    replay_path = (
        Path(args.diagnostic_replay).resolve()
        if args.diagnostic_replay
        else run_dir / "multi_q/checkpoints/step_0200000.sequences.npy"
    )
    for path in (stage2_path, step0_path, replay_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    stage2 = torch.load(stage2_path, map_location="cpu")
    step0 = torch.load(step0_path, map_location="cpu")
    if stage2.get("stage_version") != "2.2":
        raise RuntimeError("Expected Stage2.2 checkpoint")
    if stage2.get("training_target") != (
        "finite_episode_monte_carlo_return_no_bootstrap"
    ):
        raise RuntimeError("Stage2.2 MC target contract changed")
    if step0.get("stage") != "stage3-v5" or step0.get("group") != "multi_q":
        raise RuntimeError("Expected Stage3-v5 Multi step0 checkpoint")
    if any(int(step0.get(key, -1)) != 0 for key in (
        "env_steps", "updates", "actor_updates"
    )):
        raise RuntimeError("Step0 checkpoint contains updates")

    config = step0["config"]
    length = int(config["recurrent_replay"]["critic_context_length"])
    gamma = float(config["gamma"])
    if length != 10 or abs(float(stage2.get("gamma", -1)) - gamma) > 1e-12:
        raise RuntimeError("Stage2.2/Stage3 history or gamma mismatch")
    if float(config["tau"]) != 0.005:
        raise RuntimeError("Expected production tau=0.005")
    return run_dir, stage2_path, step0_path, replay_path, step0, config


def initial_contract(stage2_path, step0, episodes, first_refs, returns, device):
    moving = prior.build_agent(stage2_path, step0, device)
    fixed = prior.build_agent(stage2_path, step0, device)
    for agent in (moving, fixed):
        agent.bellman_target = types.MethodType(
            prior.replay_qmean_bellman_target, agent
        )

    contract = {
        "same_initial_online_hash": (
            prior.module_digest(moving.critic)
            == prior.module_digest(fixed.critic)
        ),
        "same_initial_target_critic_hash": (
            prior.module_digest(moving.target_critic)
            == prior.module_digest(fixed.target_critic)
        ),
        "same_initial_actor_hash": (
            prior.module_digest(moving.actor)
            == prior.module_digest(fixed.actor)
        ),
        "same_initial_target_actor_hash": (
            prior.module_digest(moving.target_actor)
            == prior.module_digest(fixed.target_actor)
        ),
        "same_initial_optimizer_state": prior.nested_state_equal(
            moving.critic_optimizer.state_dict(),
            fixed.critic_optimizer.state_dict(),
        ),
    }

    length = int(step0["config"]["recurrent_replay"]["critic_context_length"])
    sequence, diagnostic = prior.prepare_training_batch(
        episodes, first_refs, length, returns
    )
    final = prior.final_transition(sequence)
    targets = []
    for agent in (moving, fixed):
        prior.install_diagnostic_target_data(agent, diagnostic)
        b = agent._tensor_batch(final)
        targets.append(agent.bellman_target(b, sequence)[0].reshape(-1))

    oracle = torch.as_tensor(
        diagnostic["mc_current"], dtype=torch.float32, device=device
    ).reshape(-1)
    reward = torch.as_tensor(
        final["rewards"], dtype=torch.float32, device=device
    ).reshape(-1)
    terminal = torch.as_tensor(
        final["terminals"], dtype=torch.float32, device=device
    ).reshape(-1) >= 0.5
    contract["first_batch_targets_max_abs"] = float(
        (targets[0] - targets[1]).abs().max().item()
    )
    contract["first_batch_target_vs_mc_mae"] = float(
        (targets[0] - oracle).abs().mean().item()
    )
    contract["terminal_target_equals_reward_max_abs"] = float(
        (targets[0][terminal] - reward[terminal]).abs().max().item()
        if bool(terminal.any()) else 0.0
    )
    del moving, fixed
    prior.cleanup_device(device)
    return contract


def comparison_rows(moving, fixed):
    moving_by = {int(row["update"]): row for row in moving["trajectory"]}
    fixed_by = {int(row["update"]): row for row in fixed["trajectory"]}
    common = sorted(set(moving_by) & set(fixed_by))
    if not common:
        raise RuntimeError("Paired branches have no common milestone")

    rows = []
    for update in common:
        left = moving_by[update]
        right = fixed_by[update]
        left_q = left["online"]["qmin"]
        right_q = right["online"]["qmin"]
        rows.append({
            "update": update,
            "moving_online_qmin_spearman": left_q["spearman_q_return"],
            "fixed_online_qmin_spearman": right_q["spearman_q_return"],
            "fixed_minus_moving_qmin_spearman": (
                right_q["spearman_q_return"] - left_q["spearman_q_return"]
            ),
            "moving_online_qmean_spearman": (
                left["online"]["qmean"]["spearman_q_return"]
            ),
            "fixed_online_qmean_spearman": (
                right["online"]["qmean"]["spearman_q_return"]
            ),
            "moving_online_qmin_pearson": left_q["pearson_q_return"],
            "fixed_online_qmin_pearson": right_q["pearson_q_return"],
            "moving_online_qmin_mae": left_q["mae_q_return"],
            "fixed_online_qmin_mae": right_q["mae_q_return"],
            "moving_online_qmin_bias": left_q["signed_bias_q_return"],
            "fixed_online_qmin_bias": right_q["signed_bias_q_return"],
            "moving_online_qmin_mean": left_q["mean"],
            "fixed_online_qmin_mean": right_q["mean"],
            "moving_online_qmin_std": left_q["std"],
            "fixed_online_qmin_std": right_q["std"],
            "moving_online_episode_auc": (
                left["online"]["probe_episode_auc_qmin"]
            ),
            "fixed_online_episode_auc": (
                right["online"]["probe_episode_auc_qmin"]
            ),
            "moving_target_qmin_spearman": (
                left["target"]["qmin"]["spearman_q_return"]
            ),
            "fixed_target_qmin_spearman": (
                right["target"]["qmin"]["spearman_q_return"]
            ),
            "moving_target_qmin_mae": (
                left["target"]["qmin"]["mae_q_return"]
            ),
            "fixed_target_qmin_mae": (
                right["target"]["qmin"]["mae_q_return"]
            ),
            "moving_target_qmin_bias": (
                left["target"]["qmin"]["signed_bias_q_return"]
            ),
            "fixed_target_qmin_bias": (
                right["target"]["qmin"]["signed_bias_q_return"]
            ),
        })
    return rows, common


def main():
    args = arguments()
    (
        run_dir, stage2_path, step0_path, replay_path, step0, config
    ) = load_inputs(args)
    fixed_replay, episodes = prior.load_canonical(replay_path)
    length = int(config["recurrent_replay"]["critic_context_length"])
    gamma = float(config["gamma"])
    schedule = prior.make_reference_schedule(
        episodes, args.updates, args.batch_size, length, args.seed
    )
    schedule_hash = hashlib.sha256(schedule.tobytes()).hexdigest()
    returns = prior.mc_return_cache(episodes, gamma)
    probe_refs = prior.make_probe_refs(
        episodes, args.probe_size, length, args.seed + 1000003
    )
    probe_returns, probe_labels, probe_episode_ids = (
        prior.probe_reference_returns(
            episodes, probe_refs, length, gamma
        )
    )
    mc_identity = prior.canonical_mc_identity(episodes, gamma)
    device = prior.resolve_device(args.device)
    contract = initial_contract(
        stage2_path, step0, episodes, schedule[0], returns, device
    )
    print(
        f"[SETUP] episodes={len(episodes)} updates={args.updates} "
        f"batch={args.batch_size} probe={len(probe_refs)} "
        f"device={device} schedule_sha256={schedule_hash}",
        flush=True,
    )
    branch_args = (
        stage2_path, step0, episodes, schedule, returns,
        probe_refs, probe_returns, probe_labels, probe_episode_ids,
        device, prior.DEFAULT_MILESTONES, args.probe_batch_size,
    )
    moving = prior.branch_run(
        "moving_qmean_replay_bootstrap", *branch_args
    )
    frozen = prior.branch_run(
        "fixed_step0_qmean_replay_bootstrap", *branch_args
    )
    rows, common = comparison_rows(moving, frozen)
    initial_spearman = rows[0]["moving_online_qmin_spearman"]
    last = rows[-1]

    validity = {
        "same_initial_online_hash": (
            moving["initial_online_hash"] == frozen["initial_online_hash"]
            and contract["same_initial_online_hash"]
        ),
        "same_initial_target_critic_hash": (
            moving["initial_target_critic_hash"]
            == frozen["initial_target_critic_hash"]
            and contract["same_initial_target_critic_hash"]
        ),
        "same_initial_actor_hash": (
            moving["initial_actor_hash"] == frozen["initial_actor_hash"]
            and contract["same_initial_actor_hash"]
        ),
        "same_initial_target_actor_hash": (
            moving["initial_target_actor_hash"]
            == frozen["initial_target_actor_hash"]
            and contract["same_initial_target_actor_hash"]
        ),
        "same_initial_optimizer_state": contract[
            "same_initial_optimizer_state"
        ],
        "same_initial_probe_qmin_spearman": abs(
            moving["trajectory"][0]["online"]["qmin"][
                "spearman_q_return"
            ] - frozen["trajectory"][0]["online"]["qmin"][
                "spearman_q_return"
            ]
        ) <= 1e-12,
        "same_first_batch_qmean_replay_target": (
            contract["first_batch_targets_max_abs"] <= 2e-6
        ),
        "terminal_semantics_match": (
            contract["terminal_target_equals_reward_max_abs"] <= 2e-6
        ),
        "canonical_mc_identity_max_abs_le_2e_6": (
            mc_identity["abs_max"] <= 2e-6
        ),
        "moving_target_critic_changed": moving[
            "target_critic_changed"
        ],
        "fixed_target_critic_unchanged": not frozen[
            "target_critic_changed"
        ],
        "actor_unchanged_both": (
            not moving["actor_changed"] and not frozen["actor_changed"]
        ),
        "target_actor_unchanged_both": (
            not moving["target_actor_changed"]
            and not frozen["target_actor_changed"]
        ),
        "same_precomputed_batch_schedule": True,
        "step0_has_zero_updates": (
            int(step0["env_steps"]) == 0
            and int(step0["updates"]) == 0
            and int(step0["actor_updates"]) == 0
        ),
    }
    valid = all(validity.values())
    complete = moving["completed"] and frozen["completed"]
    summary = {
        "last_common_update": common[-1],
        "initial_online_qmin_spearman": initial_spearman,
        "final_moving_online_qmin_spearman": last[
            "moving_online_qmin_spearman"
        ],
        "final_fixed_online_qmin_spearman": last[
            "fixed_online_qmin_spearman"
        ],
        "final_fixed_minus_moving_qmin_spearman": last[
            "fixed_minus_moving_qmin_spearman"
        ],
        "moving_change_from_initial": (
            last["moving_online_qmin_spearman"] - initial_spearman
        ),
        "fixed_change_from_initial": (
            last["fixed_online_qmin_spearman"] - initial_spearman
        ),
        "final_moving_qmin_mae": last["moving_online_qmin_mae"],
        "final_fixed_qmin_mae": last["fixed_online_qmin_mae"],
        "final_moving_qmin_bias": last["moving_online_qmin_bias"],
        "final_fixed_qmin_bias": last["fixed_online_qmin_bias"],
    }
    output = {
        "status": "PASS" if valid and complete else (
            "PARTIAL" if valid else "INVALID"
        ),
        "experiment": "moving_vs_fixed_step0_qmean_replay_bootstrap",
        "isolation_scope": (
            "Only target-Critic Polyak feedback differs; both branches "
            "use the same step0 Critic, restored optimizer, replay-next "
            "action qmean bootstrap, batch schedule and fixed MC probe."
        ),
        "device": str(device),
        "stage2_checkpoint": str(stage2_path),
        "stage3_step0_checkpoint": str(step0_path),
        "canonical_diagnostic_replay": str(replay_path),
        "canonical_seed": fixed_replay.get("seed"),
        "canonical_episode_count": len(episodes),
        "sequence_length": length,
        "gamma": gamma,
        "tau": float(config["tau"]),
        "critic_lr": float(config["critic_lr"]),
        "critic_weight_decay": float(config["critic_weight_decay"]),
        "updates": args.updates,
        "batch_size": args.batch_size,
        "schedule_seed": args.seed,
        "schedule_sha256": schedule_hash,
        "same_precomputed_batch_schedule": True,
        "probe_size": len(probe_refs),
        "probe_seed": args.seed + 1000003,
        "milestones": list(prior.DEFAULT_MILESTONES),
        "environment_steps_performed": 0,
        "actor_updates_performed": 0,
        "training_checkpoints_written": 0,
        "optimizer_steps_requested_per_branch": args.updates,
        "moving_optimizer_steps_completed": moving[
            "updates_completed"
        ],
        "fixed_optimizer_steps_completed": frozen[
            "updates_completed"
        ],
        "initial_contract": contract,
        "canonical_mc_identity": mc_identity,
        "validity": validity,
        "moving_branch": moving,
        "fixed_branch": frozen,
        "comparison": rows,
        "comparison_scope": {
            "common_milestones": common,
            "last_common_update": common[-1],
            "moving_failure": moving["failure"],
            "fixed_failure": frozen["failure"],
        },
        "summary": summary,
    }
    out_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "testing/stage2_vs_stage3_readiness/"
        "multi_moving_vs_fixed_qmean_replay_bootstrap.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("[VALIDITY]", json.dumps(validity, sort_keys=True))
    print("[SUMMARY]", json.dumps(summary, sort_keys=True))
    print(f"[STATUS] {output['status']}")
    print(f"[SAVED] {out_path}", flush=True)
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
