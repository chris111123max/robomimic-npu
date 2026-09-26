#!/usr/bin/env python3
"""Small causal test for Stage3-v5 target-Critic bootstrap feedback.

Two branches start from the exact same Multi Stage3 step0 checkpoint and consume
the exact same deterministic sequence batches in the exact same order:

  A) polyak: production Critic update + production target-Critic Polyak update.
  B) frozen: production Critic update, but the step0 target Critic is never
             changed.

Actor learning is disabled in both branches. The target Actor is frozen and
identical. No environment is created. This test intentionally isolates the
effect of feeding the evolving online Critic back into future TD targets through
the target Critic.

The experiment is diagnostic only and writes no training checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STAGE3 = HERE.parent
for directory in (HERE, STAGE3):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from stage3_v5_actor import load_exact_actor  # noqa: E402
from stage3_v5_agent import RecurrentGMMTD3, strict_stage2_load  # noqa: E402
from stage3_v5_history_critic import encode_replay_contexts  # noqa: E402
from stage3_v5_readiness import auc, correlation, discounted_returns  # noqa: E402
from stage3_v5_replay import OnlineSequenceReplay, final_transition  # noqa: E402
from test_stage2_stage3_readiness_compare import resolve_device, sync  # noqa: E402


DEFAULT_STAGE2 = (
    "/data/home/3220251075/lerobot_workspace/training_runs/"
    "Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/"
    "stage2_2_h10_multi_20260923_150749/multi_q/checkpoints/step_00005000.pth"
)

DEFAULT_MILESTONES = (0, 100, 250, 500, 1000, 2000)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-run-dir", required=True)
    parser.add_argument("--stage2-checkpoint", default=DEFAULT_STAGE2)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--probe-size", type=int, default=8192)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument(
        "--diagnostic-replay",
        help="Canonical replay; defaults to Multi Stage3 200K sidecar.",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def cleanup_device(device):
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def module_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_canonical(path):
    replay = OnlineSequenceReplay.load(path)
    fixed = replay.fixed_critic_diagnostic_set
    if fixed is None:
        raise RuntimeError(f"No fixed_critic_diagnostic_set in {path}")
    episodes = fixed.get("episodes", [])
    if not episodes:
        raise RuntimeError("Canonical diagnostic set contains no episodes")
    return fixed, episodes


def eligible_episodes(episodes, length):
    result = [
        (index, episode)
        for index, episode in enumerate(episodes)
        if len(np.asarray(episode["actions"])) >= int(length)
    ]
    if not result:
        raise RuntimeError("No canonical episode is long enough for recurrent sampling")
    return result


def make_reference_schedule(episodes, updates, batch_size, length, seed):
    """Precompute compact (episode index, start) refs so both branches are exact."""
    eligible = eligible_episodes(episodes, length)
    rng = np.random.default_rng(int(seed))
    refs = np.empty((int(updates), int(batch_size), 2), dtype=np.int32)
    for update in range(int(updates)):
        choices = rng.integers(len(eligible), size=int(batch_size))
        for row, choice in enumerate(choices):
            episode_index, episode = eligible[int(choice)]
            count = len(episode["actions"]) - int(length) + 1
            start = int(rng.integers(count))
            refs[update, row] = (int(episode_index), start)
    return refs


def make_probe_refs(episodes, probe_size, length, seed):
    eligible = eligible_episodes(episodes, length)
    all_refs = []
    for episode_index, episode in eligible:
        for start in range(len(episode["actions"]) - int(length) + 1):
            all_refs.append((int(episode_index), int(start)))
    if not all_refs:
        raise RuntimeError("No train-eligible probe transitions")
    rng = np.random.default_rng(int(seed))
    count = min(int(probe_size), len(all_refs))
    choice = rng.choice(len(all_refs), size=count, replace=False)
    return np.asarray([all_refs[int(index)] for index in choice], dtype=np.int32)


def stack_sequence_batch(episodes, refs, length):
    keys = (
        "observations", "actions", "rewards",
        "next_observations", "terminals", "episode_steps",
    )
    result = {key: [] for key in keys}
    for episode_index, start in np.asarray(refs, dtype=np.int64):
        episode = episodes[int(episode_index)]
        stop = int(start) + int(length)
        for key in keys:
            if key not in episode:
                raise RuntimeError(
                    f"Canonical episode {episode_index} misses required key {key}")
            result[key].append(np.asarray(episode[key])[int(start):stop])
    return {key: np.stack(value) for key, value in result.items()}


def probe_reference_returns(episodes, refs, length, gamma):
    cache = {}
    values = np.empty(len(refs), dtype=np.float32)
    labels = np.empty(len(refs), dtype=np.int64)
    episode_ids = np.empty(len(refs), dtype=np.int64)
    for row, (episode_index, start) in enumerate(np.asarray(refs, dtype=np.int64)):
        episode_index = int(episode_index)
        if episode_index not in cache:
            cache[episode_index] = discounted_returns(episodes[episode_index], gamma)
        target = int(start) + int(length) - 1
        values[row] = cache[episode_index][target]
        labels[row] = int(bool(episodes[episode_index].get("success", False)))
        episode_ids[row] = episode_index
    return values, labels, episode_ids


@torch.no_grad()
def evaluate_probe(
    critic,
    episodes,
    refs,
    returns,
    labels,
    episode_ids,
    device,
    horizon,
    length,
    batch_size,
):
    critic.eval()
    qmin_parts = []
    for first in range(0, len(refs), int(batch_size)):
        selected = refs[first:first + int(batch_size)]
        sequence = stack_sequence_batch(episodes, selected, length)
        obs = torch.as_tensor(
            sequence["observations"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(
            sequence["actions"], dtype=torch.float32, device=device)
        steps = torch.as_tensor(
            sequence["episode_steps"], dtype=torch.long, device=device)
        contexts = encode_replay_contexts(
            critic, obs, actions, steps, int(horizon))
        executed = actions[:, -1]
        q1, q2 = critic.q_from_context(
            (contexts[0][:, -1], contexts[1][:, -1]), executed)
        qmin_parts.append(
            torch.minimum(q1, q2).reshape(-1).detach().cpu().numpy())

    qmin = np.concatenate(qmin_parts).astype(np.float64)
    reference = np.asarray(returns, dtype=np.float64)
    spearman, pearson = correlation(qmin, reference)

    # Episode-level AUC using mean Q over sampled probe transitions.
    unique = np.unique(episode_ids)
    episode_q = []
    episode_labels = []
    for episode_index in unique:
        mask = episode_ids == episode_index
        episode_q.append(float(qmin[mask].mean()))
        episode_labels.append(int(labels[np.flatnonzero(mask)[0]]))
    episode_q = np.asarray(episode_q, dtype=np.float64)
    episode_labels = np.asarray(episode_labels, dtype=np.int64)
    episode_auc = (
        float(auc(episode_labels, episode_q))
        if len(np.unique(episode_labels)) == 2 else None
    )

    return {
        "count": int(len(qmin)),
        "spearman_q_return": float(spearman),
        "pearson_q_return": float(pearson),
        "probe_episode_auc": episode_auc,
        "qmin_mean": float(qmin.mean()),
        "qmin_std": float(qmin.std()),
        "qmin_min": float(qmin.min()),
        "qmin_max": float(qmin.max()),
        "mae_q_return": float(np.mean(np.abs(qmin - reference))),
    }


def state_distance(left, right):
    max_abs = 0.0
    l2_sq = 0.0
    with torch.no_grad():
        for a, b in zip(left.parameters(), right.parameters()):
            diff = (a - b).float()
            max_abs = max(max_abs, float(diff.abs().max().item()))
            l2_sq += float(torch.sum(diff * diff).item())
    return {"max_abs": float(max_abs), "l2": float(np.sqrt(l2_sq))}


def build_agent(stage2_path, step0_payload, device):
    config = step0_payload["config"]
    actor, rollout, _ = load_exact_actor(config["bc_rnn_checkpoint"], device)
    del rollout
    actor.load_state_dict(step0_payload["actor"], strict=True)

    critic, _ = strict_stage2_load(stage2_path, device)
    critic.load_state_dict(step0_payload["q1_q2"], strict=True)

    normalization = step0_payload["action_normalization_stats"]
    scale = torch.as_tensor(
        normalization["scale"], dtype=torch.float32, device=device
    ).reshape(1, 1, 1, 14)
    offset = torch.as_tensor(
        normalization["offset"], dtype=torch.float32, device=device
    ).reshape(1, 1, 1, 14)

    agent = RecurrentGMMTD3(actor, critic, config, device, scale, offset)
    agent.profiler.enabled = False
    agent.actor.load_state_dict(step0_payload["actor"], strict=True)
    agent.target_actor.load_state_dict(step0_payload["target_actor"], strict=True)
    agent.critic.load_state_dict(step0_payload["q1_q2"], strict=True)
    agent.target_critic.load_state_dict(step0_payload["target_q1_q2"], strict=True)
    agent.critic_optimizer.load_state_dict(step0_payload["critic_optimizer"])
    agent.critic_updates = int(step0_payload.get("updates", 0))
    agent.actor_updates = int(step0_payload.get("actor_updates", 0))
    agent.set_actor_training_enabled(False)
    agent.actor.eval()
    agent.target_actor.eval()
    agent.target_actor.low_noise_eval = True
    agent.target_actor.requires_grad_(False)
    agent.target_critic.requires_grad_(False)
    return agent


def branch_run(
    mode,
    stage2_path,
    step0_payload,
    episodes,
    schedule,
    probe_refs,
    probe_returns,
    probe_labels,
    probe_episode_ids,
    device,
    milestones,
    probe_batch_size,
):
    if mode not in ("polyak", "frozen"):
        raise ValueError(mode)

    config = step0_payload["config"]
    length = int(config["recurrent_replay"]["critic_context_length"])
    horizon = int(config["horizon"])
    agent = build_agent(stage2_path, step0_payload, device)

    initial_target_hash = module_digest(agent.target_critic)
    initial_target_actor_hash = module_digest(agent.target_actor)
    initial_online_hash = module_digest(agent.critic)

    trajectory = []

    def record(update):
        sync(device)
        online_metrics = evaluate_probe(
            agent.critic, episodes, probe_refs, probe_returns, probe_labels,
            probe_episode_ids, device, horizon, length, probe_batch_size)
        target_metrics = evaluate_probe(
            agent.target_critic, episodes, probe_refs, probe_returns, probe_labels,
            probe_episode_ids, device, horizon, length, probe_batch_size)
        sync(device)
        trajectory.append({
            "update": int(update),
            "online": online_metrics,
            "target": target_metrics,
            "online_target_distance": state_distance(
                agent.critic, agent.target_critic),
        })
        print(
            f"[{mode}] update={update} "
            f"online_spear={online_metrics['spearman_q_return']:.6f} "
            f"target_spear={target_metrics['spearman_q_return']:.6f} "
            f"qmean={online_metrics['qmin_mean']:.6f}",
            flush=True,
        )

    record(0)

    milestone_set = set(int(value) for value in milestones)
    for index in range(len(schedule)):
        sequence = stack_sequence_batch(
            episodes, schedule[index], length)
        final = final_transition(sequence)

        agent.critic_update(final, sequence, collect_metrics=False)
        if mode == "polyak":
            # Exact production CRITIC_ONLY behavior: target Critic is Polyak
            # updated every Critic step; target Actor remains frozen because
            # actor_gate_open=False.
            agent.polyak_update()
        elif mode == "frozen":
            # Causal intervention: intentionally skip target Critic Polyak.
            pass

        completed = index + 1
        if completed in milestone_set:
            record(completed)

    final_target_hash = module_digest(agent.target_critic)
    final_target_actor_hash = module_digest(agent.target_actor)
    final_online_hash = module_digest(agent.critic)

    result = {
        "mode": mode,
        "updates_performed": int(len(schedule)),
        "initial_online_hash": initial_online_hash,
        "final_online_hash": final_online_hash,
        "initial_target_critic_hash": initial_target_hash,
        "final_target_critic_hash": final_target_hash,
        "target_critic_changed": bool(final_target_hash != initial_target_hash),
        "initial_target_actor_hash": initial_target_actor_hash,
        "final_target_actor_hash": final_target_actor_hash,
        "target_actor_changed": bool(final_target_actor_hash != initial_target_actor_hash),
        "trajectory": trajectory,
    }

    del agent
    cleanup_device(device)
    return result


def milestone_values(updates):
    base = [value for value in DEFAULT_MILESTONES if value <= int(updates)]
    if int(updates) not in base:
        base.append(int(updates))
    return tuple(sorted(set(base)))


def main():
    args = arguments()
    if args.updates < 1:
        raise ValueError("--updates must be >= 1")
    if args.batch_size != 256:
        raise ValueError(
            "Keep --batch-size 256 so the causal test uses the production Critic batch size")
    if args.probe_size < 100:
        raise ValueError("--probe-size must be >= 100")
    if args.probe_batch_size < 1:
        raise ValueError("--probe-batch-size must be >= 1")

    run_dir = Path(args.stage3_run_dir).resolve()
    stage2_path = Path(args.stage2_checkpoint).resolve()
    step0_path = run_dir / "multi_q" / "checkpoints" / "step0_transfer.pth"
    diagnostic_replay = (
        Path(args.diagnostic_replay).resolve()
        if args.diagnostic_replay
        else run_dir / "multi_q" / "checkpoints" / "step_0200000.sequences.npy"
    )

    for path in (stage2_path, step0_path, diagnostic_replay):
        if not path.exists():
            raise FileNotFoundError(path)

    step0_payload = torch.load(step0_path, map_location="cpu")
    if step0_payload.get("stage") != "stage3-v5":
        raise RuntimeError("step0 checkpoint is not Stage3-v5")
    if step0_payload.get("group") != "multi_q":
        raise RuntimeError("step0 checkpoint is not multi_q")
    if int(step0_payload.get("env_steps", -1)) != 0:
        raise RuntimeError("Reference checkpoint is not env-step zero")
    if int(step0_payload.get("updates", -1)) != 0:
        raise RuntimeError("Reference checkpoint already contains Critic updates")
    if int(step0_payload.get("actor_updates", -1)) != 0:
        raise RuntimeError("Reference checkpoint already contains Actor updates")

    config = step0_payload["config"]
    length = int(config["recurrent_replay"]["critic_context_length"])
    if length != 10:
        raise RuntimeError("Expected critic_context_length=10")

    fixed, episodes = load_canonical(diagnostic_replay)
    schedule = make_reference_schedule(
        episodes, args.updates, args.batch_size, length, args.seed)
    probe_refs = make_probe_refs(
        episodes, args.probe_size, length, args.seed + 1000003)
    probe_returns, probe_labels, probe_episode_ids = probe_reference_returns(
        episodes, probe_refs, length, float(config["gamma"]))

    milestones = milestone_values(args.updates)
    device = resolve_device(args.device)

    print(
        f"[CAUSAL TEST] updates={args.updates} batch={args.batch_size} "
        f"probe={len(probe_refs)} milestones={milestones} device={device}",
        flush=True,
    )

    # Run sequentially on one accelerator. Both branches rebuild from step0
    # and consume the same precomputed schedule.
    polyak = branch_run(
        "polyak", stage2_path, step0_payload, episodes, schedule,
        probe_refs, probe_returns, probe_labels, probe_episode_ids,
        device, milestones, args.probe_batch_size)
    frozen = branch_run(
        "frozen", stage2_path, step0_payload, episodes, schedule,
        probe_refs, probe_returns, probe_labels, probe_episode_ids,
        device, milestones, args.probe_batch_size)

    if len(polyak["trajectory"]) != len(frozen["trajectory"]):
        raise RuntimeError("Branch milestone trajectories differ")

    comparison = []
    for left, right in zip(polyak["trajectory"], frozen["trajectory"]):
        if left["update"] != right["update"]:
            raise RuntimeError("Branch milestone updates differ")
        comparison.append({
            "update": int(left["update"]),
            "polyak_online_spearman": float(
                left["online"]["spearman_q_return"]),
            "frozen_online_spearman": float(
                right["online"]["spearman_q_return"]),
            "frozen_minus_polyak_online_spearman": float(
                right["online"]["spearman_q_return"]
                - left["online"]["spearman_q_return"]),
            "polyak_target_spearman": float(
                left["target"]["spearman_q_return"]),
            "frozen_target_spearman": float(
                right["target"]["spearman_q_return"]),
            "polyak_qmin_mean": float(left["online"]["qmin_mean"]),
            "frozen_qmin_mean": float(right["online"]["qmin_mean"]),
        })

    initial_spearman = float(polyak["trajectory"][0]["online"]["spearman_q_return"])
    final_polyak = float(polyak["trajectory"][-1]["online"]["spearman_q_return"])
    final_frozen = float(frozen["trajectory"][-1]["online"]["spearman_q_return"])

    validity = {
        "same_initial_online_hash": bool(
            polyak["initial_online_hash"] == frozen["initial_online_hash"]),
        "same_initial_target_critic_hash": bool(
            polyak["initial_target_critic_hash"]
            == frozen["initial_target_critic_hash"]),
        "same_initial_target_actor_hash": bool(
            polyak["initial_target_actor_hash"]
            == frozen["initial_target_actor_hash"]),
        "same_initial_probe_spearman": bool(
            abs(
                polyak["trajectory"][0]["online"]["spearman_q_return"]
                - frozen["trajectory"][0]["online"]["spearman_q_return"]
            ) <= 1e-12
        ),
        "frozen_target_critic_unchanged": bool(
            not frozen["target_critic_changed"]),
        "polyak_target_critic_changed": bool(
            polyak["target_critic_changed"]),
        "target_actor_unchanged_both": bool(
            not polyak["target_actor_changed"]
            and not frozen["target_actor_changed"]),
    }
    valid = bool(all(validity.values()))

    output = {
        "status": "PASS" if valid else "INVALID",
        "experiment": "stage3_v5_frozen_target_causal",
        "environment_steps_performed": 0,
        "actor_updates_performed": 0,
        "training_checkpoints_written": 0,
        "optimizer_steps_per_branch": int(args.updates),
        "device": str(device),
        "stage2_architecture_checkpoint": str(stage2_path),
        "stage3_step0_checkpoint": str(step0_path),
        "canonical_diagnostic_replay": str(diagnostic_replay),
        "canonical_seed": fixed.get("seed"),
        "canonical_episode_count": int(len(episodes)),
        "sequence_length": int(length),
        "batch_size": int(args.batch_size),
        "updates": int(args.updates),
        "schedule_seed": int(args.seed),
        "same_precomputed_batch_schedule": True,
        "isolation_scope": (
            "fixed canonical online replay only; this intentionally isolates "
            "target-Critic feedback and is not a reproduction of the 50/50 "
            "offline+online production replay mixture"
        ),
        "probe_size": int(len(probe_refs)),
        "probe_seed": int(args.seed + 1000003),
        "milestones": list(map(int, milestones)),
        "gamma": float(config["gamma"]),
        "tau": float(config["tau"]),
        "critic_lr": float(config["critic_lr"]),
        "critic_weight_decay": float(config["critic_weight_decay"]),
        "intervention": {
            "polyak": "production critic_update followed by production polyak_update",
            "frozen": "production critic_update with target Critic fixed at step0",
            "target_actor": "frozen and identical in both branches",
            "only_intended_difference": "whether target Critic receives Polyak updates",
        },
        "polyak": polyak,
        "frozen": frozen,
        "comparison": comparison,
        "validity": validity,
        "summary": {
            "initial_online_spearman": initial_spearman,
            "final_polyak_online_spearman": final_polyak,
            "final_frozen_online_spearman": final_frozen,
            "polyak_change_from_initial": float(final_polyak - initial_spearman),
            "frozen_change_from_initial": float(final_frozen - initial_spearman),
            "final_frozen_minus_polyak": float(final_frozen - final_polyak),
            "frozen_target_critic_unchanged": bool(
                not frozen["target_critic_changed"]),
            "polyak_target_critic_changed": bool(
                polyak["target_critic_changed"]),
            "target_actor_unchanged_both": bool(
                not polyak["target_actor_changed"]
                and not frozen["target_actor_changed"]),
        },
    }

    out_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "testing" / "stage2_vs_stage3_readiness"
        / "multi_frozen_target_causal.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    print(
        "\nupdate\tpolyak_online\tfrozen_online\tfrozen-polyak\t"
        "polyak_target\tfrozen_target"
    )
    for row in comparison:
        print(
            f"{row['update']}\t"
            f"{row['polyak_online_spearman']:.6f}\t"
            f"{row['frozen_online_spearman']:.6f}\t"
            f"{row['frozen_minus_polyak_online_spearman']:.6f}\t"
            f"{row['polyak_target_spearman']:.6f}\t"
            f"{row['frozen_target_spearman']:.6f}"
        )

    print("\n[VALIDITY]")
    print(json.dumps(validity, indent=2, sort_keys=True))
    print("\n[SUMMARY]")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    print(f"[SAVED] {out_path}", flush=True)
    if not valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
