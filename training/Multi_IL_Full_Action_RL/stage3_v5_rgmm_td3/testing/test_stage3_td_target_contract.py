#!/usr/bin/env python3
"""Read-only Stage3-v5 TD-target contract test.

This test isolates implementation semantics, not learning quality. It compares
the production Stage3-v5 Bellman target path against an independently built
reference on a small set of real transitions from one canonical frozen replay.

Contracts checked:
  1. stored dones == terminated OR truncated for every canonical transition;
  2. terminal / truncated transitions do not bootstrap (TD target == reward);
  3. successor Critic history equals a manually shifted horizon-10 window;
  4. vectorized target-Actor reset semantics equal the stepwise reference;
  5. production expected-next-Q and TD target equal the manual reference;
  6. checkpoint target-network state is structurally sane.

No environment, rollout, optimizer.step(), Actor update, or Critic update occurs.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STAGE3 = HERE.parent
for directory in (HERE, STAGE3):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from stage3_v5_actor import load_exact_actor, target_final_distribution  # noqa: E402
from stage3_v5_agent import (  # noqa: E402
    RecurrentGMMTD3,
    _last_reset_starts_from_numpy,
    strict_stage2_load,
)
from stage3_v5_replay import OnlineSequenceReplay  # noqa: E402
from test_stage2_stage3_readiness_compare import resolve_device, sync  # noqa: E402


DEFAULT_STAGE2 = (
    "/data/home/3220251075/lerobot_workspace/training_runs/"
    "Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/"
    "stage2_2_h10_multi_20260923_150749/multi_q/checkpoints/step_00005000.pth"
)

BOUNDARY_STEPS = (0, 1, 8, 9, 10, 11, 18, 19, 20, 21)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3-run-dir", required=True)
    parser.add_argument("--stage2-checkpoint", default=DEFAULT_STAGE2)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--diagnostic-replay",
        help="Canonical .sequences.npy; defaults to Multi Stage3 200K sidecar.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=("step0_transfer.pth", "step_0100000.pth", "step_0200000.pth"),
    )
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--output")
    return parser.parse_args()


def cleanup_device(device):
    gc.collect()
    if device.type == "npu":
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def load_canonical_episodes(path):
    replay = OnlineSequenceReplay.load(path)
    fixed = replay.fixed_critic_diagnostic_set
    if fixed is None:
        raise RuntimeError(f"No fixed_critic_diagnostic_set in {path}")
    episodes = fixed.get("episodes", [])
    if not episodes:
        raise RuntimeError("Canonical diagnostic set contains no episodes")
    return fixed, episodes


def flatten_bool(episode, key):
    if key not in episode:
        return None
    return np.asarray(episode[key], dtype=bool).reshape(-1)


def terminal_arrays(episode):
    terminated = flatten_bool(episode, "terminated")
    truncated = flatten_bool(episode, "truncated")
    dones = flatten_bool(episode, "dones")
    length = len(np.asarray(episode["actions"]))

    if terminated is None:
        terminated = np.zeros(length, dtype=bool)
    if truncated is None:
        truncated = np.zeros(length, dtype=bool)
    if dones is None:
        if "terminals" in episode:
            dones = np.asarray(episode["terminals"], dtype=bool).reshape(-1)
        else:
            raise RuntimeError("Episode has neither dones nor terminals")

    if not (len(terminated) == len(truncated) == len(dones) == length):
        raise RuntimeError("Terminal arrays do not match episode length")
    return terminated, truncated, dones


def validate_terminal_contract(episodes):
    total = terminated_count = truncated_count = done_count = 0
    mismatches = []
    terminated_cases = []
    truncated_cases = []

    for episode_index, episode in enumerate(episodes):
        terminated, truncated, dones = terminal_arrays(episode)
        expected = terminated | truncated
        bad = np.flatnonzero(dones != expected)
        for target in bad[:20]:
            mismatches.append({
                "episode_index": int(episode_index),
                "target": int(target),
                "terminated": bool(terminated[target]),
                "truncated": bool(truncated[target]),
                "done": bool(dones[target]),
            })
        for target in np.flatnonzero(terminated)[:1]:
            terminated_cases.append((episode_index, int(target)))
        for target in np.flatnonzero(truncated)[:1]:
            truncated_cases.append((episode_index, int(target)))
        total += len(dones)
        terminated_count += int(terminated.sum())
        truncated_count += int(truncated.sum())
        done_count += int(dones.sum())

    return {
        "transition_count": int(total),
        "terminated_count": int(terminated_count),
        "truncated_count": int(truncated_count),
        "done_count": int(done_count),
        "mismatch_count": int(len(mismatches)),
        "mismatch_examples": mismatches,
        "terminated_cases": terminated_cases,
        "truncated_cases": truncated_cases,
    }


def select_cases(episodes, terminal_report):
    selected = []
    used = set()

    # Exact episode-step boundary probes. Prefer one episode that contains all
    # requested steps so the reset chronology is easy to interpret.
    for target in BOUNDARY_STEPS:
        found = None
        for episode_index, episode in enumerate(episodes):
            steps = np.asarray(
                episode.get("episode_steps", np.arange(len(episode["actions"]))),
                dtype=np.int64,
            ).reshape(-1)
            matches = np.flatnonzero(steps == int(target))
            if len(matches):
                found = (episode_index, int(matches[0]))
                break
        if found is not None:
            key = tuple(found)
            if key not in used:
                selected.append({
                    "label": f"episode_step_{target}",
                    "episode_index": int(found[0]),
                    "target_index": int(found[1]),
                })
                used.add(key)

    # Add one genuine terminated and one genuine truncated sample when present.
    for label, values in (
        ("terminated_case", terminal_report["terminated_cases"]),
        ("truncated_case", terminal_report["truncated_cases"]),
    ):
        if values:
            key = tuple(values[0])
            if key not in used:
                selected.append({
                    "label": label,
                    "episode_index": int(key[0]),
                    "target_index": int(key[1]),
                })
                used.add(key)

    if not selected:
        raise RuntimeError("No diagnostic transition cases could be selected")
    return selected


def state_dict_distance(left, right):
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        raise RuntimeError("State-dict keys differ")
    max_abs = 0.0
    l2_sq = 0.0
    finite = True
    for key in left_state:
        a = left_state[key].detach()
        b = right_state[key].detach()
        diff = (a - b).float()
        finite = finite and bool(torch.isfinite(a).all().item())
        finite = finite and bool(torch.isfinite(b).all().item())
        max_abs = max(max_abs, float(diff.abs().max().item()))
        l2_sq += float(torch.sum(diff * diff).item())
    return {
        "max_abs": float(max_abs),
        "l2": float(math.sqrt(l2_sq)),
        "finite": bool(finite),
    }


def distribution_diffs(production, reference):
    p_base = production.component_distribution.base_dist
    r_base = reference.component_distribution.base_dist
    p_logits = production.mixture_distribution.logits
    r_logits = reference.mixture_distribution.logits
    p_probs = production.mixture_distribution.probs
    r_probs = reference.mixture_distribution.probs

    return {
        "loc_max_abs": float((p_base.loc - r_base.loc).abs().max().item()),
        "scale_max_abs": float((p_base.scale - r_base.scale).abs().max().item()),
        "logits_max_abs": float((p_logits - r_logits).abs().max().item()),
        "probs_max_abs": float((p_probs - r_probs).abs().max().item()),
    }


def independently_shifted_successor(episode, start, target):
    """Build [o_(start+1), ..., o_(target+1)] without using the production helper."""
    observations = np.asarray(episode["observations"], dtype=np.float32)
    next_observations = np.asarray(episode["next_observations"], dtype=np.float32)
    actions = np.asarray(episode["actions"], dtype=np.float32)
    episode_steps = np.asarray(
        episode.get("episode_steps", np.arange(len(actions))), dtype=np.int64
    ).reshape(-1)

    if not (0 <= start <= target < len(actions)):
        raise ValueError("Invalid successor window indices")

    successor_obs = []
    continuity_diffs = []
    for current in range(start, target + 1):
        successor_index = current + 1
        if successor_index < len(observations):
            manual = observations[successor_index]
            continuity_diffs.append(
                float(np.max(np.abs(next_observations[current] - manual))))
        else:
            manual = next_observations[current]
        successor_obs.append(manual)

    successor_obs = np.stack(successor_obs).astype(np.float32)
    prior_actions = np.zeros((len(successor_obs), actions.shape[-1]), np.float32)
    if len(successor_obs) > 1:
        # Token j>0 corresponds to episode step start+1+j and therefore sees
        # the action executed at the immediately preceding successor step.
        prior_actions[1:] = actions[start + 1:target + 1]
    successor_steps = episode_steps[start:target + 1] + 1

    return {
        "observations": successor_obs,
        "previous_actions": prior_actions,
        "episode_steps": successor_steps.astype(np.int64),
        "continuity_max_abs": (
            float(max(continuity_diffs)) if continuity_diffs else 0.0
        ),
    }


@torch.no_grad()
def manual_reference(agent, episode, start, target, horizon, actor_horizon):
    shifted = independently_shifted_successor(episode, start, target)

    obs = torch.as_tensor(
        shifted["observations"][None], dtype=torch.float32, device=agent.device)
    previous_actions = torch.as_tensor(
        shifted["previous_actions"][None], dtype=torch.float32, device=agent.device)
    successor_steps = torch.as_tensor(
        shifted["episode_steps"][None], dtype=torch.long, device=agent.device)
    progress = successor_steps.to(dtype=torch.float32).unsqueeze(-1) / float(horizon)

    contexts, _ = agent.target_critic.encode_history(
        obs, previous_actions, progress)
    final_context = (contexts[0][:, -1], contexts[1][:, -1])

    # Independent stepwise target-Actor reference. Unlike the production
    # vectorized helper, this applies the reset check at each successor step.
    reference_distribution, _ = target_final_distribution(
        agent.target_actor, obs, successor_steps, horizon=actor_horizon)

    base = reference_distribution.component_distribution.base_dist
    means = base.loc
    probabilities = reference_distribution.mixture_distribution.probs
    action_dim = means.shape[-1]
    broadcast = (1,) * (means.ndim - 1) + (action_dim,)
    component_actions = (
        means * agent.action_scale.reshape(broadcast)
        + agent.action_offset.reshape(broadcast)
    )
    q1, q2 = agent.target_critic.q_from_context(final_context, component_actions)
    selected = torch.minimum(q1, q2).squeeze(-1)
    expected_next = (probabilities * selected).sum(-1)

    rewards = np.asarray(episode["rewards"], dtype=np.float32).reshape(-1)
    _, _, dones = terminal_arrays(episode)
    reward = torch.as_tensor(
        [rewards[target]], dtype=torch.float32, device=agent.device)
    terminal = torch.as_tensor(
        [float(dones[target])], dtype=torch.float32, device=agent.device)
    td_target = reward + float(agent.config["gamma"]) * (
        1.0 - terminal) * expected_next.reshape(-1)

    reset_positions = np.flatnonzero(
        shifted["episode_steps"] % int(actor_horizon) == 0)
    expected_reset_start = int(reset_positions[-1]) if len(reset_positions) else 0

    return {
        "shifted": shifted,
        "contexts": contexts,
        "distribution": reference_distribution,
        "expected_next": expected_next,
        "td_target": td_target,
        "expected_reset_start": expected_reset_start,
    }


def sequence_batch_from_episode(episode, target, context_length):
    actions = np.asarray(episode["actions"], dtype=np.float32)
    observations = np.asarray(episode["observations"], dtype=np.float32)
    next_observations = np.asarray(episode["next_observations"], dtype=np.float32)
    episode_steps = np.asarray(
        episode.get("episode_steps", np.arange(len(actions))), dtype=np.int64
    ).reshape(-1)
    rewards = np.asarray(episode["rewards"], dtype=np.float32).reshape(-1)
    _, _, dones = terminal_arrays(episode)

    length = min(int(context_length), int(target) + 1)
    start = int(target) - length + 1

    sequence = {
        "observations": observations[start:target + 1][None],
        "actions": actions[start:target + 1][None],
        "next_observations": next_observations[start:target + 1][None],
        "episode_steps": episode_steps[start:target + 1][None],
    }
    final = {
        "rewards": np.asarray([[rewards[target]]], dtype=np.float32),
        "terminals": np.asarray([[float(dones[target])]], dtype=np.float32),
    }
    return start, sequence, final


@torch.no_grad()
def run_case(agent, episode, case, context_length, atol):
    target = int(case["target_index"])
    start, sequence, final = sequence_batch_from_episode(
        episode, target, context_length)

    b = agent._tensor_batch(final)
    production = agent.bellman_target(b, sequence)
    (
        production_td,
        production_distribution,
        production_expected_next,
        _target_tensors,
        production_starts,
        production_contexts,
    ) = production

    if production_contexts is None:
        raise RuntimeError("History-aware target Critic did not return contexts")

    reference = manual_reference(
        agent,
        episode,
        start,
        target,
        int(agent.config["horizon"]),
        int(agent.config["actor_source_contract"]["rnn_horizon"]),
    )

    prod_final = (
        production_contexts[0][:, -1],
        production_contexts[1][:, -1],
    )
    ref_final = (
        reference["contexts"][0][:, -1],
        reference["contexts"][1][:, -1],
    )
    context_q1_diff = float((prod_final[0] - ref_final[0]).abs().max().item())
    context_q2_diff = float((prod_final[1] - ref_final[1]).abs().max().item())
    distribution_diff = distribution_diffs(
        production_distribution, reference["distribution"])
    expected_q_diff = float(
        (production_expected_next.reshape(-1)
         - reference["expected_next"].reshape(-1)).abs().max().item())
    td_diff = float(
        (production_td.reshape(-1)
         - reference["td_target"].reshape(-1)).abs().max().item())

    rewards = np.asarray(episode["rewards"], dtype=np.float32).reshape(-1)
    terminated, truncated, dones = terminal_arrays(episode)
    reward_value = float(rewards[target])
    production_td_value = float(production_td.reshape(-1)[0].item())
    terminal_reward_diff = (
        abs(production_td_value - reward_value) if bool(dones[target]) else None
    )

    original_steps = np.asarray(sequence["episode_steps"], dtype=np.int64)
    helper_starts = _last_reset_starts_from_numpy(
        original_steps,
        int(agent.config["actor_source_contract"]["rnn_horizon"]),
    )
    helper_start = int(helper_starts[0])
    production_start = int(production_starts[0])
    expected_start = int(reference["expected_reset_start"])

    max_actor_diff = max(distribution_diff.values())
    max_context_diff = max(context_q1_diff, context_q2_diff)
    checks = {
        "successor_context_match": bool(max_context_diff <= atol),
        "target_actor_distribution_match": bool(max_actor_diff <= atol),
        "expected_next_q_match": bool(expected_q_diff <= atol),
        "td_target_match": bool(td_diff <= atol),
        "reset_start_match": bool(
            production_start == helper_start == expected_start),
        "terminal_no_bootstrap": bool(
            terminal_reward_diff is None or terminal_reward_diff <= atol),
        "replay_continuity_match": bool(
            reference["shifted"]["continuity_max_abs"] <= atol),
    }

    steps = np.asarray(
        episode.get("episode_steps", np.arange(len(episode["actions"]))),
        dtype=np.int64,
    ).reshape(-1)

    return {
        "label": case["label"],
        "episode_index": int(case["episode_index"]),
        "diagnostic_episode_id": (
            int(episode["diagnostic_episode_id"])
            if "diagnostic_episode_id" in episode else None
        ),
        "target_index": target,
        "episode_step": int(steps[target]),
        "window_start_index": int(start),
        "window_length": int(target - start + 1),
        "terminated": bool(terminated[target]),
        "truncated": bool(truncated[target]),
        "done_for_td": bool(dones[target]),
        "reward": reward_value,
        "production_td": production_td_value,
        "production_expected_next_q": float(
            production_expected_next.reshape(-1)[0].item()),
        "reference_expected_next_q": float(
            reference["expected_next"].reshape(-1)[0].item()),
        "reference_td": float(reference["td_target"].reshape(-1)[0].item()),
        "production_reset_start": production_start,
        "helper_reset_start": helper_start,
        "reference_reset_start": expected_start,
        "successor_context_q1_max_abs": context_q1_diff,
        "successor_context_q2_max_abs": context_q2_diff,
        "target_actor_distribution_max_abs": distribution_diff,
        "expected_next_q_max_abs": expected_q_diff,
        "td_target_max_abs": td_diff,
        "terminal_reward_max_abs": terminal_reward_diff,
        "replay_next_obs_continuity_max_abs": float(
            reference["shifted"]["continuity_max_abs"]),
        "checks": checks,
        "pass": bool(all(checks.values())),
    }


def load_checkpoint_into_agent(agent, payload):
    for key in ("actor", "target_actor", "q1_q2", "target_q1_q2"):
        if key not in payload:
            raise RuntimeError(f"Checkpoint misses {key}")
    agent.actor.load_state_dict(payload["actor"], strict=True)
    agent.target_actor.load_state_dict(payload["target_actor"], strict=True)
    agent.critic.load_state_dict(payload["q1_q2"], strict=True)
    agent.target_critic.load_state_dict(payload["target_q1_q2"], strict=True)
    agent.actor.eval()
    agent.target_actor.eval()
    agent.target_actor.requires_grad_(False)
    agent.critic.eval()
    agent.target_critic.eval()
    agent.target_critic.requires_grad_(False)


def validate_checkpoint_contract(reference_config, payload, name):
    if payload.get("stage") != "stage3-v5":
        raise RuntimeError(f"{name}: not Stage3-v5")
    if payload.get("group") != "multi_q":
        raise RuntimeError(f"{name}: not multi_q")
    config = payload["config"]
    for key in (
        "gamma", "horizon", "recurrent_replay",
        "actor_source_contract", "bc_rnn_checkpoint",
    ):
        if config.get(key) != reference_config.get(key):
            raise RuntimeError(f"{name}: config mismatch for {key}")
    if int(config["recurrent_replay"]["critic_context_length"]) != 10:
        raise RuntimeError(f"{name}: critic_context_length is not 10")
    if int(config["actor_source_contract"]["rnn_horizon"]) != 10:
        raise RuntimeError(f"{name}: Actor horizon is not 10")


def main():
    args = arguments()
    run_dir = Path(args.stage3_run_dir).resolve()
    stage2_path = Path(args.stage2_checkpoint).resolve()
    diagnostic_replay = (
        Path(args.diagnostic_replay).resolve()
        if args.diagnostic_replay
        else run_dir / "multi_q" / "checkpoints" / "step_0200000.sequences.npy"
    )
    checkpoint_paths = [
        run_dir / "multi_q" / "checkpoints" / name
        for name in args.checkpoints
    ]

    if not stage2_path.exists():
        raise FileNotFoundError(stage2_path)
    if not diagnostic_replay.exists():
        raise FileNotFoundError(diagnostic_replay)
    for path in checkpoint_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.atol <= 0:
        raise ValueError("--atol must be > 0")

    fixed, episodes = load_canonical_episodes(diagnostic_replay)
    terminal_report = validate_terminal_contract(episodes)
    selected_cases = select_cases(episodes, terminal_report)

    payloads = [
        torch.load(path, map_location="cpu")
        for path in checkpoint_paths
    ]
    reference_config = payloads[0]["config"]
    for path, payload in zip(checkpoint_paths, payloads):
        validate_checkpoint_contract(reference_config, payload, path.name)

    device = resolve_device(args.device)

    # Build exactly the same Actor / Critic classes as production Stage3-v5.
    actor, rollout, actor_metadata = load_exact_actor(
        reference_config["bc_rnn_checkpoint"], device)
    del rollout
    critic, _ = strict_stage2_load(stage2_path, device)

    first_payload = payloads[0]
    actor.load_state_dict(first_payload["actor"], strict=True)
    critic.load_state_dict(first_payload["q1_q2"], strict=True)

    normalization = first_payload["action_normalization_stats"]
    scale = torch.as_tensor(
        normalization["scale"], dtype=torch.float32, device=device
    ).reshape(1, 1, 1, 14)
    offset = torch.as_tensor(
        normalization["offset"], dtype=torch.float32, device=device
    ).reshape(1, 1, 1, 14)

    agent = RecurrentGMMTD3(
        actor, critic, reference_config, device, scale, offset)
    agent.profiler.enabled = False

    checkpoint_results = {}
    overall_pass = terminal_report["mismatch_count"] == 0

    for path, payload in zip(checkpoint_paths, payloads):
        load_checkpoint_into_agent(agent, payload)
        sync(device)

        actor_distance = state_dict_distance(agent.actor, agent.target_actor)
        critic_distance = state_dict_distance(agent.critic, agent.target_critic)

        cases = []
        for case in selected_cases:
            episode = episodes[int(case["episode_index"])]
            result = run_case(
                agent,
                episode,
                case,
                context_length=10,
                atol=float(args.atol),
            )
            cases.append(result)
            overall_pass = overall_pass and bool(result["pass"])

        is_step0 = int(payload.get("env_steps", -1)) == 0
        step0_critic_equal = (
            critic_distance["max_abs"] <= float(args.atol)
            if is_step0 else None
        )
        frozen_actor_equal = (
            actor_distance["max_abs"] <= float(args.atol)
            if int(payload.get("actor_updates", -1)) == 0 else None
        )
        if step0_critic_equal is not None:
            overall_pass = overall_pass and bool(step0_critic_equal)
        if frozen_actor_equal is not None:
            overall_pass = overall_pass and bool(frozen_actor_equal)

        checkpoint_results[path.stem] = {
            "checkpoint": str(path),
            "env_steps": int(payload.get("env_steps", -1)),
            "critic_updates": int(payload.get("updates", -1)),
            "actor_updates": int(payload.get("actor_updates", -1)),
            "online_vs_target_actor": actor_distance,
            "online_vs_target_critic": critic_distance,
            "step0_target_critic_equals_online": step0_critic_equal,
            "frozen_target_actor_equals_online": frozen_actor_equal,
            "case_count": int(len(cases)),
            "cases": cases,
            "max_diffs": {
                "successor_context": float(max(
                    max(case["successor_context_q1_max_abs"],
                        case["successor_context_q2_max_abs"])
                    for case in cases
                )),
                "target_actor_distribution": float(max(
                    max(case["target_actor_distribution_max_abs"].values())
                    for case in cases
                )),
                "expected_next_q": float(max(
                    case["expected_next_q_max_abs"] for case in cases
                )),
                "td_target": float(max(
                    case["td_target_max_abs"] for case in cases
                )),
                "terminal_reward": float(max(
                    [case["terminal_reward_max_abs"] for case in cases
                     if case["terminal_reward_max_abs"] is not None] or [0.0]
                )),
                "replay_continuity": float(max(
                    case["replay_next_obs_continuity_max_abs"] for case in cases
                )),
            },
            "pass": bool(
                all(case["pass"] for case in cases)
                and (step0_critic_equal is not False)
                and (frozen_actor_equal is not False)
            ),
        }

    output = {
        "status": "PASS" if overall_pass else "FAIL",
        "read_only": True,
        "environment_steps_performed": 0,
        "optimizer_steps_performed": 0,
        "actor_updates_performed": 0,
        "critic_updates_performed": 0,
        "device": str(device),
        "atol": float(args.atol),
        "stage2_architecture_checkpoint": str(stage2_path),
        "canonical_diagnostic_replay": str(diagnostic_replay),
        "canonical_seed": fixed.get("seed"),
        "canonical_episode_count": int(len(episodes)),
        "terminal_contract": {
            key: value for key, value in terminal_report.items()
            if key not in ("terminated_cases", "truncated_cases")
        },
        "selected_case_labels": [case["label"] for case in selected_cases],
        "production_path": "RecurrentGMMTD3.bellman_target",
        "reference_path": (
            "manual shifted successor window + target_critic.encode_history + "
            "stepwise stage3_v5_actor.target_final_distribution"
        ),
        "actor_metadata": actor_metadata,
        "checkpoints": checkpoint_results,
    }

    out_path = (
        Path(args.output).resolve()
        if args.output
        else run_dir / "testing" / "stage2_vs_stage3_readiness"
        / "multi_td_target_contract.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")

    print(
        f"[TERMINAL CONTRACT] transitions={terminal_report['transition_count']} "
        f"terminated={terminal_report['terminated_count']} "
        f"truncated={terminal_report['truncated_count']} "
        f"done={terminal_report['done_count']} "
        f"mismatch={terminal_report['mismatch_count']}"
    )
    print(
        "checkpoint\tenv_steps\tupdates\tactor_updates\t"
        "ctx_max\tactor_max\tnextQ_max\ttd_max\tterminal_max\tpass"
    )
    for name, row in checkpoint_results.items():
        diff = row["max_diffs"]
        print(
            f"{name}\t{row['env_steps']}\t{row['critic_updates']}\t"
            f"{row['actor_updates']}\t"
            f"{diff['successor_context']:.8g}\t"
            f"{diff['target_actor_distribution']:.8g}\t"
            f"{diff['expected_next_q']:.8g}\t"
            f"{diff['td_target']:.8g}\t"
            f"{diff['terminal_reward']:.8g}\t"
            f"{row['pass']}"
        )

    print("\n[SELECTED CASES]")
    for case in selected_cases:
        print(
            f"{case['label']}: episode={case['episode_index']} "
            f"target_index={case['target_index']}"
        )
    print(f"\n[STATUS] {output['status']}")
    print(f"[SAVED] {out_path}", flush=True)

    # Return non-zero on a real contract mismatch so Luna / Codex cannot miss it.
    if not overall_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
