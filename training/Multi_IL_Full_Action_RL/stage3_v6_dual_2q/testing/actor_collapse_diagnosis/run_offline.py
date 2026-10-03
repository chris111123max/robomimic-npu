#!/usr/bin/env python3
"""Offline Stage3-v6 Actor/Q diagnosis on fixed, aligned replay contexts."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
V5 = ROOT / "stage3_v5_rgmm_td3"
V6 = ROOT / "stage3_v6_dual_2q"
for folder in (V5, V6):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
from diagnose_stage3_actor_checkpoint import parameter_drift
from stage3_v5_actor import (
    distribution_tensors,
    environment_means,
    flat_to_obs,
    load_exact_actor,
)
from stage3_v5_history_critic import component_mean_q, encode_replay_contexts
from stage3_v6_agent import strict_stage2_load, target_final_distribution_vectorized

CHECKPOINTS = {
    "actor_init": "step0_transfer.pth",
    "step100k": "step_0100000.pth",
    "critic_ready": "critic_ready.pth",
    "step200k": "step_0200000.pth",
    "step280k": "best_success.pth",
    "step300k": "step_0300000.pth",
    "last": "last.pth",
}
A2_GROUPS = {0: "bc_rnn_success", 1: "online_success", 2: "online_failure"}


def resolve_device(name):
    if name.startswith("npu"):
        import torch_npu  # noqa: F401
        if not torch.npu.is_available():
            raise RuntimeError("Requested NPU is not available")
        torch.npu.set_device(name)
    return torch.device(name)


def to_np(value):
    return value.detach().cpu().numpy()


def scalar_stats(values):
    x = np.asarray(values, np.float64).reshape(-1)
    if not x.size:
        return None
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def correlations(left, right):
    x = np.asarray(left, np.float64).reshape(-1)
    y = np.asarray(right, np.float64).reshape(-1)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {"pearson": None, "spearman": None}
    from scipy.stats import spearmanr
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(spearmanr(x, y).statistic),
    }


def a2_forward(actor, observations, device, scale, offset, batch_size=48):
    """Run the same per-step recurrent API and low-noise sampling as rollout."""
    actor.eval()
    torch.manual_seed(517341)
    if device.type == "npu":
        torch.npu.manual_seed_all(517341)
    fields = (
        "probs", "argmax_mode", "entropy", "effective_modes",
        "means_normalized", "means_env", "learned_std",
        "argmax_action_env", "expected_action_env",
        "sample_action_env", "hidden",
    )
    collected = {name: [] for name in fields}
    with torch.inference_mode():
        for lo in range(0, len(observations), batch_size):
            obs = torch.as_tensor(
                observations[lo:lo + batch_size], dtype=torch.float32, device=device
            )
            state = None
            rows = {name: [] for name in fields}
            for timestep in range(obs.shape[1]):
                distribution, state = actor.forward_train_step(
                    flat_to_obs(obs[:, timestep]), rnn_state=state
                )
                tensors = distribution_tensors(distribution)
                probs = tensors["probs"]
                means_norm = tensors["means_normalized"]
                means_env = environment_means(distribution, scale, offset)
                mode = torch.argmax(probs, dim=-1)
                gather = mode[:, None, None].expand(-1, 1, 14)
                argmax_action = means_env.gather(1, gather).squeeze(1)
                expected_action = (probs.unsqueeze(-1) * means_env).sum(1)
                sample_action = (
                    distribution.sample() * scale.reshape(1, 14)
                    + offset.reshape(1, 14)
                )
                entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
                hidden = state[0][-1] if isinstance(state, tuple) else state[-1]
                values = {
                    "probs": probs,
                    "argmax_mode": mode,
                    "entropy": entropy,
                    "effective_modes": entropy.exp(),
                    "means_normalized": means_norm,
                    "means_env": means_env,
                    "learned_std": tensors["scales"],
                    "argmax_action_env": argmax_action,
                    "expected_action_env": expected_action,
                    "sample_action_env": sample_action,
                    "hidden": hidden,
                }
                for key, value in values.items():
                    rows[key].append(to_np(value))
            for key in fields:
                collected[key].append(np.stack(rows[key], axis=1))
    return {key: np.concatenate(parts, axis=0) for key, parts in collected.items()}


def a2_diagnostics(reference, current, groups, scale):
    s = np.asarray(scale, np.float64).reshape(14)
    action_delta = current["argmax_action_env"] - reference["argmax_action_env"]
    sampled_delta = current["sample_action_env"] - reference["sample_action_env"]
    mean_delta = current["means_normalized"] - reference["means_normalized"]
    arrays = {
        "argmax_action_l2_env": np.linalg.norm(action_delta, axis=-1),
        "argmax_action_l2_normalized": np.linalg.norm(action_delta / s, axis=-1),
        "sample_action_l2_normalized": np.linalg.norm(sampled_delta / s, axis=-1),
        "component_mean_l2_normalized": np.linalg.norm(
            mean_delta.reshape(*mean_delta.shape[:2], -1), axis=-1
        ),
        "probability_l1": np.abs(
            current["probs"] - reference["probs"]
        ).sum(-1),
        "hidden_l2": np.linalg.norm(
            current["hidden"] - reference["hidden"], axis=-1
        ),
        "mode_changed": current["argmax_mode"] != reference["argmax_mode"],
    }
    summary = {}
    for group_id, label in A2_GROUPS.items():
        selected = groups == group_id
        per_step = []
        for timestep in range(10):
            per_step.append({
                "timestep": timestep,
                "argmax_action_l2_normalized": scalar_stats(
                    arrays["argmax_action_l2_normalized"][selected, timestep]
                ),
                "sample_action_l2_normalized": scalar_stats(
                    arrays["sample_action_l2_normalized"][selected, timestep]
                ),
                "component_mean_l2_normalized": scalar_stats(
                    arrays["component_mean_l2_normalized"][selected, timestep]
                ),
                "hidden_l2": scalar_stats(arrays["hidden_l2"][selected, timestep]),
                "mode_change_fraction": float(
                    np.mean(arrays["mode_changed"][selected, timestep])
                ),
                "current_entropy_mean": float(
                    np.mean(current["entropy"][selected, timestep])
                ),
            })
        summary[label] = {
            "count": int(selected.sum()),
            "all_steps": {
                "argmax_action_l2_normalized": scalar_stats(
                    arrays["argmax_action_l2_normalized"][selected]
                ),
                "sample_action_l2_normalized": scalar_stats(
                    arrays["sample_action_l2_normalized"][selected]
                ),
                "component_mean_l2_normalized": scalar_stats(
                    arrays["component_mean_l2_normalized"][selected]
                ),
                "hidden_l2": scalar_stats(arrays["hidden_l2"][selected]),
                "mode_change_fraction": float(
                    np.mean(arrays["mode_changed"][selected])
                ),
            },
            "per_step": per_step,
        }
    return arrays, summary


def final_distribution(actor, observations):
    return target_final_distribution_vectorized(
        actor, observations, horizon=10, starts=[0] * len(observations)
    )[0]


def a3_forward(init_actor, actor, critic, contexts, device, scale, offset, horizon):
    out = {key: [] for key in (
        "q_obj_init", "q_obj_current",
        "q1_expected_init", "q2_expected_init",
        "q1_expected_current", "q2_expected_current",
        "q1_init_action", "q2_init_action",
        "q1_current_action", "q2_current_action",
        "q1_replay_action", "q2_replay_action",
        "init_action_env", "current_action_env", "replay_action_env",
        "init_mode", "current_mode",
        "init_probs", "current_probs",
        "init_means_env", "current_means_env",
    )}
    with torch.inference_mode():
        for lo in range(0, len(contexts["a3_observations"]), 32):
            hi = lo + 32
            obs = torch.as_tensor(
                contexts["a3_observations"][lo:hi], dtype=torch.float32, device=device
            )
            actions = torch.as_tensor(
                contexts["a3_actions"][lo:hi], dtype=torch.float32, device=device
            )
            steps = torch.as_tensor(
                contexts["a3_episode_steps"][lo:hi], dtype=torch.long, device=device
            )
            replay_action = actions[:, -1]
            encoded = encode_replay_contexts(
                critic, obs, actions, steps, horizon
            )
            final_contexts = (encoded[0][:, -1], encoded[1][:, -1])
            din = final_distribution(init_actor, obs)
            dcur = final_distribution(actor, obs)
            pairs = {}
            for label, dist in (("init", din), ("current", dcur)):
                qobj, q1_modes, _, tensors, means_env = component_mean_q(
                    critic, final_contexts, dist, scale, offset, twin_min=False
                )
                q2_modes = critic.q2.q_from_context(
                    final_contexts[1], means_env
                ).squeeze(-1)
                probs = tensors["probs"]
                if not torch.allclose(qobj, (probs * q1_modes).sum(-1),
                                      atol=1e-5, rtol=1e-5):
                    raise RuntimeError("Production Q1 objective contract mismatch")
                mode = torch.argmax(probs, dim=-1)
                mode_index = mode[:, None]
                action = means_env.gather(
                    1, mode[:, None, None].expand(-1, 1, 14)
                ).squeeze(1)
                pairs[f"q_obj_{label}"] = qobj
                pairs[f"q1_expected_{label}"] = (probs * q1_modes).sum(-1)
                pairs[f"q2_expected_{label}"] = (probs * q2_modes).sum(-1)
                pairs[f"q1_{label}_action"] = q1_modes.gather(
                    1, mode_index
                ).squeeze(1)
                pairs[f"q2_{label}_action"] = q2_modes.gather(
                    1, mode_index
                ).squeeze(1)
                pairs[f"{label}_action_env"] = action
                pairs[f"{label}_mode"] = mode
                pairs[f"{label}_probs"] = probs
                pairs[f"{label}_means_env"] = means_env
            q1_replay, q2_replay = critic.q_from_context(
                final_contexts, replay_action
            )
            pairs["q1_replay_action"] = q1_replay.squeeze(-1)
            pairs["q2_replay_action"] = q2_replay.squeeze(-1)
            pairs["replay_action_env"] = replay_action
            for key in out:
                out[key].append(to_np(pairs[key]))
    return {key: np.concatenate(parts, axis=0) for key, parts in out.items()}


def summarize_a3(raw, success, scale):
    s = np.asarray(scale, np.float64).reshape(14)
    init_action = raw["init_action_env"]
    current_action = raw["current_action_env"]
    replay = raw["replay_action_env"]
    delta_q = raw["q_obj_current"] - raw["q_obj_init"]
    action_delta = current_action - init_action
    drift_env = np.linalg.norm(action_delta, axis=-1)
    drift_normalized = np.linalg.norm(action_delta / s, axis=-1)
    replay_init_env = np.linalg.norm(init_action - replay, axis=-1)
    replay_current_env = np.linalg.norm(current_action - replay, axis=-1)
    replay_init_normalized = np.linalg.norm((init_action - replay) / s, axis=-1)
    replay_current_normalized = np.linalg.norm((current_action - replay) / s, axis=-1)
    replay_distance_delta = replay_current_normalized - replay_init_normalized
    twin_init = np.abs(
        raw["q1_init_action"] - raw["q2_init_action"]
    )
    twin_current = np.abs(
        raw["q1_current_action"] - raw["q2_current_action"]
    )
    baseline_iqr = float(
        np.percentile(raw["q_obj_init"], 75)
        - np.percentile(raw["q_obj_init"], 25)
    )
    meaningful_threshold = max(0.1 * baseline_iqr, 1e-6)
    arrays = {
        "delta_q_obj": delta_q,
        "action_drift_l2_env": drift_env,
        "action_drift_l2_normalized": drift_normalized,
        "replay_distance_init_env": replay_init_env,
        "replay_distance_current_env": replay_current_env,
        "replay_distance_init_normalized": replay_init_normalized,
        "replay_distance_current_normalized": replay_current_normalized,
        "replay_distance_delta_normalized": replay_distance_delta,
        "twin_disagreement_init_action": twin_init,
        "twin_disagreement_current_action": twin_current,
        "high_q_more_ood": (delta_q > 0) & (replay_distance_delta > 0),
        "mode_changed": raw["init_mode"] != raw["current_mode"],
    }

    def group_result(mask):
        return {
            "count": int(np.sum(mask)),
            "q_obj_init": scalar_stats(raw["q_obj_init"][mask]),
            "q_obj_current": scalar_stats(raw["q_obj_current"][mask]),
            "delta_q_obj": scalar_stats(delta_q[mask]),
            "fraction_delta_q_positive": float(np.mean(delta_q[mask] > 0)),
            "fraction_delta_q_meaningful": float(
                np.mean(delta_q[mask] > meaningful_threshold)
            ),
            "action_drift_l2_env": scalar_stats(drift_env[mask]),
            "action_drift_l2_normalized": scalar_stats(drift_normalized[mask]),
            "replay_distance_init_normalized": scalar_stats(
                replay_init_normalized[mask]
            ),
            "replay_distance_current_normalized": scalar_stats(
                replay_current_normalized[mask]
            ),
            "replay_distance_delta_normalized": scalar_stats(
                replay_distance_delta[mask]
            ),
            "high_q_more_ood_fraction": float(
                np.mean(arrays["high_q_more_ood"][mask])
            ),
            "mode_change_fraction": float(np.mean(arrays["mode_changed"][mask])),
            "twin_disagreement_init_action": scalar_stats(twin_init[mask]),
            "twin_disagreement_current_action": scalar_stats(twin_current[mask]),
            "q1_init_action": scalar_stats(raw["q1_init_action"][mask]),
            "q2_init_action": scalar_stats(raw["q2_init_action"][mask]),
            "q1_current_action": scalar_stats(raw["q1_current_action"][mask]),
            "q2_current_action": scalar_stats(raw["q2_current_action"][mask]),
            "both_twins_current_above_init_fraction": float(np.mean(
                (raw["q1_current_action"][mask] > raw["q1_init_action"][mask])
                & (raw["q2_current_action"][mask] > raw["q2_init_action"][mask])
            )),
            "delta_q_vs_action_drift_correlation": correlations(
                delta_q[mask], drift_normalized[mask]
            ),
            "delta_q_vs_current_replay_distance_correlation": correlations(
                delta_q[mask], replay_current_normalized[mask]
            ),
            "delta_q_vs_replay_distance_change_correlation": correlations(
                delta_q[mask], replay_distance_delta[mask]
            ),
            "per_dimension_mean_absolute_action_drift_env": np.mean(
                np.abs(action_delta[mask]), axis=0
            ).tolist(),
            "per_dimension_mean_absolute_action_drift_normalized": np.mean(
                np.abs(action_delta[mask] / s), axis=0
            ).tolist(),
        }

    summary = {
        "q_objective": "Production Actor objective: GMM-probability-weighted Q1 at component means",
        "single_action": "argmax-GMM-component environment-space mean (deterministic low-noise proxy)",
        "meaningful_q_gain_threshold": meaningful_threshold,
        "threshold_basis": "10% of same-checkpoint Q_obj(init) interquartile range; 1e-6 numeric floor",
        "all": group_result(np.ones(len(delta_q), dtype=bool)),
        "success_contexts": group_result(success),
        "failure_contexts": group_result(~success),
    }
    return arrays, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, choices=tuple(CHECKPOINTS))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.checkpoints:
        if (output_dir / f"offline_{name}.json").exists() or (
            output_dir / f"offline_{name}.npz"
        ).exists():
            raise FileExistsError(f"Output for {name} already exists")
    device = resolve_device(args.device)
    contexts_npz = np.load(args.contexts.resolve())
    contexts = {key: contexts_npz[key] for key in contexts_npz.files}
    if len(contexts["a2_observations"]) != 192 or len(contexts["a3_observations"]) != 256:
        raise RuntimeError("Fixed context count changed")
    if not np.all(contexts["a2_episode_steps"][:, 0] % 10 == 0):
        raise RuntimeError("A2 Actor reset boundary changed")
    if not np.all(contexts["a3_episode_steps"][:, 0] % 10 == 0):
        raise RuntimeError("A3 Actor reset boundary changed")
    config = json.loads((run / "shared/config_resolved.json").read_text())
    source = json.loads((run / "shared/stage2_source_manifest.json").read_text())
    source_critic = source["multi_q"]["checkpoint"]
    actor, rollout, actor_metadata = load_exact_actor(
        run / "shared/bc_rnn_gmm_source.pth", device
    )
    init_payload = torch.load(
        run / "shared/actor_init.pth", map_location="cpu", weights_only=False
    )
    init_state = init_payload["actor_state_dict"]
    actor.load_state_dict(init_state, strict=True)
    init_actor = actor.eval()
    current_actor = copy.deepcopy(init_actor).to(device).eval()
    scale = torch.as_tensor(
        rollout.action_normalization_stats["actions"]["scale"],
        dtype=torch.float32, device=device
    ).reshape(1, 1, 1, 14)
    offset = torch.as_tensor(
        rollout.action_normalization_stats["actions"]["offset"],
        dtype=torch.float32, device=device
    ).reshape(1, 1, 1, 14)
    scale_np = to_np(scale).reshape(14)
    critic, _ = strict_stage2_load(source_critic, device)
    critic.eval()
    init_a2 = a2_forward(
        init_actor, contexts["a2_observations"], device, scale, offset
    )

    for name in args.checkpoints:
        checkpoint = run / "mean2q/multi_q/checkpoints" / CHECKPOINTS[name]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        current_state = init_state if name == "actor_init" else payload["actor"]
        current_actor.load_state_dict(current_state, strict=True)
        current_actor.eval()
        critic.load_state_dict(payload["q1_q2"], strict=True)
        critic.eval()
        stats = payload["action_normalization_stats"]
        if not np.allclose(stats["scale"], scale_np, rtol=0, atol=1e-7):
            raise RuntimeError("Action scale differs from BC reference")
        if not np.allclose(stats["offset"], to_np(offset).reshape(14),
                           rtol=0, atol=1e-7):
            raise RuntimeError("Action offset differs from BC reference")

        with torch.inference_mode():
            sample = torch.as_tensor(
                contexts["a3_observations"][:16], dtype=torch.float32, device=device
            )
            current_actor.train()
            train_distribution = final_distribution(current_actor, sample)
            current_actor.eval()
            eval_distribution = final_distribution(current_actor, sample)
            train_tensors = distribution_tensors(train_distribution)
            eval_tensors = distribution_tensors(eval_distribution)
            max_mean_gap = float(torch.max(torch.abs(
                train_tensors["means_normalized"]
                - eval_tensors["means_normalized"]
            )).item())
            max_prob_gap = float(torch.max(torch.abs(
                train_tensors["probs"] - eval_tensors["probs"]
            )).item())
            if max(max_mean_gap, max_prob_gap) > 1e-5:
                raise RuntimeError(
                    "Train/eval Actor means or probabilities differ; objective "
                    "and execution require separate representations"
                )
        current_a2 = a2_forward(
            current_actor, contexts["a2_observations"], device, scale, offset
        )
        a2_arrays, a2_summary = a2_diagnostics(
            init_a2, current_a2, contexts["a2_group"], scale_np
        )
        a3_raw = a3_forward(
            init_actor, current_actor, critic, contexts,
            device, scale, offset, int(config["horizon"])
        )
        a3_arrays, a3_summary = summarize_a3(
            a3_raw, contexts["a3_success"], scale_np
        )
        drift = parameter_drift(init_state, current_state)
        drift["groups"].setdefault("encoder", {
            "l2": None, "relative_l2": None, "max_abs": None, "numel": 0
        })
        def optimizer_lr(key):
            state = payload.get(key, {})
            return [float(group["lr"]) for group in state.get("param_groups", [])]
        summary = {
            "checkpoint_name": name,
            "checkpoint_path": str(checkpoint),
            "device": str(device),
            "env_steps": int(payload["env_steps"]),
            "critic_updates": int(payload["updates"]),
            "actor_updates": int(payload["actor_updates"]),
            "actor_lr": optimizer_lr("actor_optimizer"),
            "critic_lr": optimizer_lr("critic_optimizer"),
            "gate_open": bool(payload["actor_gate_open"]),
            "gate_open_step": payload.get("gate_open_step"),
            "actor_metadata": actor_metadata,
            "actor_train_eval_mean_gap_max": max_mean_gap,
            "actor_train_eval_prob_gap_max": max_prob_gap,
            "parameter_drift": drift,
            "a2_behavior": a2_summary,
            "a3_q_action": a3_summary,
        }
        raw_path = output_dir / f"offline_{name}.npz"
        json_path = output_dir / f"offline_{name}.json"
        raw = {
            **{f"a2_current_{key}": value for key, value in current_a2.items()},
            **{f"a2_init_{key}": value for key, value in init_a2.items()},
            **{f"a2_delta_{key}": value for key, value in a2_arrays.items()},
            **{f"a3_{key}": value for key, value in a3_raw.items()},
            **{f"a3_delta_{key}": value for key, value in a3_arrays.items()},
            "a3_success": contexts["a3_success"],
            "a2_group": contexts["a2_group"],
        }
        np.savez_compressed(raw_path, **raw)
        json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print(json.dumps({
            "checkpoint": name,
            "env_steps": summary["env_steps"],
            "actor_updates": summary["actor_updates"],
            "global_drift_l2": drift["l2"],
            "q_gain_mean": a3_summary["all"]["delta_q_obj"]["mean"],
            "fraction_q_gain_positive": a3_summary["all"]["fraction_delta_q_positive"],
            "high_q_more_ood_fraction": a3_summary["all"]["high_q_more_ood_fraction"],
            "output": str(json_path),
        }), flush=True)


if __name__ == "__main__":
    main()
