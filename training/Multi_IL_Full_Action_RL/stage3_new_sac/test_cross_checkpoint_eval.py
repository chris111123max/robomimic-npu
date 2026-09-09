#!/usr/bin/env python3
"""Read-only cross-checkpoint diagnosis for the Stage3 RNN-Q branch.

This is deliberately separate from every Stage3 trainer.  It loads an Actor
from one checkpoint and the *target* Twin Critic from another checkpoint,
then runs the existing deterministic handoff evaluator.  No optimizer,
replay buffer, SAC update, or parameter update is created or invoked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from stage3_new_agent import build_actor, build_critic, state_hash
from stage3_new_evaluation import build_env, close_env
from stage3_new_handoff import FrozenRNNProposer, evaluate_handoff


EXPECTED_STEPS = {50000, 300000}
EXPECTED_SEEDS = list(range(20000, 20010))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Stage3 RNN-Q Actor/target-Critic cross-checkpoint test"
    )
    parser.add_argument("--pair-run-dir", required=True)
    parser.add_argument("--branch", required=True, choices=("rnn_q",))
    parser.add_argument("--actor-step", required=True, type=int, choices=sorted(EXPECTED_STEPS))
    parser.add_argument("--target-critic-step", required=True, type=int, choices=sorted(EXPECTED_STEPS))
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the first evaluation seed and write to a separate TEST directory",
    )
    return parser.parse_args()


def checkpoint_step(payload, requested: int, path: Path) -> None:
    """Reject a mislabeled checkpoint when its metadata records env_steps."""
    if "env_steps" in payload and int(payload["env_steps"]) != int(requested):
        raise RuntimeError(
            f"Checkpoint metadata mismatch for {path}: "
            f"env_steps={payload['env_steps']}, requested={requested}"
        )


def load_cross_models(config, actor_path: Path, critic_path: Path, device: torch.device,
                      actor_step: int, critic_step: int):
    actor_payload = torch.load(actor_path, map_location=device)
    critic_payload = torch.load(critic_path, map_location=device)
    checkpoint_step(actor_payload, actor_step, actor_path)
    checkpoint_step(critic_payload, critic_step, critic_path)
    for name, payload in (("actor", actor_payload), ("target critic", critic_payload)):
        if not isinstance(payload, dict):
            raise RuntimeError(f"{name} checkpoint payload is not a dictionary")

    actor_state = actor_payload.get("actor_state_dict")
    target_state = critic_payload.get("target_critic_state_dict")
    if actor_state is None:
        raise RuntimeError(f"{actor_path} has no actor_state_dict")
    if target_state is None:
        raise RuntimeError(
            f"{critic_path} has no target_critic_state_dict; "
            "cross-checkpoint test must use the target Critic"
        )

    actor = build_actor(config, device)
    actor.load_state_dict(actor_state, strict=True)
    actor.eval()
    actor.requires_grad_(False)
    actor_hash = state_hash(actor)
    recorded_actor_hash = actor_payload.get("actor_hash")
    if recorded_actor_hash is not None and actor_hash != recorded_actor_hash:
        raise RuntimeError(f"Actor state hash mismatch in {actor_path}")

    model_config = actor_payload.get("model_config", critic_payload.get("model_config", {}))
    expected = {
        "obs_dim": 59,
        "action_dim": 14,
        "hidden_dims": list(config["hidden_dims"]),
        "activation": "relu",
        "layer_norm": True,
    }
    for key, value in expected.items():
        if key in model_config and model_config[key] != value:
            raise RuntimeError(
                f"Cross-test model contract mismatch for {key}: "
                f"checkpoint={model_config[key]!r}, expected={value!r}"
            )
    target = build_critic(
        obs_dim=59,
        action_dim=14,
        hidden_dims=config["hidden_dims"],
        activation="relu",
        layer_norm=True,
        device=device,
    )
    target.load_state_dict(target_state, strict=True)
    target.eval()
    target.requires_grad_(False)
    return actor, target, actor_hash, state_hash(target), actor_payload, critic_payload


def validate_seed_protocol(pair: Path, config) -> list[int]:
    manifest_path = pair / "shared" / "seed_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing formal seed manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    seeds = [int(value) for value in manifest.get("evaluation_seeds", [])]
    if seeds != EXPECTED_SEEDS:
        raise RuntimeError(
            "Cross-test requires the formal evaluation seed protocol "
            f"20000..20009; got {seeds!r}"
        )
    configured = config.get("evaluation_seeds")
    if configured is not None and [int(value) for value in configured] != seeds:
        raise RuntimeError("config_resolved.json and seed_manifest.json disagree on evaluation seeds")
    if int(config.get("evaluation_episodes", len(seeds))) != 10:
        raise RuntimeError("Cross-test requires exactly 10 formal evaluation episodes")
    if int(config.get("horizon", 700)) != 700:
        raise RuntimeError("Cross-test requires horizon=700")
    if not bool(config.get("terminate_on_success", True)):
        raise RuntimeError("Cross-test requires terminate_on_success=true")
    return seeds


def main() -> None:
    args = arguments()
    if args.actor_step == args.target_critic_step:
        raise ValueError("This diagnostic only accepts cross-checkpoint pairs with different steps")
    pair = Path(args.pair_run_dir).resolve()
    config = read_json(pair / "shared" / "config_resolved.json")
    if args.branch != "rnn_q":
        raise AssertionError("Only the RNN-Q branch is in scope for this test")

    actor_path = pair / args.branch / "checkpoints" / f"step_{args.actor_step:06d}.pth"
    target_path = pair / args.branch / "checkpoints" / f"step_{args.target_critic_step:06d}.pth"
    for path in (actor_path, target_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")

    bc_checkpoint = config.get("bc_rnn_checkpoint")
    if not bc_checkpoint:
        raise RuntimeError("config_resolved.json has no bc_rnn_checkpoint")
    bc_path = Path(bc_checkpoint).resolve()
    if not bc_path.is_file():
        raise FileNotFoundError(f"Missing frozen BC-RNN checkpoint: {bc_path}")
    dataset = config.get("expert_dataset")
    if not dataset:
        raise RuntimeError("config_resolved.json has no expert_dataset")
    dataset_path = Path(dataset).resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Missing evaluation dataset: {dataset_path}")

    seeds = validate_seed_protocol(pair, config)
    eval_seeds = seeds[:1] if args.smoke else seeds
    device = torch.device(args.device)
    label = f"actor{args.actor_step // 1000}k_targetq{args.target_critic_step // 1000}k_test"
    output_root = pair / ("test_cross_checkpoint_smoke" if args.smoke else "test_cross_checkpoint") / label
    result_path = output_root / "result.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic result: {result_path}")

    print(
        "[CROSS-TEST] branch=rnn_q "
        f"actor_source_step={args.actor_step} "
        f"target_critic_source_step={args.target_critic_step} "
        f"device={device} training_enabled=False",
        flush=True,
    )
    print(f"[CROSS-TEST] actor_checkpoint={actor_path}", flush=True)
    print(f"[CROSS-TEST] target_critic_checkpoint={target_path}", flush=True)
    print(f"[CROSS-TEST] bc_rnn_checkpoint={bc_path}", flush=True)
    print(f"[CROSS-TEST] evaluation_seeds={eval_seeds}", flush=True)

    actor = target = proposer = env = None
    try:
        actor, target, actor_hash, target_hash, actor_payload, target_payload = load_cross_models(
            config, actor_path, target_path, device, args.actor_step, args.target_critic_step
        )
        proposer = FrozenRNNProposer(str(bc_path), str(device))
        proposer.nets.eval()
        proposer.nets.requires_grad_(False)
        env = build_env(str(dataset_path))
        with torch.no_grad():
            metrics = evaluate_handoff(
                actor,
                target,
                proposer,
                env,
                eval_seeds,
                int(config.get("horizon", 700)),
                device,
                retries=int(config.get("sim_error_handling", {}).get("evaluation_retry_count", 1)),
            )
    finally:
        if env is not None:
            close_env(env)

    result = {
        "status": "TEST",
        "test_type": "cross_checkpoint_test",
        "branch": "rnn_q",
        "smoke": bool(args.smoke),
        "training_enabled": False,
        "updates_performed": 0,
        "replay_buffer_used": False,
        "optimizer_updates": 0,
        "actor_source_step": int(args.actor_step),
        "target_critic_source_step": int(args.target_critic_step),
        "actor_checkpoint": str(actor_path),
        "target_critic_checkpoint": str(target_path),
        "bc_rnn_checkpoint": str(bc_path),
        "expert_dataset": str(dataset_path),
        "device": str(device),
        "evaluation_seeds": eval_seeds,
        "formal_evaluation_seeds": seeds,
        "actor_state_hash": actor_hash,
        "target_critic_state_hash": target_hash,
        "actor_checkpoint_sha256": sha256(actor_path),
        "target_critic_checkpoint_sha256": sha256(target_path),
        "bc_rnn_checkpoint_sha256": sha256(bc_path),
        "actor_state_source_key": "actor_state_dict",
        "target_critic_state_source_key": "target_critic_state_dict",
        "deterministic_actor": True,
        "frozen_bc_rnn": True,
        "selector": "min(target_q1, target_q2); q_rl > q_rnn else rnn",
        "margin": 0.0,
        "tie_break": "rnn",
        "horizon": int(config.get("horizon", 700)),
        "terminate_on_success": True,
        **metrics,
    }
    write_json(result_path, result)
    print(json.dumps({"status": "TEST", "result": str(result_path), **{k: result[k] for k in ("success_rate", "rl_selected_fraction", "rnn_selected_fraction")}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
