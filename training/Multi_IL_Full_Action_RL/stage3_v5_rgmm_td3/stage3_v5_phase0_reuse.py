"""Validate a completed compatible Stage3-v3/v4 Phase-0 gate."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


PHASE0_FILES = (
    "transfer_validation.json",
    "step0_competence.json",
    "phase0_gate.json",
)

EVALUATION_CONTRACT_KEYS = (
    "obs_dim",
    "action_dim",
    "actor_source_contract",
    "actor_gate",
    "bc_rnn_checkpoint_sha256",
    "expert_dataset_sha256",
    "training_seed",
    "competence_evaluation_seeds",
    "horizon",
    "sim_error_handling",
)


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_phase0_reuse(reference_pair, target_config, target_actor_hash):
    """Return copy provenance, or reject an incompatible/partial old gate.

    Training-only settings, including policy_delay and total_env_steps, may
    differ. All inputs that determine the Phase-0 Actor and evaluation must
    match. The three completed gate artifacts are copied into the new pair.
    """
    reference = Path(reference_pair).resolve(strict=True)
    shared = reference / "shared"
    old_config = _read(shared / "config_resolved.json")
    mismatches = []
    for key in EVALUATION_CONTRACT_KEYS:
        old_value = old_config.get(key)
        if key == "competence_evaluation_seeds" and old_value is None:
            # Older v3/v4 pairs used evaluation_seeds for Phase-0 competence.
            old_value = old_config.get("evaluation_seeds")
        if old_value != target_config.get(key):
            mismatches.append(key)
    if mismatches:
        raise RuntimeError(f"Phase-0 reuse contract differs: {', '.join(mismatches)}")

    fairness = _read(shared / "pair_fairness.json")
    transfer = _read(shared / "transfer_validation.json")
    competence = _read(shared / "step0_competence.json")
    gate = _read(shared / "phase0_gate.json")
    expected_sha = target_config["bc_rnn_checkpoint_sha256"]
    required_successes = int(target_config["actor_gate"]["competence_min_successes"])
    expected_episodes = int(target_config["actor_gate"]["competence_episodes"])
    expected_seeds = list(target_config.get("competence_evaluation_seeds",
                                           target_config["evaluation_seeds"]))
    maximum_diff = float(transfer.get("max_abs_diff", float("inf")))

    if fairness.get("stage") not in ("stage3-v3", "stage3-v4") or not fairness.get("actor_hashes_identical"):
        raise RuntimeError("Phase-0 reuse source has invalid pair fairness")
    if fairness.get("actor_hash") != target_actor_hash:
        raise RuntimeError("Phase-0 reuse Actor hash differs from the new pair")
    if (transfer.get("checkpoint_sha256") != expected_sha
            or transfer.get("source_actor_hash") != target_actor_hash
            or transfer.get("new_actor_hash") != target_actor_hash
            or transfer.get("equivalence_pass") is not True
            or transfer.get("missing_keys") != []
            or transfer.get("unexpected_keys") != []
            or not math.isfinite(maximum_diff)
            or maximum_diff > float(target_config["actor_gate"]["equivalence_tolerance"])):
        raise RuntimeError("Phase-0 reuse source failed transfer equivalence")

    episodes = competence.get("episodes")
    if not isinstance(episodes, list):
        raise RuntimeError("Phase-0 reuse source lacks episode-level evidence")
    if ([row.get("seed") for row in episodes] != expected_seeds
            or competence.get("evaluation_seeds") != expected_seeds
            or competence.get("valid_episodes") != expected_episodes
            or competence.get("sim_error_episodes") != 0
            or len(episodes) != expected_episodes
            or any(row.get("sim_error") or not isinstance(row.get("success"), bool)
                   for row in episodes)
            or competence.get("success_count") != sum(row["success"] for row in episodes)
            or competence.get("success_count", 0) < required_successes
            or competence.get("minimum_successes") != required_successes
            or competence.get("competence_pass") is not True
            or competence.get("env_steps") != 0):
        raise RuntimeError("Phase-0 reuse source failed competence evidence checks")

    success_rate = competence["success_count"] / expected_episodes
    if (competence.get("success_rate") != success_rate
            or gate.get("env_steps") != 0
            or gate.get("eval_success_count") != competence["success_count"]
            or gate.get("eval_success_rate") != success_rate
            or gate.get("equivalence_pass") is not True
            or gate.get("competence_pass") is not True
            or gate.get("warmup_pass") is not False
            or gate.get("gate_open") is not False
            or gate.get("latched") is not True):
        raise RuntimeError("Phase-0 reuse source has an inconsistent gate")

    return {
        "source_pair_run_dir": str(reference),
        "source_stage": fairness["stage"],
        "actor_hash": target_actor_hash,
        "bc_rnn_checkpoint_sha256": expected_sha,
        "expert_dataset_sha256": target_config["expert_dataset_sha256"],
        "evaluation_seeds": expected_seeds,
        "success_count": competence["success_count"],
        "success_rate": success_rate,
        "artifact_sha256": {name: _sha256(shared / name) for name in PHASE0_FILES},
    }
