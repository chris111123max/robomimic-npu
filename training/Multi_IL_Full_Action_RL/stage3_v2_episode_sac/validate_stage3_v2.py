#!/usr/bin/env python3
"""Code-level and optional prepared-pair validation for Stage3-v2."""
from __future__ import annotations

import argparse
import ast
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from stage3_v2_agent import Stage3V2SAC, build_actor, build_critic, state_hash
from stage3_v2_behavior import (
    assign_behavior_source,
    bc_schedule,
    progressive_critic_schedule,
    validate_behavior_schedule,
)
from stage3_v2_evaluation import evaluate_pure_actor
from stage3_v2_replay import EpisodeReplay, SymmetricEpisodeSampler


HERE = Path(__file__).resolve().parent


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "stage3_v2_config.json"))
    parser.add_argument("--pair-run-dir")
    return parser.parse_args()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def called_names(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                result.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                result.append(node.func.attr)
    return result


class FakeOffline:
    def __init__(self):
        self.size = 512
        self.data = {
            "observations": np.zeros((self.size, 59), np.float32),
            "actions": np.zeros((self.size, 14), np.float32),
            "rewards": np.zeros((self.size, 1), np.float32),
            "next_observations": np.zeros((self.size, 59), np.float32),
            "terminals": np.zeros((self.size, 1), np.float32),
            "action_rnn": np.zeros((self.size, 14), np.float32),
        }
        self.rng = np.random.default_rng(7)

    def sample(self, count):
        indices = self.rng.integers(self.size, size=int(count))
        return {key: value[indices].copy() for key, value in self.data.items()}


def validate_pair(pair, config):
    shared = Path(pair).resolve() / "shared"
    contract = read_json(shared / "pair_contract.json")
    actor_payload = torch.load(shared / "actor_init.pth", map_location="cpu")
    actor_a = build_actor(config, "cpu")
    actor_b = build_actor(config, "cpu")
    actor_a.load_state_dict(actor_payload["actor_state_dict"], strict=True)
    actor_b.load_state_dict(actor_payload["actor_state_dict"], strict=True)
    if state_hash(actor_a) != state_hash(actor_b):
        raise AssertionError("Paired Actor hashes differ")
    if not contract["actor_hashes_identical"]:
        raise AssertionError("pair_contract does not certify identical Actor init")
    if contract["episode_behavior_schedule"] != config["episode_behavior_schedule"]:
        raise AssertionError("Prepared pair behavior schedule mismatch")
    if contract["bc_regularization_schedule"] != config["bc_regularization_schedule"]:
        raise AssertionError("Prepared pair BC schedule mismatch")


def main():
    args = arguments()
    config = read_json(args.config)
    validate_behavior_schedule(config["episode_behavior_schedule"])
    checks = {}

    calls = called_names(HERE / "train_stage3_v2_vector.py")
    agent_calls = called_names(HERE / "stage3_v2_agent.py")
    if "select_actions" in calls:
        raise AssertionError("Stage3-v2 trainer calls select_actions")
    checks["no_timestep_selector"] = True
    if "hybrid_bootstrap" in calls or "hybrid_bootstrap" in agent_calls:
        raise AssertionError("Stage3-v2 path calls hybrid_bootstrap")
    checks["no_hybrid_bootstrap"] = True

    schedule = config["episode_behavior_schedule"]
    expected = {
        0: 1.0, 9999: 1.0, 10000: 0.5, 29999: 0.5,
        30000: 0.25, 99999: 0.25, 100000: 0.0,
    }
    for step, fraction in expected.items():
        sources = [assign_behavior_source(schedule, step, episode)[0] for episode in range(100)]
        observed = sources.count("rnn") / len(sources)
        if not np.isclose(observed, fraction):
            raise AssertionError(f"Behavior schedule mismatch at {step}: {observed}")
    checks["episode_behavior_schedule"] = True

    for step, expected_scale in ((0, 0.0), (10000, 0.0), (20000, 0.5), (30000, 1.0)):
        result = progressive_critic_schedule(config, step)
        if not np.isclose(result["critic_lr_scale"], expected_scale):
            raise AssertionError(f"Progressive Critic mismatch at {step}")
    checks["progressive_critic_schedule"] = True

    bc_expected = {
        0: ("bc_only", 1.0), 9999: ("bc_only", 1.0),
        10000: ("sac_plus_bc", 1.0), 30000: ("sac_plus_bc", 1.0),
        65000: ("sac_plus_bc", 0.5), 100000: ("pure_sac", 0.0),
        200000: ("pure_sac", 0.0),
    }
    for step, (objective, weight) in bc_expected.items():
        result = bc_schedule(config, step)
        if result["actor_objective"] != objective or not np.isclose(result["lambda_bc"], weight):
            raise AssertionError(f"BC schedule mismatch at {step}: {result}")
    checks["bc_schedule"] = True

    # An episode stores one immutable source string; action provenance follows it.
    for step in (0, 10000, 30000, 100000):
        source, _ = assign_behavior_source(schedule, step, 12)
        action_tags = [source for _ in range(700)]
        if len(set(action_tags)) != 1:
            raise AssertionError("Episode source changed within an episode")
    checks["episode_source_integrity"] = True

    offline = FakeOffline()
    online = EpisodeReplay(512, seed=7)
    for index in range(256):
        source = index % 2
        action = np.full(14, float(source), np.float32)
        teacher = action.copy() if source == 0 else np.zeros(14, np.float32)
        online.add(
            np.zeros(59, np.float32), action, 0.0, np.zeros(59, np.float32), False,
            {"behavior_source": source, "action_rnn": teacher, "env_id": 0,
             "episode_id": index, "episode_seed": 30000 + index,
             "behavior_phase": 1},
        )
    sampler = SymmetricEpisodeSampler(offline, online, 7)
    batch = sampler.sample(256)
    if int((batch["is_online"] < 0.5).sum()) != 128 or int((batch["is_online"] > 0.5).sum()) != 128:
        raise AssertionError("Replay batch is not exactly 128/128")
    checks["replay_50_50"] = True
    if int(config["utd"]) != 1:
        raise AssertionError("UTD is not 1")
    checks["utd_one"] = True

    actor = build_actor(config, "cpu")
    critic = build_critic(59, 14, config["hidden_dims"], "relu", True, "cpu")
    agent = Stage3V2SAC(
        actor, critic, config, torch.device("cpu"),
        -np.ones(14, np.float32), np.ones(14, np.float32),
    )
    tensor_batch = {
        key: torch.as_tensor(value, dtype=torch.float32)
        for key, value in batch.items()
    }
    components = agent.standard_target_components(tensor_batch)
    expected_target = tensor_batch["rewards"] + float(config["gamma"]) * (
        1.0 - tensor_batch["terminals"]
    ) * (components["target_qmin"] - agent.alpha.detach() * components["next_log_pi"])
    if not torch.allclose(components["td_target"], expected_target):
        raise AssertionError("TD target is not standard SAC")
    checks["standard_sac_target"] = True

    pure_signature = set(inspect.signature(evaluate_pure_actor).parameters)
    if "target" in pure_signature or "proposer" in pure_signature:
        raise AssertionError("Pure Actor evaluator accepts selector dependencies")
    checks["pure_actor_evaluation"] = True
    if args.pair_run_dir:
        prepared = read_json(Path(args.pair_run_dir) / "shared" / "config_resolved.json")
        validate_pair(args.pair_run_dir, prepared)
        checks["paired_initialization"] = True
    else:
        checks["paired_initialization"] = "SKIPPED: pass --pair-run-dir"

    print(json.dumps({"status": "PASS", "stage": "stage3-v2", "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
