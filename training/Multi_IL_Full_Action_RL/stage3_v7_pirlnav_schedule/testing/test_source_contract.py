#!/usr/bin/env python3
"""Read-only AST/source guard. No Torch, no NPU, no simulator, no training."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
trainer = ROOT / "train_stage3_v7_vector.py"
schedule = ROOT / "stage3_v7_schedule.py"
launcher = ROOT / "launch_stage3_v7.py"

for path in (trainer, schedule, launcher, ROOT / "testing" / "test_schedule.py"):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    ast.parse(path.read_text(encoding="utf-8"))

text = trainer.read_text(encoding="utf-8")
scheduler = schedule.read_text(encoding="utf-8")
launch = launcher.read_text(encoding="utf-8")
checks = {
    "one_npu": 'choices=("npu:0",)' in text,
    "one_group": 'choices=("multi_q",)' in text,
    "only_random2q": 'choices=("random2q",)' in text,
    "v6_source_adapter": 'stage3_v6_dual_2q' in text,
    "v6_agent_reuse": 'from stage3_v6_agent import RecurrentGMMTD3' in text,
    "v5_replay_reuse": 'from stage3_v5_replay import' in text,
    "v7_handoff": 'CriticHandoffV7(config' in text,
    "restore_v6_100k": 'HandoffStateV7.from_v6_100k' in text,
    "restore_target_rng": 'agent.load_target_selector_state_dict' in text,
    "save_target_rng": '"target_selector_state": agent.target_selector_state_dict()' in text,
    "save_online_replay": "online.save(replay_path)" in text,
    "v7_output_isolated": 'stage3_v7_pirlnav_schedule' in text,
    "no_fourcard_map": 'manual_mapping = {' not in text,
    "no_300k_stop": "def warning_if_timed_out" in scheduler
                       and "return False" in scheduler,
    "frozen_critic_decay": "TrainingStateV7.CRITIC_DECAY" in scheduler
                           and "actor_enabled=False" in scheduler,
    "200k_decay": "critic_decay_env_steps=200000" in text,
    "100k_actor_warmup": "actor_warmup_env_steps=100000" in text,
    "launch_only_npu0": '"npu:0"' in launch,
}
for key, passed in checks.items():
    print(("PASS" if passed else "FAIL") + ": " + key)
assert all(checks.values()), [k for k,v in checks.items() if not v]
print("SOURCE_CONTRACT_PASS", len(checks), "checks")
