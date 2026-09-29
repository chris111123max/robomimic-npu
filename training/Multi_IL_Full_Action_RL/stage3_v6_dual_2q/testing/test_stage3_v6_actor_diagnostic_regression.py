"""Regression guard for the first Actor update after the V6 gate opens.

The V6 agent inherits the V5 Actor update.  The recurrent Critic branch must
evaluate the learned-std diagnostic on the same final horizon context and final
GMM distribution as the Actor RL objective.  A stale `distribution` name here
caused the formal run to crash immediately after the gate opened.
"""
from __future__ import annotations

import ast
from pathlib import Path


V5_AGENT = (
    Path(__file__).resolve().parents[2]
    / "stage3_v5_rgmm_td3"
    / "stage3_v5_agent.py"
)


def _actor_update_node(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "actor_update":
            return node
    raise AssertionError("actor_update not found")


def test_recurrent_actor_learned_std_uses_final_context_and_distribution():
    source = V5_AGENT.read_text()
    compile(source, str(V5_AGENT), "exec")
    actor_update = _actor_update_node(source)

    sampled_calls = [
        node
        for node in ast.walk(actor_update)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sampled_q"
    ]
    assert len(sampled_calls) == 1

    call = sampled_calls[0]
    assert len(call.args) >= 3
    assert isinstance(call.args[1], ast.Name)
    assert isinstance(call.args[2], ast.Name)
    assert call.args[1].id == "final_contexts"
    assert call.args[2].id == "final_distribution"


if __name__ == "__main__":
    test_recurrent_actor_learned_std_uses_final_context_and_distribution()
    print("PASS: recurrent Actor diagnostic uses final_contexts/final_distribution")
