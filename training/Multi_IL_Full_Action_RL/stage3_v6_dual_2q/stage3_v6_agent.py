"""Stage3-v6 agent: V5 training with mean-2Q or random-one-2Q targets.

Only the Bellman target estimator changes. Actor objective, replay, handoff,
optimizers, Polyak updates and all other Stage3-v5 mechanics are inherited.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
V5 = HERE.parent / "stage3_v5_rgmm_td3"
if str(V5) not in sys.path:
    sys.path.insert(0, str(V5))

from stage3_v5_agent import (  # noqa: E402,F401
    RecurrentGMMTD3 as _V5RecurrentGMMTD3,
    _last_reset_starts_from_numpy,
    strict_stage2_load,
    target_final_distribution_vectorized,
)
from stage3_v5_history_critic import (  # noqa: E402
    component_mean_q,
    encode_replay_contexts,
)
from stage3_v6_target import TARGET_MODES, TargetSelector2Q  # noqa: E402


class RecurrentGMMTD3V6(_V5RecurrentGMMTD3):
    """V5 agent with exactly one controlled change: Critic target estimator."""

    def __init__(self, actor, critic, config, device, action_scale, action_offset):
        super().__init__(actor, critic, config, device, action_scale, action_offset)
        mode = config.get("critic_target_mode")
        if mode not in TARGET_MODES:
            raise RuntimeError(
                f"Stage3-v6 requires critic_target_mode in {TARGET_MODES}, got {mode!r}"
            )
        selector_cfg = config.get("random_one_selector", {})
        seed = int(config["training_seed"]) + int(selector_cfg["seed_offset"])
        self.target_selector = TargetSelector2Q(mode, seed)
        self._active_target_q = None
        self._last_update_selected_q = None

    @property
    def critic_target_mode(self):
        return self.target_selector.mode

    def target_selector_state_dict(self):
        if self._active_target_q is not None:
            raise RuntimeError("cannot checkpoint Stage3-v6 inside a Critic update")
        return self.target_selector.state_dict()

    def load_target_selector_state_dict(self, state):
        if self._active_target_q is not None:
            raise RuntimeError("cannot restore selector inside a Critic update")
        self.target_selector.load_state_dict(state)

    def target_selector_audit(self):
        result = self.target_selector.audit()
        result["last_update_selected_q"] = (
            None if self._last_update_selected_q is None
            else int(self._last_update_selected_q) + 1
        )
        return result

    def _combine_target_members(self, q1_expected, q2_expected):
        if self.critic_target_mode == "mean2q":
            return self.target_selector.combine(q1_expected, q2_expected, None)
        if self._active_target_q is None:
            # Readiness/fixed diagnostics must not consume selector randomness.
            return self.target_selector.diagnostic_expectation(
                q1_expected, q2_expected
            )
        return self.target_selector.combine(
            q1_expected, q2_expected, self._active_target_q
        )

    @torch.no_grad()
    def bellman_target(self, b, target_sequence):
        """V6 Bellman target with no hard clipped-min target."""
        horizon = int(self.config["actor_source_contract"]["rnn_horizon"])
        target_starts = _last_reset_starts_from_numpy(
            target_sequence["episode_steps"], horizon
        )
        target_keys = ["next_observations"]
        if hasattr(self.target_critic, "encode_history"):
            target_keys += ["observations", "actions", "episode_steps"]
        with self.profiler.measure("host_to_device_ms", device=True):
            target = self._tensor_batch(
                {key: target_sequence[key] for key in target_keys}
            )

        distribution, _ = target_final_distribution_vectorized(
            self.target_actor,
            target["next_observations"],
            horizon=horizon,
            starts=target_starts,
        )

        if hasattr(self.target_critic, "encode_history"):
            target_contexts = encode_replay_contexts(
                self.target_critic,
                target["observations"],
                target["actions"],
                target["episode_steps"],
                self.config["horizon"],
                next_observations=target["next_observations"],
            )
            _, q1_modes, q2_modes, tensors, _ = component_mean_q(
                self.target_critic,
                (target_contexts[0][:, -1], target_contexts[1][:, -1]),
                distribution,
                self.action_scale,
                self.action_offset,
            )
        else:
            target_contexts = None
            _, q1_modes, q2_modes, tensors, _ = self._expected_q(
                self.target_critic,
                target["next_observations"][:, -1],
                distribution,
            )

        probabilities = tensors["probs"]
        q1_expected = (probabilities * q1_modes).sum(-1)
        q2_expected = (probabilities * q2_modes).sum(-1)
        expected_next = self._combine_target_members(q1_expected, q2_expected)

        td_target = b["rewards"] + float(self.config["gamma"]) * (
            1.0 - b["terminals"]
        ) * expected_next.reshape(-1, 1)
        return (
            td_target,
            distribution,
            expected_next,
            target,
            target_starts,
            target_contexts,
        )

    def critic_update(self, batch, target_sequence, collect_metrics=True):
        """Draw random-one exactly once here, never inside diagnostics."""
        selected = self.target_selector.begin_update()
        self._active_target_q = selected
        self._last_update_selected_q = selected
        try:
            metrics = super().critic_update(
                batch, target_sequence, collect_metrics=collect_metrics
            )
        finally:
            self._active_target_q = None

        if collect_metrics:
            audit = self.target_selector.audit()
            metrics.update({
                "critic_target_mode_mean2q": int(self.critic_target_mode == "mean2q"),
                "critic_target_mode_random2q": int(self.critic_target_mode == "random2q"),
                "random_one_selected_q": (
                    0 if selected is None else int(selected) + 1
                ),
                "random_one_q1_updates": int(audit["q1_updates"]),
                "random_one_q2_updates": int(audit["q2_updates"]),
            })
        return metrics


# Trainer imports this name to keep the rest of the V5 loop unchanged.
RecurrentGMMTD3 = RecurrentGMMTD3V6
