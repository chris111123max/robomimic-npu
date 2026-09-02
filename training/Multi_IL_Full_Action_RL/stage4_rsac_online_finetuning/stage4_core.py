"""Shared Stage4 recurrent SAC runtime built on the vendored pomdp-baselines."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
VENDOR = PROJECT / "third_party" / "pomdp_baselines"
STAGE3R = PROJECT / "stage3_r_bc_rnn_to_rsac"
STAGE2R = PROJECT / "stage2_r_recurrent_critic_pretraining"
STAGE1 = PROJECT / "stage1_rollout_collection"
STAGE3 = PROJECT / "stage3_actor_initialization"
for path in (HERE, VENDOR, STAGE3R, STAGE2R, STAGE1, STAGE3):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from buffers.seq_replay_buffer_efficient import RAMEfficient_SeqReplayBuffer  # noqa: E402
import torchkit.pytorch_utils as ptu  # noqa: E402
from stage2_r_critic import SAC, architecture, make_pair  # noqa: E402
from stage3_r_actor import load_actor  # noqa: E402


GROUPS = ("random_critic", "rnn_only_critic", "multi_il_critic")
GROUP_DEVICES = {"random_critic": "npu:0", "rnn_only_critic": "npu:1", "multi_il_critic": "npu:2"}


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False); handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def append_csv(path, row):
    path = Path(path); exists = path.exists(); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists: writer.writeheader()
        writer.writerow(row); handle.flush(); os.fsync(handle.fileno())


def seed_all(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if hasattr(torch, "npu") and torch.npu.is_available(): torch.npu.manual_seed_all(int(seed))


def select_device(name):
    if name.startswith("npu"):
        import torch_npu  # noqa: F401
        if not torch.npu.is_available(): raise RuntimeError("Ascend NPU is unavailable")
        torch.npu.set_device(name)
    device = torch.device(name); ptu.set_device(device); return device


def state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(name.encode()); digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def safe_gradient_norm(parameters):
    """Compute a diagnostic norm without a large NPU float32 reduction."""
    norms=[]
    for parameter in parameters:
        if parameter.grad is None: continue
        gradient=parameter.grad.detach();maximum=float(gradient.abs().max().item())
        if maximum==0.0:norms.append(0.0);continue
        norms.append(maximum*math.sqrt(float((gradient/maximum).square().sum().item())))
    return math.hypot(*norms)


def phase_at(env_step, config):
    step = int(env_step); freeze = int(config["actor_freeze_steps"]); end = int(config["actor_warmup_end"])
    if step <= freeze:
        return {"phase": "critic_only", "actor_lr": 0.0, "critic_lr": float(config["critic_phase1_lr"]), "actor_updates": False}
    if step < end:
        fraction = (step - freeze) / float(end - freeze)
        critic_lr = float(config["critic_phase1_lr"]) + fraction * (float(config["critic_joint_lr"]) - float(config["critic_phase1_lr"]))
        return {"phase": "actor_lr_warmup", "actor_lr": fraction * float(config["actor_target_lr"]), "critic_lr": critic_lr, "actor_updates": True}
    return {"phase": "joint_rsac", "actor_lr": float(config["actor_target_lr"]), "critic_lr": float(config["critic_joint_lr"]), "actor_updates": True}


def assert_phase_contract(config):
    freeze = int(config["actor_freeze_steps"]); end = int(config["actor_warmup_end"])
    assert not phase_at(freeze - 1, config)["actor_updates"]
    assert phase_at(freeze, config)["actor_lr"] == 0.0
    middle = phase_at((freeze + end) // 2, config)
    assert middle["actor_updates"] and 0.0 < middle["actor_lr"] < float(config["actor_target_lr"])
    assert phase_at(end, config)["actor_lr"] == float(config["actor_target_lr"])
    assert phase_at(end + 1, config)["phase"] == "joint_rsac"


class Stage3RSACAdapter(nn.Module):
    """Give Stage3RActor the exact recurrent actor signature expected by upstream SAC."""
    def __init__(self, actor):
        super().__init__(); self.actor = actor

    def forward(self, prev_actions, rewards, observs):
        del prev_actions, rewards
        features = self.actor.forward_sequence(observs.transpose(0, 1), reset_interval=True)
        actions, _, _, log_probs = self.actor.actions_from_features(features, deterministic=False, return_log_prob=True)
        return actions.transpose(0, 1), log_probs.transpose(0, 1)


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value, dict): return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list): return [cpu_tree(item) for item in value]
    if isinstance(value, tuple): return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def rng_state():
    result = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if hasattr(torch, "npu") and torch.npu.is_available(): result["npu"] = torch.npu.get_rng_state()
    return result


def restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if "npu" in state and hasattr(torch, "npu"): torch.npu.set_rng_state(state["npu"])


class OnlineSequenceReplay:
    def __init__(self, config):
        self.buffer = RAMEfficient_SeqReplayBuffer(int(config["replay_capacity"]), 59, 14, int(config["sequence_length"]), float(config["sample_weight_baseline"]), np.float32)
        self.transitions = 0; self.episodes = 0; self.episode_terminal_metadata=[]

    def add_episode(self, obs, actions, rewards, dones, next_obs, terminated=None, truncated=None):
        self.buffer.add_episode(np.asarray(obs, np.float32), np.asarray(actions, np.float32), np.asarray(rewards, np.float32).reshape(-1, 1), np.asarray(dones, np.uint8).reshape(-1, 1), np.asarray(next_obs, np.float32))
        length=len(actions);self.transitions += length; self.episodes += 1
        term=np.zeros(length,np.uint8) if terminated is None else np.asarray(terminated,np.uint8);trunc=np.zeros(length,np.uint8) if truncated is None else np.asarray(truncated,np.uint8)
        if not np.array_equal(np.logical_or(term,trunc).astype(np.uint8),np.asarray(dones,np.uint8)):raise RuntimeError("terminated OR truncated differs from replay done mask")
        self.episode_terminal_metadata.append({"length":length,"terminated_indices":np.flatnonzero(term).tolist(),"truncated_indices":np.flatnonzero(trunc).tolist()})

    def sample(self, count, device):
        values = self.buffer.random_episodes(int(count))
        return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in values.items()}

    def save(self, path):
        b = self.buffer; payload = {"top": b._top, "size": b._size, "transitions": self.transitions, "episodes": self.episodes,"episode_terminal_metadata":self.episode_terminal_metadata}
        arrays = {name: getattr(b, name) for name in ("_observations", "_actions", "_rewards", "_terminals", "_ends", "_valid_starts")}
        np.savez_compressed(path, metadata=np.asarray(json.dumps(payload)), **arrays)

    def load(self, path):
        b = self.buffer
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"])); b._top = int(meta["top"]); b._size = int(meta["size"])
            self.transitions = int(meta["transitions"]); self.episodes = int(meta["episodes"]);self.episode_terminal_metadata=list(meta.get("episode_terminal_metadata",[]))
            for name in ("_observations", "_actions", "_rewards", "_terminals", "_ends", "_valid_starts"): getattr(b, name)[:] = data[name]


class Stage4SAC:
    def __init__(self, actor, critic, target, config, device):
        self.actor = actor; self.actor_adapter = Stage3RSACAdapter(actor); self.critic = critic; self.target = target; self.config = config; self.device = device
        self.algo = SAC(entropy_alpha=float(config["initial_entropy_alpha"]), automatic_entropy_tuning=bool(config["automatic_entropy_tuning"]), target_entropy=float(config["target_entropy"]), alpha_lr=float(config["alpha_lr"]), action_dim=14)
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(config["actor_target_lr"]))
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(config["critic_joint_lr"]))
        self.update_index = 0; target.requires_grad_(False); actor.requires_grad_(False)

    @staticmethod
    def aligned(batch):
        obs, obs2, actions, rewards, dones = batch["obs"], batch["obs2"], batch["act"], batch["rew"], batch["term"]
        _, size, _ = obs.shape; zero_a = torch.zeros((1, size, 14), device=obs.device); zero_r = torch.zeros((1, size, 1), device=obs.device)
        return torch.cat((obs[[0]], obs2), 0), torch.cat((zero_a, actions), 0), torch.cat((zero_r, rewards), 0), torch.cat((zero_r, dones), 0)

    def _debug_failure(self, root, env_step, batch, diagnostics):
        directory = Path(root) / f"step_{int(env_step):08d}_update_{self.update_index:08d}"; directory.mkdir(parents=True, exist_ok=False)
        torch.save(cpu_tree(batch), directory / "batch.pt")
        torch.save(cpu_tree(self.actor.state_dict()), directory / "actor.pth"); torch.save(cpu_tree(self.critic.state_dict()), directory / "critic.pth"); torch.save(cpu_tree(self.target.state_dict()), directory / "target_critic.pth")
        torch.save(cpu_tree(self.actor_optimizer.state_dict()), directory / "actor_optimizer.pth"); torch.save(cpu_tree(self.critic_optimizer.state_dict()), directory / "critic_optimizer.pth")
        if self.algo.automatic_entropy_tuning: torch.save(cpu_tree(self.algo.alpha_entropy_optim.state_dict()), directory / "alpha_optimizer.pth")
        atomic_json(directory / "config.json", self.config); atomic_json(directory / "diagnostics.json", diagnostics); return directory

    def update(self, batch, env_step, debug_root, phase_step=None):
        critic_started=time.perf_counter()
        self.update_index += 1; phase = phase_at(env_step if phase_step is None else phase_step, self.config)
        self.actor.requires_grad_(phase["actor_updates"])
        for group in self.actor_optimizer.param_groups: group["lr"] = phase["actor_lr"]
        for group in self.critic_optimizer.param_groups: group["lr"] = phase["critic_lr"]
        observs, actions, rewards, dones = self.aligned(batch); mask = batch["mask"]; valid = torch.clamp(mask.sum(), min=1.0)
        (q1, q2), target_q = self.algo.critic_loss(False, False, self.actor_adapter, None, self.critic, self.target, observs, actions, rewards, dones, float(self.config["gamma"]))
        loss1 = (((q1 - target_q) ** 2) * mask).sum() / valid; loss2 = (((q2 - target_q) ** 2) * mask).sum() / valid; critic_loss = loss1 + loss2
        tensors = (q1, q2, target_q, critic_loss)
        if not all(torch.isfinite(value).all() for value in tensors):
            directory = self._debug_failure(debug_root, env_step, batch, {"reason": "nonfinite_forward", "update": self.update_index}); raise RuntimeError(f"Non-finite Stage4 critic forward; saved {directory}")
        self.critic_optimizer.zero_grad(set_to_none=True); critic_loss.backward()
        bad = [name for name, parameter in self.critic.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
        if bad:
            directory = self._debug_failure(debug_root, env_step, batch, {"reason": "nonfinite_critic_gradient", "bad_parameters": bad, "update": self.update_index}); raise RuntimeError(f"Non-finite Stage4 critic gradient; saved {directory}")
        try: grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.config["critic_max_grad_norm"]), error_if_nonfinite=True, foreach=False)
        except RuntimeError as error:
            directory=self._debug_failure(debug_root,env_step,batch,{"reason":"nonfinite_critic_global_norm","error":str(error),"update":self.update_index});raise RuntimeError(f"Non-finite Stage4 critic norm; saved {directory}") from error
        self.critic_optimizer.step();critic_seconds=time.perf_counter()-critic_started
        if any(not torch.isfinite(parameter).all() for parameter in self.critic.parameters()):
            directory=self._debug_failure(debug_root,env_step,batch,{"reason":"nonfinite_critic_parameter","update":self.update_index});raise RuntimeError(f"Non-finite Stage4 critic parameter; saved {directory}")
        actor_loss_value = alpha_loss_value = entropy = actor_grad_norm = None;actor_seconds=0.0
        if phase["actor_updates"]:
            actor_started=time.perf_counter()
            for parameter in self.critic.parameters(): parameter.requires_grad_(False)
            policy_loss, log_probs = self.algo.actor_loss(False, False, self.actor_adapter, None, self.critic, self.target, observs, actions, rewards)
            actor_loss = (policy_loss * mask).sum() / valid
            if not torch.isfinite(actor_loss):
                directory = self._debug_failure(debug_root, env_step, batch, {"reason": "nonfinite_actor_loss", "update": self.update_index}); raise RuntimeError(f"Non-finite Stage4 actor loss; saved {directory}")
            self.actor_optimizer.zero_grad(set_to_none=True); actor_loss.backward()
            bad_actor=[name for name,parameter in self.actor.named_parameters() if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
            if bad_actor:
                directory=self._debug_failure(debug_root,env_step,batch,{"reason":"nonfinite_actor_gradient","bad_parameters":bad_actor,"update":self.update_index});raise RuntimeError(f"Non-finite Stage4 actor gradient; saved {directory}")
            actor_grad_norm = safe_gradient_norm(self.actor.parameters())
            if not math.isfinite(actor_grad_norm):
                directory=self._debug_failure(debug_root,env_step,batch,{"reason":"nonfinite_actor_global_norm","update":self.update_index});raise RuntimeError(f"Non-finite Stage4 actor norm; saved {directory}")
            self.actor_optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in self.actor.parameters()):
                directory=self._debug_failure(debug_root,env_step,batch,{"reason":"nonfinite_actor_parameter","update":self.update_index});raise RuntimeError(f"Non-finite Stage4 actor parameter; saved {directory}")
            for parameter in self.critic.parameters(): parameter.requires_grad_(True)
            mean_log_prob = float(((log_probs[:-1] * mask).sum() / valid).detach().item()); entropy = -mean_log_prob
            alpha_loss_value = float((-self.algo.log_alpha_entropy.exp().detach() * (mean_log_prob + self.algo.target_entropy)).item()) if self.algo.automatic_entropy_tuning else 0.0
            self.algo.update_others(mean_log_prob); actor_loss_value = float(actor_loss.item());actor_seconds=time.perf_counter()-actor_started
        if self.update_index % int(self.config["target_update_interval"]) == 0: ptu.soft_update_from_to(self.critic, self.target, float(self.config["tau"]))
        if any(not torch.isfinite(parameter).all() for parameter in self.target.parameters()):
            directory=self._debug_failure(debug_root,env_step,batch,{"reason":"nonfinite_target_critic_parameter","update":self.update_index});raise RuntimeError(f"Non-finite Stage4 target Critic; saved {directory}")
        if not np.isfinite(float(self.algo.alpha_entropy)):
            directory=self._debug_failure(debug_root,env_step,batch,{"reason":"nonfinite_entropy_alpha","update":self.update_index});raise RuntimeError(f"Non-finite Stage4 entropy alpha; saved {directory}")
        td = torch.minimum(q1, q2) - target_q
        self.last_update_timing={"critic_update_seconds":critic_seconds,"actor_update_seconds":actor_seconds}
        return {"phase": phase["phase"], "actor_lr": phase["actor_lr"], "critic_lr": phase["critic_lr"], "actor_loss": actor_loss_value, "critic_loss": float(critic_loss.item()), "alpha": float(self.algo.alpha_entropy), "alpha_loss": alpha_loss_value, "entropy": entropy, "Q1_mean": float(q1[mask.bool()].mean().item()), "Q2_mean": float(q2[mask.bool()].mean().item()), "target_q_mean": float(target_q[mask.bool()].mean().item()), "TD_error_mean": float(td[mask.bool()].mean().item()), "TD_error_std": float(td[mask.bool()].std(unbiased=False).item()), "critic_grad_norm": float(grad_norm.item()), "critic_fraction_clipped": float(float(grad_norm.item()) > float(self.config["critic_max_grad_norm"])), "actor_grad_norm": actor_grad_norm}


def initialize_models(group, config, device):
    actor, actor_payload = load_actor(config["actor_checkpoint"], device); actor.train(); actor.requires_grad_(True)
    if int(actor_payload["epoch"]) != 150: raise RuntimeError("Stage4 requires Stage3-R epoch150 Actor")
    critic, target = make_pair(config, device); source = None
    if group == "rnn_only_critic": source = config["rnn_critic_checkpoint"]; expected = int(config["rnn_critic_expected_update"]); expected_mse = float(config["rnn_critic_expected_val_td_mse"])
    elif group == "multi_il_critic": source = config["multi_critic_checkpoint"]; expected = int(config["multi_critic_expected_update"]); expected_mse = float(config["multi_critic_expected_val_td_mse"])
    if source:
        payload = torch.load(source, map_location=device)
        if int(payload["update"]) != expected: raise RuntimeError(f"{group} checkpoint update={payload['update']}, expected={expected}")
        validation=payload.get("validation",{}); actual_mse=float(validation.get("val_td_mse",float("nan"))); actual_ranking=float(validation.get("pairwise_ranking_accuracy",float("nan")))
        if not math.isclose(actual_mse,expected_mse,rel_tol=0.0,abs_tol=1e-9): raise RuntimeError(f"{group} validation TD MSE mismatch: {actual_mse} != {expected_mse}")
        if not math.isclose(actual_ranking,float(config["pretrained_critic_expected_ranking"]),rel_tol=0.0,abs_tol=1e-12): raise RuntimeError(f"{group} ranking mismatch: {actual_ranking}")
        if payload["architecture"] != architecture(config): raise RuntimeError(f"{group} architecture mismatch")
        critic.load_state_dict(payload["critic_state_dict"], strict=True)
    target.load_state_dict(critic.state_dict(), strict=True); target.requires_grad_(False)
    if state_hash(target) != state_hash(critic): raise RuntimeError("Target Critic initialization mismatch")
    return actor, actor_payload, critic, target, source


def model_audit(group, actor, critic, target, source, config):
    result = {"group": group, "actor_hash": state_hash(actor), "critic_hash": state_hash(critic), "target_critic_hash": state_hash(target), "critic_checkpoint": source, "actor_checkpoint": config["actor_checkpoint"], "pomdp_baselines_commit": config["pomdp_baselines_commit"]}
    if result["critic_hash"] != result["target_critic_hash"]: raise RuntimeError("Online and target Critic hashes differ at initialization")
    return result
