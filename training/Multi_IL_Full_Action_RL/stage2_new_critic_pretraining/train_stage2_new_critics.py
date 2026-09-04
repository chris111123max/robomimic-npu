#!/usr/bin/env python3
"""Train actor-independent RNN-Q and/or policy-balanced Multi-Q."""
from __future__ import annotations
import argparse, copy, json, random, tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

from critic_network import build_critic, load_stage2_critic_checkpoint, model_config_from
from stage2_new_dataset import POLICIES, data_audit, load_split_datasets, split_manifest
from stage2_new_evaluation import (action_gradient_diagnostics, evaluate_critic, gradient_cosine, make_probes, probe_arrays)
from stage2_new_sampler import MultiPolicyBalancedSampler, RNNTransitionSampler

HERE = Path(__file__).resolve().parent

def read_json(path):
    with open(path, encoding="utf-8") as stream: return json.load(stream)
def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, prefix=".tmp_") as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n"); temporary = stream.name
    Path(temporary).replace(path)
def append_jsonl(path, value):
    with open(path, "a", encoding="utf-8") as stream: stream.write(json.dumps(value, sort_keys=True) + "\n")
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available(): npu.manual_seed_all(seed)
def select_device(name):
    if name != "auto":
        if str(name).startswith("npu"):
            try: import torch_npu  # noqa: F401
            except ImportError as exc: raise RuntimeError("NPU requested but torch_npu cannot be imported") from exc
            if not hasattr(torch, "npu") or not torch.npu.is_available(): raise RuntimeError("NPU requested but unavailable")
            torch.npu.set_device(name)
        return torch.device(name)
    if hasattr(torch, "npu") and torch.npu.is_available(): return torch.device("npu:0")
    return torch.device("cpu")
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "stage2_new_config.json")); parser.add_argument("--dataset-root"); parser.add_argument("--output-root")
    parser.add_argument("--device", default=None); parser.add_argument("--mode", choices=("both", "rnn_q", "multi_q"), default="both")
    parser.add_argument("--max-updates", type=int); parser.add_argument("--batch-size", type=int); parser.add_argument("--run-id")
    parser.add_argument("--validate-only", action="store_true"); parser.add_argument("--rnn-checkpoint"); parser.add_argument("--multi-checkpoint")
    return parser.parse_args()
def tensor_batch(batch, device): return {key: torch.as_tensor(batch[key], device=device) for key in ("state", "action", "return")}
def checkpoint_payload(critic, optimizer, config, update, metric):
    return {"critic_state_dict": critic.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "model_config": model_config_from(config),
            "training_config": config, "gamma": float(config["gamma"]), "weight_decay": float(config["weight_decay"]), "step": int(update), "validation_metric": float(metric)}
def selection_metric(label, evaluation): return evaluation["bc_rnn"]["twin_mean_mse"] if label == "rnn_q" else evaluation["balanced_aggregate"]["twin_mean_mse"]

def train_variant(label, sampler, base_state, train, val, config, device, output):
    output.mkdir(parents=True); (output / "checkpoints").mkdir()
    critic = build_critic(config["obs_dim"], config["action_dim"], config["hidden_dims"], config["activation"], config["layer_norm"], device)
    critic.load_state_dict(copy.deepcopy(base_state))
    for key, value in base_state.items():
        if not torch.equal(value.cpu(), critic.state_dict()[key].cpu()):
            raise AssertionError(f"{label} did not receive identical initial weights")
    optimizer = torch.optim.AdamW(critic.parameters(), lr=float(config["critic_lr"]), weight_decay=float(config["weight_decay"]))
    best_metric, best_eval = float("inf"), None
    for update in range(1, int(config["max_updates"]) + 1):
        critic.train(); batch = tensor_batch(sampler.sample(config["batch_size"]), device); q1, q2 = critic(batch["state"], batch["action"])
        loss1, loss2 = torch.nn.functional.mse_loss(q1, batch["return"]), torch.nn.functional.mse_loss(q2, batch["return"]); loss = loss1 + loss2
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        row = {"update": update, "train_q1_loss": float(loss1.item()), "train_q2_loss": float(loss2.item()), "train_mean_loss": float(loss.item()/2), "learning_rate": float(optimizer.param_groups[0]["lr"])}
        if label == "multi_q": row["sampled_policy_proportions"] = sampler.proportions()
        if update % int(config["eval_interval"]) == 0 or update == int(config["max_updates"]):
            evaluation = evaluate_critic(critic, val, device, config["evaluation_batch_size"]); metric = selection_metric(label, evaluation); row["validation"] = evaluation; row["selection_metric"] = metric
            if metric < best_metric:
                best_metric, best_eval = metric, evaluation; torch.save(checkpoint_payload(critic, optimizer, config, update, metric), output / "checkpoints" / "best.pth")
        append_jsonl(output / "train_metrics.jsonl", row)
        if update % int(config["checkpoint_interval"]) == 0 or update == int(config["max_updates"]): torch.save(checkpoint_payload(critic, optimizer, config, update, selection_metric(label, evaluate_critic(critic, val, device, config["evaluation_batch_size"]))), output / "checkpoints" / "last.pth")
    if best_eval is None: raise RuntimeError("No validation evaluation was performed")
    best_payload = torch.load(output / "checkpoints" / "best.pth", map_location=device)
    critic.load_state_dict(best_payload["critic_state_dict"])
    atomic_json(output / "final_validation.json", best_eval)
    return critic, best_eval, best_metric

def diagnostics(models, val, probes, device):
    output, raw = {}, {}
    for label, critic in models.items():
        output[label], raw[label] = {}, {}
        for name, rows in probes.items():
            states, actions = probe_arrays(rows, val); raw[label][name] = action_gradient_diagnostics(critic, states, actions, device)
            output[label][name] = {key: value for key, value in raw[label][name].items() if not key.endswith("gradients")}
    comparison = {name: gradient_cosine(raw["rnn_q"][name], raw["multi_q"][name]) for name in probes} if set(models) == {"rnn_q", "multi_q"} else {}
    return output, comparison

def main():
    args = parse_args(); config = read_json(args.config)
    for key, value in (("dataset_root",args.dataset_root),("output_root",args.output_root),("device",args.device),("max_updates",args.max_updates),("batch_size",args.batch_size)):
        if value is not None: config[key] = value
    config["obs_dim"], config["action_dim"] = int(config["obs_dim"]), int(config["action_dim"])
    if config["gamma"] != 0.99 or config["weight_decay"] != 1e-4 or not config["layer_norm"]: raise RuntimeError("Stage2-new fixed gamma/weight_decay/LayerNorm contract was changed")
    train_seeds = list(range(int(config["train_seed_start"]), int(config["train_seed_end"])+1)); val_seeds = list(range(int(config["val_seed_start"]), int(config["val_seed_end"])+1))
    device = select_device(config["device"]); seed_all(config["training_seed"])
    train, val = load_split_datasets(config["dataset_root"], train_seeds, val_seeds, config["gamma"])
    timestamp = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S"); run = Path(config["output_root"]) / timestamp
    if run.exists(): raise FileExistsError(run)
    run.mkdir(parents=True); config["resolved_device"] = str(device); atomic_json(run / "config_resolved.json", config); atomic_json(run / "data_audit.json", data_audit(train,val)); atomic_json(run / "split_manifest.json", split_manifest(train,"train") + split_manifest(val,"validation"))
    probes, probe_manifest = make_probes(val, int(config["probe_max_transitions"])); atomic_json(run / "probe_manifest.json", probe_manifest)
    labels = ("rnn_q", "multi_q") if args.mode == "both" else (args.mode,); models, evaluations, metrics = {}, {}, {}
    if args.validate_only:
        paths = {"rnn_q":args.rnn_checkpoint, "multi_q":args.multi_checkpoint}
        for label in labels:
            if not paths[label]: raise ValueError(f"--{label}-checkpoint is required with --validate-only")
            models[label], _ = load_stage2_critic_checkpoint(paths[label], device)
            evaluations[label] = evaluate_critic(models[label], val, device, config["evaluation_batch_size"]); metrics[label] = selection_metric(label, evaluations[label])
            atomic_json(run / label / "final_validation.json", evaluations[label])
    else:
        base = build_critic(config["obs_dim"], config["action_dim"], config["hidden_dims"], config["activation"], config["layer_norm"], device)
        base_state = copy.deepcopy(base.state_dict()); torch.save({"critic_state_dict":base_state,"model_config":model_config_from(config)}, run / "initial_critic_state.pth")
        for label in labels:
            sampler = RNNTransitionSampler(train["bc_rnn"], config["training_seed"]) if label == "rnn_q" else MultiPolicyBalancedSampler(train, config["training_seed"])
            models[label], evaluations[label], metrics[label] = train_variant(label, sampler, base_state, train, val, config, device, run / label)
    grad, cosine = diagnostics(models, val, probes, device); comparison = {"validation_selection_metric":metrics, "return_regression":evaluations, "action_gradient":grad, "gradient_cosine_rnn_vs_multi":cosine}
    artifacts = {f"{label}_best": f"{label}/checkpoints/best.pth" for label in labels if not args.validate_only}
    atomic_json(run / "critic_comparison.json", comparison); atomic_json(run / "stage2_new_summary.json", {"stage":"stage2-new","run_dir":str(run),"mode":args.mode,"validate_only":args.validate_only,"selection_metrics":metrics,"artifacts":artifacts})
    print(json.dumps({"run_dir":str(run),"selection_metrics":metrics}, indent=2))
if __name__ == "__main__": main()
