"""Testing-only: does Ready twin-Q rank TWO RELATIVELY SUPPORTED actions correctly?
One preregistered BC-vs-625 action at each of four pre-existing BC histories;
one action is replaced, then the SAME BC policy continues. No fitting.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.dont_write_bytecode = True

import numpy as np

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
RL = HERE.parents[2]
AUDIT = TEST / "critic_plateau_action_coverage"
CONSISTENCY = TEST / "actor_critic_improvement_consistency"
TRACE = TEST / "mean_multi_collapse_diagnosis" / "round1" / "READY_trajectories.jsonl"
ACTOR625 = TEST / "mean_multi_collapse_diagnosis" / "round1" / "actor_625.pth"
OUT = HERE / "output"
SEEDS = (20008, 20002, 20005, 20007)
GAMMA = 0.99
HORIZON = 700
RNG_SEED = 20007
MIN_Q_DIFFERENCE = 1e-5
MIN_ACTION_DIFFERENCE = 1e-4

for parent in (RL / "stage3_v5_rgmm_td3", RL / "stage3_v6_dual_2q",
               TEST / "mean_multi_collapse_diagnosis",
               TEST / "final_collapse_rootcause"):
    sys.path.insert(0, str(parent))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def traces_by_seed():
    rows = [json.loads(s) for s in TRACE.read_text().splitlines() if s.strip()]
    result = {}
    for seed in SEEDS:
        trace = sorted((r for r in rows if int(r["seed"]) == seed),
                       key=lambda r: int(r["timestep"]))
        if not trace or any(int(r["timestep"]) != i for i, r in enumerate(trace)):
            raise RuntimeError("Missing/noncontiguous original BC prefix for seed %d" % seed)
        result[seed] = trace
    return result


def load_models():
    import torch
    from core import setup, module_hash
    from stage3_v5_agent import target_final_distribution_vectorized
    from stage3_v5_history_critic import encode_replay_contexts

    device, ready, bc, critic, scale, offset = setup()
    if str(device) != "npu:0":
        raise RuntimeError("ONLY npu:0 is permitted")
    bc.eval().requires_grad_(False)
    critic.eval().requires_grad_(False)
    updated = copy.deepcopy(bc)
    payload = torch.load(ACTOR625, map_location="cpu", weights_only=False)
    if int(payload["actor_virtual_updates"]) != 625:
        raise RuntimeError("Not the frozen 625-update Actor")
    updated.load_state_dict(payload["actor"], strict=True)
    updated.eval().requires_grad_(False)
    return (torch, device, bc, updated, critic, scale, offset,
            module_hash, target_final_distribution_vectorized, encode_replay_contexts)


def values_for_candidates(critic, encode, torch, device, obs, acts, steps, actions):
    with torch.no_grad():
        o = torch.as_tensor(obs, device=device, dtype=torch.float32)
        a = torch.as_tensor(acts, device=device, dtype=torch.float32)
        t = torch.as_tensor(steps, device=device, dtype=torch.long)
        context = encode(critic, o, a, t, HORIZON)
        features = tuple(x[:, -1] for x in context)
        result = []
        for candidate in actions:
            q1, q2 = critic.q_from_context(
                features, torch.as_tensor(candidate, device=device, dtype=torch.float32))
            result.append(np.concatenate(
                (q1.detach().cpu().numpy(), q2.detach().cpu().numpy()), axis=-1))
        return np.stack(result, axis=1)


def prepare():
    """All selection uses prior relative-support labels and frozen Q, never fork returns."""
    if OUT.exists():
        raise FileExistsError("Output directory already exists: " + str(OUT))
    traces = traces_by_seed()
    selection = np.load(AUDIT / "query_selection.npz")
    support = np.load(AUDIT / "conditional_support_records.npz")
    cache = np.load(CONSISTENCY / "frozen_numeric_evidence.npz")
    metadata = json.loads((CONSISTENCY / "context_metadata.json").read_text())
    ids = selection["cache_indices"]
    primary, sensitivity = support["primary_status"], support["sensitivity_status"]
    if not (len(ids) == len(primary[0]) == len(sensitivity[0])):
        raise RuntimeError("Prior audit arrays have inconsistent shapes")
    models = load_models()
    (torch, device, bc, updated, critic, scale, offset, module_hash, dist_fn, encode) = models
    candidates = []
    excluded = {"not_BC_history": 0, "outside_window": 0, "not_supported": 0,
                "out_of_bounds_or_same": 0}
    for j, cid in enumerate(ids):
        cid = int(cid)
        if cid < 4096:
            excluded["not_BC_history"] += 1
            continue
        m = metadata[cid]
        if int(m["owner"]) != 0 or int(m["seed"]) not in SEEDS:
            excluded["not_BC_history"] += 1
            continue
        seed, step = int(m["seed"]), int(m["step"])
        trace = traces[seed]
        if step < 79 or step > min(499, len(trace) - 21) or step % 10 != 9:
            excluded["outside_window"] += 1
            continue
        modes = [int(np.argmax(cache[str(u) + "_probs"][cid])) for u in (0, 625)]
        if any(int(status[si, j, mode]) != 2
               for status in (primary, sensitivity)
               for si, mode in enumerate(modes)):
            excluded["not_supported"] += 1
            continue
        bc_action = np.asarray(cache["0_means"][cid, modes[0]], np.float32)
        new_action = np.asarray(cache["625_means"][cid, modes[1]], np.float32)
        if (max(np.abs(bc_action).max(), np.abs(new_action).max()) > 1.0
                or np.max(np.abs(new_action - bc_action)) < MIN_ACTION_DIFFERENCE):
            excluded["out_of_bounds_or_same"] += 1
            continue
        window = trace[step - 9:step + 1]
        candidates.append(dict(
            seed=seed, step=step, cache_index=cid, modes=modes,
            actions=np.stack([bc_action, new_action]),
            observations=np.asarray([r["observation_flat"] for r in window], np.float32),
            history_actions=np.asarray([r["action"] for r in window], np.float32),
            steps=np.arange(step - 9, step + 1, dtype=np.int64),
            prereg_source="original_ready_BC_trajectory",
        ))
    if not candidates:
        raise RuntimeError("No prior-audit-supported candidate; no simulation permitted")

    q = values_for_candidates(
        critic, encode, torch, device,
        np.stack([x["observations"] for x in candidates]),
        np.stack([x["history_actions"] for x in candidates]),
        np.stack([x["steps"] for x in candidates]),
        [np.stack([x["actions"][k] for x in candidates]) for k in range(2)])
    for i, candidate in enumerate(candidates):
        candidate["q"] = q[i]
        candidate["delta_q"] = q[i, 1] - q[i, 0]

    # Four independent seed-context pairs, selected BEFORE any paired simulation.
    # Prefer nontrivial, twin-agreed Q ordering; never filter using real fork outcomes.
    selected = []
    counts = {}
    for seed in SEEDS:
        pool = [c for c in candidates if c["seed"] == seed
                and abs(c["delta_q"][0]) > MIN_Q_DIFFERENCE
                and abs(c["delta_q"][1]) > MIN_Q_DIFFERENCE
                and c["delta_q"][0] * c["delta_q"][1] > 0]
        counts[str(seed)] = len(pool)
        if not pool:
            raise RuntimeError(
                "No supported twin-agreed candidate for seed %d; STOP before simulation. "
                "Candidate counts: %s" % (seed, counts))
        # Fixed deterministic rule; no MC/terminal result consulted.
        selected.append(sorted(pool, key=lambda c:
                               (-abs(float(c["delta_q"][0])),
                                -abs(float(c["delta_q"][1])), c["step"]))[0])

    # Verify cached component means against the actual stored 625/BC models.
    for c in selected:
        o = torch.as_tensor(c["observations"][None], device=device)
        steps = torch.as_tensor(c["steps"][None], device=device)
        for k, actor in enumerate((bc, updated)):
            with torch.no_grad():
                d, _ = dist_fn(actor, o, steps, 10)
                means = (d.component_distribution.base_dist.loc * scale + offset)
                exact = means[0, c["modes"][k]].detach().cpu().numpy()
            if not np.allclose(exact, c["actions"][k], rtol=0, atol=1e-5):
                raise RuntimeError("Cached candidate differs from frozen model; STOP")
    paths = [TRACE, ACTOR625, AUDIT / "conditional_support_records.npz",
             AUDIT / "query_selection.npz", CONSISTENCY / "frozen_numeric_evidence.npz",
             CONSISTENCY / "context_metadata.json"]
    reg = {
        "purpose": "within-relative-support action ranking only",
        "support_rule": "both chosen categorical-MAP component means have label 2 at 95% AND 99%",
        "support_caveat": "nearby historical actions, NOT guaranteed identical state-action training samples",
        "reference_policy": "frozen Ready BC-RNN-GMM",
        "candidate_policy": "frozen testing Actor 625",
        "selection": "per seed largest |delta Q1| with twin-agreed nonzero sign; return-blind",
        "prediction_semantics": "Q(h,a), one current component-mean action, BC continuation",
        "seeds": list(SEEDS), "future_rng_seed": RNG_SEED,
        "critic_updates": 0, "actor_updates": 0,
        "input_sha256": {str(p): sha256(p) for p in paths},
        "model_hashes": {name: module_hash(model) for name, model in
                         (("BC", bc), ("Actor625", updated), ("ReadyCritic", critic))},
        "eligible_counts": counts, "excluded": excluded,
        "pairs": [dict(seed=c["seed"], step=c["step"], cache_index=c["cache_index"],
                       modes=c["modes"], actions=c["actions"].tolist(),
                       predicted_q=c["q"].tolist(), predicted_delta_q=c["delta_q"].tolist(),
                       observations=c["observations"].tolist(),
                       history_actions=c["history_actions"].tolist(),
                       episode_steps=c["steps"].tolist()) for c in selected],
    }
    OUT.mkdir(parents=True)
    dump_new(OUT / "preregistration.json", reg)
    print("PREPARED 4 supported pairs; no simulation", [(c["seed"], c["step"],
          c["delta_q"].tolist()) for c in selected], flush=True)


def hidden_hash(executor, worker):
    state = executor.hidden[worker]
    items = [] if state is None else (state if isinstance(state, tuple) else [state])
    return hashlib.sha256(
        b"".join(t.detach().cpu().numpy().tobytes() for t in items)
        + str(executor.counters[worker]).encode()).hexdigest()


def run():
    import torch
    from core import setup, module_hash, StaggeredVectorEnv, DATASET
    from stage3_v5_actor import BatchedGMMExecutor, obs_to_flat
    from helpers import snapshot_hash
    from snapshot_worker import snapshot_worker
    import stage3_v5_vector_env as vector

    prereg = json.loads((OUT / "preregistration.json").read_text())
    if (prereg["seeds"] != list(SEEDS) or len(prereg["pairs"]) != 4
            or len({p["seed"] for p in prereg["pairs"]}) != 4):
        raise RuntimeError("Preregistered four-seed contract failed")
    for path, digest in prereg["input_sha256"].items():
        if sha256(path) != digest:
            raise RuntimeError("Input changed after preregistration: " + path)
    if (OUT / "results.json").exists():
        raise FileExistsError("Results already exist; do not overwrite/repeat")
    if (OUT / "run_started.json").exists():
        raise FileExistsError("This test has already started; do not silently rerun")

    device, ready, bc, critic, scale, offset = setup()
    if str(device) != "npu:0":
        raise RuntimeError("Only npu:0 allowed")
    bc.eval().requires_grad_(False)
    critic.eval().requires_grad_(False)
    if (module_hash(bc) != prereg["model_hashes"]["BC"]
            or module_hash(critic) != prereg["model_hashes"]["ReadyCritic"]):
        raise RuntimeError("Frozen BC/Critic changed since preparation")
    model_before = (module_hash(bc), module_hash(critic))
    traces = traces_by_seed()
    vector._worker = snapshot_worker
    torch.set_num_threads(1)
    dump_new(OUT / "run_started.json", {"time": time.time(), "four_workers": True,
                                       "branches": ["BC", "ACTOR625"], "npu": "npu:0"})
    vec = None
    baseline = {}
    outcomes = {}
    contract = {"initialized": 0, "used": 0, "closed": 0}
    try:
        vec = StaggeredVectorEnv(DATASET, 4, RNG_SEED, delay=.5, timeout=120,
                                 startup_parallelism=4, shared_memory=False)
        contract["initialized"] = len(vec.initial_observations)
        contract["worker_pids"] = [p.pid for p in vec.processes]
        if contract["initialized"] != 4 or vec.alive_worker_count() != 4:
            raise RuntimeError("Not exactly 4 live parallel simulator workers")

        for branch in ("BC", "ACTOR625"):
            obs_map = vec.reset_many({i: s for i, s in enumerate(SEEDS)})
            obs = [obs_map[i] for i in range(4)]
            executor = BatchedGMMExecutor(bc, scale, offset, 4, horizon=10)
            torch.manual_seed(RNG_SEED)
            torch.npu.manual_seed_all(RNG_SEED)
            active = [True] * 4
            episode_rows = [[] for _ in SEEDS]
            pair_checks = {}
            for t in range(HORIZON):
                actions = executor.actions_for(
                    list(range(4)), obs, 0.0, None, None)
                for i, spec in enumerate(prereg["pairs"]):
                    if not active[i]:
                        continue
                    seed, fork = spec["seed"], spec["step"]
                    if t <= fork:
                        difference = float(np.max(np.abs(
                            obs_to_flat(obs[i]) - np.asarray(
                                traces[seed][t]["observation_flat"], np.float32))))
                        if difference > 1e-6:
                            raise AssertionError(
                                "Original BC prefix mismatch: %s %s %s %s" %
                                (branch, seed, t, difference))
                    if t < fork:
                        actions[i] = np.asarray(traces[seed][t]["action"], np.float32)
                    elif t == fork:
                        vec.connections[i].send(("snapshot", None))
                        message = vec._recv(i, 120, "known_support_fork_snapshot")
                        if message[0] != "SNAPSHOT":
                            raise RuntimeError("No valid fork snapshot: " + str(message[0]))
                        h = hashlib.sha256(
                            torch.get_rng_state().numpy().tobytes()
                            + torch.npu.get_rng_state().cpu().numpy().tobytes()).hexdigest()
                        fingerprint = {
                            "physics_and_controller": snapshot_hash(message[1]),
                            "BC_hidden": hidden_hash(executor, i),
                            "parent_torch_rng": h,
                            "saved_history": hashlib.sha256(
                                np.asarray(spec["observations"], np.float32).tobytes()
                                + np.asarray(spec["history_actions"], np.float32)[:-1].tobytes()
                                + np.asarray(spec["episode_steps"], np.int64).tobytes()).hexdigest()
                        }
                        if branch == "BC":
                            baseline[seed] = fingerprint
                        if baseline.get(seed) != fingerprint:
                            raise AssertionError("STRICT_PAIR_MISMATCH seed=" + str(seed))
                        pair_checks[seed] = fingerprint
                        k = 0 if branch == "BC" else 1
                        actions[i] = np.asarray(spec["actions"][k], np.float32)
                ids = [i for i in range(4) if active[i]]
                if not ids:
                    break
                messages = vec.step([actions[i] for i in ids], ids)
                for i, msg in messages:
                    if msg[0] != "OK":
                        raise RuntimeError("Simulator error: " + str(msg[0]))
                    _, next_obs, reward, done, won, _ = msg
                    episode_rows[i].append({
                        "timestep": t, "reward": float(reward),
                        "success": bool(won),
                        "action": np.asarray(actions[i], np.float32).tolist(),
                    })
                    obs[i] = next_obs
                    active[i] = not (done or won or t == HORIZON - 1)
                if t % 100 == 0:
                    print("SIM_PROGRESS", branch, t, "active", sum(active), flush=True)
            if len(pair_checks) != 4:
                raise RuntimeError("Not all four preregistered contexts were reached")
            contract["used"] = 4
            episode_stats = []
            for i, spec in enumerate(prereg["pairs"]):
                fork = spec["step"]
                rows = episode_rows[i]
                if len(rows) <= fork:
                    raise RuntimeError("Episode ended before fork")
                tail = [r["reward"] for r in rows[fork:]]
                ret = sum(float(reward) * GAMMA ** k
                          for k, reward in enumerate(tail))
                episode_stats.append({
                    "seed": spec["seed"], "fork_step": fork,
                    "return_from_fork": ret, "success": rows[-1]["success"],
                    "length": len(rows), "pair_fingerprint": pair_checks[spec["seed"]],
                    "chosen_action": spec["actions"][0 if branch == "BC" else 1],
                })
            outcomes[branch] = episode_stats
            print("BRANCH_COMPLETE", branch,
                  [(e["seed"], e["return_from_fork"]) for e in episode_stats],
                  flush=True)
    finally:
        if vec is not None:
            vec.close()
            contract["closed"] = sum(not p.is_alive() and p.exitcode == 0
                                     for p in vec.processes)
            contract["exitcodes"] = [p.exitcode for p in vec.processes]
    if any(contract[k] != 4 for k in ("initialized", "used", "closed")):
        raise RuntimeError("Four-worker contract FAILED: " + str(contract))
    if (module_hash(bc), module_hash(critic)) != model_before:
        raise RuntimeError("Frozen model changed during test")
    paired = []
    for spec, a, b in zip(prereg["pairs"], outcomes["BC"], outcomes["ACTOR625"]):
        if a["seed"] != b["seed"] or a["pair_fingerprint"] != b["pair_fingerprint"]:
            raise AssertionError("Pairing changed")
        delta_real = b["return_from_fork"] - a["return_from_fork"]
        dq1, dq2 = map(float, spec["predicted_delta_q"])
        identifiable = abs(delta_real) > 1e-6
        paired.append({
            "seed": spec["seed"], "fork_step": spec["step"],
            "delta_q1": dq1, "delta_q2": dq2,
            "delta_real_MC": delta_real,
            "real_return_identifiable": identifiable,
            "q1_order_correct": bool(dq1 * delta_real > 0) if identifiable else None,
            "q2_order_correct": bool(dq2 * delta_real > 0) if identifiable else None,
            "BC": a, "Actor625_single_action": b,
        })
    informative = [p for p in paired if p["real_return_identifiable"]]
    misranked = [p for p in informative
                 if not p["q1_order_correct"] or not p["q2_order_correct"]]
    label = ("LOCAL_PAIR_MISORDER_OBSERVED" if misranked else
             "LOCAL_PAIR_ORDER_AGREED_ON_IDENTIFIABLE_PAIRS" if informative else
             "INCONCLUSIVE_NO_DISTINGUISHABLE_RETURNS")
    results = {
        "classification": label,
        "pairs": paired, "worker_contract": contract,
        "informative_pairs": len(informative),
        "mismatches": len(misranked),
        "model_hashes_unchanged": True,
        "actual_actor_updates": 0, "actual_critic_updates": 0,
        "tested_only_prior_audit_supported_actions": True,
        "limitations": [
            "Only four episode contexts and one common future RNG stream.",
            "Support means nearby historical actions, not exact training tuples.",
            "A correct result does NOT establish that all Critic action rankings are right.",
            "A correct result does NOT establish that unseen states cause Actor collapse.",
            "Same frozen BC continuation tests one-action rankings, NOT repeatedly executing new Actor.",
            "Nonzero realized return differences are single-seed outcomes, not expected-Q ground truth.",
        ]
    }
    dump_new(OUT / "results.json", results)
    report = [
        "# 已有相对数据支持区域内的 Q 动作排序检验",
        "",
        "**结果：%s**" % label,
        "",
        "范围：4 个预注册 BC history；每个仅比较 BC 与 Actor625 的一个已支持分量均值动作；",
        "只替换当前这一个动作，后续统一使用冻结 BC；没有训练或检查陌生动作。",
        "",
        "| Seed | Fork | ΔQ1 | ΔQ2 | Δ真实折扣回报 | 可辨识 | Q1/Q2 排序 |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for p in paired:
        outcome = ("一致" if p["q1_order_correct"] and p["q2_order_correct"]
                   else "发现误排序") if p["real_return_identifiable"] else "不可判断"
        report.append("| %d | %d | %+.6g | %+.6g | %+.6g | %s | %s |" %
                      (p["seed"], p["fork_step"], p["delta_q1"], p["delta_q2"],
                       p["delta_real_MC"], "是" if p["real_return_identifiable"] else "否",
                       outcome))
    report += [
        "",
        "有效真实回报对比：%d/4；其中观察到的排序冲突：%d。" %
        (len(informative), len(misranked)),
        "",
        "四个模拟器 worker 初始化／使用／正常关闭：%d/%d/%d。" %
        (contract["initialized"], contract["used"], contract["closed"]),
        "",
        "**解释边界：** 即使全部一致，也仅说明这四个局部动作对未发现错误；",
        "不能证明 Critic 整体正确，更不能由此推出状态样本太少是根因。",
        "即使发现误排序，也需进一步重复随机未来轨迹才能确定期望 Q 的系统性错误。",
    ]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("TEST_COMPLETE", label, "initialized/used/closed=4/4/4", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "run"))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    else:
        run()
