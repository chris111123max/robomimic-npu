"""Strict reader for the fixed Experiment 00 run and its paired results."""

from pathlib import Path

from utils.result_utils import atomic_json, bool_value, read_csv, read_json


POLICIES = ("bc", "bc_gmm", "bc_gmm_rnn", "bc_gmm_transformer")
RNN = "bc_gmm_rnn"
TRANSFORMER = "bc_gmm_transformer"


def _matrix_from_results(source_run):
    path = Path(source_run) / "raw_results" / "rollout_results.csv"
    rows = read_csv(path)
    matrix = {}
    for row in rows:
        sid = int(row["initial_state_id"])
        policy = row["policy_name"]
        if policy in POLICIES:
            matrix.setdefault(sid, {})[policy] = int(bool_value(row["success"]))
    seed_manifest = read_json(Path(source_run) / "seed_manifest.json")
    output = []
    for sid, seed in enumerate(seed_manifest["environment_seeds"]):
        values = matrix.get(sid, {})
        if set(values) != set(POLICIES):
            raise RuntimeError(f"Experiment 00 raw results incomplete for state {sid}: {sorted(values)}")
        output.append({"initial_state_id": sid, "seed": int(seed), **values,
                       "any_success": int(any(values.values()))})
    return output


def _matrix_from_analysis(source_run):
    rows = read_csv(Path(source_run) / "analysis" / "success_matrix.csv")
    result = []
    for row in rows:
        parsed = {"initial_state_id": int(row["initial_state_id"]),
                  "seed": int(row.get("seed", row.get("environment_seed")))}
        for policy in POLICIES:
            parsed[policy] = int(bool_value(row[policy]))
        parsed["any_success"] = int(any(parsed[p] for p in POLICIES))
        result.append(parsed)
    return sorted(result, key=lambda row: row["initial_state_id"])


def read_success_matrix(source_run):
    source_run = Path(source_run)
    analysis = source_run / "analysis" / "success_matrix.csv"
    raw = source_run / "raw_results" / "rollout_results.csv"
    if not analysis.exists() and not raw.exists():
        raise FileNotFoundError("Experiment 00 has neither success_matrix.csv nor rollout_results.csv")
    primary = _matrix_from_analysis(source_run) if analysis.exists() else _matrix_from_results(source_run)
    if analysis.exists() and raw.exists():
        secondary = _matrix_from_results(source_run)
        if primary != secondary:
            raise RuntimeError("Experiment 00 success_matrix.csv and rollout_results.csv disagree")
    return primary


def compute_statistics(rows, runtime_errors):
    n = len(rows)
    counts = {policy: sum(row[policy] for row in rows) for policy in POLICIES}
    both_success = sum(row[RNN] and row[TRANSFORMER] for row in rows)
    rnn_only = sum(row[RNN] and not row[TRANSFORMER] for row in rows)
    transformer_only = sum(not row[RNN] and row[TRANSFORMER] for row in rows)
    both_fail = sum(not row[RNN] and not row[TRANSFORMER] for row in rows)
    return {
        "num_initial_conditions": n,
        "runtime_errors": int(runtime_errors),
        "policy_success_counts": counts,
        "rnn_transformer": {
            "both_success": both_success,
            "rnn_success_transformer_fail": rnn_only,
            "rnn_fail_transformer_success": transformer_only,
            "both_fail": both_fail,
            "oracle_success": n - both_fail,
        },
        "all_policy_oracle_success": sum(any(row[p] for p in POLICIES) for row in rows),
    }


def count_runtime_errors(source_run):
    errors_path = Path(source_run) / "raw_results" / "errors.csv"
    if errors_path.exists():
        return len(read_csv(errors_path))
    count = 0
    episodes = Path(source_run) / "raw_results" / "episodes"
    for path in episodes.glob("*/*.json") if episodes.exists() else ():
        try:
            count += int(read_json(path).get("status") == "error")
        except Exception:
            count += 1
    return count


def validate_source(config, source_run, output_manifest=None):
    source_run = Path(source_run).expanduser().resolve()
    required = [source_run / "config.json", source_run / "seed_manifest.json",
                source_run / "initial_state_manifest.json", source_run / "initial_states"]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required Experiment 00 input missing: {path}")
    source_config = read_json(source_run / "config.json")
    if str(Path(source_config["dataset_path"])) != str(Path(config["dataset_path"])):
        raise RuntimeError("Experiment 00 dataset differs from Experiment 02 fixed dataset")
    for policy in (RNN, TRANSFORMER):
        actual = str(Path(source_config["policies"][policy]["checkpoint_path"]))
        expected = str(Path(config["policies"][policy]["checkpoint_path"]))
        if actual != expected:
            raise RuntimeError(f"{policy} checkpoint mismatch: source={actual}, expected={expected}")
    if source_config.get("expected_environment_name") != config["environment_name"]:
        raise RuntimeError("Experiment 00 environment name mismatch")
    if int(source_config.get("horizon")) != int(config["horizon"]):
        raise RuntimeError("Experiment 00 horizon mismatch")

    rows = read_success_matrix(source_run)
    statistics = compute_statistics(rows, count_runtime_errors(source_run))
    expected = config["expected_source_statistics"]
    if config.get("strict_source_statistics", True) and statistics != expected:
        raise RuntimeError(f"Experiment 00 fixed statistics mismatch:\nactual={statistics}\nexpected={expected}")

    seed_manifest = read_json(source_run / "seed_manifest.json")
    state_manifest = read_json(source_run / "initial_state_manifest.json")
    states = sorted(state_manifest["states"], key=lambda x: int(x["initial_state_id"]))
    if len(rows) != 100 or len(states) != 100 or len(seed_manifest["environment_seeds"]) != 100:
        raise RuntimeError("Experiment 00 must contain exactly 100 paired initial states")
    for sid, (row, entry, seed) in enumerate(zip(rows, states, seed_manifest["environment_seeds"])):
        if int(row["initial_state_id"]) != sid or int(entry["initial_state_id"]) != sid:
            raise RuntimeError(f"Initial-state ids are not contiguous at {sid}")
        if int(row["seed"]) != int(seed) or int(entry["environment_seed"]) != int(seed):
            raise RuntimeError(f"Seed mapping mismatch for initial state {sid}")
        if not (source_run / entry["state_file"]).is_file():
            raise FileNotFoundError(f"Initial-state file missing: {entry['state_file']}")

    manifest = {
        "source_run": str(source_run), "source_config": str(source_run / "config.json"),
        "dataset_path": config["dataset_path"], "environment_name": config["environment_name"],
        "horizon": int(config["horizon"]), "action_dimension": int(config["action_dimension"]),
        "checkpoints": {p: config["policies"][p]["checkpoint_path"] for p in (RNN, TRANSFORMER)},
        "statistics": statistics, "strict_source_statistics": bool(config.get("strict_source_statistics", True)),
        "seed_manifest": str(source_run / "seed_manifest.json"),
        "initial_state_manifest": str(source_run / "initial_state_manifest.json"),
    }
    if output_manifest:
        atomic_json(output_manifest, manifest)
    return rows, states, seed_manifest, manifest
