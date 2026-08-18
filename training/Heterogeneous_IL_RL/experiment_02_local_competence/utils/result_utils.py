"""Crash-safe persistence and table helpers."""

import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _replace(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path, payload):
    def write(handle):
        handle.write((json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    _replace(path, write)


def atomic_csv(path, fieldnames, rows):
    rows = list(rows)
    def write(handle):
        text = os.fdopen(os.dup(handle.fileno()), "w", encoding="utf-8", newline="", closefd=True)
        try:
            writer = csv.DictWriter(text, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            text.flush()
        finally:
            text.close()
    _replace(path, write)


def atomic_npz(path, compressed=True, **arrays):
    def write(handle):
        (np.savez_compressed if compressed else np.savez)(handle, **arrays)
    _replace(path, write)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Corrupt JSONL {path}:{number}") from exc
    return rows


def stable_seed(base, *parts):
    import hashlib
    material = ":".join([str(int(base)), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "little") & 0x7FFFFFFF


def bool_value(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes"):
        return True
    if text in ("0", "false", "no"):
        return False
    raise ValueError(f"Not a boolean value: {value!r}")


def ensure_dirs(run_dir):
    root = Path(run_dir)
    children = [
        "logs", "recheck/trajectories", "source_trajectories/rnn_fail_transformer_success",
        "source_trajectories/transformer_fail_rnn_success",
        "branch_states/rnn_fail_transformer_success",
        "branch_states/transformer_fail_rnn_success", "reconstruction",
        "branch_results/workers", "branch_results/trajectories", "analysis/plots",
    ]
    root.mkdir(parents=True, exist_ok=True)
    for child in children:
        (root / child).mkdir(parents=True, exist_ok=True)
    return root
