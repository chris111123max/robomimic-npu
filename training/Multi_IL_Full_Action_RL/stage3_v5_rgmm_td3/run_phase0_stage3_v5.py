#!/usr/bin/env python3
"""Removed V5 Phase-0 entry point.

Stage3-v5 starts from the immutable BC-RNN Actor and audited Stage2 Critic
checkpoints. Readiness is replay-only and is owned by ``CriticHandoff``;
there is no separate environment-evaluation gate to run or reuse.
"""
from __future__ import annotations

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-run-dir")
    parser.parse_args()
    raise SystemExit(
        "Stage3-v5 has no Phase-0 run. Prepare the pair and start a branch "
        "with train_stage3_v5_vector.py."
    )


if __name__ == "__main__":
    main()
