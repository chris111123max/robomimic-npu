# Stage 1.5 — Runtime Dataset Discovery and Audit

Stage 1.5 answers one question: what trajectory data was actually collected by Stage 1? It reads
`training_runs` recursively and never writes into that tree. It does not launch rollouts, load
checkpoints, train models, alter rewards, or start Stage 2.

## Workflow

Run discovery first:

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic

python -u training/Multi_IL_Full_Action_RL/stage1_5_dataset_audit/discover_stage1_data.py \
  --training-runs-root /data/home/3220251075/lerobot_workspace/training_runs
```

Discovery scans `.hdf5`, `.h5`, `.json`, `.jsonl`, `.npz`, `.pkl`, and `.pickle`, while excluding
checkpoint, model, TensorBoard, log, and video paths. A filename or directory name can suggest a
policy for inventory purposes, but automatic selection requires policy identity from file content.
An HDF5 candidate is a rollout only when its contents recover state, action, reward, next state,
and an episode-ending signal.

The selection algorithm groups content-validated HDF5 datasets by run family. It selects only an
unambiguous family containing BC-RNN, BC-Transformer, and BC-GMM with exactly 100 unique,
duplicate-free, identical seeds. Multiple equally valid formal runs produce `status=ambiguous`;
analysis then stops instead of silently choosing one.

Inspect the selected schemas independently when desired:

```bash
python -u training/Multi_IL_Full_Action_RL/stage1_5_dataset_audit/inspect_dataset_schema.py
```

Run the audit after discovery reports `status=selected`:

```bash
python -u training/Multi_IL_Full_Action_RL/stage1_5_dataset_audit/analyze_stage1_data.py \
  --training-runs-root /data/home/3220251075/lerobot_workspace/training_runs
```

Use `--rediscover` to refresh inventory and selection before analysis. All absolute paths can be
overridden with CLI arguments; run each script with `--help` for the complete list.

## Outputs

Only `training/Multi_IL_Full_Action_RL/analysis` is written:

```text
stage1_5_dataset_inventory.json
stage1_5_selected_datasets.json
stage1_5_dataset_report.json
stage1_5_policy_summary.csv
same_seed_outcomes.csv
same_seed_pattern_summary.csv
stage1_5_schema_report.txt
```

HDF5 files are opened with `h5py.File(path, "r")`. Numeric arrays are processed one episode and,
where needed, one chunk at a time. Pickles are inventoried but deliberately not deserialized because
unpickling can execute code; they cannot be automatically selected without a safe external format
conversion or explicit future adapter.

TwoArmTransport progress flags are used only when the real dataset provides named fields or an
explicit `progress_observation_schema` mapping. No hard-coded `object` indices are assumed. Missing
seed, success, next-state, progress, or other fields are reported as `missing_field`; they are never
fabricated from expected behavior.
