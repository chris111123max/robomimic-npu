# Multi-IL + Full-Action RL

**STATUS: Stage 1 and 1.5 complete; Stage 2.0 target-policy sanity check implemented; Critic training not started**

This is a separate long-lived research project. It does not reuse or overwrite `training/IL+RL`,
and it never modifies the three Pure IL checkpoints.

## Fixed research route

1. **Stage 1:** heterogeneous BC-GMM, BC-RNN-GMM, and BC-Transformer-GMM rollout collection.
2. **Stage 2:** equal-budget Single-IL / Multi-IL critic pretraining and value anchoring.
3. **Stage 3:** initialize a SAC-compatible actor from the strongest IL policy.
4. **Stage 4:** non-residual full-action online SAC.

The final execution rule is `a_exec = a_SAC`, not `a_IL + delta_a`. Stage 2, Stage 3, and Stage 4
are intentionally placeholders; this implementation does not train a critic, actor, or SAC agent.

Stage 1.5 is a strictly read-only bridge between collection and later training. It recursively
discovers runtime artifacts from `training_runs`, validates candidates from their contents, selects
the formal three-policy 100-seed family only when the choice is unambiguous, and streams an audit of
schema, seeds, outcomes, progress flags, sparse rewards, and NaN/Inf values. It performs no rollout,
training, reward modification, transition-budget choice, or critic preparation. See
`stage1_5_dataset_audit/README.md` for commands and output definitions.

Stage 2.0 provides only a five-seed frozen BC-RNN target-policy sanity check. It reuses the official
checkpoint `RolloutPolicy`, freezes every network parameter, replays stored RNN histories, and checks
numerical behavior on Transformer and GMM histories. It creates no replay buffer, critic, actor, SAC
agent, rollout, or model update. Run `stage2_critic_pretraining/validate_frozen_rnn_target.py`; its
only result is written under `analysis/stage2`.

## Stage 1 implementation

The collector reuses the official robomimic path:

- `FileUtils.policy_from_checkpoint` restores the native `RolloutPolicy`, observation/action
  normalization, and checkpoint model.
- `FileUtils.env_from_checkpoint` reconstructs the checkpoint environment and applies config
  wrappers.
- `policy.start_episode()` resets BC-RNN hidden state before every episode.
- BC-Transformer retains its native 10-frame `FrameStackWrapper` context.
- the stored action is exactly the full action passed to `env.step`.

The policy input and dataset state are deliberately different concepts. A policy receives its
native checkpoint input (including normalization or frame stack). The dataset always stores the
same canonical current-frame low-dimensional environment keys:

```text
robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos,
robot1_eef_pos, robot1_eef_quat, robot1_gripper_qpos, object
```

RNN hidden state, Transformer embeddings, and GMM parameters are not stored as critic state.

### TwoArmTransport progress flags

The canonical `object` vector already contains both `payload_in_target_bin` and
`trash_in_trash_bin`. They are native TwoArmTransport object-modality observables; robosuite
concatenates that modality into `object-state`, and `EnvRobosuite` exposes the same vector as
canonical `object`. Therefore Stage 1 does **not** duplicate them as transition datasets.

Their flat indices must not be hard-coded because the observable order differs between robosuite
versions. At collection time, Stage 1 derives both indices from the installed environment's active
object-observable order and writes the result to the HDF5 root attribute
`progress_observation_schema` and each policy's `metadata.json`. A downstream analyzer can recover
either boolean at transition `t` as:

```python
schema = json.loads(hdf5_file.attrs["progress_observation_schema"])
index = schema["fields"]["payload_in_target_bin"]["flat_index"]
payload_in_target_bin = bool(episode["obs/object"][t].reshape(-1)[index])
```

The same applies to `next_obs/object` and `trash_in_trash_bin`. The validator requires both mapped
columns to contain only `0.0` or `1.0`. These flags are analysis-only: collection does not alter the
environment reward, construct shaped reward, stop on either partial flag, or change policy rollout.

## Same-seed initial conditions

Stage 1 does more than call `numpy.random.seed`:

1. seed Python, NumPy, Torch, and the visible NPU/CUDA runtime;
2. call an environment `seed` API when one is exposed;
3. reset the reference checkpoint environment once;
4. save its model XML, episode metadata, and flattened simulator state;
5. reset every policy-specific environment to that exact saved state;
6. require exact state-vector equality and compare canonical initial observations.

Consequently, same-seed policies share the same persisted simulator state, object randomization,
robot state, and model XML. `initial_states.hdf5` and `initial_state_manifest.json` retain the audit
trail.

Remaining limitations are explicit:

- robosuite does not expose a universal environment seed API in every version;
- NPU kernels and third-party simulator internals can still have platform-level nondeterminism;
- GMM sampling is controlled by the recorded policy seed, but exact reproducibility still depends
  on the installed Torch/CANN versions;
- only canonical observations are compared; unobserved simulator caches are represented only to
  the extent supported by official `get_state` / `reset_to`.

## Transition schema

Every policy gets its own `transitions.hdf5`. Each episode contains equal-length transition arrays:

```text
obs/<key>, actions, rewards, next_obs/<key>,
dones, terminated, truncated,
policy_id, episode_id, initial_seed, timestep,
episode_success, episode_return, episode_length
```

Optional `mc_return` is written only when a gamma is explicitly configured. It never replaces raw
environment reward.

The HDF5 root also contains `progress_observation_schema`, which maps the two Transport progress
booleans into `obs/object` and `next_obs/object`. No duplicate progress arrays are written.

For this fixed-horizon robosuite wrapper, `terminated` is the raw environment `done`. `truncated`
means the collector ended at its configured horizon or stopped after success while raw `done` was
false. `dones = terminated OR truncated`, so the last saved transition is always an episode boundary.
Both successful and failed episodes are retained.

## Output layout

Runtime artifacts live outside the source tree under the workspace-level `training_runs` directory.
Each run is isolated by timestamp:

```text
/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/
└── stage1_rollout_collection/
    ├── datasets/<run-id>/
    │   ├── initial_states.hdf5
    │   ├── seed_list.json
    │   ├── same_seed_outcomes.json
    │   ├── collection_summary.json
    │   ├── bc_gmm/
    │   │   ├── transitions.hdf5
    │   │   ├── episodes.json
    │   │   └── metadata.json
    │   ├── bc_rnn/
    │   └── bc_transformer/
    ├── runs/<run-id>/
    │   ├── run_manifest.json
    │   ├── initial_state_manifest.json
    │   ├── collection_summary.json
    │   ├── <policy>_metadata.json
    │   └── worker_shards/
    │       ├── logs/worker_00.log ... worker_03.log
    │       └── seeds/
    └── launcher_logs/
```

The final merged dataset retains the original single-run layout. Worker HDF5/JSON shards are
deleted automatically after a successful merge; only worker logs and seed assignments remain:

```text
datasets/<run-id>/
├── initial_states.hdf5
├── seed_list.json
├── same_seed_outcomes.json
├── collection_summary.json
├── bc_gmm/
│   ├── transitions.hdf5
│   ├── episodes.json
│   └── metadata.json
├── bc_rnn/
└── bc_transformer/

runs/<run-id>/
├── run_manifest.json
├── initial_state_manifest.json
├── seed_list.json
├── collection_summary.json
└── <policy>_metadata.json
```

`same_seed_outcomes.json` and `collection_summary.json` report all `000` through `111` success
patterns in policy order `(GMM, RNN, Transformer)`. This is descriptive only; Stage 1 performs no
gating, ranking, pair mining, voting, or policy selection.

## Smoke test

The smoke test runs three seeds for every policy and validates the generated datasets:

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
NPU_ID=0 bash training/Multi_IL_Full_Action_RL/scripts/run_stage1_smoke_test.sh
```

It writes only under `/tmp/multi_il_full_action_rl_stage1_smoke`. The log prints policy ID,
checkpoint, algorithm type, observation keys, action dimension, episode length, return, success,
and termination reason.

## Formal collection

The formal collector partitions seeds round-robin across four independent NPU workers and merges
their shards back into one standard dataset. The number of episodes is a runtime argument:

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
NPU_IDS=0,1,2,3 bash training/Multi_IL_Full_Action_RL/scripts/run_stage1_collection.sh 100 10000
```

Arguments are `NUM_EPISODES` and optional `SEED_START`. For a custom explicit seed list, invoke the
collector directly with `--seed-list path/to/seeds.json`.

Long-running collection can be detached safely:

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
LOG_ROOT="/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage1_rollout_collection/launcher_logs"
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/multi_il_stage1_$(date +%Y%m%d_%H%M%S).out"
nohup env NPU_IDS=0,1,2,3 bash training/Multi_IL_Full_Action_RL/scripts/run_stage1_collection.sh 100 10000 \
  > "$LOG" 2>&1 < /dev/null &
echo "PID=$!"
echo "LOG=$LOG"
```

## Standalone validation

Validation requires only Python, NumPy, and h5py; it does not load policies or robosuite:

```bash
python training/Multi_IL_Full_Action_RL/stage1_rollout_collection/validate_dataset.py \
  --dataset-root /data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage1_rollout_collection/datasets/<run-id>
```

It verifies lengths, episode continuity, IDs, seeds, action shape, shared canonical observation
schema, NaN/Inf absence, metadata consistency, same-seed state hashes, outcome statistics, and that
no success filter was enabled. `--require-both-outcomes` may be used for a large formal run when
both natural successes and natural failures are expected; it is intentionally not mandatory for a
three-episode smoke test.
