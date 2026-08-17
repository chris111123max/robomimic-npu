# Experiment 00: Heterogeneous IL policy complementarity

This is a read-only policy evaluation experiment. It does **not** train BC or RL models, create a critic, alter rewards, modify checkpoints, ensemble actions, or switch policies during an episode.

## Scientific question

The experiment asks whether BC, BC-GMM, BC-GMM-RNN, and BC-GMM-Transformer policies trained from the same demonstrations have complementary task competence on TwoArmTransport. Each policy independently executes one complete episode for each initial condition. With the formal configuration this is 100 initial conditions × 4 policies = 400 rollouts.

Checkpoint success values embedded in filenames are not used as measurements. Every success rate is recomputed on this experiment's paired state bank.

## Why paired simulator states are required

Using the same integer seed separately is not treated as proof of an identical physical initial condition. `build_initial_state_bank.py` uses the dataset's environment metadata, resets the environment, and persists the official `EnvRobosuite.get_state()` representation: MuJoCo model XML, flattened simulator state, and robosuite episode metadata. It also stores the complete low-dimensional initial observation.

Every policy later starts by loading the same file and calling `EnvRobosuite.reset_to(...)`. Before inference, the evaluator verifies:

- the persisted full-state, state-vector, and observation hashes against the manifest;
- exact equality of the restored simulator state vector;
- exact equality of the restored hash for all seven observation keys declared by the checkpoints.

State-bank construction compares two independently repeated `reset_to` operations, both started from the same deterministic environment RNG stream. It intentionally does not require every extra diagnostic observation returned by the raw environment (such as unused joint velocities or site quaternions) to match the transient cache produced by the preceding random `reset()`. Those fields are not policy inputs. The policy adapter filters inputs to the checkpoint-declared keys, while trajectories may retain the full raw low-dimensional environment observation for later diagnostics.

This includes TwoArmTransport model randomization encoded in model XML and `ep_meta`; raw qpos/qvel are not manipulated directly. State-bank and manifest writes are atomic and resumable.

## Policies

The four checkpoint paths live only in `config/experiment_config.json`, under the stable output names `bc`, `bc_gmm`, `bc_gmm_rnn`, and `bc_gmm_transformer`.

`utils/policy_loader.py` calls the official `FileUtils.policy_from_checkpoint`. Architecture, observation configuration, normalization, GMM evaluation behavior, RNN state, and Transformer history therefore come from each original checkpoint. Every rollout calls `RolloutPolicy.start_episode()`, whose project implementation switches to evaluation mode and calls the underlying policy's `reset()`. Temporal state cannot leak between episodes.

The unified loader also reproduces checkpoint-defined rollout frame stacking at the policy-input boundary. In particular, the Transformer checkpoint declares `train.frame_stack=10` and `context_length=10`, so its first action receives ten copies of the restored initial observation and later actions receive the official sliding ten-frame window. This history is cleared by every `start_episode()`. The shared environment remains unwrapped for strict simulator-state verification, and saved trajectories remain comparable raw single-frame low-dimensional data.

The standalone validation and state-bank stages initialize robomimic's process-global observation modality registry from the BC checkpoint config before the first `EnvRobosuite.reset()`. This mirrors the initialization side effect of the official policy-first evaluator while still creating the common environment from dataset metadata.

Checkpoint compatibility does not require byte-identical metadata because checkpoint and dataset files can legitimately carry different environment-version, rendering, or optional-default fields. Validation strictly checks the TwoArmTransport name and environment type plus shared robot, controller, control-frequency, and gripper semantics. It then validates every checkpoint's observation keys and shapes, action dimension, horizon, and one native inference result against the single dataset-metadata environment. Non-semantic metadata differences are printed for auditability.

Policy sampling gets a deterministic per-(policy, initial-state) RNG seed, independent of evaluation order. This preserves the checkpoint's native stochastic evaluation semantics; it does not replace GMM sampling with a mean action. Environment initialization remains fixed by the restored state bank.

## Files

- `launch.py`: validation and unified `validate`, `build`, `eval`, `analyze`, and `all` stages.
- `scripts/build_initial_state_bank.py`: fixed seed manifest and atomic simulator state bank.
- `scripts/evaluate_policy_bank.py`: one-policy-at-a-time paired rollouts, trajectories, errors, and resume.
- `scripts/analyze_complementarity.py`: success matrix, policy summary, oracle, rescue, and overlap analysis.
- `utils/env_utils.py`: dataset environment construction, seeding, official restore, and success checks.
- `utils/policy_loader.py`: official native policy loading, device selection, and RNG isolation.
- `utils/result_utils.py`: fingerprints and atomic JSON, NPZ, and CSV persistence.

## Metrics

The **Episode-Level Selection Oracle** succeeds on an initial condition when at least one of the four independently executed policies succeeds. It is only a success-set union and a measure of complementarity potential. It is not an executable policy, state-level switching method, or ensemble.

`Oracle Gap = Episode Oracle Success Rate - Best Single Policy Success Rate`.

The best single policy is selected from this paired evaluation, not assumed from checkpoint filenames. For every other policy, unique rescue relative to the best counts `other succeeds AND best fails`. Conditional rescue divides that count by the number of best-policy failures. Rescue cases from different policies may overlap and must not be summed to estimate Oracle Gap.

`rescue_matrix.csv` contains both-success, both-fail, directional rescue, directional miss, and agreement counts and rates for all 16 ordered policy pairs.

## Outputs and resume

Runs are written outside the source tree under:

`/data/home/3220251075/lerobot_workspace/training_runs/Heterogeneous_IL_RL/experiment_00_policy_complementarity/<timestamp>/`

Each episode writes its compressed low-dimensional trajectory and per-episode JSON atomically, then rebuilds aggregate result and error CSV files atomically. A completed pair is skipped only when its result hash and trajectory structure validate. Runtime exceptions are stored with tracebacks as errors, never converted to failures. Analysis marks `ERROR` / `MISSING`, writes an incomplete summary, and returns nonzero instead of presenting incomplete data as a valid 100×4 experiment.

Use `--run-dir <existing-run>` to resume. Do not pass `--force-rebuild` or `--force-eval` unless intentional recomputation is desired. A smoke run and formal run must use separate newly created timestamp directories.

When all 400 pairs are valid, inspect:

- `analysis/success_matrix.csv`
- `analysis/policy_summary.csv`
- `analysis/rescue_matrix.csv`
- `analysis/complementarity_summary.json`

Trajectories contain low-dimensional observations, actions, rewards, environment-done flags, task-success flags, and step indices. Simulator states can be added per step with `trajectory.save_sim_states`, which defaults to `false` to limit storage.

## Server usage

Initialize the `robosuite_npu` environment and CANN first, export the project `PYTHONPATH`, and run `launch.py`. `ASCEND_RT_VISIBLE_DEVICES` may be set before launch; the project NPU helper then sees that physical device as logical `npu:0`.

Examples:

```bash
python -u training/Heterogeneous_IL_RL/experiment_00_policy_complementarity/launch.py --stage validate
python -u training/Heterogeneous_IL_RL/experiment_00_policy_complementarity/launch.py --stage all --num-seeds 2
python -u training/Heterogeneous_IL_RL/experiment_00_policy_complementarity/launch.py --stage all --num-seeds 100
```

To resume, repeat the applicable stage and supply the printed run directory:

```bash
python -u training/Heterogeneous_IL_RL/experiment_00_policy_complementarity/launch.py \
  --stage all --run-dir /absolute/path/to/existing/run --num-seeds 100
```
