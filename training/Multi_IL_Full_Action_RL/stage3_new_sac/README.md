# Stage3-new: paired standard SAC, RNN-Q versus Multi-Q

This independent route tests whether the two Stage2-new Critic initializations
change subsequent online policy optimization.  The first formal experiment is
RNN-Q versus Multi-Q only; Random-Q is intentionally deferred.  No Stage3-R,
RSAC, Actor distillation, Actor/Critic freeze, BC loss, high UTD, ensemble, or
Stage1 rollout replay is used.

## Shared standard SAC

The Actor is a randomly initialized RLKit `TanhGaussianPolicy`: 59-D canonical
low-dimensional observation, hidden layers `[256, 256]` with ReLU, state-dependent
log standard deviation clamped to `[-20, 2]`, and a 14-D post-tanh environment
action.  `shared/actor_init.pth` is created once; both processes strict-load the
same state dict and verify its hash.  Evaluation uses deterministic `tanh(mu)`.

The online and target Twin Critics use the exact Stage2-new public builder:

```
concat(state 59, action 14)
Linear(73,256) -> LayerNorm -> ReLU
Linear(256,256) -> LayerNorm -> ReLU
Linear(256,1)
```

Stage2 checkpoints are checked for dimensions, hidden sizes, ReLU, LayerNorm,
and gamma 0.99, then loaded with `strict=True`.  The target Critic starts as an
exact copy.  Online Critic optimization uses AdamW with weight decay `1e-4`;
Actor and alpha optimizers use zero weight decay.

The TD target and Actor objective are standard clipped-double-Q SAC with
automatic entropy tuning, target entropy `-14`, initial alpha `0.01`, soft target
updates (`tau=0.005`), and UTD=1.  After each environment transition the code
immediately inserts one online transition and, once replay has 1000 transitions,
performs one joint SAC update.  There are no episode-end update bursts.

## Expert/offline data

Each batch is composed by sampling two independent buffers: 50% uniform
transitions from the original TwoArmTransport proficient-human low-dimensional
demonstration HDF5 and 50% uniform transitions from online replay.  Stage1
BC-RNN/Transformer/GMM rollout HDF5 files are explicitly rejected because their
information must enter Stage3 only through Critic initialization.

The repository identifies the official Transport PH low-dim source through the
Pure-IL BC-RNN checkpoint, but does not contain the server HDF5 absolute path.
`prepare_stage3_new_pair.py --expert-checkpoint ...` reads the checkpoint's real
`train.data` at runtime, validates the HDF5 environment and schema, and records
the resolved path.  `--expert-dataset` is also supported when the exact path is
already known.  There is no observation normalization or extra action scaling.
Official robomimic demonstration files may contain additional low-dimensional
observation datasets; they are ignored while the required seven canonical keys
are selected in the fixed Stage1/Stage2 order to produce exactly 59 dimensions.
Stored robomimic demonstration `dones` are used as the offline terminal masks;
online time-limit truncations retain SAC bootstrap while genuine environment or
success terminals do not.

## Budget, evaluation, probes, and resume

The earlier `alpha_init=1.0` pair is retained unchanged as the
`alpha_init_1p0_diagnostic` run. It records the early soft-Q scale transient and
must not be deleted, overwritten, or resumed for the corrected experiment. The
new `alpha_init_0p01` pair starts again from the original Stage2 best Critics,
an empty online replay, fresh optimizers, and the exact original shared Actor
and seed manifest. Automatic entropy tuning remains enabled; only
`log_alpha` initialization changes from `log(1)` to `log(0.01)`.

The first run uses 300,000 actual environment transitions per group.  Evaluations
use 10 fixed seeds `20000..20009` at steps 0, 25k, 50k, 100k, 150k, 200k, and
300k.  Evaluation is deterministic and its transitions are never inserted into
replay.  Training reset number `k` uses seed `30000+k` in both processes.

Stage2's saved deterministic probe manifest is reused at steps 0, 1k, 5k, 10k,
25k, 50k, 100k, 200k, and 300k.  NPZ artifacts retain Q1/Q2/Qmin, deterministic
Actor actions, and both action gradients.  Pair comparison reports head-wise
absolute Q differences and Pearson/Spearman correlations, gradient cosines,
Actor-action L2 divergence, and the two success-rate learning curves.

Every SAC update logs reward mean/std/min/max, target-Qmin mean/std, entropy
bonus mean/std/min/max, final TD-target mean/std/min/max, current Q1/Q2/Qmin
scale, alpha/log-alpha/loss, target entropy, and policy entropy. Pair comparison
reports Q1/Q2 action-gradient cosine distributions (mean/std/median/p10/p90),
Actor divergence, absolute and relative Qmin scale differences, and
centered/z-score Pearson and Spearman geometry diagnostics. These are analysis
metrics only and do not alter training.

Milestone checkpoints and `latest.pth` contain all models, optimizers, alpha,
RNG state, episode/environment context, and a separate complete online replay
snapshot.  Resume restores them.  Extending to 500k requires explicitly resuming
both groups with the same new budget; it is never automatic.

## Local synthetic smoke test

```bash
python training/Multi_IL_Full_Action_RL/stage3_new_sac/validate_stage3_new.py
```

This exercises strict Stage2 loading, identical Actors, LayerNorm placement,
target copying/soft update, automatic alpha, per-transition replay, the 1000-step
threshold, UTD accounting, 50/50 sampling, SAC backward passes, evaluation
isolation, deterministic actions, checkpoint/resume, probes, gradients, and pair
comparison.  It does not use robosuite or server artifacts.

## Formal server run

Set the completed Stage2-new run and the retained alpha=1 diagnostic pair. A
new output directory is mandatory:

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
conda activate robosuite_npu
STAGE2_RUN=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage2_new_critic_pretraining/stage2new_formal_001
REFERENCE_PAIR_RUN=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_new_sac/stage3new_pair_002
PAIR_ID=stage3new_alpha0p01_001
python training/Multi_IL_Full_Action_RL/stage3_new_sac/prepare_stage3_new_pair.py \
  --run-id "$PAIR_ID" \
  --reference-pair-run-dir "$REFERENCE_PAIR_RUN" \
  --alpha-init 0.01 \
  --stage2-run-dir "$STAGE2_RUN" \
  --rnn-q-checkpoint "$STAGE2_RUN/rnn_q/checkpoints/best.pth" \
  --multi-q-checkpoint "$STAGE2_RUN/multi_q/checkpoints/best.pth" \
  --expert-checkpoint "/data/home/3220251075/lerobot_workspace/training_runs/Pure IL/two_arm_transport_bc_rnn_official_ph_low_dim/two_arm_transport_bc_rnn_official_ph_low_dim/20260811151701/models/model_epoch_1000_low_dim_v15_success_0.9.pth"
```

The prepare command prints `PAIR_RUN_DIR`.  In terminal 1 / NPU 0:

```bash
PAIR_RUN_DIR=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_new_sac/stage3new_alpha0p01_001
STAGE2_RUN=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage2_new_critic_pretraining/stage2new_formal_001
python -u training/Multi_IL_Full_Action_RL/stage3_new_sac/train_stage3_new.py \
  --group rnn_q --device npu:0 --pair-run-dir "$PAIR_RUN_DIR" \
  --critic-init-checkpoint "$STAGE2_RUN/rnn_q/checkpoints/best.pth" \
  2>&1 | tee "$PAIR_RUN_DIR/rnn_q/console.log"
```

In terminal 2 / NPU 1:

```bash
PAIR_RUN_DIR=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_new_sac/stage3new_alpha0p01_001
STAGE2_RUN=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage2_new_critic_pretraining/stage2new_formal_001
python -u training/Multi_IL_Full_Action_RL/stage3_new_sac/train_stage3_new.py \
  --group multi_q --device npu:1 --pair-run-dir "$PAIR_RUN_DIR" \
  --critic-init-checkpoint "$STAGE2_RUN/multi_q/checkpoints/best.pth" \
  2>&1 | tee "$PAIR_RUN_DIR/multi_q/console.log"
```

Resume both to 500k (run both commands, one per NPU):

```bash
python -u training/Multi_IL_Full_Action_RL/stage3_new_sac/train_stage3_new.py --group rnn_q --device npu:0 --pair-run-dir "$PAIR_RUN_DIR" --critic-init-checkpoint "$STAGE2_RUN/rnn_q/checkpoints/best.pth" --resume "$PAIR_RUN_DIR/rnn_q/checkpoints/latest.pth" --total-env-steps 500000
python -u training/Multi_IL_Full_Action_RL/stage3_new_sac/train_stage3_new.py --group multi_q --device npu:1 --pair-run-dir "$PAIR_RUN_DIR" --critic-init-checkpoint "$STAGE2_RUN/multi_q/checkpoints/best.pth" --resume "$PAIR_RUN_DIR/multi_q/checkpoints/latest.pth" --total-env-steps 500000
```

After both processes reach the same budget:

```bash
python training/Multi_IL_Full_Action_RL/stage3_new_sac/compare_stage3_new_pair.py --pair-run-dir "$PAIR_RUN_DIR"
```

Formal output is isolated under
`.../stage3_new_sac/<PAIR_ID>/{shared,rnn_q,multi_q,pair_comparison}`.

## Read-only terminal and Q-source audits

`audit_stage3_terminal_semantics.py` inspects the real expert HDF5 schema and
reports exactly how its stored `dones` become the SAC bootstrap mask. It does
not infer missing terminated or truncated fields. It also checks whether a
demo's final stored `next_obs` matches the next demo's first observation.

`audit_stage3_q_source_decomposition.py` loads a saved Stage3 checkpoint and
its referenced `.replay.npz`. The completed run is recoverable because every
checkpoint records `replay_path`, and `last.pth` points to `last.replay.npz`.
It compares dataset/behavior, current-policy, and fixed-seed random actions on
expert and online states. TD-target components call the same
`Stage3SAC.target_components` helper used by training. The scripts are
read-only with respect to models, datasets, checkpoints, and replay; they only
create files under `<PAIR_RUN>/audits/`.

## Stage3-new CQL-lite

`stage3_new_cqllite_config.json` enables the paired stabilized experiment while
the baseline config explicitly keeps `cql.enabled=false`. For both expert and
online halves, each replay action is the support anchor. Each state receives
one detached stochastic current-policy action and ten uniform actions sampled
from the real robosuite `action_spec` bounds. The two head penalties are
`logsumexp(Q_i(s, a_ood)) - Q_i(s, a_data)` and their sum is weighted by the
fixed `lambda=0.1` in the existing Critic optimizer. Actor loss, TD target,
automatic alpha, UTD, replay threshold, 50/50 sampling, and terminal semantics
are unchanged.

The CQL run must be prepared fresh with the vanilla alpha0p01 pair supplied as
`--reference-pair-run-dir`; this byte-copies only its original shared Actor and
seed manifest, while both Critics reload their Stage2 best checkpoints. Runtime
logs include raw/weighted CQL loss and support-vs-OOD Q statistics. Fixed
source-separated diagnostics are written every 5000 environment steps (plus
step 1000) to each group's `source_diagnostics.jsonl`.

Synthetic validation:

```bash
python training/Multi_IL_Full_Action_RL/stage3_new_sac/validate_stage3_cqllite.py
```

Preparation uses the existing CLI with the CQL config:

```bash
python training/Multi_IL_Full_Action_RL/stage3_new_sac/prepare_stage3_new_pair.py \
  --config training/Multi_IL_Full_Action_RL/stage3_new_sac/stage3_new_cqllite_config.json \
  --reference-pair-run-dir "$BASELINE_PAIR_RUN" --alpha-init 0.01 \
  --run-id "$PAIR_ID" --stage2-run-dir "$STAGE2_RUN" \
  --rnn-q-checkpoint "$STAGE2_RUN/rnn_q/checkpoints/best.pth" \
  --multi-q-checkpoint "$STAGE2_RUN/multi_q/checkpoints/best.pth" \
  --expert-checkpoint "$EXPERT_CHECKPOINT"
```

After both groups finish, run `compare_stage3_new_pair.py` for the within-CQL
pair and `compare_stage3_cql_baseline.py --cql-pair-run-dir ...
--baseline-pair-run-dir ...` for milestone comparison with vanilla SAC.

## CQL-lite + frozen Stage2 value-geometry anchor

`stage3_new_cqllite_anchor_config.json` adds a frozen copy of the corresponding
Stage2 Twin Critic. Student and teacher start from the same checkpoint. Every
Critic update independently samples 256 Stage2 training transitions and matches
each Q head after batch-wise z-scoring (`unbiased=False`, epsilon `1e-6`). The
raw anchor is the mean of the two head losses and has fixed weight `0.1`; it
never updates the Actor and never participates in TD or CQL.

The audited Stage2 loader and split are reused. RNN-Q samples BC-RNN only.
Multi-Q reuses the rotating policy-balanced RNN/Transformer/GMM sampler.
Validation transitions are diagnostics only. `step0_anchor_audit.json` verifies
student/teacher equality; `anchor_diagnostics.jsonl` records train/validation
correlations and fixed-probe success-minus-failure geometry.

Only `mujoco.FatalError` is recovered during physics stepping. The corrupt step
is discarded without consuming an environment step, the episode is excluded
from normal outcome statistics, and the incident is recorded in
`sim_fatal_errors.jsonl`. Five consecutive failures stop the run. Evaluation
retries an affected seed once and excludes persistent simulator errors from the
success-rate denominator.

```bash
python training/Multi_IL_Full_Action_RL/stage3_new_sac/validate_stage3_cqllite_anchor.py
```

## Frozen BC-RNN proposal handoff

`stage3_new_rnn_handoff_config.json` defines the separate main experiment that
tests whether RNN-only versus heterogeneous-IL Stage2 Critic initialization
improves action arbitration and downstream RL handoff. This is related to
existing IL-proposal / RL-proposal arbitration approaches; no novelty claim is
made for the basic proposal mechanism.

The SAC Actor remains the shared random feed-forward policy. A frozen official
robomimic `RolloutPolicy` loaded from the BC-RNN checkpoint advances exactly
once per real observation and resets at every episode. Target Twin-Q selects
between its deterministic proposal and the stochastic SAC proposal during
training (ties select RNN). Replay stores the executed action plus both proposal
actions, selector values, source, and the already-computed recurrent proposal
for the next state. The Bellman bootstrap repeats the proposal arbitration;
only an RL-winning bootstrap receives the SAC entropy term.

Expert recurrent proposals are precomputed once in temporal demo order into a
shared cache by pair preparation. The handoff Actor loss uses deterministic
Actor means only for online replay rows that actually selected RNN. CQL-lite is
unchanged and the value-geometry anchor is disabled. Phase A is deliberately a
no-update 20-seed arbitration report and is separate from formal 300k training.

Synthetic validation:

```bash
python training/Multi_IL_Full_Action_RL/stage3_new_sac/validate_stage3_handoff.py
```

## Critic-protected progressive unfreezing

`stage3_new_rnn_handoff_progressive_config.json` keeps both Stage2 online and
target Critics bit-exactly frozen before 10k valid environment steps. In that
protected phase the Actor receives only stored-proposal handoff imitation and
alpha stays fixed at 0.01. From 10k through 30k, Critic LR and target Polyak tau
increase linearly from zero to their unchanged base values. Standard SAC Actor
updates, automatic entropy tuning, CQL-lite, handoff imitation, and the existing
hybrid bootstrap run from 10k onward. Detailed milestone diagnostics remain;
`train_metrics.jsonl` writes one scalar aggregate per 100 update attempts.
