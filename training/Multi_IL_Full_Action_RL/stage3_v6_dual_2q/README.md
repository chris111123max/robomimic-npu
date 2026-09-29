# Stage3-v6: Dual 2Q Target A/B

Stage3-v6 keeps the Stage3-v5 architecture, replay, Actor objective,
optimizers, UTD, policy delay, Polyak update and evaluation protocol unchanged.
The mean-vs-random controlled variable is the **Critic Bellman target estimator**.
Readiness V2 is a V6-only handoff change shared by both target modes; Stage3-v5
readiness behavior is not modified.

## Target mechanisms

`mean2q`:

```text
expected_next = sum_k p'_k * 0.5 * (Q1'(h', mu'_k) + Q2'(h', mu'_k))
target = r + gamma * (1-terminal) * expected_next
```

`random2q`:

```text
J ~ Uniform{1,2} once per Critic optimizer update
expected_next = sum_k p'_k * QJ'(h', mu'_k)
target = r + gamma * (1-terminal) * expected_next
```

The selected `J` applies to the entire 256-transition Critic minibatch.
Both online Q networks regress the same scalar target. Hard clipped-min is
never used as the Stage3-v6 training target.

Random-one uses a dedicated NumPy Generator. It does not consume global Python,
NumPy, Torch or NPU training RNG, so the target selector itself does not shift
environment / replay / Actor random streams. Its RNG state and counts are saved
in every checkpoint and restored exactly.

Diagnostics outside a Critic optimizer update use the selector expectation
`0.5*(Q1'+Q2')` and do not advance the random-one selector.

The Actor objective is intentionally unchanged from Stage3-v5 (Q1
component-mean objective). This prevents the formal mean-vs-random comparison
from changing both the Critic target and Actor objective at once.

## Fixed four-NPU mapping

```text
NPU0 = mean2q   / multi_q
NPU1 = mean2q   / rnn_q
NPU2 = random2q / multi_q
NPU3 = random2q / rnn_q
```

The trainer validates this mapping and refuses a mismatched device assignment.

## Startup order

The four jobs are **not** allowed to create simulator environments at the same
time. The launcher uses a two-wave startup barrier:

```text
Wave 1:
  NPU0 = mean2q   / multi_q
  NPU2 = random2q / multi_q

Wait until BOTH Multi trainers report READY after all of their requested
simulator envs and trainer runtime have initialized.

Wave 2:
  NPU1 = mean2q   / rnn_q
  NPU3 = random2q / rnn_q
```

For formal runs, each trainer still uses 16 envs with
`startup_parallelism=4`, i.e. its own simulator creation proceeds
4 + 4 + 4 + 4. The second RNN wave is launched only after both Multi runs have
completed this startup phase. A startup-ready marker is written by the trainer
and validated by the launcher; no fixed sleep is used.

## Workflow

First prepare one immutable four-run directory with
`prepare_stage3_v6_quad.py`. It contains a shared Actor source, Stage2.2
Critic-source manifest, seed manifest and four run directories.

Then launch all four jobs with:

```bash
python training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/launch_stage3_v6_4npu.py \
  --quad-run-dir <prepared_stage3v6_run_dir>
```

For a four-card smoke run:

```bash
python training/Multi_IL_Full_Action_RL/stage3_v6_dual_2q/launch_stage3_v6_4npu.py \
  --quad-run-dir <prepared_stage3v6_run_dir> \
  --smoke
```

Formal runs use 16 simulator envs per training process, matching Stage3-v5.

## Critic Readiness V2 (formal)

Prerequisite `DATA_READY`: at least 100K env steps, 150 completed episodes,
30 successes and 30 failures. Hard conditions are `RANK_READY` (Spearman of
Qmean vs finite episode MC return >= 0.70), `TD_HEALTHY` (fixed-set Emax finite
and no two consecutive 10K Emax increases each >= 35%; first two checks are
NOT_READY), and `NUMERIC_SAFE` (finite Q/target/TD and recorded loss/grad norm;
Qmean mean shift <= 0.5 previous/reference standard deviations, symmetric
standard-deviation ratio <= 1.5). Two consecutive full passes enter
`ACTOR_WARMUP`. The Stage2.2-initialized Critic is evaluated on the first
frozen readiness set for the Qmean reference; the current 100K Critic is never
used as a surrogate. Mean and random share identical gate code and thresholds;
readiness does not advance random selector RNG.

Twin disagreement, AUC, old TD plateau, and OOD remain diagnostic-only. An
unready Critic at 300K produces a warning and continues `CRITIC_ONLY`; a true
NUMERIC_SAFE catastrophe remains fatal. Legacy `critic_readiness` config keys
exist solely for the old diagnostic calculations. The active gate is
`critic_readiness_v2`.

For the two-card Multi-only formal variant, use `--manual-multi-two-card` with
mean2q/multi_q on logical `npu:0` and random2q/multi_q on logical `npu:1`.
Launch each trainer manually from an isolated runtime working directory; wait
for mean's verified 16/16 READY marker before launching random. No RNN trainer
or four-task launcher is part of this variant.
