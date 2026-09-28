# Stage3-v6: Dual 2Q Target A/B

Stage3-v6 keeps the Stage3-v5 architecture, replay, handoff, Actor objective,
optimizers, UTD, policy delay, Polyak update and evaluation protocol unchanged.
The controlled variable is the **Critic Bellman target estimator**.

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
