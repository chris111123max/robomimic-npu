# Codex Task: Stage3-v5 Critic Ranking Drift — Autonomous Root-Cause Loop (max 4 iterations)

## Mission

Work autonomously on the current Stage3-v5 **Multi Critic ranking-drift** investigation.

Your job is not only to run one existing test. You must execute a closed diagnostic loop:

1. read all relevant existing evidence;
2. form one narrow causal hypothesis;
3. run the smallest decisive test;
4. read the produced JSON / logs completely;
5. decide whether the evidence is sufficient;
6. if not, design and implement the next diagnostic **under the testing directory only**;
7. repeat for at most **4 diagnostic iterations total**.

If a sufficiently specific causal mechanism is established before iteration 4, **stop early** and write the final report. Do not continue merely to reach four iterations.

The goal is to explain why Stage3-v5 Critic learning regresses from the good Stage2.2 MC-return geometry after Stage3 Bellman updates.

---

# 0. Hard safety / scope rules

Repository:

```text
/data/home/3220251075/lerobot_workspace/robomimic
```

Branch:

```text
fix/stage3-v5-horizon10-critic
```

Use:

```text
npu:0
```

There is a pre-existing local modification:

```text
M fusion_result.json
```

**Never touch it.**

Forbidden:

```bash
git reset --hard
git clean
git checkout -- fusion_result.json
git restore fusion_result.json
git add -A
git add .
```

Do not modify any production Stage1 / Stage2.2 / Stage3 source file.

All new or modified diagnostic source code, notes, reports, helper scripts, and task-state files must live under:

```text
training/Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3/testing/
```

Do not place new diagnostic code elsewhere.

Do not start:

- robosuite environment rollouts;
- formal Stage3 training;
- Actor training;
- new data collection;
- checkpoint-producing training jobs.

Diagnostic in-memory Critic optimizer steps are allowed when required by a causal test.

Never write or overwrite formal training checkpoints.

Do not alter existing training checkpoints or replay sidecars.

Do not automatically expand into unrelated experiments.

---

# 1. Environment

Use:

```bash
WORKSPACE="/data/home/3220251075/lerobot_workspace"

source "$WORKSPACE/miniconda3/etc/profile.d/conda.sh"
conda activate robosuite_npu

source "$WORKSPACE/Ascend/ascend-toolkit/set_env.sh"

export PYTHONPATH="$WORKSPACE/robomimic:$WORKSPACE/robomimic/rlkit:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

cd "$WORKSPACE/robomimic"
```

Before doing anything else:

```bash
git status --short
git branch --show-current
git log -8 --oneline
```

The current branch must remain:

```text
fix/stage3-v5-horizon10-critic
```

---

# 2. Fixed artifacts

Stage2.2 Multi checkpoint:

```text
/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/stage2_2_h10_multi_20260923_150749/multi_q/checkpoints/step_00005000.pth
```

Stage3-v5 run:

```text
/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3/stage3v5_h10_20260923_161337
```

Stage3 step0:

```text
$RUN/multi_q/checkpoints/step0_transfer.pth
```

Stage3 canonical replay:

```text
$RUN/multi_q/checkpoints/step_0200000.sequences.npy
```

Canonical paired-test defaults unless a test has a compelling reason otherwise:

```text
seed = 20260926
Critic batch = 256
probe size = 8192
probe batch = 1024
milestones = 0,100,250,500,1000,2000
```

If only probe evaluation OOMs, reduce probe batch to 512 or 256. Do not silently change the optimization batch, replay schedule, seed, or update count.

---

# 3. Facts already established — do not retest without a concrete contradiction

Treat the following as the current evidence base.

## 3.1 Stage2.2 readiness

Stage2.2 Multi on the canonical replay has approximately:

```text
Spearman ≈ 0.717
Pearson  ≈ 0.946
AUC      ≈ 0.992
```

Stage3 step0 is identical to Stage2.2.

Stage3 later degrades:

```text
100K Spearman ≈ 0.552
200K Spearman ≈ 0.418
```

Q1 and Q2 become more mutually consistent while MC ranking worsens.

Therefore ordinary twin divergence is not the explanation.

## 3.2 Implementation contract

The Stage3 TD target implementation, history slicing, reset semantics, terminal/truncation mask, transfer/loading contract, and target/online checkpoint relationships were audited.

No material implementation mismatch was found.

Do not reopen these unless new evidence directly contradicts them.

## 3.3 Target Actor continuation

Replay-next-action and target-Actor component-mean targets are almost identical on the canonical diagnostic set.

The target-Actor contribution is tiny compared with the growing target-Critic bootstrap mismatch.

Therefore target Actor continuation is low priority.

## 3.4 Moving target feedback

A frozen-target causal fork strongly reduced the ranking degradation relative to normal Polyak feedback.

This is strong evidence that moving target-Critic feedback matters.

It does not by itself identify the deepest cause.

## 3.5 Target timescale

Production `tau=0.005` is standard SAC/TD3-like behavior.

Slowing the target by about 10x only slightly helped and did not solve the problem.

Do not treat "tau is simply too fast" as the primary explanation.

## 3.6 clipped double-Q min

A paired min-vs-mean test showed:

- removing `min(Q1,Q2)` reduces negative bias and modestly protects ranking;
- mean-target still clearly drifts;
- target Critic ranking still declines.

Therefore `min` is a secondary pessimistic seed / amplifier, not a necessary cause.

Do not spend an iteration merely retesting min vs mean.

## 3.7 Stage2.2 initial Bellman closure

The latest pure-forward test:

```text
testing/test_stage2_bellman_closure.py
```

showed approximately:

```text
Qmean MC MAE      ≈ 0.03204
Qmean closure MAE ≈ 0.00336

Qmin MC MAE       ≈ 0.03243
Qmin closure MAE  ≈ 0.00393
```

Self-bootstrap at step0 is not worse than direct Q against MC.

Current and successor MC regression errors are highly correlated (~0.99 Pearson), and the closure residual is exactly reconstructed by:

```text
(Q_t - G_t) - gamma * (Q_next - G_next)
```

Therefore a large initial one-step Bellman-closure mismatch is **not supported**.

Finite-history non-Markov / aliasing is not proven and is currently low priority.

---

# 4. Central unresolved causal question

The unresolved mechanism is now narrow:

```text
good Stage2.2 MC geometry
        ↓
repeated learned-Q Bellman updates
        ↓
moving target Critic changes
        ↓
future learned-Q targets change
        ↓
online Q1/Q2 jointly drift away from empirical MC returns
        ↓
Polyak feeds that geometry back into future targets
```

We need to distinguish:

### Hypothesis A — learned-Q bootstrap feedback is the main causal source

The moving learned target generates and amplifies approximation bias.

### Hypothesis B — generic repeated optimization / finite replay projection is enough

Even with exact MC supervision, repeatedly updating this network on the fixed canonical replay would substantially degrade the original MC ranking.

### Hypothesis C — both contribute

Exact MC supervision also drifts somewhat, but learned-Q bootstrap causes most of the damage.

---

# 5. Iteration protocol

Maximum: **4 diagnostic iterations**.

An "iteration" means:

1. state one narrow hypothesis;
2. inspect/reuse existing code if possible;
3. run one decisive diagnostic;
4. fully read the result JSON / logs;
5. write an iteration report;
6. decide STOP or CONTINUE.

Do not count syntax checks as an iteration.

Do not run multiple broad branches in one iteration unless they are the two paired arms of one causal test.

Each iteration must create/update a report:

```text
training/Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3/testing/CODEX_STAGE3_ROOT_CAUSE_REPORT.md
```

Append:

```text
## Iteration N
Hypothesis
Test
Validity
Key numbers
Interpretation
Decision: STOP / CONTINUE
Next hypothesis if CONTINUE
```

Also maintain a machine-readable state file:

```text
training/Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3/testing/codex_stage3_root_cause_state.json
```

Suggested fields:

```json
{
  "iterations_completed": 0,
  "stopped_early": false,
  "current_best_causal_statement": "",
  "confidence": "",
  "remaining_alternatives": [],
  "iteration_results": []
}
```

Do not put large raw arrays into this state file.

---

# 6. Iteration 1 — MUST begin with the existing bootstrap-vs-oracle experiment

Existing test:

```text
training/Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3/testing/test_stage3_bootstrap_vs_oracle_mc_causal.py
```

Expected result path:

```text
$RUN/testing/stage2_vs_stage3_readiness/multi_bootstrap_vs_oracle_mc_causal.json
```

A prior task may already have run this experiment accidentally.

**First inspect the existing JSON if it exists. Do not automatically rerun it.**

### 6.1 Validate the existing result

The result is usable only if all relevant validity checks are true, including:

```text
same_initial_online_hash
same_initial_target_critic_hash
same_initial_actor_hash
same_initial_target_actor_hash
same_initial_probe_qmin_spearman
target_critic_changed_both
actor_unchanged_both
target_actor_unchanged_both
same_initial_optimizer_state
initial_contract_hashes_match
canonical_mc_identity_max_abs_le_2e_6
oracle_target_equals_mc_max_abs_le_2e_6
first_batch_mc_identity_max_abs_le_2e_6
terminal_semantics_match
```

Also verify branch completion / failures.

If the JSON is valid and contains the intended paired experiment, use it as Iteration 1.

If it is missing, invalid, stale, or does not correspond to the current code/contract, run:

```bash
RUN="/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3/stage3v5_h10_20260923_161337"

STAGE2="/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/stage2_2_h10_multi_20260923_150749/multi_q/checkpoints/step_00005000.pth"

CANONICAL="$RUN/multi_q/checkpoints/step_0200000.sequences.npy"

python training/Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3/testing/test_stage3_bootstrap_vs_oracle_mc_causal.py   --stage3-run-dir "$RUN"   --stage2-checkpoint "$STAGE2"   --diagnostic-replay "$CANONICAL"   --device npu:0   --updates 2000   --batch-size 256   --probe-size 8192   --probe-batch-size 1024   --seed 20260926
```

### 6.2 Meaning of the branches

Bootstrap branch intentionally isolates moving learned-Q feedback:

```text
y =
r + gamma*(1-terminal)
    * mean(Q1_target,Q2_target)(successor_history, replay_next_action)
```

It deliberately removes:

- target Actor continuation;
- clipped twin min.

Oracle branch:

```text
y = G_t
```

Both branches keep:

- identical step0;
- identical optimizer state;
- identical deterministic batch schedule;
- identical gradient-step count;
- identical Polyak update;
- identical probe.

### 6.3 Iteration-1 interpretation

If oracle remains close to initial MC geometry while moving bootstrap substantially degrades:

```text
Strongly supports learned-Q bootstrap feedback as the dominant causal source.
```

If both degrade similarly:

```text
Generic repeated optimization / replay projection remains a major cause.
```

If oracle degrades mildly but bootstrap much more:

```text
Both contribute, with learned-Q feedback the dominant amplifier.
```

Do not infer from one final number alone. Inspect the trajectory at:

```text
0 / 100 / 250 / 500 / 1000 / 2000
```

and compare:

- Qmin Spearman;
- Qmean Spearman;
- Qmin MAE;
- Qmin signed bias;
- target Critic Spearman;
- branch completion/non-finite behavior.

---

# 7. Early-stop criterion after Iteration 1

You MAY stop after Iteration 1 only if the causal evidence is already unusually decisive.

A valid early stop requires all of:

1. oracle-MC branch remains approximately stable or improves under the same repeated optimizer steps;
2. learned-Q bootstrap branch shows a large, systematic loss of MC ranking / increase in MAE;
3. the separation appears across multiple milestones, not just one noisy point;
4. paired validity is clean;
5. existing frozen-target evidence is directionally consistent;
6. no plausible generic-optimization explanation remains at comparable magnitude.

If these are satisfied, write a final causal statement such as:

```text
The dominant mechanism is repeated moving learned-Q bootstrap feedback:
the Critic is stable under exact MC supervision but drifts when its own moving
target estimates become the supervision source. This creates a self-reinforcing
projection / approximation-bias loop.
```

Do not overclaim a mathematical theorem or unique universal cause.

If evidence is not this strong, continue.

---

# 8. Iterations 2–4 — adaptive decision tree

Do not pre-run every test below.

Choose the next test based on the previous result.

All new files must be under:

```text
training/Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3/testing/
```

Prefer reusing helpers from existing causal tests rather than duplicating production logic.

## Path A — Oracle stable, moving bootstrap drifts

This is the most likely path.

### Preferred Iteration 2

Test:

```text
moving qmean replay bootstrap
vs
fixed-step0 qmean replay bootstrap
```

Purpose:

Isolate whether the damaging factor is specifically **feedback from an evolving target Critic**, rather than using an approximate Q target at all.

Both branches:

- same step0;
- same optimizer;
- same replay-next action;
- same qmean aggregation;
- same schedule;
- same 2000 updates;
- Actor unused;
- min unused.

Difference:

```text
A: target Critic receives production Polyak updates
B: target Critic remains exactly frozen at step0
```

This is cleaner than the older frozen-target test because it removes target Actor and min from both arms.

Suggested filename:

```text
testing/test_stage3_moving_vs_fixed_qmean_replay_bootstrap.py
```

Strong result:

```text
moving drifts strongly;
fixed-step0 remains near the initial geometry.
```

If this happens together with stable oracle MC, the causal chain is strong enough to STOP EARLY:

```text
exact MC stable
fixed learned-Q teacher stable
moving learned-Q teacher unstable
=> evolving self-bootstrap feedback is the dominant mechanism.
```

### If fixed-step0 also drifts strongly

Then approximate-Q supervision itself, not only moving feedback, may be enough.

For Iteration 3, compare target error structure rather than changing tau again.

A useful narrow test is to quantify whether repeated regression to the fixed Q teacher causes a systematic projection change in states where the fixed teacher has larger MC error.

Possible diagnostic:

```text
bucket transitions by |Q_step0 - G|
and track per-bucket ranking / signed drift under fixed-teacher updates.
```

Do not broaden beyond this without evidence.

---

## Path B — Oracle MC also drifts materially

Then generic repeated optimization is not ruled out.

### Preferred Iteration 2

Test the optimizer-state hypothesis:

```text
oracle MC + inherited Stage3 step0 Critic optimizer state
vs
oracle MC + freshly initialized Adam optimizer
```

Keep all other hyperparameters and batch schedule identical.

Purpose:

Determine whether inherited optimizer moments / state are driving continued movement away from the Stage2.2 solution.

Suggested filename:

```text
testing/test_stage3_oracle_mc_optimizer_state_causal.py
```

If fresh optimizer removes most drift:

```text
optimizer-state carryover is a concrete causal contributor.
```

If both drift similarly:

```text
optimizer-state carryover is not the main cause.
```

### Preferred Iteration 3 if both oracle branches still drift

Test sampling / finite-replay projection, not another bootstrap variant.

Use exact MC targets in both branches.

Compare:

```text
A: deterministic stochastic batches of 256 (current schedule)
B: deterministic coverage-balanced / near-full-dataset update schedule
```

The goal is to determine whether repeated sampling from the finite canonical set changes the fitted solution because of sampling imbalance.

Keep the total number of processed transition examples comparable.

Do not create environment data.

Suggested filename:

```text
testing/test_stage3_oracle_mc_sampling_projection_causal.py
```

If balanced coverage is stable but random repeated batches drift:

```text
finite replay sampling / projection is causal.
```

If both drift:

```text
function approximation / optimization geometry itself remains.
```

### Iteration 4 if still unresolved

Use the smallest test that discriminates the remaining two mechanisms.

Do not start a large hyperparameter sweep.

Examples:

- freeze recurrent encoder and train only Q heads under oracle MC;
- freeze Q heads and train encoder only under oracle MC;
- compare parameter-block drift and MC geometry.

Only do this if the previous oracle tests show that exact-MC repeated optimization itself is unstable.

Suggested filename:

```text
testing/test_stage3_oracle_mc_parameter_block_causal.py
```

This can localize whether representation drift or value-head drift is responsible.

---

## Path C — Bootstrap branch is not worse than oracle

If moving learned-Q bootstrap does **not** materially worsen ranking relative to oracle MC, do not force the bootstrap hypothesis.

Reassess the evidence.

Use the next iteration to test the strongest generic optimization alternative, normally:

```text
oracle MC inherited optimizer
vs
oracle MC fresh optimizer
```

Then follow Path B.

---

# 9. What counts as "root cause found"

Stop early only when there is a causal statement that is both specific and experimentally isolated.

Examples that qualify:

### Root cause level 1 — moving learned-Q feedback

```text
Exact MC repeated training stays stable;
fixed learned-Q teacher stays stable;
moving learned-Q teacher drifts.
```

This is sufficient to conclude:

```text
The dominant cause is evolving learned-Q self-bootstrap feedback, which
self-reinforces approximation/projection bias.
```

### Root cause level 2 — optimizer state

```text
Oracle MC with inherited optimizer drifts;
oracle MC with fresh optimizer does not.
```

This supports optimizer-state carryover as a concrete causal mechanism.

### Root cause level 3 — sampling projection

```text
Oracle MC random repeated batches drift;
coverage-balanced oracle MC stays stable.
```

This supports finite replay sampling/projection as a concrete mechanism.

### Not enough

The following do NOT by themselves count as root cause:

- "the loss changed";
- "Q became negative";
- "tau affects it a little";
- "min adds pessimism";
- "one branch has a different final mean";
- one non-monotonic milestone;
- a correlation without a paired intervention;
- a test with invalid initial-state / optimizer / batch matching.

---

# 10. Required metrics for every optimizer-based paired test

At minimum record for each milestone:

```text
0
100
250
500
1000
2000
```

For online Critic:

- Qmin Spearman vs empirical MC;
- Qmean Spearman vs empirical MC;
- Qmin Pearson;
- Qmin MAE;
- Qmin RMSE if convenient;
- Qmin signed bias;
- Qmin mean/std;
- episode AUC if already available.

For target Critic:

- Qmin Spearman;
- Qmin MAE;
- signed bias when useful.

Also record:

- branch completion;
- first non-finite update if any;
- online/target hashes;
- Actor/target Actor hashes;
- optimizer-state identity at start;
- exact batch-schedule identity;
- number of optimizer steps;
- environment steps = 0;
- Actor updates = 0;
- training checkpoints written = 0.

If a branch becomes non-finite, let the paired branch continue when safe, preserve the last valid milestone, and mark the result `PARTIAL` rather than silently rerunning with different parameters.

---

# 11. Test implementation rules

Before writing a new test, inspect existing helpers in:

```text
testing/test_stage3_frozen_target_causal.py
testing/test_stage3_min_vs_mean_target_causal.py
testing/test_stage3_bootstrap_vs_oracle_mc_causal.py
testing/test_stage2_bellman_closure.py
testing/test_stage3_target_update_timescale.py
testing/test_stage3_twin_bootstrap_decomposition.py
```

Reuse:

- step0 reconstruction;
- fixed batch schedules;
- canonical probe construction;
- MC return helpers;
- metric functions;
- NPU cleanup;
- validity patterns.

Do not modify production agent behavior to make a diagnostic easier.

Use testing-only monkeypatching / local target functions if needed.

Run:

```bash
python -m py_compile <new_test.py>
```

before executing each new test.

When a test writes a result, prefer a clearly named JSON.

Runtime result JSON may live under the existing run:

```text
$RUN/testing/stage2_vs_stage3_readiness/
```

but every **new source file and human-readable diagnosis/report** must remain under the repository `testing/` directory.

---

# 12. No meaningless four-round quota

"Maximum four iterations" means exactly that: a ceiling, not a quota.

Examples:

- Iteration 1 decisive -> stop after 1.
- Iteration 1 + Iteration 2 establish moving self-feedback -> stop after 2.
- Three iterations isolate optimizer-state carryover -> stop after 3.
- Use iteration 4 only if there is still a well-defined unresolved fork.

Do not invent an extra experiment merely to consume the remaining iteration count.

---

# 13. Final report requirements

At the end, update:

```text
training/Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3/testing/CODEX_STAGE3_ROOT_CAUSE_REPORT.md
```

The final section must be:

```text
# Final conclusion
```

and contain:

## A. Iterations completed

A compact table:

| Iteration | Hypothesis | Test | Status | Main result | Decision |
|---|---|---|---|---|---|

## B. Best causal explanation

Use one of:

```text
STRONG SUPPORT
PARTIAL SUPPORT
NOT SUPPORTED
INCONCLUSIVE
```

Then state the narrowest defensible mechanism.

## C. Evidence chain

List the actual numeric paired evidence supporting the mechanism.

Do not rely on verbal impressions.

## D. Alternatives ruled low

Explicitly mention whichever are supported by evidence, for example:

- transfer/load bug;
- terminal/history bug;
- target Actor continuation;
- clipped min as primary cause;
- target tau simply being too fast;
- initial Stage2 Bellman non-closure.

## E. Remaining uncertainty

State what is still not proven.

## F. Production implication

Do **not** modify production code.

Only state what mechanism a later production change should target.

Examples:

```text
If moving learned-Q feedback is isolated as dominant, future fixes should target
bootstrap supervision / anchoring / target construction, not min, Actor
continuation, or tau alone.
```

Do not start that fix automatically.

---

# 14. Git discipline

During diagnostics, do not stage unrelated files.

At the end, if Git credentials are available and the diagnostic work is complete, you may commit/push only the files you created or intentionally modified under:

```text
training/Multi_IL_Full_Action_RL/stage3_v5_rgmm_td3/testing/
```

Use explicit paths with `git add`.

Never include `fusion_result.json`.

If push is unavailable, do not block the diagnosis. Leave the testing files in the worktree and report the exact paths.

---

# 15. Final answer to the user

When the autonomous loop finishes, report only:

1. how many iterations were actually used;
2. whether root cause was isolated;
3. the strongest causal conclusion;
4. the 3–6 most decisive numeric results;
5. paths to the final report and result JSON(s);
6. whether any branch became non-finite;
7. confirmation that no environment rollout, Actor training, formal training checkpoint write, or production source modification occurred.

Do not dump raw logs unless required to explain a failure.

Do not propose a fifth iteration.
