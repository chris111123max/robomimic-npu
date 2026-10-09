# Stage3-V7: PIRLNav-inspired dynamic handoff (single NPU)

This is a **separate experiment**, not a modification of Stage3-V6. Reuses
the V6 vector learner, Stage2.2 history-aware Twin-Q, BC-RNN-GMM actor,
random2q target selector, balanced multi_q replay, UTD=0.25, policy_delay=4,
and Polyak target updates. Formal mode uses **one NPU (npu:0)** with 16
CPU simulator environments; sixteen environments **do not require 16 NPUs**.

PIRLNav is the *inspiration for phased learning rates*, not a claim that this
is its exact PPO implementation. V7 still uses a TD3-style off-policy Twin-Q
and Q1 Actor loss; does not become PPO.

## Readiness V7

Every 10,000 aggregate env steps from 100K onward, check:

- At least 150 completed online episodes, at least 30 successful, 30 failed.
- Ready Qmean versus MC Spearman >= 0.7 on the existing frozen diagnostic set.
- Finite diagnostic/target values and finite observed Critic training metrics.
- Two consecutive passes start the next phase.

No plateau gate, old Q-scale/TD-worsening hard gates, or 300K Critic-only
deadline. These are *basic readiness screens*, **not** proof of convergence,
unbiased action ranking, or independent held-out performance. Any non-finite
training/diagnostic metric is a hard stop; insufficient data/rank is not.

If readiness passes at R env steps:

| Phase | Env-step range | Critic LR | Actor LR |
|---|---|---|---|
| CRITIC_ONLY | before R | 3e-4 | 0 |
| CRITIC_DECAY | R to R+200K | linear 3e-4 to 7.5e-5 | 0, **no Actor updates** |
| ACTOR_WARMUP | R+200K to R+300K | 7.5e-5 | linear 0 to 2e-6 |
| JOINT_RL | after R+300K | 7.5e-5 | 2e-6 |

Actor optimizer updates continue to respect the inherited policy_delay=4.
A gate flag being enabled precisely at warmup start can initially accompany
**zero LR**; the first nonzero update happens later. Eval starts at Actor
unlock, then every 20K aggregate env steps. Checkpoints every 50K plus
readiness event and final/last. The *whole-run* default 3M env-step budget
still applies (separate from the removed 300K Critic-only cutoff).

## Preflight checks (no simulator, no NPU)

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
python training/Multi_IL_Full_Action_RL/stage3_v7_pirlnav_schedule/testing/test_source_contract.py
python training/Multi_IL_Full_Action_RL/stage3_v7_pirlnav_schedule/testing/test_schedule.py
```

## Launch (only after preflight succeeds)

The launcher discovers an existing V6 *prepared run* with original BC and
Stage2.2 source manifests. If there are multiple plausible source runs, it
will stop and ask for an explicit `--v6-run-dir` rather than guess.

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
python training/Multi_IL_Full_Action_RL/stage3_v7_pirlnav_schedule/launch_stage3_v7.py --v6-run-dir /ABS/PATH/TO/PREPARED_V6_RUN
```

When omitted, `--v6-run-dir` is auto-discovered if unambiguous. The trainer
first looks **only** under that selected prepared run for:

`random2q/multi_q/checkpoints/step_0100000.pth`

and its required `step_0100000.sequences.npy` online Replay file. If both
are present, it verifies checkpoint stage, group, mode, 100K env steps, zero
Actor updates, source hashes, optimizer/model/replay compatibility, and
restores RNG including the random2q selector. Any incompatible/corrupt
*present* snapshot fails closed (never silently restarts).

**If the complete 100K snapshot is absent**, the training run initializes
from the same prepared original Stage1 BC actor and Stage2.2 multi-Q Critic
and trains from zero. This requires a prepared V6 shared source directory
even when the *100K snapshot* is missing. It must not substitute a mean2q or
rnn_q checkpoint.

V7 checkpoints/logs are isolated under
`.../training_runs/Multi_IL_Full_Action_RL/stage3_v7_pirlnav_schedule/<v6_run_id>/random2q/multi_q/`.
Existing V6 training checkpoints and diagnostics remain untouched.
To resume an interrupted V7 run, pass `--resume /ABS/PATH/TO/V7_CHECKPOINT.pth`
in addition to the same source run. As in V6, partial vector episodes are
reset and cannot be replayed as exact continuous simulator trajectories.

## Smoke-only diagnostic

```bash
python training/Multi_IL_Full_Action_RL/stage3_v7_pirlnav_schedule/launch_stage3_v7.py --v6-run-dir /ABS/PATH/TO/PREPARED_V6_RUN --smoke --num-envs 2 --total-env-steps 2000
```

Smoke shortens readiness timing and warmup **only inside the isolated smoke
runtime**, never changing the formal rules. A readiness transition is not
guaranteed if the simulator has not produced usable complete episodes.

For now: **run preflight and smoke first; read outputs; do not start formal
training until these checks pass.**

### Implementation note

`train_stage3_v7_vector.py` is the audited V6 trainer fork with only V7
scheduling, readiness integration, hardware mapping, source-selection,
checkpoint-compatibility and output-isolation changes. All environment,
Actor/target Q, replay/updates and optimizer math is imported from V6/V5.
V6 production files have not been edited.
