# Stage3-v6 mean2q/Multi Actor collapse — final diagnosis

All new code and output stayed under `stage3_v6_dual_2q/testing/actor_collapse_diagnosis/`. No production code, trainer, gate, checkpoint, or fusion_result.json was changed. The formal trainers remained stopped.

## Baseline and collapse window

Prior same-evaluator 10-seed test (horizon 700, seeds 20000–20009): actor_init 6/10 success, mean length 552.4; critic_ready 6/10, exactly identical Actor parameters and per-seed outcomes; 280K 0/10, all failures at 700 steps without simulator errors.

The new paired closed-loop B1/B2 used four initial-success seeds 20008, 20002, 20005, 20007. Eight environments were initialized strictly one after another, kept resident, reused, and closed. Results:

| seed | actor_init | critic_ready | 200K | 280K |
|---:|---|---|---|---|
| 20008 | success 460 | success 460 | fail 700 | fail 700 |
| 20002 | success 423 | success 423 | fail 700 | fail 700 |
| 20005 | success 479 | success 479 | fail 700 | fail 700 |
| 20007 | success 431 | success 431 | fail 700 | fail 700 |

Thus the demonstrated performance-collapse window is **140K–200K**. No 160K/240K Actor checkpoints exist, so the onset cannot be localized more tightly from this run.

## Actual checkpoint timeline

| checkpoint | filename | env steps | Critic updates | Actor updates | Actor LR | Critic LR | gate |
|---|---|---:|---:|---:|---:|---:|---|
| actor_init | step0_transfer.pth | 0 | 0 | 0 | 0 | 3e-4 | closed |
| step100k | step_0100000.pth | 100000 | 24750 | 0 | 0 | 3e-4 | closed |
| critic_ready | critic_ready.pth | 140000 | 34750 | 0 | 0 | 3e-4 | closed |
| step200k | step_0200000.pth | 200000 | 49722 | 3743 | 8.57e-7 | 2.04e-4 | open |
| step280k | best_success.pth | 280000 | 69750 | 8750 | 2e-6 | 7.5e-5 | open |
| step300k | step_0300000.pth | 300000 | 74730 | 9995 | 2e-6 | 7.5e-5 | open |
| last | last.pth | 315904 | 78726 | 10994 | 2e-6 | 7.5e-5 | open |

## A1: parameter drift (unique tensors)

The Actor state_dict has duplicate aliases for the GMM heads under `nets.decoder` and `nets.rnn.per_step_net`; equality was verified at every checkpoint. The values below count each shared tensor once, unlike the original raw parameter_drift JSON. No separate encoder parameter exists.

| checkpoint | global L2 | RNN | GMM mean | logits | std |
|---|---:|---:|---:|---:|---:|
| actor_init / 100K / ready | 0 | 0 | 0 | 0 | 0 |
| 200K | 0.8695 | 0.8566 | 0.1431 | 0.0429 | 0 |
| 280K | 2.1344 | 2.1021 | 0.3573 | 0.0959 | 0 |
| 300K | 2.5182 | 2.4846 | 0.3940 | 0.1138 | 0 |
| last | 2.8039 | 2.7669 | 0.4379 | 0.1193 | 0 |

Relative L2 at 200K/280K/last: global 0.80%/1.97%/2.58%; RNN 0.82%/2.01%/2.65%; mean head 1.59%/3.97%/4.87%; logits 0.25%/0.56%/0.70%.

## A2: fixed sequence Actor output

192 fixed, Actor-horizon-aligned 10-step windows: 64 BC-RNN success, 64 frozen online success, 64 frozen online failure. All checkpoints used identical sequence indices and fixed RNG.

| checkpoint | online-success action drift, normalized L2 | hidden L2 | mode-change fraction |
|---|---:|---:|---:|
| ready | 0 | 0 | 0 |
| 200K | 1.164 | 1.127 | 20.9% |
| 280K | 1.964 | 2.248 | 32.0% |
| last | 2.025 | 2.885 | 27.3% |

Drift starts at recurrent step 0, then amplifies. At 200K, step-0/step-9 action drift is 0.556/1.272 and hidden drift 0.163/1.534; at 280K these are 0.933/2.266 and 0.300/2.915. It is not a pure mode-collapse or late-RNN-only effect.

## A3–A6: same frozen contexts, Critic exploitation and action support

256 aligned contexts, balanced 128 original success/128 original failure, came from a frozen readiness reservoir of 116 online episodes. Production history encoding and Actor objective were reused. The actual `mean2q` branch uses **probability-weighted Q1 at GMM component means** for Actor updates (`twin_min=False`); `mean2q` only chooses the Bellman target aggregation.

| checkpoint | mean ΔQ1 objective | ΔQ1>0 | mean normalized replay distance init→current | higher Q and farther from replay | twin disagreement median init→current |
|---|---:|---:|---:|---:|---:|
| ready | 0 | 0% | 0.177→0.177 | 0% | 0.0014→0.0014 |
| 200K | +0.00891 | 90.6% | 0.177→1.434 | 88.3% | 0.0073→0.0092 |
| 280K | +0.01890 | 91.0% | 0.177→2.242 | 88.3% | 0.0068→0.0129 |
0.0199 |

At 280K, original success contexts independently show ΔQ1=+0.01862, high-Q/more-OOD=87.5%, and replay distance 0.142→2.152. Original failure contexts show +0.01918, 89.1%, 0.212→2.333. This is not confined to old failures.


Twin Q is *not* a clean shared-high-value story: 280K mean expected ΔQ2 is about −0.00091, only 46.5% of contexts improve under both twins. Twin disagreement current median/p95 is 0.0129/0.0939, versus init 0.0068/0.0328.

## Closed-loop first divergence

The init-versus-ready control has identical actions and robot positions at every paired timestep. Against init, all four 200K seeds have action L2 0.436–0.446 at timestep 0; both arm end-effectors differ by more than 1 cm at timestep 3. At 280K, first-action L2 is 0.814–0.842 and 1 cm physical divergence arrives at timestep 2. The 0.1 action threshold is far above the approximately 1e-4 learned GMM std and the exact zero control.

On the **failed trajectory states themselves**, mean same-state ΔQ1 is +0.00498 at 200K (positive on 79.75% of 2800 steps) and +0.00582 at 280K (positive on 78.29%). Mean normalized action drift is 0.727/1.518. Importantly, all four 280K first-step ΔQ values are negative. The mechanism is cumulative Q1-driven policy drift, **not** the assertion that every action on every failed trajectory has elevated Q. Mean Q1 on already-diverged failed states is only 0.101 at 200K and 0.126 at 280K; the Critic does recognize many downstream low-value states. Training-batch Q around 0.42 is not directly comparable across these different state distributions.

## Autonomous iteration 1: conservative twin-objective counterfactual

Question: Does a conservative twin objective still favor the learned off-support action? On exactly the same 256 frozen contexts, we compared probability-weighted Q1, twin mean, and per-component twin minimum for init versus learned Actor. We also interpolated the single argmax-mode action from init to current. No optimizer, environment, or checkpoint was changed.

| checkpoint | ΔQ1 mean / positive | ΔQmean mean / positive | ΔQmin mean / positive |
|---|---:|---:|---:|
| 200K | +0.00891 / 90.6% | +0.00414 / 73.0% | +0.00255 / 71.9% |
| 280K | +0.01890 / 91.0% | +0.00900 / 72.7% | +0.00175 / 59.8% |
| last | +0.01811 / 94.9% | +0.00678 / 78.9% | −0.00240 / 45.3% |

0.5421; Qmin peaks around λ=0.5 at 0.5412 and finishes at 0.5396. **Confirmed:** the actual Q1-only objective provides the strongest apparent incentive to leave support. **Not confirmed:** switching to twin-min alone would prevent early collapse, because its 200K advantage remains positive in 71.9% of contexts. No second autonomous iteration was needed.

## Ranked root cause

**PRIMARY — CRITIC_EXPLOITATION driving ACTION_DISTRIBUTION_SHIFT.** There is no Actor BC/support anchor: after gate, Q1 optimization pushes actions away from the previously successful replay support. Direct numbers: unchanged gate-before Actor 6/10 and 4/4 versus post-gate 200K/280K 0/4 and 280K 0/10; replay distance 0.177→1.434 already at 200K and 2.242 at 280K; ΔQ1>0 on 90.6%/91.0% of frozen contexts with high-Q/more-OOD on 88.3%; the four same-seed trajectories diverge in action at timestep 0, in physical state at timestep 2–3, and all fail.

**SECONDARY — RECURRENT_POLICY_DRIFT and COMPONENT_MEAN_DRIFT.** Unique RNN L2 is 0.857/2.102 and mean-head L2 is 0.143/0.357 at 200K/280K; hidden/action differences grow within each 10-step block. They are the routes for the Actor policy change, not independently proven initiating causes.

**NOT PRIMARY — GMM_MODE_COLLAPSE, TWIN_Q_SHARED_EXTRAPOLATION_ERROR, plain Actor LR, and transfer/checkpoint/evaluator faults.** Mode changes are partial; entropy is not monotonic. Q2 does not share the mean Q1 gain, and twin-min sharply weakens it. LR may change speed, not explain the preference for off-support actions. The init/ready exact control excludes pre-gate transfer and evaluator problems. This is the strongest supported mechanism, not mathematical proof of unique causality or of oracle-Q error at every state.

## One next minimal fix experiment — NOT implemented

From the same 140K critic_ready state/replay, A/B only one change: add a weak action-space trust-region penalty to the **actor_init** policy on existing horizon-10 Actor contexts; leave Critic, gate, architecture, optimizer schedule, and evaluator unchanged. Calibrate its coefficient to hold mean normalized Actor-to-init action drift ≤0.5 on the fixed successful contexts, below the already-failed 200K value 1.164. Initially run only 60K additional environment steps (to 200K). At 160K/180K/200K compare identical fixed seeds/context metrics against the original branch: four-seed success, action/replay drift, ΔQ1, and high-Q/more-OOD. Stop immediately if drift exceeds 0.5 or four-seed success falls below 3/4; extend only if both behavior and value metrics improve. This directly constrains the observed support escape; simply reducing Actor LR or using twin-min alone is less targeted.

## Artifacts and cleanup

See the sibling fixed_context_indices.json, fixed_contexts.npz, seven offline_<checkpoint>.json/.npz pairs, corrected_unique_parameter_drift.json, closed_loop_seeds.json, closed_loop.json, closed_loop_steps.jsonl, closed_loop_analysis.json, and twin_actor_objective_counterfactual.json. Diagnostic environments and processes exited; both NPUs reported no running process.

FORMAL TRAINING REMAINS STOPPED
