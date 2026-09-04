# Stage2-new: standard SAC-compatible Critic pretraining

Stage2-new asks whether heterogeneous IL rollouts supply broader value
information for a Critic initialization.  It contains no Actor, target Actor,
entropy term, TD bootstrap, replay buffer, or online environment interaction.
It regresses both Q heads directly onto finite-episode Monte-Carlo returns.

This is independent of the older `stage2_critic_pretraining` route (which
uses frozen-RNN TD targets) and the Stage2-R / RSAC routes (which use recurrent
models).  Those historical routes are not inputs or outputs of this program.

## Data and split

The formal default Stage1 root is
`/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage1_rollout_collection/datasets/20260820_160603`.
It expects the audited Stage1 v2 files under `bc_rnn/`, `bc_transformer/`, and
`bc_gmm/`, each with `transitions.hdf5` and `/episodes/episode_*` groups.
The loader requires the explicit `initial_seed` fields, the canonical 59-D
observation key order, 14-D stored environment actions, finite values, and
valid final `done == terminated | truncated` boundaries.  It fails rather than
guessing any field or seed mapping.

Training uses seeds 10000–10079 and validation uses 10080–10099.  Splitting is
by complete seed/episode, never by transition.  `split_manifest.json` records
the policy, seed, episode id, outcome, length, and assignment.

## Variants and objective

`RNN-Q` samples only BC-RNN transitions.  `Multi-Q` samples transitions
uniformly within each of BC-RNN, BC-Transformer, and BC-GMM, while allocating
each batch approximately 1/3 per policy; batch remainders rotate so no policy
is favored over time.  There is deliberately no success balancing or episode
balancing.

For every episode, returns are recomputed from the stored rewards:

`G_t = r_t + 0.99 G_(t+1)`.

The recursion is reset at every stored episode boundary.  Both `terminated`
and `truncated` end the finite episode; there is no value bootstrap.

Both variants begin from the same saved initial Twin-Critic state dict and use
the same update count, batch size, AdamW optimizer, learning rate, random-seed
protocol, and checkpoint policy.  The only intended experimental difference is
the trajectory source.

## Critic

The shared implementation source is
`training/Multi_IL_Full_Action_RL/stage2_critic_pretraining/critic_network.py`:
the repository's existing 59+14 feed-forward twin Q network and the same
two hidden 256-wide ReLU convention used by its SAC-style baseline.  Stage2-new
uses its backward-compatible optional LayerNorm extension:

```
state (59) + action (14) = 73
Q1: Linear(73,256) -> LayerNorm -> ReLU -> Linear(256,256) -> LayerNorm -> ReLU -> Linear(256,1)
Q2: independent copy of Q1
```

There is no input normalization, action rescaling, output activation, or
LayerNorm on the scalar Q.  Initialization is PyTorch `nn.Linear` default
initialization inherited from that shared source.  The formal config records
`hidden_dims=[256,256]`, `activation=relu`, `critic_lr=3e-4` (the old Stage2
standard critic config), `gamma=0.99`, and `weight_decay=1e-4`.

## Outputs and evaluation

Each formal run writes a fresh timestamped directory under
`/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage2_new_critic_pretraining/`.
It includes the resolved config, data audit, split/probe manifests, separate
`rnn_q` and `multi_q` checkpoint directories, metrics JSONL, final validation,
`critic_comparison.json`, and `stage2_new_summary.json`.

Best RNN-Q uses RNN validation Twin-mean MSE.  Best Multi-Q uses the unweighted
mean of the three per-policy validation Twin-mean MSE values.  Final validation
always reports Q1/Q2/twin MSE and MAE, success/failure min-Q gap, min-Q ROC-AUC,
Pearson/Spearman correlation with return, and safely records null AUC/correlation
for degenerate subsets.  Deterministic probes cover RNN-success, other-policy
failure, and balanced validation data.  It records Q1/Q2 action-gradient norm
statistics and RNN-Q vs Multi-Q gradient cosines.

## Local smoke test

The local smoke test creates a temporary synthetic HDF5 fixture matching the
audited Stage1 schema.  It validates loading, MC returns, seed split, sampler
proportions, LayerNorm placement, CPU forward/backward/AdamW, checkpoint
round-trip, action gradients, safe evaluation, and CLI config overrides:

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
python training/Multi_IL_Full_Action_RL/stage2_new_critic_pretraining/validate_stage2_new.py
```

It does not read or validate the real server dataset.

## Server commands

After uploading this code to the server:

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
conda activate robosuite_npu
python training/Multi_IL_Full_Action_RL/stage2_new_critic_pretraining/train_stage2_new_critics.py --device npu:0 --mode both
```

RNN-Q only:

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
conda activate robosuite_npu
python training/Multi_IL_Full_Action_RL/stage2_new_critic_pretraining/train_stage2_new_critics.py --device npu:0 --mode rnn_q
```

Multi-Q only:

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
conda activate robosuite_npu
python training/Multi_IL_Full_Action_RL/stage2_new_critic_pretraining/train_stage2_new_critics.py --device npu:0 --mode multi_q
```

Validation only (write a new diagnostic run):

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
conda activate robosuite_npu
python training/Multi_IL_Full_Action_RL/stage2_new_critic_pretraining/train_stage2_new_critics.py --device npu:0 --mode both --validate-only --rnn-checkpoint /ABSOLUTE/RUN/rnn_q/checkpoints/best.pth --multi-checkpoint /ABSOLUTE/RUN/multi_q/checkpoints/best.pth
```

For later Stage3-new use, construct the same public `TwinCritic` through
`build_critic(...)` and load `checkpoint["critic_state_dict"]`; no Actor or
recurrent wrapper is needed.  Local Codex cannot validate the server HDF5,
Ascend NPU, or formal training run.
