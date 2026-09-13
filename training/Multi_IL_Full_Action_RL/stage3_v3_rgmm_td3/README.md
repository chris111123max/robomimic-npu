# Stage3-v3: Recurrent GMM Actor + TD3-style RL

This experiment directly reuses robomimic's native `RNNGMMActorNetwork` from
the authoritative BC-RNN-GMM checkpoint. It does not distill into an MLP and it
does not use the old handoff selector or proposal cache.

## Audited Actor contract

- observation: 59D low-dimensional dictionary (`41 + 3 + 4 + 2 + 3 + 4 + 2`)
- recurrent core: two-layer `LSTM(59, 400)`, batch-first
- output: five 14D Gaussian component means/scales and five mixture logits
- `min_std=1e-4`, `softplus`, `use_tanh=False`, `low_noise_eval=True`
- rollout hidden state resets at episode start and every 10 actions
- total unique parameter count reported by the checkpoint: 2,078,945

The prepared Actor is created from the original policy object and strict-loads
the complete original state dict. Phase 0 compares parameters, component means,
scales, logits, probabilities, final LSTM state, and a seeded sampled action.

## Algorithm

The memoryless Stage2 twin MLP Critic is unchanged. The critic target is

```text
y = r + gamma * (1-terminal) *
    sum_k p_target[k] * min(Q1_target(s_next, mu_target[k]),
                            Q2_target(s_next, mu_target[k]))
```

There is no target policy smoothing. The delayed Actor objective is

```text
L_actor = -alpha * mean(sum_k p[k] * Q1(s, mu[k]))
          / stop_gradient(mean(abs(Q1(s, a_replay))))
          + adaptive_lambda_bc * mean_offline(-log GMM(a_demo | history))
```

The RL term updates component means and mixture logits. The offline GMM NLL also
maintains the scale head. Online CQL, SAC entropy/alpha, Q filters, AWAC,
recurrent Critics, fallback policies, and handoff selectors are disabled.
The delayed Actor update uses `policy_delay=2`, meaning one Actor update after
every two Critic updates once the 10k competence gate is open. The RL term uses
TD3+BC-style batch Q-scale normalization (`alpha=2.5`). The BC coefficient
starts at 1 and changes only after valid fixed-seed evaluations using a
performance/decline feedback rule, bounded to `[0, 1]`. Its controller state is
saved in checkpoints and restored on resume. These coefficients are experiment
settings, not values proven optimal for TwoArmTransport.

Rollout Actor inference is vectorized across all active environments: recurrent
hidden states are packed into one batch and actions are copied from the NPU to
the host once per vector step.
Recurrent hidden-state resets use a device-side mask, avoiding a host/device
sync at each timestep of the Critic target Actor's context reconstruction.

Each Critic update uses 128 offline and 128 online boundary-safe sequences. The
last transition is the memoryless Critic sample; the preceding recurrent
context reconstructs the target Actor hidden state. Actor updates use ten
burn-in steps without gradient and ten BPTT learning steps. Horizon-10 hidden
resets use the episode timestep, so sampled windows match rollout semantics.

## 1. Synthetic validator

```bash
cd /data/home/3220251075/lerobot_workspace/robomimic
conda activate robosuite_npu

SCRIPT_DIR=training/Multi_IL_Full_Action_RL/stage3_v3_rgmm_td3
BC_RNN_CHECKPOINT="/data/home/3220251075/lerobot_workspace/training_runs/Pure IL/two_arm_transport_bc_rnn_official_ph_low_dim/two_arm_transport_bc_rnn_official_ph_low_dim/20260811151701/models/model_epoch_1000_low_dim_v15_success_0.9.pth"

python "$SCRIPT_DIR/validate_stage3_v3.py" \
  --bc-rnn-checkpoint "$BC_RNN_CHECKPOINT" \
  --device npu:0
```

## 2. Prepare a pair

```bash
OUTPUT_ROOT=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage3_v3_rgmm_td3
STAGE2_RUN=/data/home/3220251075/lerobot_workspace/training_runs/Multi_IL_Full_Action_RL/stage2_new_critic_pretraining/stage2new_formal_001
RNN_Q_CHECKPOINT="$STAGE2_RUN/rnn_q/checkpoints/best.pth"
MULTI_Q_CHECKPOINT="$STAGE2_RUN/multi_q/checkpoints/best.pth"
PAIR_ID=stage3v3_rgmm_td3_$(date +%Y%m%d_%H%M%S)
PAIR_RUN_DIR="$OUTPUT_ROOT/$PAIR_ID"

python "$SCRIPT_DIR/prepare_stage3_v3_pair.py" \
  --run-id "$PAIR_ID" \
  --bc-rnn-checkpoint "$BC_RNN_CHECKPOINT" \
  --rnn-q-checkpoint "$RNN_Q_CHECKPOINT" \
  --multi-q-checkpoint "$MULTI_Q_CHECKPOINT"
```

`--expert-dataset` is optional. By default the authoritative path embedded in
the checkpoint is used.

## 3. Phase 0 or validated reuse

This performs exact transfer validation and 20 closed-loop episodes. It makes
no RL update and fails unless at least 10 of 20 episodes succeed.
If the same pair already has a complete passing Phase 0, rerunning this
command returns the recorded result without repeating the episodes.

```bash
python "$SCRIPT_DIR/run_phase0_stage3_v3.py" \
  --pair-run-dir "$PAIR_RUN_DIR" \
  --device npu:0
```

Inspect:

```bash
cat "$PAIR_RUN_DIR/shared/transfer_validation.json"
cat "$PAIR_RUN_DIR/shared/step0_competence.json"
cat "$PAIR_RUN_DIR/shared/phase0_gate.json"
```

For a new pair with the same BC checkpoint, expert dataset, Actor structure,
evaluation seeds, horizon, and gate requirements, pass an existing completed
Stage3-v3 pair to `prepare_stage3_v3_pair.py`:

```bash
REFERENCE_PHASE0_PAIR=/path/to/completed/stage3_v3_pair
python "$SCRIPT_DIR/prepare_stage3_v3_pair.py" \
  --run-id "$PAIR_ID" \
  --bc-rnn-checkpoint "$BC_RNN_CHECKPOINT" \
  --rnn-q-checkpoint "$RNN_Q_CHECKPOINT" \
  --multi-q-checkpoint "$MULTI_Q_CHECKPOINT" \
  --reuse-phase0-from "$REFERENCE_PHASE0_PAIR"
```

The prepare command verifies the original episode-level evidence, Actor hash,
checkpoint and dataset hashes, and evaluation contract before copying the three
Phase-0 artifacts. Its output contains `"phase0_reused": true`; the new pair
also records `shared/phase0_reuse_manifest.json`. Training settings such as
`policy_delay` may change without repeating Phase 0. Skip the Phase-0 command
for this new pair and launch training directly.

## 4. Smoke

Use a fresh prepared pair. A 12k smoke crosses the real 10k latch boundary;
below 10k the Actor hash is checked on every collected transition.

```bash
mkdir -p "$PAIR_RUN_DIR/multi_q"

python -u "$SCRIPT_DIR/train_stage3_v3_vector.py" \
  --group multi_q \
  --device npu:0 \
  --pair-run-dir "$PAIR_RUN_DIR" \
  --critic-init-checkpoint "$MULTI_Q_CHECKPOINT" \
  --num-envs 2 \
  --total-env-steps 12000 \
  --smoke
```

## 5. Formal paired launch

Run these after the validator and a passing Phase 0 (fresh or reused). Prepare
a new pair for formal training. The commands below do not run automatically.

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "$PAIR_RUN_DIR/rnn_q" "$PAIR_RUN_DIR/multi_q"

nohup python -u "$SCRIPT_DIR/train_stage3_v3_vector.py" \
  --group rnn_q \
  --device npu:0 \
  --pair-run-dir "$PAIR_RUN_DIR" \
  --critic-init-checkpoint "$RNN_Q_CHECKPOINT" \
  --num-envs 16 \
  --total-env-steps 3000000 \
  > "$PAIR_RUN_DIR/rnn_q/console.log" 2>&1 &
RNN_PID=$!
echo "$RNN_PID" > "$PAIR_RUN_DIR/rnn_q/train.pid"

nohup python -u "$SCRIPT_DIR/train_stage3_v3_vector.py" \
  --group multi_q \
  --device npu:1 \
  --pair-run-dir "$PAIR_RUN_DIR" \
  --critic-init-checkpoint "$MULTI_Q_CHECKPOINT" \
  --num-envs 16 \
  --total-env-steps 3000000 \
  > "$PAIR_RUN_DIR/multi_q/console.log" 2>&1 &
MULTI_PID=$!
echo "$MULTI_PID" > "$PAIR_RUN_DIR/multi_q/train.pid"

echo "RNN-Q PID=$RNN_PID"
echo "Multi-Q PID=$MULTI_PID"
echo "PAIR_RUN_DIR=$PAIR_RUN_DIR"
```

## Outputs

Shared gate artifacts are under `shared/`. Each branch contains
`train_metrics.jsonl`, `episode_metrics.jsonl`, `gate_metrics.jsonl`,
`throughput_metrics.jsonl`, `evaluations/`, `diagnostics/`, and `checkpoints/`.

`stage_timing.jsonl` records one synchronized profiling round approximately
every 1000 aggregate environment steps. It separates batched Actor inference,
vector environment stepping, replay sampling, one Critic update, one Actor
update when the gate is open, and one target-network Polyak update. The
Critic and Actor times are for one sampled update, not the whole vector round;
the profiler does not change UTD or policy delay. Inspect recent samples with
`tail -n 5 "$PAIR_RUN_DIR/multi_q/stage_timing.jsonl"`.
Checkpoints include `step0_transfer.pth`, `gate_open.pth`, `best_success.pth`,
milestone checkpoints, `latest.pth`, and `last.pth`. Resume with
`--resume /path/to/last.pth`; partial simulator episodes are deliberately reset.

Compare available paired evaluations without changing a run:

```bash
python "$SCRIPT_DIR/compare_stage3_v3_pair.py" --pair-run-dir "$PAIR_RUN_DIR"
```
