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
L_actor = -mean(sum_k p[k] * Q1(s, mu[k]))
          + lambda_bc * mean_offline(-log GMM(a_demo | history))
```

The RL term updates component means and mixture logits. The offline GMM NLL also
maintains the scale head. Online CQL, SAC entropy/alpha, Q filters, AWAC,
recurrent Critics, fallback policies, and handoff selectors are disabled.

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

## 3. Mandatory Phase 0

This performs exact transfer validation and 20 closed-loop episodes. It makes
no RL update and fails unless at least 10 of 20 episodes succeed.

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

Only run these after the validator, Phase 0, and smoke pass. Prepare a new pair
for formal training. The commands below do not run automatically.

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
Checkpoints include `step0_transfer.pth`, `gate_open.pth`, `best_success.pth`,
milestone checkpoints, `latest.pth`, and `last.pth`. Resume with
`--resume /path/to/last.pth`; partial simulator episodes are deliberately reset.

Compare available paired evaluations without changing a run:

```bash
python "$SCRIPT_DIR/compare_stage3_v3_pair.py" --pair-run-dir "$PAIR_RUN_DIR"
```
