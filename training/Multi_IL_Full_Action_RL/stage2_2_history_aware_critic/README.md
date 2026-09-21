# Stage2.2 History-Aware Twin-Q Critic Pretraining

An isolated hypothesis experiment for `Q(h_t,a_t)`. It never modifies or loads
Stage2.1 weights and is not connected to any Stage3 implementation.

Run structural validation first:

```bash
python training/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/validate_temporal_alignment.py
python training/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/audit_stage2_2_dataset.py
```

Replay the two legacy failure batches without changing data:

```bash
python training/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/replay_stage2_2_bad_batch.py --mode rnn_q --step 1626 --dataset-root "$DATASET_ROOT" --output rnn_1626.json --dump-npz rnn_1626.npz
python training/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/replay_stage2_2_bad_batch.py --mode multi_q --step 1161 --dataset-root "$DATASET_ROOT" --output multi_1161.json --dump-npz multi_1161.npz
```

After the real dataset audit, run a short sequential smoke on one device:

```bash
python training/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/train_stage2_2.py --device npu:0 --mode rnn_q --max-updates 10 --run-id smoke_rnn_<timestamp>
python training/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/train_stage2_2.py --device npu:0 --mode multi_q --max-updates 10 --run-id smoke_multi_<timestamp>
```

After the two 10-update smokes, run at most 2000 stability updates. A 50K run is
forbidden until the finite audit, exact replay, both smokes, all finite guards,
checkpoint round-trip, sampling audit, shared holdout evaluation and the 2K
stability run pass. Train the matched controls with `--mode matched_rnn_q` or
`--mode matched_multi_q`; they are Stage2.2-only experimental controls.
