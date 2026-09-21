# Stage2.2 History-Aware Twin-Q Critic Pretraining

An isolated hypothesis experiment for `Q(h_t,a_t)`. It never modifies or loads
Stage2.1 weights and is not connected to any Stage3 implementation.

Run structural validation first:

```bash
python training/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/validate_temporal_alignment.py
python training/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/audit_stage2_2_dataset.py
```

After the real dataset audit, run a short sequential smoke on one device:

```bash
python training/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/train_stage2_2.py --device npu:0 --mode rnn_q --max-updates 10 --run-id smoke_rnn_<timestamp>
python training/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/train_stage2_2.py --device npu:0 --mode multi_q --max-updates 10 --run-id smoke_multi_<timestamp>
```

Do not start a 50K formal run until both smokes, checkpoint round-trip, sampling
audit, shared holdout evaluation, progress slices, and aliasing report pass.
