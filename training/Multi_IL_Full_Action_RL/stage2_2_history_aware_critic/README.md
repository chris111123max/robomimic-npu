# Stage2.2 History-Aware Twin-Q Critic Pretraining

Stage2.2 now trains `Q(h_t, a_t)` with a **maximum recurrent context of 10
steps**, matching the temporal window size used by the Stage1 BC-RNN setup.
The environment rollout horizon remains 700; it is not the LSTM context length.

Current recurrent contract:

- context for transition `t`: `max(0,t-9) ... t`;
- LSTM initial state: zero for every sampled context;
- first token previous action: zero;
- candidate current action: Q head only, never recurrent input;
- supervision: final valid transition only;
- current training batch: 256 target transitions;
- Q1 and Q2 remain independent;
- finite-episode MC targets and source splits are unchanged.

Run the structural checks with:

```bash
python training/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/validate_temporal_alignment.py
python training/Multi_IL_Full_Action_RL/stage2_2_history_aware_critic/audit_stage2_2_dataset.py
```

Historical failure replay remains available through
`replay_stage2_2_bad_batch.py`; it keeps the old 32+16 sampler and historical
batch size 16 and is not used for current training.

Train the two current branches with `train_stage2_2.py --mode rnn_q` and
`--mode multi_q`. Full-prefix Stage2.2 checkpoints are a different history
contract and are deliberately rejected by the new checkpoint loader.
