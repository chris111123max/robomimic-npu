# Stage 2 critic pretraining

Critic pretraining is not implemented or started. The only implemented component is the Stage 2.0
frozen BC-RNN target-policy sanity check:

```bash
python training/Multi_IL_Full_Action_RL/stage2_critic_pretraining/validate_frozen_rnn_target.py
```

It reads five fixed episodes per Stage 1 policy, never modifies the HDF5 files, and writes only
`training/Multi_IL_Full_Action_RL/analysis/stage2/frozen_rnn_target_validation.json`.
