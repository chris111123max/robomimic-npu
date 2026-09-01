# Stage3-R: BC-RNN to pomdp-baselines Recurrent SAC Actor

This stage transfers all eight tensors of the frozen two-layer BC-RNN LSTM into
a recurrent actor whose Gaussian policy head is the vendored
`pomdp-baselines` `TanhGaussianPolicy`. It trains only the new deterministic
mean behavior against inferred sampled-component clean means. It does not run
SAC, train critics, start Stage2-R, or start Stage4.

The actor keeps the original ten-step hidden reset. HDF5 canonical object-last
states are explicitly reordered to the BC-RNN object-first LSTM input.

Smoke:

```bash
python -u training/Multi_IL_Full_Action_RL/stage3_r_bc_rnn_to_rsac/run_stage3_r.py --smoke-test --device npu:0
```

Equivalent launcher:

```bash
NPU_ID=0 bash training/Multi_IL_Full_Action_RL/stage3_r_bc_rnn_to_rsac/run.sh smoke
```

Formal run executes preflight, all 300 epochs, then screens unique candidates
serially. Each candidate uses 16 independent environment workers.
