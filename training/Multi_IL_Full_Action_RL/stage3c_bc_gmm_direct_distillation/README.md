# Stage 3C — BC-GMM to SAC Direct Distillation

This independent stage distils the saved Stage 1 BC-GMM environment actions into a freshly
initialized 59-256-256-14 SAC-compatible actor. The BC-GMM teacher is frozen and is only
restored by the mandatory compatibility audit; training never resamples teacher actions.

Smoke test (audit plus two epochs, then stop):

```bash
python -u training/Multi_IL_Full_Action_RL/stage3c_bc_gmm_direct_distillation/run_stage3c.py \
  --smoke-test --device npu:0
```

Formal run (audit, training, 16-environment held-out candidate screening, selection, then stop):

```bash
python -u training/Multi_IL_Full_Action_RL/stage3c_bc_gmm_direct_distillation/run_stage3c.py \
  --device npu:0
```

No Stage 2 critic is loaded and Stage 4 is never started by this workflow.
