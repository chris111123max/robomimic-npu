# Stage3-v2: Episode Behavior + Standard SAC

Stage3-v2 is independent from `stage3_new_sac`. It compares RNN-Q and Multi-Q
Stage2 Critic initialization while keeping a shared random Actor, shared data,
shared seeds, CQL-lite, and all SAC hyperparameters identical.

The environment behavior source is chosen once at episode start with a
deterministic episode-id pattern. No timestep target-Q selector exists. The TD
target is standard entropy-regularized SAC, and formal evaluation uses only the
deterministic SAC Actor.

## Workflow

1. Run `prepare_stage3_v2_pair.py` using a completed/prepared Stage3-new pair as
   the immutable source of Actor initialization, seed manifest, BC-RNN path,
   expert dataset, proposal cache, and Stage2 source manifest.
2. Run `validate_stage3_v2.py --pair-run-dir ...`.
3. Run one short trainer invocation with `--smoke --num-envs 2
   --total-env-steps 2000` in a dedicated smoke pair.
4. Launch `rnn_q` on NPU 0 and `multi_q` on NPU 1 for formal training.

Outputs are written beneath the configured independent root:

```text
training_runs/Multi_IL_Full_Action_RL/stage3_v2_episode_sac/<run-id>/
```

The shared frozen BC-RNN baseline is created once by the `rnn_q` process.
Checkpoint resume restores all models, optimizers, replay, sampling RNG and
counters. Because worker environments do not expose portable vector state,
resume deliberately abandons partial episodes and starts the next deterministic
episode generation for every environment.
