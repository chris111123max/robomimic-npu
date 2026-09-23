# Stage2.2 horizon-10 Critic design

Stage2.2 keeps the existing finite-episode Monte-Carlo target, Twin-Q MSE,
AdamW optimizer, seed split, source definitions, and checkpoint selection. The
change is the temporal state contract.

At transition `t`, the Critic receives at most ten recurrent tokens:

```
[max(0, t-9), ..., t]
```

Each token remains `[o_k, previous_executed_action_k, k/H]`, with `H=700`
used only for normalized episode progress. The recurrent state is initialized
to zero for every sampled context. The first token in every context uses a zero
previous-action vector, so information before the ten-step window cannot leak
through `a_(k-1)`. The candidate action `a_t` never enters recurrence and is
concatenated only in the Q head.

Training samples one target transition per sequence and supervises only the
final valid context position. `sequence_batch_size=256`, so each optimizer
update still contains 256 supervised transitions while recurrent compute is
bounded by `256 x 10` tokens. Early episode transitions naturally use shorter
contexts and right padding; padding occurs only after the supervised position.

Held-out evaluation reconstructs the exact same <=10-step zero-state context
for every transition. It never evaluates the recurrent Critic with an
episode-start full-prefix history.

Q1 and Q2 retain independent token encoders, LSTMs, and heads. The matched
memoryless control still evaluates the current token plus candidate action
without recurrence. RNN-Q uses only BC-RNN Stage1 trajectories; Multi-Q uses
the balanced BC-RNN / BC-Transformer / BC-GMM source rotation.

The historical 32+16 arbitrary-window sampler is retained only by
`LegacyWindowSampler` for replaying old failures. Its historical batch size is
fixed separately at 16 and is not the current training contract.

Existing full-prefix Stage2.2 checkpoints are intentionally incompatible with
this contract and must not be resumed or used as initialization.
