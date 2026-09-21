# Stage2.2 frozen design candidate

This isolated experiment changes only the Critic state representation. Stage2.1's
finite-episode Monte-Carlo target, twin MSE, AdamW settings, seed split, update
budget, and checkpoint selection remain unchanged.

At transition `t`, the temporal token is `[o_t, a_(t-1), t/H]`; `a_(t-1)` is
zero at episode start. The candidate `a_t` bypasses recurrence and enters only
the Q head. Q1 and Q2 own independent token encoders, LSTMs, and heads.

The revised history definition is **full episode prefix**. For a supervised
block beginning at `s`, the recurrent encoder unrolls episode steps `0..s-1`
and then `s..stop-1`; a learning mask applies loss only to the latter block.
The old fixed 32-step arbitrary-window burn-in is not used for training (the
configuration value remains only to exactly replay the failed legacy batches).
Blocks beginning at zero supervise early transitions, and episodes shorter than
16 steps are represented by a masked short block rather than discarded.
Validation uses the same episode-start unroll. Targets remain finite Monte-Carlo
returns; termination and truncation both end the episode without bootstrap.

The matched control is `Q(o_t, a_(t-1), t/H, a_t)`: it uses the same data,
targets, optimizer, budget and evaluation but has no multi-step recurrence.
Its two independent MLPs use hidden widths `[240, 256]`, close in parameter
count to the recurrent model.

This architecture is not compatible with existing Stage3 code by design. No
Stage3 adapter, Actor, target network, Polyak update, or online RL exists here.
