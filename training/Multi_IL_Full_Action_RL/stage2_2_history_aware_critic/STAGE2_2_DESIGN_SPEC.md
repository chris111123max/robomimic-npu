# Stage2.2 frozen design candidate

This isolated experiment changes only the Critic state representation. Stage2.1's
finite-episode Monte-Carlo target, twin MSE, AdamW settings, seed split, update
budget, and checkpoint selection remain unchanged.

At transition `t`, the temporal token is `[o_t, a_(t-1), t/H]`; `a_(t-1)` is
zero at episode start. The candidate `a_t` bypasses recurrence and enters only
the Q head. Q1 and Q2 own independent token encoders, LSTMs, and heads.

Windows use 32 burn-in tokens and 16 supervised tokens. Sixteen windows yield
256 supervised transitions per update, matching Stage2.1's batch size. All legal
starts are precomputed inside one episode. Targets are finite Monte-Carlo returns;
both termination and truncation stop the episode and no bootstrap is used.

This architecture is not compatible with existing Stage3 code by design. No
Stage3 adapter, Actor, target network, Polyak update, or online RL exists here.
