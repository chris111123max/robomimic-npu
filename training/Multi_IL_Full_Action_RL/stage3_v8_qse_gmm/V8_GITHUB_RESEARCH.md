# V8 GitHub Source Research

Status: COMPLETE. Research completed through GitHub plugin before any SSH access for this task. All links pin the actual fetched commit. No third-party source code is copied into V8.

## hengyuan-hu/ibrl

Commit: `3c457da3689cc90f881bcdf42a0a578f4a1f60aa`.

- [rl/q_agent.py](https://github.com/hengyuan-hu/ibrl/blob/3c457da3689cc90f881bcdf42a0a578f4a1f60aa/rl/q_agent.py)
- [train_rl.py](https://github.com/hengyuan-hu/ibrl/blob/3c457da3689cc90f881bcdf42a0a578f4a1f60aa/train_rl.py)
- [rl/replay.py](https://github.com/hengyuan-hu/ibrl/blob/3c457da3689cc90f881bcdf42a0a578f4a1f60aa/rl/replay.py)

Functions inspected: QAgent.add_bc_policy/_act_ibrl/update_critic/_compute_actor_loss/_compute_actor_bc_loss/update_actor_rft; Workspace._setup_replay/rl_train; ReplayBuffer._push_episode/sample_rl_bc.

Actual implementation: Actor: -E min(Q1,Q2) for RL sampled actions; RFT adds c*r*MSE(data actions). r=fraction(Qref>Qcurrent), optional. Target: reward+discount*min target-Q(BC/RL candidate chosen by target-Q when bootstrap_method=ibrl). BC policy is separately loaded, eval/no-grad, not in RL optimizer. Train step alternates configurable prior/online minibatches; dedicated BC replay accepts completed successful episodes, optionally frozen.

Transfer decision: Borrow independent success replay and additive data supervision. Do not borrow Q-max candidate selection, target min rule, or dynamic Q-gated imitation; uncertain action ordering can suppress the very safeguard needed here.

License audit: No LICENSE/COPYING found in root or recursive tree at pinned commit; no source copied.

## rail-berkeley/rlkit

Commit: `ac45a9db24b89d97369bef302487273bcc3e3d84`.

- [rlkit/torch/sac/awac_trainer.py](https://github.com/rail-berkeley/rlkit/blob/ac45a9db24b89d97369bef302487273bcc3e3d84/rlkit/torch/sac/awac_trainer.py)

Functions inspected: AWACTrainer.train_from_torch/run_bc_batch.

Actual implementation: Q target: r+(1-d)gamma[min target-Q(next a)-alpha log pi(next a)]. V is min Q at MLE action or sampled policy actions, optionally K-sample average. A=Q(data a)-V. score optionally clipped BEFORE exponent; weights softmax(A/beta) over batch, whitened exp, raw exp or step-function variants. Actor term -mean(log pi(data a)*B*stopgrad(weights)); optional entropy/reparameterized Q and separate demonstration BC loss. Replay data actions can include prior and online data; trainer itself receives the combined batch.

Transfer decision: Borrow log-probability of actual data actions and detached interpretable weights normalized over valid positions. Default equal source weights, no advantage weighting. Full AWAC replacement is distinct from V8's unchanged Q1 component-mean objective plus auxiliary success NLL.

License audit: MIT, copyright Vitchyr Pong; preserve notice if copying substantial code; independent implementation only.

## ikostrikov/rlpd

Commit: `c90fd4baf28c9c9ef40a81460a2e395092844f88`.

- [rlpd/agents/sac/sac_learner.py](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/rlpd/agents/sac/sac_learner.py)
- [train_finetuning.py](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/train_finetuning.py)

Functions inspected: SACLearner.update_actor/update_critic/update; train_finetuning online/offline sampling.

Actual implementation: Actor E[alpha log pi(a)-mean ensemble Q(a)]; target min of configurable target ensemble subset, optional entropy backup; critic MSE averaged across heads. Learner slices a batch into utd_ratio critic batches, then one actor+temperature update on last minibatch. Main samples batch*utd*offline_ratio prior + remaining online (default0.5), preserving truncation bootstrap mask. Prior data continue supporting RL rather than freezing actor.

Transfer decision: Retain existing V7 50/50 critic sampling and delayed actor schedule. Do not import SAC, temperature, larger ensembles, layer normalization or high UTD.

License audit: MIT, copyright Kostrikov/Ball/Smith; no source copied.

## rail-berkeley/hil-serl

Commit: `c32939bccb65f3b8c43a9f9add3d322d4ab0264a`.

- [examples/train_rlpd.py](https://github.com/rail-berkeley/hil-serl/blob/c32939bccb65f3b8c43a9f9add3d322d4ab0264a/examples/train_rlpd.py)
- [serl_launcher/serl_launcher/agents/continuous/sac.py](https://github.com/rail-berkeley/hil-serl/blob/c32939bccb65f3b8c43a9f9add3d322d4ab0264a/serl_launcher/serl_launcher/agents/continuous/sac.py)

Functions inspected: actor/learner; SACAgent.policy_loss_fn/critic_loss_fn/update/sample_actions.

Actual implementation: Learner consumes half demo and half online batches. cta_ratio-1 critic-only updates precede one combined actor/critic/temperature update; network publication is asynchronous. Actual intervention action overrides proposed action before replay insert. New online and intervention transitions continuously enter separate stores. SACActor alpha logpi-meanQ, target min-subsetQ optional entropy; success is not inferred from intervention membership.

Transfer decision: Borrow explicit executed-action labels, source-separated data organization and actor/critic schedule reuse. Demo/intervention membership is not full Transport success; exclude robot hardware, intervention API and SAC objectives.

License audit: Apache2.0, copyright Luo/Xu/Wu; any future source redistribution must retain license/notices and mark changes; none copied.

## twni2016/pomdp-baselines

Commit: `e7c19c32a20033d75414b29fbc466c77c211e968`.

- [policies/models/policy_rnn.py](https://github.com/twni2016/pomdp-baselines/blob/e7c19c32a20033d75414b29fbc466c77c211e968/policies/models/policy_rnn.py)
- [policies/rl/sac.py](https://github.com/twni2016/pomdp-baselines/blob/e7c19c32a20033d75414b29fbc466c77c211e968/policies/rl/sac.py)
- [buffers/seq_replay_buffer_vanilla.py](https://github.com/twni2016/pomdp-baselines/blob/e7c19c32a20033d75414b29fbc466c77c211e968/buffers/seq_replay_buffer_vanilla.py)

Functions inspected: ModelFreeOffPolicy_Separate_RNN.forward/update/act; SAC.actor_loss/critic_loss; SeqReplayBuffer.add_episode/random_episodes/_generate_masks.

Actual implementation: Separate recurrent actor/critic; T+1 zero-prefixed previous actions/rewards align histories. SAC per-position actor loss -minQ+alpha logpi is masked and divided by valid count, not fixed padded batch length; hidden state explicit at execution. Replay stores whole episodes and masks boundary crossing, warns terminal != episode boundary on timeout.

Transfer decision: Borrow masked full-sequence supervised loss/BPTT and explicit episode boundaries; use existing V7 actor zero reset at actualsteps0,10,..., not arbitrary subsequences. Do not reuse all-position Q or their actor/critic history convention.

License audit: MIT, copyright Tianwei Ni; independent implementation.

## amazon-far/residual-offpolicy-rl

Commit: `66262b06a0caf159fb001504949b7b48ccda5f64`.

- [resfit/rl_finetuning/off_policy/rl/q_agent.py](https://github.com/amazon-far/residual-offpolicy-rl/blob/66262b06a0caf159fb001504949b7b48ccda5f64/resfit/rl_finetuning/off_policy/rl/q_agent.py)

Functions inspected: QAgent._compute_actor_loss/_compute_actor_bc_loss/update_actor_rft/update_critic.

Actual implementation: Actor calls policy with stddev0, evaluates composed/clamped base+residual action if residual mode, -Q+action magnitude L2. BC MSE explicitly asserts non-residual mode. RFT adds coef*ratio*BC, ratio optional fraction Qref>Qcurrent. The target reduction delegates to critic_target.q_value; this q_agent call does not itself perform random-two selection. Critic internals were not audited here, so no internal reduction formula is claimed.

Transfer decision: Borrow additive RL+data learning concept only; do not borrow residual composition, action L2, distributional Q or critic-dependent adaptive BC coefficient.

License audit: CC BY-NC4.0 (not MIT); attribution/noncommercial applies to reuse. No code copied; source independently reimplemented.

## Four routes and decision

1. Full AWAC replaces actor improvement with advantage-weighted data likelihood; unsuitable as first version because advantage ordering is not established and it changes the approved Q1 objective.
2. Full IBRL selects BC/RL actions and bootstraps with their Q maximum; requires reliable counterfactual action ordering, currently not established.
3. Residual Actor preserves a frozen base policy but changes architecture/action composition and checkpoint semantics; excluded.
4. Q1-guided successful-experience GMM fine-tuning retains V7 last-position Q1 objective, adds ten-position likelihood on verified successful actions, and learns new successful online episodes. Selected as QSE-GMM.

The mechanism is intended to preserve supported skills while keeping Q improvement active. Success replay is outcome-filtered, not a collection of high-Q failures. Fixed supervision variance and initial auxiliary strength must be calibrated on remote real data after source/checkpoint audit. No claim of improved closed-loop success is justified by source research or offline tests alone.
