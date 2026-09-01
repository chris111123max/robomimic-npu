# Recurrent SAC Framework Selection & Compatibility Audit

## Executive decision

**Recommended framework: NONE — NO SUITABLE EXISTING RECURRENT SAC FOUND.**

No audited framework simultaneously provides a mature continuous recurrent SAC path with sequence replay, burn-in, recurrent-state propagation during off-policy optimization, a recurrent actor plus an independent feed-forward critic, strict reuse of the existing Stage 2 `TwinCritic`, and practical Ascend NPU support. Filling the missing pieces would amount to implementing the recurrent SAC replay/training machinery prohibited by this audit.

## Candidate comparison

| Framework | Existing continuous SAC | Existing recurrent actor | Recurrent off-policy SAC end-to-end | Actor recurrent + critic FF | Custom actor / critic | Sequence replay | Burn-in | Ascend NPU | BC-RNN LSTM transfer | Stage 2 critic reuse | Recommendation |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| Tianshou 2.0.1 | Yes | Yes | **No verified mature path** | API can accept separate modules | High flexibility | `stack_num` stacks observations, but is not a complete recurrent SAC sequence learner | No native SAC burn-in path found | Small-to-medium device adaptation, but algorithmic gap remains | Easy at tensor level | Wrapper possible, but Tianshou expects two critic modules rather than this `TwinCritic` container | Reject |
| RLlib 2.58.0 | Yes | Stateful RLModule / legacy recurrent Model support | Strongest generic sequence-replay machinery | Not demonstrated as a stable supported SAC configuration with this split | Possible through substantial custom RLModule/SAC model work | Yes | Yes | **Major incompatibility**: CUDA/GPU resource and device management is not torch-NPU-native | Possible only inside a custom stateful module | Difficult; SAC expects its own critic module contracts and target handling | Reject |
| SB3 + SB3-Contrib | Yes in SB3 | RecurrentPPO only | No Recurrent SAC | No applicable implementation | SAC customization does not supply recurrent replay | No recurrent SAC sequence replay | No | PyTorch-level patches would not fix missing algorithm | Not applicable | Not applicable | Reject |
| CleanRL | Continuous SAC script | LSTM examples are PPO, not SAC | No | No existing implementation | Not a modular importable framework | No recurrent SAC replay | No | SAC script hard-codes CUDA-oriented switches and still lacks recurrent SAC | Would require rewriting the script | Would require rewriting loss/training integration | Reject |

Primary evidence: [Tianshou repository and supported algorithms](https://github.com/thu-ml/tianshou), [Tianshou SAC implementation](https://github.com/thu-ml/tianshou/blob/master/tianshou/algorithm/modelfree/sac.py), [Tianshou recurrent sequence guidance using `stack_num`](https://github.com/thu-ml/tianshou/discussions/698), [RLlib SAC documentation](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#soft-actor-critic-sac), [RLlib sequence replay and burn-in API](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.utils.replay_buffers.multi_agent_replay_buffer.MultiAgentReplayBuffer.__init__.html), [SB3-Contrib algorithm list](https://stable-baselines3.readthedocs.io/en/v2.3.0/guide/sb3_contrib.html), and [CleanRL algorithm list](https://github.com/vwxyzjn/cleanrl/blob/master/docs/rl-algorithms/overview.md).

## Why Tianshou is not selected

Tianshou is the closest lightweight candidate. Its SAC implementation supports continuous `Box` actions, Gaussian reparameterization, tanh correction, entropy regularization, independently supplied actor and critics, collectors, and recurrent actor state during environment collection. A custom actor could expose `(mu, sigma), state`; a critic wrapper could expose each half of the existing `TwinCritic`.

The blocking gap is training semantics. The documented recurrent replay mechanism is buffer `stack_num`, which supplies stacked contiguous observations. It does not establish a complete SAC implementation with stored/reconstructed LSTM state, burn-in before the optimized segment, masked losses across padded/episode boundaries, and target-action recurrent-state propagation. `RecurrentActorProb` existing beside `SACPolicy` is therefore insufficient evidence of a correct end-to-end recurrent SAC algorithm. Supplying these missing pieces would violate the prohibition on writing recurrent replay, sequence sampling, hidden propagation, and recurrent SAC loss/training logic.

No Tianshou version is recommended for installation for this experiment. In particular, do not install latest Tianshou into the existing CANN environment merely to test this route.

## Why RLlib is not selected

RLlib has the strongest generic recurrent replay facilities among the audited frameworks. Its replay buffer exposes sequence length, burn-in, zero/previous initial-state control, episode boundaries, and sequence storage. However, satisfying this experiment would require a custom stateful SAC RLModule (or legacy SAC model), custom policy distribution wiring, a custom bridge from the SAC critic API to the existing `TwinCritic`, and validation of target-critic ownership/update semantics.

More importantly, RLlib's accelerator discovery, worker resource assignment, learner devices, and distributed tensor placement are designed around CPU/CUDA. Treating `npu:0` as a first-class accelerator would be a framework-level port rather than a small local adapter. This is a major incompatibility for the current single-node Ascend environment.

## Confirmed model adapters if a suitable engine appears later

### Observation adapter

The HDF5/critic canonical order is robot-first and object-last. The transferred BC-RNN LSTM expects object-first:

```text
canonical [r0_pos, r0_quat, r0_grip, r1_pos, r1_quat, r1_grip, object]
    -> reorder
BC-RNN   [object, r0_pos, r0_quat, r0_grip, r1_pos, r1_quat, r1_grip]
```

The actor adapter must perform this permutation before the LSTM. The critic must continue receiving the original canonical object-last 59D state.

### LSTM weight mapping

An actor using a native `torch.nn.LSTM(59, 400, num_layers=2, batch_first=True)` can copy tensors one-to-one:

| BC-RNN parameter | Target recurrent actor parameter | Shape |
|---|---|---|
| `nets.rnn.nets.weight_ih_l0` | `lstm.weight_ih_l0` | `[1600, 59]` |
| `nets.rnn.nets.weight_hh_l0` | `lstm.weight_hh_l0` | `[1600, 400]` |
| `nets.rnn.nets.bias_ih_l0` | `lstm.bias_ih_l0` | `[1600]` |
| `nets.rnn.nets.bias_hh_l0` | `lstm.bias_hh_l0` | `[1600]` |
| `nets.rnn.nets.weight_ih_l1` | `lstm.weight_ih_l1` | `[1600, 400]` |
| `nets.rnn.nets.weight_hh_l1` | `lstm.weight_hh_l1` | `[1600, 400]` |
| `nets.rnn.nets.bias_ih_l1` | `lstm.bias_ih_l1` | `[1600]` |
| `nets.rnn.nets.bias_hh_l1` | `lstm.bias_hh_l1` | `[1600]` |

Hidden state remains `(h, c)`, each `[2, B, 400]`. New heads are `Linear(400,14)` for `mu` and `log_std`; the original GMM heads are not transferred.

### Critic adapter

The immutable Stage 2 model remains two `73->256->256->1` Q networks. A framework adapter may expose `q1(state, action)` and `q2(state, action)` separately while retaining the original `TwinCritic` object and loading `critic_state_dict` with `strict=True`. The framework must not recreate, reshape, or retrain these weights merely for import.

## Hidden-state and replay decision

The original BC-RNN rollout resets `(h,c)` every 10 action calls, not only at episode boundaries. A future integration should initially preserve this behavior during collection for behavior equivalence. Training, however, requires a framework-defined sequence method with episode-safe contiguous samples and burn-in. Replacing the 10-step reset with full-episode recurrence or burn-in is an experimental choice, not a transparent adapter.

No audited engine provides all required semantics on Ascend without implementing or substantially modifying the recurrent SAC framework. Therefore the project must stop at this audit under the stated constraints.

## Environment and mutation status

- Existing Python 3.10 / PyTorch-NPU / robosuite / robomimic environment was not changed.
- No dependency was installed or upgraded.
- Stage 2 Critic does not need modification: **NO**.
- A custom recurrent SAC implementation is not authorized and was not created: **NO**.
- Training was not started.
- Stage 4 was not started or modified.

