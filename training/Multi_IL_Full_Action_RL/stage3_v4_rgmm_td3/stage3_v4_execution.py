"""Execution optimizations that retain sequential optimizer steps."""
import numpy as np
import torch


def prepare_round_batches(offline, online, count, config, device, actor_ready):
    from stage3_v4_replay import symmetric_sequence_batch, final_transition, aligned_sequence_batch
    critics, actors = [], []
    for _ in range(count):
        sequence = symmetric_sequence_batch(
            offline, online, config["batch_size"],
            config["recurrent_replay"]["critic_context_length"])
        critics.append((final_transition(sequence), sequence))
        if actor_ready:
            actors.append(aligned_sequence_batch(
                offline, online, config["recurrent_replay"]["actor_sequence_batch_size"],
                config["recurrent_replay"]["train_seq_len"], horizon=10))
    # One transfer per field for the whole round; each optimizer still sees
    # its own original-sized minibatch and the latest model parameters.
    fields = {key: torch.as_tensor(np.stack([batch[key] for batch, _ in critics]),
                                  dtype=torch.float32, device=device)
              for key in critics[0][0]}
    next_obs = torch.as_tensor(np.stack([seq["next_observations"] for _, seq in critics]),
                               dtype=torch.float32, device=device)
    prepared = []
    for index, (_, seq) in enumerate(critics):
        prepared.append(({key: value[index] for key, value in fields.items()},
                         {"next_observations": next_obs[index],
                          "episode_steps": seq["episode_steps"]}))
    actor_fields = {}
    if actors:
        actor_fields = {key: torch.as_tensor(np.stack([batch[key] for batch in actors]),
                                            dtype=torch.long if key == "episode_steps" else torch.float32,
                                            device=device)
                        for key in ("observations", "episode_steps")}
    return prepared, [{key: value[index] for key, value in actor_fields.items()}
                      for index in range(len(actors))]


def enable_npu_compile(agent):
    """Compile Q kernels, including Actor Q1, without wrapping saved modules.

    Recurrent Actor execution remains native; TorchAir support is dependent
    on the server software. Compilation errors are intentionally not hidden.
    """
    if agent.device.type != "npu":
        raise RuntimeError("TorchAir compilation requires an NPU device")
    import torchair
    backend = torchair.get_npu_backend()
    for module in (agent.critic, agent.target_critic):
        module.forward = torch.compile(module.forward, backend=backend, dynamic=False)
    agent.critic.q1.forward = torch.compile(
        agent.critic.q1.forward, backend=backend, dynamic=False)
    print("[STAGE3-V4] TorchAir Q compilation enabled; LSTM and optimizers remain native", flush=True)
