"""Execution optimizations that retain sequential optimizer steps."""
import numpy as np
import torch
import time


def prepare_round_batches(offline, online, count, config, device, actor_ready,
                          update_count=None, critic_updates=0, profiler=None, transfer=True):
    from stage3_v5_replay import symmetric_sequence_batch, final_transition, aligned_sequence_batch
    critics, actors = [], {}
    # The vector round may contain 16 transitions while the configured update
    # stride is four.  Prefetch exactly the number of optimizer rounds that
    # will actually be consumed; sampling 16 batches here would waste CPU/NPU
    # transfer time and leave most prefetched batches unused.
    # The collector round size is not the update schedule.  UTD credit is the
    # only source of truth; callers pass floor(previous_credit + n*UTD).
    if update_count is None:
        update_count = int(np.floor(int(count) * float(config["utd"])))
    update_count = max(0, int(update_count))
    if not update_count:
        return [], {}
    for index in range(update_count):
        sequence = symmetric_sequence_batch(
            offline, online, config["batch_size"],
            config["recurrent_replay"]["critic_context_length"], profiler=profiler)
        critics.append((final_transition(sequence), sequence))
        if actor_ready and (int(critic_updates) + index + 1) % int(config["policy_delay"]) == 0:
            actors[index] = aligned_sequence_batch(
                offline, online, config["recurrent_replay"]["actor_sequence_batch_size"],
                config["recurrent_replay"]["train_seq_len"], horizon=10, profiler=profiler)
    if not transfer:
        return critics, actors
    if profiler:
        profiler.synchronize()
    transfer_started = time.perf_counter()
    # Full-prefix batches have variable padded time dimensions across updates,
    # so transfer each prepared critic batch independently instead of stacking
    # different prefix lengths into one round tensor.
    prepared = []
    for batch, seq in critics:
        prepared_batch = {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in batch.items()
        }
        prepared_sequence = {
            key: torch.as_tensor(seq[key], dtype=torch.float32, device=device)
            for key in ("observations", "actions", "next_observations")
        }
        prepared_sequence.update({
            "episode_steps": seq["episode_steps"],
            "sequence_lengths": seq["sequence_lengths"],
            "sample_window_starts": seq["sample_window_starts"],
        })
        prepared.append((prepared_batch, prepared_sequence))
    actor_fields = {}
    if actors:
        actor_fields = {key: torch.as_tensor(np.stack([batch[key] for batch in actors.values()]),
                                            dtype=torch.long if key == "episode_steps" else torch.float32,
                                            device=device)
                        for key in ("observations", "actions", "episode_steps")}
    actor_prepared = {update_index: {key: value[position] for key, value in actor_fields.items()}
                      for position, update_index in enumerate(actors)}
    if profiler and profiler.enabled:
        profiler.synchronize()
        name = "host_to_device_ms"
        profiler.totals[name] = profiler.totals.get(name, 0.0) + (time.perf_counter()-transfer_started)*1000
        profiler.counts[name] = profiler.counts.get(name, 0) + 1
    return prepared, actor_prepared


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
    print("[STAGE3-V5] TorchAir Q compilation enabled; LSTM and optimizers remain native", flush=True)
