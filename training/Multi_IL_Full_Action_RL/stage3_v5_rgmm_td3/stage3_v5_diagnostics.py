"""Frozen, stratified readiness samples created by independent NumPy RNG."""
import copy
from contextlib import contextmanager
import random
import numpy as np
from stage3_v5_replay import _pad_prefix_batch, _sample_prefix_sequence_batch


@contextmanager
def isolated_training_rng(online=None, offline=None):
    """Defensive isolation, including sampler states and every active device."""
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    npu_state = (torch.npu.get_rng_state_all() if hasattr(torch, "npu")
                 and torch.npu.is_available() else None)
    online_state = copy.deepcopy(online.rng.bit_generator.state) if online is not None else None
    offline_state = copy.deepcopy(offline.state_dict()) if offline is not None else None
    try:
        yield
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if npu_state is not None:
            torch.npu.set_rng_state_all(npu_state)
        if online_state is not None:
            online.rng.bit_generator.state = online_state
        if offline_state is not None:
            offline.load_state_dict(offline_state)


def frozen_set(online, config, sample_count=256):
    if online.fixed_critic_diagnostic_set is not None:
        return online.fixed_critic_diagnostic_set
    rules = config["critic_readiness"]
    reservoirs = online.diagnostic_reservoir
    if (len(reservoirs[True]) < rules["min_success_episodes"] or
            len(reservoirs[False]) < rules["min_failure_episodes"]):
        return None
    episodes = copy.deepcopy(reservoirs[True] + reservoirs[False])
    length = int(config["recurrent_replay"]["critic_context_length"])
    eligible = [episode for episode in episodes if len(episode["actions"]) >= length]
    if not eligible:
        return None
    seed = int(config["training_seed"]) + 734911
    rng = np.random.default_rng(seed)
    count = min(256, int(sample_count))
    # Use an independent RNG exactly as before, but retain episode prefixes so
    # readiness TD diagnostics use the same recurrent state semantics as training.
    sequences = _pad_prefix_batch(
        _sample_prefix_sequence_batch(eligible, rng, count, length))
    indices = list(zip(
        np.zeros(count, dtype=np.int64),
        sequences["sample_window_starts"].tolist()))
    radii = rules["ood_radii"]
    count = min(128, len(indices))
    noise = np.stack([rng.uniform(-radius, radius, (count, 14)).astype(np.float32)
                      for radius in radii])
    online.fixed_critic_diagnostic_set = {
        "seed": seed, "source": "branch-local online stratified reservoir",
        "episodes": episodes, "sequences": sequences, "indices": indices,
        "ood_noise": noise, "created_from_episode_count": sum(online.diagnostic_seen.values()),
        "success_episode_count": len(reservoirs[True]),
        "failure_episode_count": len(reservoirs[False]),
    }
    return online.fixed_critic_diagnostic_set
