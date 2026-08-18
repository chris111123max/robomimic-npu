"""Capture and restore Python, NumPy, Torch CPU, and accelerator RNG states."""

import pickle
import random

import numpy as np


def seed_all(seed):
    import torch
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    accelerator = "cpu"
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
        accelerator = "npu"
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        accelerator = "cuda"
    return accelerator


def seed_policy_rng(seed):
    """Seed only Torch streams; do not perturb robosuite's Python/NumPy stream."""
    import torch
    seed = int(seed)
    torch.manual_seed(seed)
    accelerator = "cpu"
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
        accelerator = "npu"
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        accelerator = "cuda"
    return accelerator


def capture_rng_state():
    import torch
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu().numpy(),
        "accelerator": None,
        "accelerator_states": None,
    }
    try:
        if hasattr(torch, "npu") and torch.npu.is_available() and hasattr(torch.npu, "get_rng_state_all"):
            state["accelerator"] = "npu"
            state["accelerator_states"] = [item.cpu().numpy() for item in torch.npu.get_rng_state_all()]
        elif torch.cuda.is_available():
            state["accelerator"] = "cuda"
            state["accelerator_states"] = [item.cpu().numpy() for item in torch.cuda.get_rng_state_all()]
    except Exception as exc:
        state["accelerator_capture_error"] = f"{type(exc).__name__}: {exc}"
    return state


def restore_rng_state(state):
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8, device="cpu"))
    kind, values = state.get("accelerator"), state.get("accelerator_states")
    if kind is None or values is None:
        return False, state.get("accelerator_capture_error", "accelerator RNG API unavailable")
    tensors = [torch.as_tensor(item, dtype=torch.uint8, device="cpu") for item in values]
    try:
        if kind == "npu" and hasattr(torch.npu, "set_rng_state_all"):
            torch.npu.set_rng_state_all(tensors)
        elif kind == "cuda":
            torch.cuda.set_rng_state_all(tensors)
        else:
            return False, f"{kind} RNG restoration API unavailable"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def encode_rng_state(state):
    return np.frombuffer(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL), dtype=np.uint8)


def decode_rng_state(array):
    return pickle.loads(np.asarray(array, dtype=np.uint8).tobytes())
