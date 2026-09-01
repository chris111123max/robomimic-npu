"""BC-GMM component action conversion used by the Stage3-R target cache."""
from __future__ import annotations

import torch

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.python_utils as PyUtils
import robomimic.utils.torch_utils as TorchUtils


def component_means_to_environment_space(policy, means):
    """Apply the exact RolloutPolicy post-processing to each GMM component mean."""
    if policy.action_normalization_stats is None:
        return means

    original_shape = tuple(means.shape)
    if len(original_shape) != 3:
        raise RuntimeError(f"Expected component means [B,M,A], got {original_shape}")

    flat = means.detach().cpu().numpy().reshape(-1, original_shape[-1])
    action_keys = policy.policy.global_config.train.action_keys
    statistics = policy.action_normalization_stats
    action_shapes = {
        key: statistics[key]["offset"].shape[1:] for key in statistics
    }
    action_dict = PyUtils.vector_to_action_dict(
        flat, action_shapes=action_shapes, action_keys=action_keys
    )
    action_dict = ObsUtils.unnormalize_dict(
        action_dict, normalization_stats=statistics
    )

    action_config = policy.policy.global_config.train.action_config
    for key, value in action_dict.items():
        if action_config[key].get("format") != "rot_6d":
            continue
        rotation_6d = torch.from_numpy(value)
        conversion = action_config[key].get(
            "convert_at_runtime", "rot_axis_angle"
        )
        if conversion == "rot_axis_angle":
            action_dict[key] = TorchUtils.rot_6d_to_axis_angle(
                rot_6d=rotation_6d
            ).numpy()
        elif conversion == "rot_euler":
            action_dict[key] = TorchUtils.rot_6d_to_euler_angles(
                rot_6d=rotation_6d, convention="XYZ"
            ).numpy()
        else:
            raise RuntimeError(
                f"Unsupported rot_6d runtime conversion: {conversion}"
            )

    transformed = PyUtils.action_dict_to_vector(
        action_dict, action_keys=action_keys
    ).reshape(original_shape[0], original_shape[1], -1)
    if transformed.shape[-1] != 14:
        raise RuntimeError(
            f"Environment component action dimension is not 14: {transformed.shape}"
        )
    return torch.as_tensor(
        transformed, device=means.device, dtype=means.dtype
    )
