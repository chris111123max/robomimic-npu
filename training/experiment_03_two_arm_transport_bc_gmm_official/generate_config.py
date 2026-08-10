"""Generate the official-style BC-GMM reproduction config for TwoArmTransport PH low_dim."""

from pathlib import Path
import sys


EXPERIMENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = EXPERIMENT_DIR.parents[2]
ROBOMIMIC_DIR = WORKSPACE_DIR / "robomimic"
DATASET_DIR = WORKSPACE_DIR / "datasets"
OUTPUT_DIR = WORKSPACE_DIR / "training_runs"

DATASET_PATH = (
    WORKSPACE_DIR
    / "datasets"
    / "transport"
    / "PH"
    / "low_dim_v15.hdf5"
)

if str(ROBOMIMIC_DIR) not in sys.path:
    sys.path.insert(0, str(ROBOMIMIC_DIR))

from robomimic.config import config_factory  # noqa: E402
from robomimic.scripts.generate_paper_configs import (  # noqa: E402
    modify_bc_config_for_dataset,
    modify_config_for_dataset,
    modify_config_for_default_low_dim_exp,
)


def build_config():
    """Build robomimic's official Transport-PH low_dim BC-GMM paper configuration."""
    config = config_factory(algo_name="bc")

    config = modify_config_for_default_low_dim_exp(config)
    config = modify_config_for_dataset(
        config=config,
        task_name="transport",
        dataset_type="ph",
        hdf5_type="low_dim",
        base_dataset_dir=str(DATASET_DIR),
    )
    config = modify_bc_config_for_dataset(
        config=config,
        task_name="transport",
        dataset_type="ph",
        hdf5_type="low_dim",
    )

    # Keep the official algorithm / training settings, but adapt machine-specific
    # paths and experiment identity to this Ascend workspace.
    with config.experiment.values_unlocked():
        config.experiment.name = "two_arm_transport_bc_gmm_official_ph_low_dim"
        config.experiment.ckpt_path = None

    with config.train.values_unlocked():
        # The server's actual directory is uppercase "PH".
        config.train.data = str(DATASET_PATH)
        config.train.output_dir = str(
            OUTPUT_DIR / "two_arm_transport_bc_gmm_official_ph_low_dim"
        )

    # Defensive assertions: this experiment exists specifically to reproduce
    # the official-style feed-forward BC-GMM setting.
    assert config.algo.gmm.enabled is True
    assert config.algo.gmm.num_modes == 5
    assert config.algo.gaussian.enabled is False
    assert config.algo.rnn.enabled is False
    assert config.train.batch_size == 100
    assert config.train.num_epochs == 2000
    assert config.experiment.epoch_every_n_steps == 100
    assert config.experiment.rollout.enabled is True
    assert config.experiment.rollout.n == 50
    assert config.experiment.rollout.rate == 50
    assert config.experiment.rollout.horizon == 700

    return config


def main():
    config_path = EXPERIMENT_DIR / "config.json"
    config = build_config()
    config.dump(filename=str(config_path))

    print("Generated official-style BC-GMM config:")
    print(f"  config      : {config_path}")
    print(f"  experiment  : {config.experiment.name}")
    print(f"  dataset     : {config.train.data}")
    print(f"  output_dir  : {config.train.output_dir}")
    print(f"  batch_size  : {config.train.batch_size}")
    print(f"  num_epochs  : {config.train.num_epochs}")
    print(f"  steps/epoch : {config.experiment.epoch_every_n_steps}")
    print(f"  rollout     : {config.experiment.rollout.enabled}")
    print(f"  rollout_n   : {config.experiment.rollout.n}")
    print(f"  rollout_rate: {config.experiment.rollout.rate}")
    print(f"  gmm         : {config.algo.gmm.enabled}")
    print(f"  gmm_modes   : {config.algo.gmm.num_modes}")


if __name__ == "__main__":
    main()
