"""Generate Experiment 04 Vanilla BC from Experiment 03 BC-GMM config.

Experiment 04 is the direct Vanilla BC counterpart of Experiment 03.
It copies Experiment 03's current training / validation / rollout settings
and changes only the experiment identity, output directory, and GMM switch.
"""

import json
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = EXPERIMENT_DIR.parent
ROBOMIMIC_DIR = TRAINING_DIR.parent
WORKSPACE_DIR = ROBOMIMIC_DIR.parent

SOURCE_CONFIG = (
    TRAINING_DIR
    / "experiment_03_two_arm_transport_bc_gmm_official"
    / "config.json"
)

OUTPUT_CONFIG = EXPERIMENT_DIR / "config.json"

DATASET_PATH = (
    WORKSPACE_DIR
    / "datasets"
    / "transport"
    / "PH"
    / "low_dim_v15.hdf5"
)

OUTPUT_DIR = (
    WORKSPACE_DIR
    / "training_runs"
    / "two_arm_transport_bc_official_ph_low_dim"
)


def build_config():
    if not SOURCE_CONFIG.is_file():
        raise FileNotFoundError(
            f"Experiment 03 config not found: {SOURCE_CONFIG}"
        )

    with SOURCE_CONFIG.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config["experiment"]["name"] = "two_arm_transport_bc_official_ph_low_dim"
    config["experiment"]["ckpt_path"] = None

    config["train"]["data"] = str(DATASET_PATH)
    config["train"]["output_dir"] = str(OUTPUT_DIR)

    # Experiment 04 = vanilla deterministic BC.
    # All other Experiment 03 settings are preserved.
    config["algo"]["gmm"]["enabled"] = False
    config["algo"]["gaussian"]["enabled"] = False
    config["algo"]["rnn"]["enabled"] = False
    config["algo"]["vae"]["enabled"] = False
    config["algo"]["transformer"]["enabled"] = False

    return config


def main():
    config = build_config()

    with OUTPUT_CONFIG.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        f.write("\n")

    print("Generated Experiment 04 from Experiment 03:")
    print("  source      :", SOURCE_CONFIG)
    print("  output      :", OUTPUT_CONFIG)
    print("  experiment  :", config["experiment"]["name"])
    print("  dataset     :", config["train"]["data"])
    print("  output_dir  :", config["train"]["output_dir"])
    print("  batch_size  :", config["train"]["batch_size"])
    print("  num_epochs  :", config["train"]["num_epochs"])
    print("  steps/epoch :", config["experiment"]["epoch_every_n_steps"])
    print("  validate    :", config["experiment"]["validate"])
    print("  rollout     :", config["experiment"]["rollout"]["enabled"])
    print("  rollout_n   :", config["experiment"]["rollout"]["n"])
    print("  rollout_rate:", config["experiment"]["rollout"]["rate"])
    print("  gaussian    :", config["algo"]["gaussian"]["enabled"])
    print("  gmm         :", config["algo"]["gmm"]["enabled"])


if __name__ == "__main__":
    main()
