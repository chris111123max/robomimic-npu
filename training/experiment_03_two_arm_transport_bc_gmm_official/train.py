"""Launch official-style robomimic BC-GMM for Transport PH low-dimensional data."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid


EXPERIMENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = EXPERIMENT_DIR.parents[2]
ROBOMIMIC_TRAIN = WORKSPACE_DIR / "robomimic" / "robomimic" / "scripts" / "train.py"
CONFIG_PATH = EXPERIMENT_DIR / "config.json"

EXPECTED_DATASET = (
    WORKSPACE_DIR
    / "datasets"
    / "transport"
    / "PH"
    / "low_dim_v15.hdf5"
)

OUTPUT_DIR = (
    WORKSPACE_DIR
    / "training_runs"
    / "two_arm_transport_bc_gmm_official_ph_low_dim"
)

CACHE_DIR = WORKSPACE_DIR / ".cache"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def make_runtime_config(config):
    """Replace only machine-specific paths; preserve all algorithm parameters."""
    config = json.loads(json.dumps(config))
    config["train"]["data"] = str(EXPECTED_DATASET)
    config["train"]["output_dir"] = str(OUTPUT_DIR)
    return config


def make_smoke_config(config):
    """Small launch check only; it is not an official training run."""
    config = json.loads(json.dumps(config))
    config["experiment"]["name"] += f"_smoke_{uuid.uuid4().hex[:8]}"
    config["experiment"]["epoch_every_n_steps"] = 3
    config["experiment"]["validation_epoch_every_n_steps"] = 3
    config["experiment"]["rollout"]["enabled"] = False
    config["experiment"]["render_video"] = False
    config["experiment"]["validate"] = False
    config["train"]["num_epochs"] = 2
    config["train"]["output_dir"] = str(WORKSPACE_DIR / "training_runs" / "smoke")
    return config


def validate_runtime_config(config):
    """Fail early if this experiment drifts away from the intended official setting."""
    expected = {
        "batch_size": 100,
        "num_epochs": 2000,
        "steps_per_epoch": 100,
        "gmm_enabled": True,
        "gmm_modes": 5,
        "gaussian_enabled": False,
        "rnn_enabled": False,
    }

    actual = {
        "batch_size": config["train"]["batch_size"],
        "num_epochs": config["train"]["num_epochs"],
        "steps_per_epoch": config["experiment"]["epoch_every_n_steps"],
        "gmm_enabled": config["algo"]["gmm"]["enabled"],
        "gmm_modes": config["algo"]["gmm"]["num_modes"],
        "gaussian_enabled": config["algo"]["gaussian"]["enabled"],
        "rnn_enabled": config["algo"]["rnn"]["enabled"],
    }

    if actual != expected:
        raise RuntimeError(
            "Experiment 03 configuration no longer matches the intended "
            f"official-style BC-GMM setup.\nExpected: {expected}\nActual: {actual}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if not ROBOMIMIC_TRAIN.is_file():
        raise FileNotFoundError(
            f"robomimic training entry point not found: {ROBOMIMIC_TRAIN}"
        )

    if not EXPECTED_DATASET.is_file():
        raise FileNotFoundError(
            f"Transport dataset not found: {EXPECTED_DATASET}"
        )

    runtime_config = make_runtime_config(load_config())

    if args.smoke_test:
        runtime_config = make_smoke_config(runtime_config)
    else:
        validate_runtime_config(runtime_config)

    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="two_arm_transport_bc_gmm_official_runtime_",
        dir=EXPERIMENT_DIR,
        encoding="utf-8",
        delete=False,
    )
    temporary_path = Path(temporary.name)

    with temporary:
        json.dump(runtime_config, temporary, ensure_ascii=False, indent=4)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "HF_HOME": str(CACHE_DIR / "huggingface"),
            "MPLCONFIGDIR": str(CACHE_DIR / "matplotlib"),
            "TORCH_HOME": str(CACHE_DIR / "torch"),
        }
    )

    print("==============================================")
    print("Experiment 03: official-style BC-GMM")
    print("==============================================")
    print("dataset      :", EXPECTED_DATASET)
    print("output_dir   :", runtime_config["train"]["output_dir"])
    print("batch_size   :", runtime_config["train"]["batch_size"])
    print("num_epochs   :", runtime_config["train"]["num_epochs"])
    print("steps/epoch  :", runtime_config["experiment"]["epoch_every_n_steps"])
    print("rollout      :", runtime_config["experiment"]["rollout"]["enabled"])
    print("rollout_n    :", runtime_config["experiment"]["rollout"]["n"])
    print("rollout_rate :", runtime_config["experiment"]["rollout"]["rate"])
    print("GMM          :", runtime_config["algo"]["gmm"]["enabled"])
    print("GMM modes    :", runtime_config["algo"]["gmm"]["num_modes"])
    print("==============================================")

    command = [
        sys.executable,
        str(ROBOMIMIC_TRAIN),
        "--config",
        str(temporary_path),
    ]

    if args.resume:
        command.append("--resume")

    try:
        raise SystemExit(
            subprocess.call(
                command,
                cwd=str(WORKSPACE_DIR / "robomimic"),
                env=env,
            )
        )
    finally:
        temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
