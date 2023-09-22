"""
Replicates all published results from scratch.
Copyright (c) 2023, Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""
import os

from ann2snn import run_and_rename
from config import HERA_PARAMS, get_output_dir
from main import main, main_sweep_threshold, main_sweep_noise
from post_processing import post_process


def replicate(num_trials=10):
    """
    Replicates all published results from scratch.
    Runs through Threshold, Noise and Standard experiments.
    Converts all models to SNNs and evaluates.
    Generates result plots from all of this.
    """
    # Check data directory is present
    if not os.path.exists(os.path.join("data", "HERA_04-03-2022_all.pkl")):
        print("Data file does not exist")
    # Threshold models
    main_sweep_threshold(num_trials)
    os.rename(
        os.path.join(get_output_dir(), "DAE"),
        os.path.join(get_output_dir(), "DAE-THRESHOLD"),
    )
    # Noise models
    main_sweep_noise(num_trials)
    os.rename(
        os.path.join(get_output_dir(), "DAE"),
        os.path.join(get_output_dir(), "DAE-NOISE"),
    )
    # Train standard models
    config_vals = HERA_PARAMS
    for trial in range(num_trials):
        config_vals["trial"] = trial + 1
        main(config_vals)
    # SNN Conversions
    run_and_rename(f"./{get_output_dir()}/DAE/MISO", "SDAE")
    run_and_rename(f"./{get_output_dir()}/DAE-NOISE/MISO", "SDAE-NOISE")
    run_and_rename(f"./{get_output_dir()}/DAE-THRESHOLD/MISO", "SDAE-THRESHOLD")
    # Plotting and reporting
    post_process("DAE-THRESHOLD", "DAE-NOISE", "SDAE-THRESHOLD", "SDAE-NOISE")


if __name__ == "__main__":
    replicate()
