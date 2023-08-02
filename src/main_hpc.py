import os
import sys

import config
from main import main


def print_incorrect_usage(var_range: list, num_trials: int):
    print("Incorrect usage")
    print(f"Need at least {len(var_range) * num_trials} tasks")
    print(f"Number of tasks does not work with number of trials")


def main_hpc():
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT"))
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    task_type = os.environ.get("TASK_TYPE")
    num_trials = int(os.environ.get("NUM_TRIALS"))
    config.OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
    config.INPUT_DIR = os.environ.get("INPUT_DIR")
    config.DATA_DIR = os.environ.get("DATA_DIR")
    config_vals = config.STANDARD_PARAMS
    if task_type == "NOISE":
        rfi_exclusion_vals = [
            None,
            "rfi_stations",
            "rfi_dtv",
            "rfi_impulse",
            "rfi_scatter",
        ]
        if num_tasks != len(rfi_exclusion_vals) * num_trials:
            print_incorrect_usage(rfi_exclusion_vals, num_trials)
            sys.exit(1)
        rfi_index = task_id // num_trials
        config_vals["excluded_rfi"] = rfi_exclusion_vals[rfi_index]
        config_vals["trial"] = task_id % num_trials + 1
    elif task_type == "THRESHOLD":
        threshold_range = [0.5, 1, 3, 5, 7, 9, 10, 20, 50, 100, 200]
        if num_tasks != len(threshold_range) * num_trials:
            print_incorrect_usage(threshold_range, num_trials)
            sys.exit(1)
        threshold_index = task_id // num_trials
        config_vals["threshold"] = threshold_range[threshold_index]
        config_vals["trial"] = task_id % num_trials + 1
    else:  # Standard
        config_vals["trial"] = task_id + 1
    import json

    print(json.dumps(config_vals, indent=4))
    print(config.OUTPUT_DIR)
    main(config_vals)


if __name__ == "__main__":
    main_hpc()
