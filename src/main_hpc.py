import glob
import json
import os
import sys

import config
from ann2snn import main_snn
from main import main
from optuna_trials import main_optuna


def print_incorrect_usage(var_range: list, num_trials: int):
    print("Incorrect usage")
    print(f"Need at least {len(var_range) * num_trials} tasks")
    print(f"Number of tasks does not work with number of trials")


def main_hpc():
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT"))
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
    task_type = os.environ.get("TASK_TYPE")
    num_trials = int(os.environ.get("NUM_TRIALS"))
    dataset = os.environ.get("DATASET")
    config_vals = config.STANDARD_PARAMS
    config_vals["dataset"] = dataset
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
        config_vals["model_type"] = "DAE-NOISE"
    elif task_type == "THRESHOLD":
        threshold_range = [0.5, 1, 3, 5, 7, 9, 10, 20, 50, 100, 200]
        if num_tasks != len(threshold_range) * num_trials:
            print_incorrect_usage(threshold_range, num_trials)
            sys.exit(1)
        threshold_index = task_id // num_trials
        config_vals["threshold"] = threshold_range[threshold_index]
        config_vals["trial"] = task_id % num_trials + 1
        config_vals["model_type"] = "DAE-THRESHOLD"
    elif task_type == "SNN":
        model_type = os.environ.get("MODEL_TYPE", "DAE")
        time_length = int(
            os.environ.get("TIME_LENGTH", config.SNN_PARAMS["time_length"])
        )
        average_n = int(os.environ.get("AVERAGE_N", config.SNN_PARAMS["average_n"]))
        out_model_type = "S" + model_type
        model_trials = sorted(
            glob.glob(os.path.join(config.get_output_dir(), model_type, "MISO", "*"))
        )
        if num_tasks > len(model_trials):
            print_incorrect_usage(model_trials, num_tasks)
            sys.exit(1)
        input_dir = model_trials[task_id]
        print(input_dir)
        main_snn(
            input_dir,
            out_model_type=out_model_type,
            plot=False,
            time_length=time_length,
            average_n=average_n,
        )
        sys.exit(0)
    elif task_type == "OPTUNA":
        main_optuna()
        sys.exit(0)
    else:  # Standard
        config_vals["trial"] = task_id + 1
    print(json.dumps(config_vals, indent=4))
    print(config.get_output_dir())
    print(config.get_data_dir())
    print(task_id)
    main(config_vals)


if __name__ == "__main__":
    main_hpc()
