import glob
import json

import pandas as pd
import os
import csv

from utils import model_name_to_config_vals

MODELS = ["SDAE", "DAE"]


def write_csv_output_from_dict(outputdir: str, filename: str, data: list, headers: list):
    os.makedirs(outputdir, exist_ok=True)
    with open(f"{outputdir}{os.sep}{filename}.csv", 'w') as ofile:
        csv_writer = csv.DictWriter(ofile, fieldnames=headers)
        csv_writer.writeheader()
        for row in data:
            csv_writer.writerow(row)


def collate_results(outputdir: str) -> list:
    results = []
    for model in MODELS:
        model_outputdir = os.path.join(outputdir, model, "MISO")
        for filename in os.listdir(model_outputdir):
            trial_vals = model_name_to_config_vals(filename)
            for result_filename in glob.glob(
                    os.path.join(model_outputdir, filename, 'metrics.json')):
                with open(result_filename, 'r') as f:
                    result_data = json.load(f)
                    if "nln" in result_data:
                        trial_vals.update(result_data["nln"])
                    else:
                        trial_vals.update(result_data)
            print(trial_vals)
            results.append(trial_vals)
    return results


def collate_reuslts():
    results = collate_results("outputs")
    write_csv_output_from_dict("outputs", "results", results, results[0].keys())


if __name__ == "__main__":
    collate_reuslts()
