import glob
import json

import pandas as pd
import os
import csv
import matplotlib.pyplot as plt

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


def make_threshold_plot(results: pd.DataFrame):
    print(results.head())
    print(len(results))
    sub_results = results[pd.isna(results["excluded_rfi"])]
    print(results["excluded_rfi"].unique())
    print(len(sub_results))
    models = results["model_type"].unique()
    width = 0.25  # 1/len(models)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    i = -1
    # TODO: Multiple trials and therefore std-dev error bars
    # TODO: Filter results for best time_length and average_n
    for model in models:
        model_results = sub_results[sub_results["model_type"] == model].groupby("threshold").mean(
            "auroc", "auprc", "f1")
        axs[0].bar(model_results.index - width / 2 * i, model_results["auroc"], width=width,
                   label=model)
        axs[1].bar(model_results.index - width / 2 * i, model_results["auprc"], width=width,
                   label=model)
        axs[2].bar(model_results.index - width / 2 * i, model_results["f1"], width=width,
                   label=model)
        i = -i
        print(model_results)
    axs[0].legend()
    axs[0].set_ylabel("AUROC")
    axs[1].set_ylabel("AUPRC")
    axs[2].set_ylabel("F1")
    for ax in axs:
        ax.set_xlabel("AOFlagger Threshold")
    plt.savefig("outputs/threshold_plot.png", dpi=300)


def collate_reuslts():
    results = collate_results("outputs")
    write_csv_output_from_dict("outputs", "results", results, results[0].keys())


if __name__ == "__main__":
    make_threshold_plot(pd.read_csv("outputs/results.csv"))
