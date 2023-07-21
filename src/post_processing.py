import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["SDAE", "DAE"]


def write_csv_output_from_dict(
    outputdir: str, filename: str, data: list, headers: list
):
    os.makedirs(outputdir, exist_ok=True)
    with open(f"{outputdir}{os.sep}{filename}.csv", "w") as ofile:
        csv_writer = csv.DictWriter(ofile, fieldnames=headers)
        csv_writer.writeheader()
        for row in data:
            csv_writer.writerow(row)


def collate_results(outputdir: str) -> list:
    results = []
    for model in MODELS:
        model_outputdir = os.path.join(outputdir, model, "MISO")
        if not os.path.exists(model_outputdir):
            continue
        for filename in os.listdir(model_outputdir):
            trial_vals = {}
            config_filename = os.path.join(model_outputdir, filename, "config.json")
            if not os.path.exists(config_filename):
                continue
            with open(config_filename, "r") as f:
                config_data = json.load(f)
                trial_vals.update(config_data)
            metric_filename = os.path.join(model_outputdir, filename, "metrics.json")
            with open(metric_filename, "r") as f:
                result_data = json.load(f)
                if "nln" in result_data:
                    trial_vals.update(result_data["nln"])
                else:
                    trial_vals.update(result_data)
            print(trial_vals)
            results.append(trial_vals)
    return results


def make_threshold_plot(results: pd.DataFrame):
    sub_results = results[pd.isna(results["excluded_rfi"])]
    models = results["model_type"].unique()
    width = 0.25  # 1/len(models)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    i = -1
    # TODO: Multiple trials and therefore std-dev error bars
    # TODO: Filter results for best time_length and average_n
    # TODO: Better legend placement
    for model in models:
        model_results = (
            sub_results[sub_results["model_type"] == model]
            .groupby("threshold")
            .mean("auroc", "auprc", "f1")
        )
        axs[0].bar(
            model_results.index + width / 2 * i,
            model_results["auroc"],
            width=width,
            label=model,
        )
        axs[1].bar(
            model_results.index + width / 2 * i,
            model_results["auprc"],
            width=width,
            label=model,
        )
        axs[2].bar(
            model_results.index + width / 2 * i,
            model_results["f1"],
            width=width,
            label=model,
        )
        i = -i
    axs[0].legend()
    axs[0].set_ylabel("AUROC")
    axs[1].set_ylabel("AUPRC")
    axs[2].set_ylabel("F1")
    for ax in axs:
        ax.set_xlabel("AOFlagger Threshold")
    plt.savefig("outputs/threshold_plot.png", dpi=300)
    plt.close("all")


def make_ood_plot(results: pd.DataFrame):
    sub_results = results[pd.notna(results["excluded_rfi"])]
    models = results["model_type"].unique()
    width = 0.25  # 1/len(models)

    fig, axs = plt.subplots(3, 1, figsize=(5, 10), sharex=True)
    i = -1
    index = np.arange(len(sub_results["excluded_rfi"].unique()))
    for model in models:
        model_results = (
            sub_results[sub_results["model_type"] == model]
            .groupby("excluded_rfi")
            .mean("auroc", "auprc", "f1")
            .sort_values("excluded_rfi")
        )
        axs[0].bar(
            index + width / 2 * i, model_results["auroc"], width=width, label=model
        )
        axs[1].bar(
            index + width / 2 * i, model_results["auprc"], width=width, label=model
        )
        axs[2].bar(index + width / 2 * i, model_results["f1"], width=width, label=model)
        i = -i

    axs[0].legend()
    axs[0].set_ylabel("AUROC")
    axs[1].set_ylabel("AUPRC")
    axs[2].set_ylabel("F1")
    axs[2].set_xlabel("Excluded RFI")
    axs[2].set_xticks(index)
    axs[2].set_xticklabels(sub_results["excluded_rfi"].unique())
    plt.savefig("outputs/ood_plot.png", dpi=300)
    plt.close("all")


def make_inferencetime_plot(results: pd.DataFrame):
    filter = (
        (results.model_type == "SDAE")
        & (results.excluded_rfi.isna())
        & (results.threshold == 10)
        & (results.latent_dimension == 32)
        & (results.num_layers == 2)
    )
    sub_results = results[filter]
    print(len(sub_results))
    fig, axs = plt.subplots(3, 1, figsize=(5, 10), sharex=True)
    width = 0.5  # 1/len(sub_results["average_n"].unique())
    i = -1
    for time_length in sub_results["time_length"].unique():
        inference_results = sub_results[sub_results["time_length"] == time_length]
        inference_results = inference_results.groupby("average_n").mean(
            "auroc", "auprc", "f1"
        )
        axs[0].bar(
            inference_results.index + width / 2 * i,
            inference_results["auroc"],
            width=width,
            label=time_length,
        )
        axs[1].bar(
            inference_results.index + width / 2 * i,
            inference_results["auprc"],
            width=width,
            label=time_length,
        )
        axs[2].bar(
            inference_results.index + width / 2 * i,
            inference_results["f1"],
            width=width,
            label=time_length,
        )
        i = -i
    axs[0].legend()
    axs[0].set_ylabel("AUROC")
    axs[1].set_ylabel("AUPRC")
    axs[2].set_ylabel("F1")
    axs[2].set_xlabel("Slice Length")
    axs[2].set_xticks(sub_results["average_n"].unique())
    plt.savefig("outputs/inferencetime_plot.png", dpi=300)
    plt.close("all")


def collate_reuslts():
    result_set = collate_results("outputs")
    write_csv_output_from_dict("outputs", "results", result_set, result_set[0].keys())


if __name__ == "__main__":
    collate_reuslts()
    results = pd.read_csv("outputs/results.csv")
    make_threshold_plot(results)
    make_ood_plot(results)
    make_inferencetime_plot(results)
