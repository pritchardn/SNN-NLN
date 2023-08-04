"""
Contains post-processing plot generating functions for the project.
Copyright (c) 2023, Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""

import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import get_output_dir


def write_csv_output_from_dict(
    outputdir: str, filename: str, data: list, headers: list
):
    """
    Writes a csv file from a list of dictionaries.
    :filename: The name of the file without file extension.
    """
    os.makedirs(outputdir, exist_ok=True)
    with open(f"{outputdir}{os.sep}{filename}.csv", "w", encoding="utf-8") as ofile:
        csv_writer = csv.DictWriter(ofile, fieldnames=headers)
        csv_writer.writeheader()
        for row in data:
            csv_writer.writerow(row)


def collate_results(outputdir: str, models: list) -> list:
    """
    Collates results from the output directory.
    Only collates results for models in the models list.
    :returns: A list of dictionaries containing the results.
    """
    result_list = []
    for model in models:
        model_outputdir = os.path.join(outputdir, model, "MISO")
        if not os.path.exists(model_outputdir):
            continue
        for filename in os.listdir(model_outputdir):
            trial_vals = {}
            config_filename = os.path.join(model_outputdir, filename, "config.json")
            if not os.path.exists(config_filename):
                continue
            with open(config_filename, "r", encoding="utf-8") as config_file:
                config_data = json.load(config_file)
                trial_vals.update(config_data)
            metric_filename = os.path.join(model_outputdir, filename, "metrics.json")
            with open(metric_filename, "r", encoding="utf-8") as config_file:
                result_data = json.load(config_file)
                if "nln" in result_data:
                    trial_vals.update(result_data["nln"])
                else:
                    trial_vals.update(result_data)
            print(trial_vals)
            result_list.append(trial_vals)
    return result_list


def make_threshold_plot(dframe: pd.DataFrame):
    """
    Makes a plot of the results for different thresholds.
    """
    sub_results = dframe[pd.isna(dframe["excluded_rfi"])]
    models = dframe["model_type"].unique()
    width = 0.25  # 1/len(models)
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    i = -1
    # TODO: Multiple trials and therefore std-dev error bars
    # TODO: Filter dframe for best time_length and average_n
    # TODO: Better legend placement
    num_ticks = len(sub_results["threshold"].unique())
    xticks = np.arange(0, num_ticks, 1)
    for model in models:
        model_results = (
            sub_results[sub_results["model_type"] == model]
            .groupby("threshold")
            .agg(
                {
                    "auroc": ["mean", "std"],
                    "auprc": ["mean", "std"],
                    "f1": ["mean", "std"],
                }
            )
        )
        xvals = np.arange(0, len(model_results.index), 1)
        axes[0].bar(
            xvals + width / 2 * i,
            model_results["auroc"]["mean"],
            width=width,
            label=model,
            yerr=model_results["auroc"]["std"],
        )
        axes[1].bar(
            xvals + width / 2 * i,
            model_results["auprc"]["mean"],
            width=width,
            label=model,
            yerr=model_results["auroc"]["std"],
        )
        axes[2].bar(
            xvals + width / 2 * i,
            model_results["f1"]["mean"],
            width=width,
            label=model,
            yerr=model_results["auroc"]["std"],
        )
        i = -i
    axes[1].legend()
    axes[0].set_ylabel("AUROC")
    axes[1].set_ylabel("AUPRC")
    axes[2].set_ylabel("F1")
    xticklabels = sorted(sub_results["threshold"].unique().astype(int))
    xticklabels[0] = 0.5
    for axis in axes:
        axis.set_xticks(xticks)
        axis.set_xticklabels(xticklabels)
        axis.set_xlabel("AOFlagger Threshold")
    plt.savefig(os.path.join(get_output_dir(), "threshold_plot.png"), dpi=300)
    plt.close("all")


def make_ood_plot(dframe: pd.DataFrame):
    """
    Makes a plot of out-of-distrubtion results.
    """
    sub_results = dframe[pd.notna(dframe["excluded_rfi"])]
    models = dframe["model_type"].unique()
    width = 0.25  # 1/len(models)

    _, axes = plt.subplots(3, 1, figsize=(5, 10), sharex=True)
    i = -1
    index = np.arange(len(sub_results["excluded_rfi"].unique()))
    for model in models:
        model_results = (
            sub_results[sub_results["model_type"] == model]
            .groupby("excluded_rfi")
            .agg(
                {
                    "auroc": ["mean", "std"],
                    "auprc": ["mean", "std"],
                    "f1": ["mean", "std"],
                }
            )
            .sort_values("excluded_rfi")
        )
        axes[0].bar(
            index + width / 2 * i,
            model_results["auroc"]["mean"],
            width=width,
            label=model,
            yerr=model_results["auroc"]["std"],
        )
        axes[1].bar(
            index + width / 2 * i,
            model_results["auprc"]["mean"],
            width=width,
            label=model,
            yerr=model_results["auprc"]["std"],
        )
        axes[2].bar(
            index + width / 2 * i,
            model_results["f1"]["mean"],
            width=width,
            label=model,
            yerr=model_results["f1"]["std"],
        )
        i = -i

    axes[0].legend()
    axes[0].set_ylabel("AUROC")
    axes[1].set_ylabel("AUPRC")
    axes[2].set_ylabel("F1")
    axes[2].set_xlabel("Excluded RFI")
    axes[2].set_xticks(index)
    axes[2].set_xticklabels(sub_results["excluded_rfi"].unique())
    plt.savefig(os.path.join(get_output_dir(), "ood_plot.png"), dpi=300)
    plt.close("all")


def make_inferencetime_plot(dframe: pd.DataFrame):
    """
    Makes a plot of inference time results. Only for SDAE.
    """
    pandas_filter = (
        (dframe.model_type == "SDAE")
        & (dframe.excluded_rfi.isna())
        & (dframe.threshold == 10)
    )
    sub_results = dframe[pandas_filter]
    print(len(sub_results))
    _, axes = plt.subplots(3, 1, figsize=(5, 10), sharex=True)
    width = 0.5  # 1/len(sub_results["average_n"].unique())
    i = -1
    for time_length in sub_results["time_length"].unique():
        inference_results = sub_results[sub_results["time_length"] == time_length]
        inference_results = inference_results.groupby("average_n").mean(
            "auroc", "auprc", "f1"
        )
        axes[0].bar(
            inference_results.index + width / 2 * i,
            inference_results["auroc"],
            width=width,
            label=time_length,
        )
        axes[1].bar(
            inference_results.index + width / 2 * i,
            inference_results["auprc"],
            width=width,
            label=time_length,
        )
        axes[2].bar(
            inference_results.index + width / 2 * i,
            inference_results["f1"],
            width=width,
            label=time_length,
        )
        i = -i
    axes[0].legend()
    axes[0].set_ylabel("AUROC")
    axes[1].set_ylabel("AUPRC")
    axes[2].set_ylabel("F1")
    axes[2].set_xlabel("Slice Length")
    axes[2].set_xticks(sub_results["average_n"].unique())
    plt.savefig(os.path.join(get_output_dir(), "inferencetime_plot.png"), dpi=300)
    plt.close("all")


def collate_results_to_file(models: list, output_filename: str = "results"):
    """
    Collates results and writes to csv file.
    :param: output_filename: Output filename without file extension.
    """
    result_set = collate_results("outputs", models)
    write_csv_output_from_dict(
        "outputs", output_filename, result_set, result_set[0].keys()
    )


def post_process(
    dae_threshold_name,
    dae_noise_name,
    sdae_threshold_name,
    sdae_noise_name,
    sdae_name=None,
):
    # Make threshold plot
    collate_results_to_file(
        [dae_threshold_name, sdae_threshold_name], output_filename="results_threshold"
    )
    results = pd.read_csv("outputs/results_threshold.csv")
    make_threshold_plot(results)
    # Make ood plot
    collate_results_to_file(
        [dae_noise_name, sdae_noise_name], output_filename="results_noise"
    )
    results = pd.read_csv("outputs/results_noise.csv")
    make_ood_plot(results)
    # Make inference time plot
    sdae_models = [sdae_threshold_name, sdae_noise_name]
    if sdae_name:
        sdae_models.append(sdae_name)
    collate_results_to_file(sdae_models, output_filename="results_inferencetime")
    results = pd.read_csv("outputs/results_inferencetime.csv")
    make_inferencetime_plot(results)


if __name__ == "__main__":
    post_process(
        "DAE-THRESHOLD-08-01",
        "DAE-NOISE-08-01",
        "SDAE-THRESHOLD-08-01",
        "SDAE-NOISE-08-01",
        "SDAE",
    )
