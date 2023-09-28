"""
Contains post-processing plot generating functions for the project.
Copyright (c) 2023, Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""

import csv
import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import get_output_dir


def translate_rfi(xticks: list):
    outputs = []
    for label in xticks:
        if label == "rfi_stations":
            outputs.append("Narrow-band Burst")
        if label == "rfi_scatter":
            outputs.append("Blips")
        if label == "rfi_impulse":
            outputs.append("Broad-band Transient")
        if label == "rfi_dtv":
            outputs.append("Broad-band Continuous")
    return outputs


def model_to_label(model: str):
    out = model.replace("SDAE", "SNLN")
    out = out.replace("DAE", "NLN")
    out = out.replace("-NOISE", "")
    out = out.replace("-THRESHOLD", "")
    return out


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
            if trial_vals["time_length"]:
                trial_vals[
                    "model_type"
                ] = f"{trial_vals['model_type']}-{trial_vals['time_length']}"
            if trial_vals["average_n"]:
                trial_vals[
                    "model_type"
                ] = f"{trial_vals['model_type']}-{trial_vals['average_n']}"
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
            label=model_to_label(model),
            yerr=model_results["auroc"]["std"],
        )
        axes[1].bar(
            xvals + width / 2 * i,
            model_results["auprc"]["mean"],
            width=width,
            label=model_to_label(model),
            yerr=model_results["auroc"]["std"],
        )
        axes[2].bar(
            xvals + width / 2 * i,
            model_results["f1"]["mean"],
            width=width,
            label=model_to_label(model),
            yerr=model_results["auroc"]["std"],
        )
        i = -i
    axes[2].legend(loc="upper right", framealpha=1, bbox_to_anchor=(1.0, 1.15))
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
        )
        axes[0].bar(
            index + width / 2 * i,
            model_results["auroc"]["mean"],
            width=width,
            label=model_to_label(model),
            yerr=model_results["auroc"]["std"],
        )
        axes[1].bar(
            index + width / 2 * i,
            model_results["auprc"]["mean"],
            width=width,
            label=model_to_label(model),
            yerr=model_results["auprc"]["std"],
        )
        axes[2].bar(
            index + width / 2 * i,
            model_results["f1"]["mean"],
            width=width,
            label=model_to_label(model),
            yerr=model_results["f1"]["std"],
        )
        i = -i

    axes[0].legend(loc="upper right", framealpha=1, bbox_to_anchor=(1.0, 1.25))
    axes[0].set_ylabel("AUROC")
    axes[1].set_ylabel("AUPRC")
    axes[2].set_ylabel("F1")
    axes[2].set_xlabel("Excluded RFI")
    axes[2].set_xticks(index)
    axes[2].set_xticklabels(
        translate_rfi(sub_results["excluded_rfi"].unique()), rotation=20
    )
    plt.savefig(os.path.join(get_output_dir(), "ood_plot.png"), dpi=300)
    plt.close("all")


def make_inferencetime_plot(dframe: pd.DataFrame):
    """
    Makes a plot of inference time results. Only for SDAE.
    """
    pandas_filter = (dframe.excluded_rfi.isna()) & (dframe.threshold == 10)
    sub_results = dframe[pandas_filter]
    print(len(sub_results))
    _, axes = plt.subplots(3, 1, figsize=(5, 10), sharex=True)
    width = 0.25  # 1/len(sub_results["average_n"].unique())
    i = -1
    index = np.arange(len(sub_results["average_n"].unique()))
    for time_length in sub_results["time_length"].unique():
        inference_results = sub_results[sub_results["time_length"] == time_length]
        inference_results = inference_results.groupby("average_n").agg(
            {
                "auroc": ["mean", "std"],
                "auprc": ["mean", "std"],
                "f1": ["mean", "std"],
            }
        )
        axes[0].bar(
            index + width / 2 * i,
            inference_results["auroc"]["mean"],
            yerr=inference_results["auroc"]["std"],
            width=width,
            label=time_length,
        )
        axes[1].bar(
            index + width / 2 * i,
            inference_results["auprc"]["mean"],
            yerr=inference_results["auprc"]["std"],
            width=width,
            label=time_length,
        )
        axes[2].bar(
            index + width / 2 * i,
            inference_results["f1"]["mean"],
            yerr=inference_results["f1"]["std"],
            width=width,
            label=time_length,
        )
        i = -i
    axes[0].legend()
    axes[0].set_ylabel("AUROC")
    axes[1].set_ylabel("AUPRC")
    axes[2].set_ylabel("F1")
    axes[2].set_xlabel("Slice Length")
    axes[2].set_xticks(index)
    axes[2].set_xticklabels(sub_results["average_n"].unique())
    plt.savefig(os.path.join(get_output_dir(), "inferencetime_plot.png"), dpi=300)
    plt.close("all")


def make_performance_table(dframe: pd.DataFrame):
    """
    Makes a table of the results.
    """
    sub_results = dframe[pd.isna(dframe["excluded_rfi"])]
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    results = sub_results.groupby("model_type").agg(
        {
            "auroc": ["mean", "std"],
            "auprc": ["mean", "std"],
            "f1": ["mean", "std"],
            "mse": ["mean", "std"],
        }
    )
    with open("outputs/performance_table.csv", "w", encoding="utf-8") as ofile:
        results.to_csv(ofile)


def collate_results_to_file(
    input_dir: str, models: list, output_filename: str = "results"
):
    """
    Collates results and writes to csv file.
    :param: output_filename: Output filename without file extension.
    """
    result_set = collate_results(input_dir, models)
    write_csv_output_from_dict(
        "outputs", output_filename, result_set, result_set[-1].keys()
    )


def post_process(
    input_dir: str,
    dae_names: list,
    dae_threshold_names: list,
    dae_noise_names: list,
    sdae_threshold_names: list,
    sdae_noise_names: list,
    sdae_names: list,
):
    # Make threshold plot
    collate_results_to_file(
        input_dir,
        [
            filename
            for filename in itertools.chain.from_iterable(
                [dae_threshold_names, sdae_threshold_names]
            )
        ],
        output_filename="results_threshold",
    )
    results = pd.read_csv("outputs/results_threshold.csv")
    make_threshold_plot(results)
    # Make ood plot
    collate_results_to_file(
        input_dir,
        [
            filename
            for filename in itertools.chain.from_iterable(
                [dae_noise_names, sdae_noise_names]
            )
        ],
        output_filename="results_noise",
    )
    results = pd.read_csv("outputs/results_noise.csv")
    make_ood_plot(results)
    # Make inference time plot
    sdae_models = []
    if sdae_names:
        sdae_models.extend(sdae_names)
    collate_results_to_file(
        input_dir, sdae_models, output_filename="results_inferencetime"
    )
    results = pd.read_csv("outputs/results_inferencetime.csv")
    make_inferencetime_plot(results)
    # Make performance table
    collate_results_to_file(
        input_dir,
        [
            filename
            for filename in itertools.chain.from_iterable([dae_names, sdae_names])
        ],
        output_filename="results",
    )
    results = pd.read_csv("outputs/results.csv")
    make_performance_table(results)


if __name__ == "__main__":
    post_process(
        "outputs/setonix-september-2/outputs/",
        ["DAE"],
        ["DAE-THRESHOLD"],
        ["DAE-NOISE"],
        [
            "SDAE-THRESHOLD-256-128"
        ],  # , "SDAE-THRESHOLD-256-256", "SDAE-THRESHOLD-512-128", "SDAE-THRESHOLD-512-256"],
        [
            "SDAE-NOISE-256-128"
        ],  # , "SDAE-NOISE-256-256", "SDAE-NOISE-512-128", "SDAE-NOISE-512-256"],
        ["SDAE-256-128"],  # , "SDAE-256-256", "SDAE-512-128", "SDAE-512-256"],
    )
