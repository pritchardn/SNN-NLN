"""
Contains simple functions used throughout the project.
Copyright (c) 2023 Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""
import json
import os

from coolname import generate_slug


def generate_model_name(config_vals: dict) -> str:
    """
    Generates a unique filename for a model based on the config values.
    :param config_vals: The values used to configure the model
    :return: A unique filename for the model
    """
    model_name = (
        f'{config_vals["model_type"]}_{config_vals["anomaly_type"]}_'
        f'{config_vals["dataset"]}_{config_vals["latent_dimension"]}_'
        f'{config_vals["num_layers"]}_'
        f'{config_vals["threshold"]}'
    )
    if config_vals["excluded_rfi"]:
        model_name += f'_{config_vals["excluded_rfi"]}'
    if config_vals["time_length"]:
        model_name += f'_{config_vals["time_length"]}'
    if config_vals["average_n"]:
        model_name += f'_{config_vals["average_n"]}'
    if config_vals["trial"]:  # WARNING: Will not work if trial is 0
        model_name += f'_trial_{config_vals["trial"]}'
    model_name += f"_{generate_slug(2)}"
    return model_name


def generate_output_dir(config_vals: dict) -> str:
    """
    Generates an output directory for a model based on the config values. Groups models together
    by model type, anomaly type and model name.
    :param config_vals: The config values used to configure the model
    :return: An output directory
    """
    output_dir = os.path.join(
        "outputs",
        config_vals["model_type"],
        config_vals["anomaly_type"],
        config_vals["model_name"],
    )
    return output_dir


def save_config(config: dict, output_dir: str):
    """
    Saves the config to a json file.
    """
    with open(
        os.path.join(output_dir, "config.json"), "w", encoding="utf-8"
    ) as config_file:
        json.dump(config, config_file, indent=4)
