"""
Contains simple functions used throughout the project.
Copyright (c) 2023 Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""
import json
import os

import numpy as np
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


def save_json(data: dict, output_dir: str, filename: str):
    """
    Saves a dictionary to a json file at output_dir as filename.json.
    """
    with open(
        os.path.join(output_dir, f"{filename}.json"), "w", encoding="utf-8"
    ) as ofile:
        json.dump(data, ofile, indent=4)


def load_config(input_dir: str):
    """
    Loads the config file from the input directory
    """
    config_file_path = os.path.join(input_dir, "config.json")
    with open(config_file_path, "r", encoding="utf-8") as ifile:
        config = json.load(ifile)
    return config


def scale_image(image):
    """
    Scales an image between its min and max values.
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))
