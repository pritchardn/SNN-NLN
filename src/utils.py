import os

from coolname import generate_slug


def generate_model_name(config_vals: dict):
    model_name = f'{config_vals["model_type"]}_{config_vals["anomaly_type"]}_' \
                 f'{config_vals["dataset"]}_{config_vals["latent_dimension"]}_' \
                 f'{config_vals["num_layers"]}_' \
                 f'{config_vals["threshold"]}'
    if config_vals['excluded_rfi']:
        model_name += f'_{config_vals["excluded_rfi"]}'
    if config_vals['time_length']:
        model_name += f'_{config_vals["time_length"]}'
    if config_vals['average_n']:
        model_name += f'_{config_vals["average_n"]}'
    if config_vals['trial']:  # WARNING: Will not work if trial is 0
        model_name += f'_trial_{config_vals["trial"]}'
    model_name += f'_{generate_slug(2)}'
    return model_name


def generate_output_dir(config_vals: dict):
    output_dir = os.path.join("outputs", config_vals["model_type"], config_vals["anomaly_type"],
                              config_vals["model_name"])
    return output_dir
