import os


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
    return model_name


def generate_output_dir(config_vals: dict):
    output_dir = os.path.join("outputs", config_vals["model_type"], config_vals["anomaly_type"],
                              config_vals["model_name"])
    return output_dir
