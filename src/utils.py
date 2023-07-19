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


def model_name_to_config_vals(dirname: str) -> dict:
    traits = dirname.split("_")
    config_vals = {"model_type": traits[0],
                   "anomaly_type": traits[1], "dataset": traits[2],
                   "latent_dimension": int(traits[3]),
                   "num_layers": int(traits[4]), "threshold": int(traits[5])}
    if config_vals["model_type"] == "DAE":
        if len(traits) > 6:
            config_vals["excluded_rfi"] = traits[6].join("_").join(traits[6:])
        else:
            config_vals["excluded_rfi"] = ""
        config_vals["time_length"] = ""
        config_vals["average_n"] = ""
    elif config_vals["model_type"] == "SDAE":
        if len(traits) > 8:
            config_vals["excluded_rfi"] = traits[6].join("_").join(traits[6:8])
            config_vals["time_length"] = int(traits[8])
            config_vals["average_n"] = int(traits[9])
        else:
            config_vals["time_length"] = int(traits[6])
            config_vals["average_n"] = int(traits[7])
            config_vals["excluded_rfi"] = ""
    return config_vals


def generate_output_dir(config_vals: dict):
    output_dir = os.path.join("outputs", config_vals["model_type"], config_vals["anomaly_type"],
                              config_vals["model_name"])
    return output_dir
