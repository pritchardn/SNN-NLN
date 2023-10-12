"""
Contains global variables used throughout the project.
Copyright (c) 2023, Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""
import os

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VERBOSE = True

HERA_PARAMS = {
    "batch_size": 32,
    "epochs": 17,
    "ae_learning_rate": 0.001589310185217798,
    "gen_learning_rate": 0.0017737115327998176,
    "disc_learning_rate": 0.00016792476164863897,
    "optimizer": "Adam",
    "num_layers": 2,
    "latent_dimension": 64,
    "num_filters": 16,
    "neighbours": 21,
    "patch_size": 32,
    "patch_stride": 32,
    "threshold": 10,
    "anomaly_type": "MISO",
    "dataset": "HERA",
    "model_type": "DAE",
    "regularize": False,
    "excluded_rfi": None,
    "time_length": None,
    "average_n": None,
    "trial": 1,
}

LOFAR_PARAMS = {
    "batch_size": 16,
    "epochs": 79,
    "ae_learning_rate": 0.007166799047204213,
    "gen_learning_rate": 0.002089832836503927,
    "disc_learning_rate": 0.0033327432207109144,
    "optimizer": "RMSprop",
    "num_layers": 2,
    "latent_dimension": 16,
    "num_filters": 16,
    "neighbours": 22,
    "patch_size": 32,
    "patch_stride": 32,
    "threshold": 10,
    "anomaly_type": "MISO",
    "dataset": "LOFAR",
    "model_type": "DAE",
    "regularize": True,
    "excluded_rfi": None,
    "time_length": None,
    "average_n": None,
    "convert_threshold": None,
    "trial": 1,
}

TABASCAL_PARAMS = {
    "batch_size": 16,
    "epochs": 13,
    "ae_learning_rate": 0.00042580710933189434,
    "gen_learning_rate": 0.009392349641041532,
    "disc_learning_rate": 0.005050946755870783,
    "optimizer": "Adam",
    "num_layers": 2,
    "latent_dimension": 16,
    "num_filters": 16,
    "neighbours": 22,
    "patch_size": 32,
    "patch_stride": 32,
    "threshold": 10,
    "anomaly_type": "MISO",
    "dataset": "TABASCAL",
    "model_type": "DAE",
    "regularize": False,
    "excluded_rfi": None,
    "time_length": None,
    "average_n": None,
    "trial": 1,
}


def get_dataset_params(dataset: str):
    if dataset == "HERA":
        return HERA_PARAMS
    elif dataset == "LOFAR":
        return LOFAR_PARAMS
    elif dataset == "TABASCAL":
        return TABASCAL_PARAMS
    else:
        raise ValueError(f"Unknown dataset {dataset}")


SNN_PARAMS = {
    "average_n": 128,
    "time_length": 256,
    "convert_threshold": "99.9%",
}


def get_output_dir():
    return os.environ.get("OUTPUT_DIR", "outputs")


def get_data_dir():
    return os.environ.get("DATA_DIR", "data")
