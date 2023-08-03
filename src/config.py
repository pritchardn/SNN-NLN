"""
Contains global variables used throughout the project.
Copyright (c) 2023, Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""
import os

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STANDARD_PARAMS = {
    "batch_size": 64,
    "epochs": 15,
    "ae_learning_rate": 0.0004215152200409049,
    "gen_learning_rate": 0.0008860574544781892,
    "disc_learning_rate": 0.0009173638064067504,
    "optimizer": "RMSprop",
    "num_layers": 5,
    "latent_dimension": 32,
    "num_filters": 16,
    "neighbours": 17,
    "patch_size": 32,
    "patch_stride": 32,
    "threshold": 10,
    "anomaly_type": "MISO",
    "dataset": "HERA",
    "model_type": "DAE",
    "excluded_rfi": None,
    "time_length": None,
    "average_n": None,
    "trial": 1,
}


def get_output_dir():
    return os.environ.get("OUTPUT_DIR", "outputs")


def get_data_dir():
    return os.environ.get("DATA_DIR", "data")
