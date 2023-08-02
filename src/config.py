"""
Contains global variables used throughout the project.
Copyright (c) 2023, Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STANDARD_PARAMS = {
    "batch_size": 16,
    "epochs": 5,
    "ae_learning_rate": 1.89e-4,
    "gen_learning_rate": 7.90e-4,
    "disc_learning_rate": 9.49e-4,
    "optimizer": "RMSprop",
    "num_layers": 5,
    "latent_dimension": 32,
    "num_filters": 16,
    "neighbours": 20,
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
