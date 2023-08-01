"""
Contains global variables used throughout the proejct.
Copyright (c) 2023, Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
