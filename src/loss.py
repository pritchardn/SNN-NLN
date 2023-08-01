"""
Loss functions for the model.
Copyright (c) 2023, Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""

import torch
from torch import nn

mse = nn.MSELoss()


def ae_loss(x_data, y_data):
    """
    Calculates the loss for the autoencoder.
    """
    return mse(x_data, y_data)


def discriminator_loss(x_data, y_data, weight):
    """
    Calculates the loss for the discriminator.
    """
    real_loss = mse(torch.ones_like(x_data), x_data)
    fake_loss = mse(torch.ones_like(y_data), y_data)
    total_loss = real_loss + fake_loss
    return total_loss * weight


def generator_loss(y_data, weight):
    """
    Calculates the loss for the generator.
    """
    return torch.mean(mse(torch.ones_like(y_data), y_data)) * weight
