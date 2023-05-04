import torch
import torch.nn as nn

mse = nn.MSELoss()
mae = nn.L1Loss()
bce = nn.BCELoss()


def ae_loss(x, y):
    return mse(x, y)


def discriminator_loss(x, y, weight):
    real_loss = mse(torch.ones_like(x), x)
    fake_loss = mse(torch.ones_like(y), y)
    total_loss = real_loss + fake_loss
    return total_loss * weight


def generator_loss(y, weight):
    return torch.mean(torch.ones_like(y), y) * weight
