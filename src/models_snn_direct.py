from collections import OrderedDict

import torch
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        base_channels: int,
        latent_dimension: int,
        regularize: bool,
        time_steps: int,
        tau: float,
    ):
        super().__init__()
        self.time_steps = time_steps
        layers = OrderedDict(
            [
                (
                    "conv_0",
                    layer.Conv2d(
                        num_input_channels,
                        2 * base_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                ),
                ("norm_0", layer.BatchNorm2d(num_features=2 * base_channels)),
                ("LIF_0", neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())),
                ("dropout_0", layer.Dropout(0.05)),
                (
                    "conv_1",
                    layer.Conv2d(
                        2 * base_channels,
                        base_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                ),
                ("norm_1", layer.BatchNorm2d(num_features=base_channels)),
                ("LIF_1", neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())),
                ("dropout_1", layer.Dropout(0.05)),
                ("flatten", layer.Flatten()),
                ("linear", layer.Linear((base_channels ** 3) // 4, latent_dimension)),
                ("LIF_2", neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())),
            ]
        )
        if not regularize:
            layers.pop("norm_0")
            layers.pop("dropout_0")
            layers.pop("norm_1")
            layers.pop("dropout_1")
        self.net = nn.Sequential(layers)

        functional.set_step_mode(self, step_mode="m")

    def forward(self, x: torch.Tensor):
        functional.reset_net(self)
        return self.net(x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1))


class Decoder(nn.Module):
    """
    Decoder for the autoencoder.
    Linear layer, followed by 5 layer CNN. Output is sigmoid otherwise ReLU is used.
    """

    def __init__(
        self,
        num_input_channels: int,
        base_channels: int,
        latent_dimension: int,
        regularize: bool,
        time_steps: int,
        tau: float,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.linear = nn.Sequential(
            layer.Linear(latent_dimension, 16 * 16 * base_channels),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        )
        layers = OrderedDict(
            [
                (
                    "convt_0",
                    layer.ConvTranspose2d(
                        base_channels,
                        base_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                ),
                ("norm_0", layer.BatchNorm2d(base_channels)),
                ("LIF_0", neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())),
                ("dropout_0", layer.Dropout(0.05)),
                (
                    "convt_1",
                    layer.ConvTranspose2d(
                        base_channels, num_input_channels, kernel_size=3, padding=1
                    ),
                ),
                ("LIF_1", neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan())),
            ]
        )
        if not regularize:
            layers.pop("norm_0")
            layers.pop("dropout_0")
        self.net = nn.Sequential(layers)

        functional.set_step_mode(self, step_mode="m")

    def forward(self, x: torch.Tensor):
        functional.reset_net(self)
        return self.net(self.linear(x).reshape(self.time_steps, x.shape[1], -1, 16, 16))


class SDAutoEncoder(nn.Module):
    """
    Autoencoder model comprised of an encoder and decoder defined here.
    """

    def __init__(
        self,
        num_input_channels: int,
        base_channels: int,
        latent_dimension: int,
        regularize: bool,
        time_steps: int,
        tau: float,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_input_channels,
            base_channels,
            latent_dimension,
            regularize,
            time_steps,
            tau,
        )
        self.decoder = Decoder(
            num_input_channels,
            base_channels,
            latent_dimension,
            regularize,
            time_steps,
            tau,
        )

    def forward(self, input_data):
        """
        Forward pass for the autoencoder.
        """
        functional.reset_net(self)
        return nn.functional.normalize(torch.sum(self.decoder(self.encoder(input_data)), dim=0))


class SDDiscriminator(nn.Module):
    """
    Discriminator model comprised of an encoder and linear layer defined here.
    """

    def __init__(
        self,
        num_input_channels: int,
        base_channels: int,
        latent_dimension: int,
        regularize: bool,
        time_steps: int,
        tau: float,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.encoder = Encoder(
            num_input_channels,
            base_channels,
            latent_dimension,
            regularize,
            time_steps,
            tau,
        )
        self.flatten = nn.Flatten()
        self.output = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        """
        Forward pass for the discriminator.
        """
        functional.reset_net(self)
        latent = self.encoder(x)
        latent = nn.functional.normalize(torch.sum(latent, dim=0))
        return latent, None