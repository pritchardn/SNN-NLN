"""
Contains model definitions for the project.
Copyright (c) 2023, Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""

from torch import nn


class Encoder(nn.Module):
    """
    Encoder for the autoencoder.
    5 layer CNN with linear layer at the end.
    """

    def __init__(
        self, num_input_channels: int, base_channels: int, latent_dimension: int
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                num_input_channels,
                2 * base_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.Dropout(0.05),
            nn.Conv2d(
                2 * base_channels, base_channels, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.Dropout(0.05),
            nn.Flatten(),
            nn.LazyLinear(latent_dimension),
        )

    def forward(self, input_data):
        """
        Forward pass for the encoder.
        """
        return self.net(input_data)


class Decoder(nn.Module):
    """
    Decoder for the autoencoder.
    Linear layer, followed by 5 layer CNN. Output is sigmoid otherwise ReLU is used.
    """

    def __init__(
        self, num_input_channels: int, base_channels: int, latent_dimension: int
    ):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(latent_dimension, 16 * 16 * base_channels),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.LazyConvTranspose2d(
                base_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.Dropout(0.05),
            nn.LazyConvTranspose2d(num_input_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, input_data):
        """
        Forward pass for the decoder.
        """
        input_data = self.linear(input_data)
        reshaped = input_data.view(input_data.shape[0], -1, 16, 16)
        reconstruction = self.net(reshaped)
        return reconstruction


class AutoEncoder(nn.Module):
    """
    Autoencoder model comprised of an encoder and decoder defined here.
    """

    def __init__(
        self, num_input_channels: int, base_channels: int, latent_dimension: int
    ):
        super().__init__()
        self.encoder = Encoder(num_input_channels, base_channels, latent_dimension)
        self.decoder = Decoder(num_input_channels, base_channels, latent_dimension)

    def forward(self, input_data):
        """
        Forward pass for the autoencoder.
        """
        latent = self.encoder(input_data)
        output = self.decoder(latent)
        return output


class Discriminator(nn.Module):
    """
    Discriminator model comprised of an encoder and linear layer defined here.
    """

    def __init__(
        self, num_input_channels: int, base_channels: int, latent_dimension: int
    ):
        super().__init__()
        self.encoder = Encoder(num_input_channels, base_channels, latent_dimension)
        self.flatten = nn.Flatten()
        self.output = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        """
        Forward pass for the discriminator.
        """
        latent = self.encoder(input_data)
        classification = self.flatten(latent)
        output = self.output(classification)
        output = self.sigmoid(output)
        return latent, output
