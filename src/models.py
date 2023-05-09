from collections import OrderedDict

import torch.nn as nn
import torchlayers as tl


class Encoder(nn.Module):

    def __init__(self, num_layers: int, latent_dimension: int,
                 num_filters: int):
        super().__init__()
        layers = []
        for n in range(num_layers):
            layers.append(("conv_{}".format(n), tl.Conv((num_layers-n) * num_filters)))
            layers.append(("batch_norm_{}".format(n), tl.BatchNorm()))
            layers.append(("dropout_{}".format(n), tl.Dropout(0.05)))
        self.cnn = nn.Sequential(OrderedDict(layers))
        self.flatten = tl.Flatten()
        self.linear = tl.Linear(latent_dimension)

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class Decoder(nn.Module):

    def __init__(self, input_shape: tuple, num_layers: int,
                 num_filters: int, latent_dimension: int):
        super().__init__()
        self.linear = tl.Linear(input_shape[0] * input_shape[1] * latent_dimension)
        self.reshape = tl.Reshape(32, input_shape[0], input_shape[1])
        layers = []
        for n in range(num_layers - 1):
            layers.append(("conv_{}".format(n), tl.ConvTranspose((n + 1) * num_filters)))
            layers.append(("batch_norm_{}".format(n), tl.BatchNorm()))
            layers.append(("dropout_{}".format(n), tl.Dropout(0.05)))
        self.cnn = nn.Sequential(OrderedDict(layers))
        self.cnn_output = tl.ConvTranspose(1)

    def forward(self, x):
        x = self.linear(x)
        x = self.reshape(x)
        x = self.cnn(x)
        x = self.cnn_output(x)
        return x


class Autoencoder(nn.Module):

    def __init__(self, num_layers: int, latent_dimension: int, num_filters: int,
                 input_shape: tuple):
        super().__init__()
        self.encoder = Encoder(num_layers, latent_dimension, num_filters)
        self.decoder = Decoder(input_shape, num_layers, num_filters, latent_dimension)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def build(self, x):
        tl.build(self.encoder, x)
        tl.build(self.decoder, self.encoder(x))
        print(self.encoder)
        print(self.decoder)


class Discriminator(nn.Module):

    def __init__(self, num_layers: int, latent_dimension: int, num_filters: int):
        super().__init__()
        self.encoder = Encoder(num_layers, latent_dimension, num_filters)
        self.flatten = nn.Flatten()
        self.output = nn.Linear(num_filters, 1)

    def forward(self, x):
        z = self.encoder(x)
        classification = self.flatten(z)
        output = self.output(classification)
        return z, output

    def build(self, x):
        tl.build(self.encoder, x)
