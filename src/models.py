from collections import OrderedDict

import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, num_layers: int, latent_dimension: int,
                 num_filters: int):
        super().__init__()
        layers = []
        layers.append(('conv1', nn.Conv2d(1, 64, 3, 2)))
        layers.append(('batchnorm1', nn.BatchNorm2d(64)))
        layers.append(('dropout1', nn.Dropout(0.05)))

        layers.append(('conv2', nn.Conv2d(64, num_filters, 3, 2)))
        layers.append(('batchnorm2', nn.BatchNorm2d(num_filters)))
        layers.append(('dropout2', nn.Dropout(0.05)))
        self.conv_block = nn.Sequential(OrderedDict(layers))
        self.latent_dimension = latent_dimension
        self.num_filters = num_filters

    def forward(self, x):
        output = self.conv_block(x)
        batch_size = output.size(0)
        output = output.view(batch_size, -1)
        print(output.shape)
        output = nn.LazyLinear(self.latent_dimension * batch_size)(output)
        return output


class Decoder(nn.Module):

    def __init__(self, input_shape: tuple, num_layers: int,
                 num_filters: int, latent_dimension: int):
        super().__init__()
        layers = []
        self.in_dim_1 = input_shape[0] // 2 ** (num_layers - 1)
        self.in_dim_2 = input_shape[1] // 2 ** (num_layers - 1)
        self.num_filters = num_filters
        num_input_neurons = self.in_dim_1 * self.in_dim_2 * self.num_filters
        self.dense = nn.Linear(latent_dimension, num_input_neurons)
        for n in range(num_layers - 1):
            num_inputs = num_input_neurons if n == 0 else (n + 1) * num_filters
            layers.append(
                (f'conv{n}', nn.ConvTranspose2d(num_inputs, (n + 1) * num_filters, 3, 2)))
            layers.append((f'batchnorm{n}', nn.BatchNorm2d((n + 1) * num_filters)))
            layers.append((f'dropout{n}', nn.Dropout(0.05)))
        self.conv_block = nn.Sequential(OrderedDict(layers))
        self.conv_output = nn.ConvTranspose2d(num_filters ** 2, input_shape[-1], 3)

    def forward(self, x):
        curr = self.dense(x)
        reshaped = torch.reshape(curr, (32, self.in_dim_1, self.in_dim_2, self.num_filters))
        output = self.conv_block(reshaped)
        output = self.conv_output(output)
        return output


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
