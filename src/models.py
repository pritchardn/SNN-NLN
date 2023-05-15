from collections import OrderedDict
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, num_layers: int, latent_dimension: int,
                 num_filters: int):
        super().__init__()
        layers = []
        for n in range(num_layers):
            in_size = 1 if n == 0 else (num_layers - n + 1) * num_filters
            out_size = (num_layers - n) * num_filters
            layers.append((f"conv_{n}", nn.Conv2d(in_size, out_size, 3, 2)))
            # layers.append(("relu_{}".format(n), nn.ReLU()))
            layers.append(("batch_norm_{}".format(n), nn.BatchNorm2d(out_size)))
            layers.append(("dropout_{}".format(n), nn.Dropout(0.05)))
        self.cnn = nn.Sequential(OrderedDict(layers))
        self.flatten = nn.Flatten()
        self.linear = nn.LazyLinear(latent_dimension)
        # self.dense_act = nn.ReLU()

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.linear(x)
        # x = self.dense_act(x)
        return x


class Decoder(nn.Module):

    def __init__(self, input_shape: tuple, num_layers: int,
                 num_filters: int, latent_dimension: int):
        super().__init__()
        self.num_filters = num_filters
        self.in_dim = input_shape[-1] // (2 ** num_layers) - 1
        self.linear = nn.Linear(latent_dimension, self.in_dim * self.in_dim * num_filters)
        self.dense_in = nn.ReLU()
        layers = []
        for n in range(num_layers - 1):
            layers.append(("conv_{}".format(n),
                           nn.ConvTranspose2d((n + 1) * num_filters, (n + 2) * num_filters, 3, 2)))
            # layers.append(("relu_{}".format(n), nn.ReLU()))
            layers.append(("batch_norm_{}".format(n), nn.BatchNorm2d((n + 2) * num_filters)))
            layers.append(("dropout_{}".format(n), nn.Dropout(0.05)))
        self.cnn = nn.Sequential(OrderedDict(layers))
        self.cnn_output = nn.ConvTranspose2d(num_layers * num_filters, 1, 3, 2, output_padding=1)
        # self.sigmoid_out = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        # x = self.dense_in(x)
        x = x.view(-1, self.num_filters, self.in_dim, self.in_dim)
        x = self.cnn(x)
        x = self.cnn_output(x)
        # x = self.sigmoid_out(x)
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


class Discriminator(nn.Module):

    def __init__(self, num_layers: int, latent_dimension: int, num_filters: int):
        super().__init__()
        self.encoder = Encoder(num_layers, latent_dimension, num_filters)
        self.flatten = nn.Flatten()
        self.output = nn.LazyLinear(1)

    def forward(self, x):
        z = self.encoder(x)
        classification = self.flatten(z)
        output = self.output(classification)
        return z, output
