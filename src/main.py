import torch

from data import load_data, process_into_dataset
from loss import ae_loss, generator_loss, discriminator_loss
from models import Autoencoder, Discriminator

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_step(auto_encoder, discriminator, x, y, ae_optimizer, disc_optimizer,
               generator_optimizer):
    auto_encoder.train()
    discriminator.train()
    x_hat = auto_encoder(x)
    real_output = discriminator(x)
    fake_output = discriminator(x_hat)

    auto_loss = ae_loss(x, x_hat)
    disc_loss = discriminator_loss(real_output[0], fake_output[0], 1)
    gen_loss = generator_loss(fake_output[0], 1)

    ae_optimizer.zero_grad()
    disc_optimizer.zero_grad()
    generator_optimizer.zero_grad()

    auto_loss.backward(retain_graph=True)
    disc_loss.backward(retain_graph=True)
    gen_loss.backward()

    ae_optimizer.step()
    disc_optimizer.step()
    generator_optimizer.step()

    return auto_loss, disc_loss, gen_loss


def train_model(auto_encoder, discriminator, train_dataset, ae_optimizer, disc_optimizer,
                generator_optimizer, epochs):
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-----------")
        for batch, (x, y) in enumerate(train_dataset):
            # x, y = x.to(device), y.to(device)

            ae_loss, disc_loss, gen_loss = train_step(auto_encoder, discriminator, x, y,
                                                      ae_optimizer, disc_optimizer,
                                                      generator_optimizer)
            print(f"Batch {batch + 1}\n-----------")
            print(f"Autoencoder loss: {ae_loss}")
            print(f"Discriminator loss: {disc_loss}")
            print(f"Generator loss: {gen_loss}")
            print(f"-----------")
    return 1.0


if __name__ == "__main__":
    batch_size = 32
    epochs = 1
    learning_rate = 1e-3
    optimizer_choice = "Adam"
    num_layers = 2
    latent_dimension = 32
    num_filters = 32
    train_x, train_y, test_x, test_y, rfi_models = load_data()
    train_dataset = process_into_dataset(train_x, train_y, batch_size=batch_size)
    test_dataset = process_into_dataset(test_x, test_y, batch_size=batch_size)
    # Create model
    auto_encoder = Autoencoder(num_layers, latent_dimension, num_filters, train_x[0][0].shape)
    auto_encoder.build(torch.randn(32, 1, 512, 512))
    discriminator = Discriminator(num_layers, latent_dimension, num_filters)
    # Create optimizer
    ae_optimizer = getattr(torch.optim, optimizer_choice)(auto_encoder.parameters(),
                                                          lr=learning_rate)
    disc_optimizer = getattr(torch.optim, optimizer_choice)(discriminator.parameters(),
                                                            lr=learning_rate)
    generator_optimizer = getattr(torch.optim, optimizer_choice)(auto_encoder.decoder.parameters(),
                                                                 lr=learning_rate)
    # Train model
    accuracy = train_model(auto_encoder, discriminator, train_dataset, ae_optimizer, disc_optimizer,
                           generator_optimizer, epochs)
    # Test model

    # Save model
