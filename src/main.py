import torch
import wandb

from data import load_data, process_into_dataset
from loss import ae_loss, generator_loss, discriminator_loss
from models import Autoencoder, Discriminator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

WANDB_ACTIVE = False


def train_step(auto_encoder, discriminator, x):
    auto_encoder.train()
    discriminator.train()
    x_hat = auto_encoder(x)
    real_output = discriminator(x)
    fake_output = discriminator(x_hat)

    auto_loss = ae_loss(x, x_hat)
    disc_loss = discriminator_loss(real_output[0], fake_output[0], 1)
    gen_loss = generator_loss(fake_output[0], 1)

    return auto_loss, disc_loss, gen_loss


def train_model(auto_encoder, discriminator, train_dataset, ae_optimizer, disc_optimizer,
                generator_optimizer, epochs):
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-----------")
        for batch, (x, y) in enumerate(train_dataset):
            # x, y = x.to(device), y.to(device)

            ae_loss, disc_loss, gen_loss = train_step(auto_encoder, discriminator, x)
            print(f"Batch {batch + 1}\n-----------")
            print(f"Autoencoder loss: {ae_loss}")
            print(f"Discriminator loss: {disc_loss}")
            print(f"Generator loss: {gen_loss}")
            print(f"-----------")
            if WANDB_ACTIVE:
                wandb.log({"autoencoder_loss": ae_loss, "discriminator_loss": disc_loss,
                           "generator_loss": gen_loss})
            ae_optimizer.zero_grad()
            disc_optimizer.zero_grad()
            generator_optimizer.zero_grad()

            ae_loss.backward(retain_graph=True)
            disc_loss.backward(retain_graph=True)
            gen_loss.backward()

            ae_optimizer.step()
            disc_optimizer.step()
            generator_optimizer.step()
    return 1.0


if __name__ == "__main__":
    config_vals = {'batch_size': 32, 'epochs': 15, 'learning_rate': 1e-3, 'optimizer': 'Adam',
                   'num_layers': 2, 'latent_dimension': 32, 'num_filters': 32}
    if WANDB_ACTIVE:
        wandb.init(project='snn-nln-1', config=config_vals)
    train_x, train_y, test_x, test_y, rfi_models = load_data()
    train_dataset = process_into_dataset(train_x, train_y, batch_size=config_vals['batch_size'])
    test_dataset = process_into_dataset(test_x, test_y, batch_size=config_vals['batch_size'])
    # Create model
    auto_encoder = Autoencoder(config_vals['num_layers'], config_vals['latent_dimension'],
                               config_vals['num_filters'], train_x[0][0].shape)
    auto_encoder.build(torch.randn(32, 1, 512, 512))
    discriminator = Discriminator(config_vals['num_layers'], config_vals['latent_dimension'],
                                  config_vals['num_filters'])
    # Create optimizer
    ae_optimizer = getattr(torch.optim, config_vals['optimizer'])(auto_encoder.parameters(),
                                                                  lr=config_vals['learning_rate'])
    disc_optimizer = getattr(torch.optim, config_vals['optimizer'])(discriminator.parameters(),
                                                                    lr=config_vals['learning_rate'])
    generator_optimizer = getattr(torch.optim, config_vals['optimizer'])(
        auto_encoder.decoder.parameters(),
        lr=config_vals['learning_rate'])
    # Train model
    accuracy = train_model(auto_encoder, discriminator, train_dataset, ae_optimizer, disc_optimizer,
                           generator_optimizer, config_vals['epochs'])
    # Test model

    # Save model
    if WANDB_ACTIVE:
        wandb.finish()
