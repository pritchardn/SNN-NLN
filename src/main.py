import torch
import wandb

from config import WANDB_ACTIVE, DEVICE
from data import load_data, process_into_dataset
from evaluation import evaluate_model, plot_loss_history
from loss import ae_loss, generator_loss, discriminator_loss
from models import Autoencoder, Discriminator
from plotting import plot_intermediate_images


def train_step(auto_encoder, discriminator, x, ae_optimizer, disc_optimizer, generator_optimizer):
    auto_encoder.train()
    discriminator.train()
    x_hat = auto_encoder(x)
    real_output, _ = discriminator(x)
    fake_output, _ = discriminator(x_hat)

    auto_loss = ae_loss(x, x_hat)
    disc_loss = discriminator_loss(real_output, fake_output, 1)
    gen_loss = generator_loss(fake_output, 1)

    del fake_output
    del real_output
    del x_hat

    auto_loss.backward(retain_graph=True)
    disc_loss.backward(retain_graph=True)
    gen_loss.backward()

    ae_optimizer.step()
    disc_optimizer.step()
    generator_optimizer.step()

    return auto_loss, disc_loss, gen_loss


def train_model(auto_encoder, discriminator, train_dataset, ae_optimizer, disc_optimizer,
                generator_optimizer, epochs):
    ae_loss_history = []
    disc_loss_history = []
    gen_loss_history = []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-----------")
        running_ae_loss = 0.0
        running_disc_loss = 0.0
        running_gen_loss = 0.0
        for batch, (x, y) in enumerate(train_dataset):
            x, y = x.to(DEVICE), y.to(DEVICE)

            ae_loss, disc_loss, gen_loss = train_step(auto_encoder, discriminator, x, ae_optimizer,
                                                      disc_optimizer, generator_optimizer)
            ae_optimizer.zero_grad()
            disc_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            running_ae_loss += ae_loss.item() * len(x)
            running_disc_loss += disc_loss.item() * len(x)
            running_gen_loss += gen_loss.item() * len(x)

        interim_ae_loss = running_ae_loss / len(train_dataset)
        interim_disc_loss = running_disc_loss / len(train_dataset)
        interim_gen_loss = running_gen_loss / len(train_dataset)

        if WANDB_ACTIVE:
            wandb.log({"autoencoder_train_loss": interim_ae_loss,
                       "discriminator_train_loss": interim_disc_loss,
                       "generator_train_loss": interim_gen_loss})

        ae_loss_history.append(interim_ae_loss)
        disc_loss_history.append(interim_disc_loss)
        gen_loss_history.append(interim_gen_loss)
        print("Autoencoder Loss: ", interim_ae_loss)
        print("Discriminator Loss: ", interim_disc_loss)
        print("Generator Loss: ", interim_gen_loss)

        plot_intermediate_images(auto_encoder, train_dataset, t + 1, 'DAE', '.',
                                 train_dataset.batch_size)
    return 1.0, auto_encoder, discriminator, ae_loss_history, disc_loss_history, gen_loss_history


if __name__ == "__main__":
    config_vals = {'batch_size': 64, 'epochs': 100, 'learning_rate': 1e-3, 'optimizer': 'Adam',
                   'num_layers': 2, 'latent_dimension': 32, 'num_filters': 32, 'neighbours': 20,
                   'patch_size': 32, 'patch_stride': 32, 'threshold': 10, 'anomaly_type': "MISO",
                   'dataset': 'HERA'}
    if WANDB_ACTIVE:
        wandb.init(project='snn-nln-1', config=config_vals)
    train_x, train_y, test_x, test_y, rfi_models = load_data()
    train_dataset = process_into_dataset(train_x, train_y, batch_size=config_vals['batch_size'],
                                         mode='HERA', threshold=config_vals['threshold'],
                                         patch_size=config_vals['patch_size'],
                                         stride=config_vals['patch_stride'],
                                         filter=True)
    test_dataset = process_into_dataset(test_x, test_y, batch_size=config_vals['batch_size'],
                                        mode='HERA', threshold=config_vals['threshold'],
                                        patch_size=config_vals['patch_size'],
                                        stride=config_vals['patch_stride'])
    # Create model
    auto_encoder = Autoencoder(config_vals['num_layers'], config_vals['latent_dimension'],
                               config_vals['num_filters'], train_dataset.dataset[0][0].shape).to(
        DEVICE)
    discriminator = Discriminator(config_vals['num_layers'], config_vals['latent_dimension'],
                                  config_vals['num_filters']).to(DEVICE)
    # Create optimizer
    ae_optimizer = getattr(torch.optim, config_vals['optimizer'])(auto_encoder.parameters(),
                                                                  lr=config_vals['learning_rate'])
    disc_optimizer = getattr(torch.optim, config_vals['optimizer'])(discriminator.parameters(),
                                                                    lr=config_vals['learning_rate'])
    generator_optimizer = getattr(torch.optim, config_vals['optimizer'])(
        auto_encoder.decoder.parameters(),
        lr=config_vals['learning_rate'])
    # Train model
    accuracy, auto_encoder, discriminator, ae_loss_history, disc_loss_history, gen_loss_history = \
        train_model(
            auto_encoder, discriminator, train_dataset,
            ae_optimizer, disc_optimizer,
            generator_optimizer, config_vals['epochs'])
    auto_encoder.eval()
    discriminator.eval()
    # Plot loss history
    plot_loss_history(ae_loss_history, disc_loss_history, gen_loss_history, '.')
    # Test model
    evaluate_model(auto_encoder, test_y, test_dataset,
                   config_vals.get('neighbours'), config_vals.get('batch_size'),
                   config_vals.get('latent_dimension'),
                   train_x[0].shape[0], config_vals.get('patch_size'), 'dae', 'DAE',
                   config_vals.get("anomaly_type"), config_vals.get("dataset"))
    # Save model
    if WANDB_ACTIVE:
        wandb.finish()
