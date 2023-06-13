import json
import os

import torch
import wandb

from config import WANDB_ACTIVE, DEVICE
from data import load_data, process_into_dataset
from evaluation import evaluate_model, plot_loss_history
from loss import ae_loss, generator_loss, discriminator_loss
from models import Autoencoder, Discriminator
from plotting import plot_intermediate_images


def save_config(config: dict, output_dir: str):
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


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
                generator_optimizer, epochs, model_type, output_dir):
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

        plot_intermediate_images(auto_encoder, train_dataset, t + 1, model_type, output_dir,
                                 train_dataset.batch_size)
    return 1.0, auto_encoder, discriminator, ae_loss_history, disc_loss_history, gen_loss_history


def main(config_vals: dict):
    config_vals['model_name'] = f'{config_vals["model_type"]}_{config_vals["anomaly_type"]}_' \
                                f'{config_vals["dataset"]}_{config_vals["latent_dimension"]}_' \
                                f'{config_vals["num_layers"]}_' \
                                f'{config_vals["threshold"]}'
    if config_vals['excluded_rfi']:
        config_vals['model_name'] += f'_{config_vals["excluded_rfi"]}'
    output_dir = f'./outputs/{config_vals["model_type"]}/{config_vals["anomaly_type"]}/' \
                 f'{config_vals["model_name"]}/'
    if WANDB_ACTIVE:
        wandb.init(project='snn-nln-1', config=config_vals)
    train_x, train_y, test_x, test_y, rfi_models = load_data(
        excluded_rfi=config_vals['excluded_rfi'])
    train_dataset, _ = process_into_dataset(train_x, train_y,
                                            batch_size=config_vals['batch_size'],
                                            mode='HERA',
                                            threshold=config_vals['threshold'],
                                            patch_size=config_vals['patch_size'],
                                            stride=config_vals['patch_stride'],
                                            filter=True, shuffle=True)
    test_dataset, test_masks_original = process_into_dataset(test_x, test_y,
                                                             batch_size=config_vals['batch_size'],
                                                             mode='HERA',
                                                             threshold=config_vals['threshold'],
                                                             patch_size=config_vals['patch_size'],
                                                             stride=config_vals['patch_stride'],
                                                             shuffle=False,
                                                             get_orig=True)
    # Create model
    auto_encoder = Autoencoder(config_vals['num_layers'], config_vals['latent_dimension'],
                               config_vals['num_filters'], train_dataset.dataset[0][0].shape).to(
        DEVICE)
    discriminator = Discriminator(config_vals['num_layers'], config_vals['latent_dimension'],
                                  config_vals['num_filters']).to(DEVICE)
    # Create optimizer
    ae_optimizer = getattr(torch.optim, config_vals['optimizer'])(auto_encoder.parameters(),
                                                                  lr=config_vals[
                                                                      'ae_learning_rate'])
    disc_optimizer = getattr(torch.optim, config_vals['optimizer'])(discriminator.parameters(),
                                                                    lr=config_vals[
                                                                        'disc_learning_rate'])
    generator_optimizer = getattr(torch.optim, config_vals['optimizer'])(
        auto_encoder.decoder.parameters(),
        lr=config_vals['gen_learning_rate'])
    # Train model
    accuracy, auto_encoder, discriminator, ae_loss_history, disc_loss_history, gen_loss_history = \
        train_model(
            auto_encoder, discriminator, train_dataset,
            ae_optimizer, disc_optimizer,
            generator_optimizer, config_vals['epochs'], config_vals['model_type'], output_dir)
    auto_encoder.eval()
    discriminator.eval()
    # Plot loss history
    plot_loss_history(ae_loss_history, disc_loss_history, gen_loss_history, output_dir)
    # Test model
    evaluate_model(auto_encoder, test_masks_original, test_dataset, train_dataset,
                   config_vals.get('neighbours'), config_vals.get('batch_size'),
                   config_vals.get('latent_dimension'),
                   train_x[0].shape[0], config_vals.get('patch_size'), config_vals['model_name'],
                   config_vals['model_type'],
                   config_vals.get("anomaly_type"), config_vals.get("dataset"))
    torch.save(auto_encoder.state_dict(), os.path.join(output_dir, 'autoencoder.pt'))
    save_config(config_vals, output_dir)
    # convert_to_snn(auto_encoder, train_dataset, test_dataset)
    if WANDB_ACTIVE:
        wandb.finish()


if __name__ == "__main__":
    SWEEP = True
    num_layers_vals = [2, 3]
    rfi_exclusion_vals = [None, 'rfi_stations', 'rfi_dtv', 'rfi_impulse', 'rfi_scatter']
    config_vals = {'batch_size': 64, 'epochs': 120, 'ae_learning_rate': 1e-4,
                   'gen_learning_rate': 1e-5, 'disc_learning_rate': 1e-5, 'optimizer': 'Adam',
                   'num_layers': 3, 'latent_dimension': 32, 'num_filters': 32, 'neighbours': 20,
                   'patch_size': 32, 'patch_stride': 32, 'threshold': 10, 'anomaly_type': "MISO",
                   'dataset': 'HERA', 'model_type': 'DAE', 'excluded_rfi': 'rfi_scatter'}
    i = 0
    if SWEEP:
        for num_layers in num_layers_vals:
            for rfi_excluded in rfi_exclusion_vals:
                config_vals['num_layers'] = num_layers
                config_vals['excluded_rfi'] = rfi_excluded
                print(config_vals)
                main(config_vals)
    else:
        main(config_vals)
