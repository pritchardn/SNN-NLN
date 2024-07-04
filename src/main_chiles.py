import os

import numpy as np
import torch
from torchsummary import summary

from config import get_output_dir, DEVICE, get_dataset_params
from data import load_data, process_into_dataset, reconstruct_patches
from evaluation import infer, get_error_dataset, nln, nln_errors, get_dists
from main import train_model
from models import AutoEncoder, Discriminator
from plotting import plot_loss_history
from utils import generate_model_name, save_json


def evaluate_chiles(
        model,
        train_dataset,
        paddings,
        neighbours,
        latent_dimension,
        original_size,
        patch_size,
        model_name,
        model_type,
        anomaly_type,
        dataset,
):
    # Run the test dataset through the nln process
    z_train = infer(model.encoder, train_dataset, latent_dimension, True)
    x_hat_train = infer(model, train_dataset, patch_size, False)
    neighbours_dist, neighbours_idx, neighbour_mask = nln(z_train, z_train, neighbours)
    del z_train
    nln_error = nln_errors(
        train_dataset, x_hat_train, x_hat_train, neighbours_idx, neighbour_mask
    )
    del x_hat_train
    del train_dataset
    nln_error_recon = reconstruct_patches(nln_error, original_size, patch_size)
    # Save the models output as flags
    np.save(
        f"{get_output_dir()}/{model_type}/{anomaly_type}/{model_name}/flags.npy",
        nln_error_recon,
    )


def main(config_vals: dict):
    """
    Main training routine for the DAE model. Loads data creates model, trains and reports.
    """
    config_vals["model_name"] = generate_model_name(config_vals)
    print(config_vals["model_name"])
    output_dir = os.path.join(
        get_output_dir(),
        config_vals["model_type"],
        config_vals["anomaly_type"],
        config_vals["model_name"],
    )
    train_x, train_y, _, _, paddings = load_data(config_vals)
    train_dataset, _ = process_into_dataset(
        train_x,
        train_y,
        batch_size=config_vals["batch_size"],
        mode=config_vals["dataset"],
        threshold=config_vals["threshold"],
        patch_size=config_vals["patch_size"],
        stride=config_vals["patch_stride"],
        filter_rfi_patches=True,
        shuffle=True,
        limit=config_vals.get("limit", None),
    )
    # Create model
    auto_encoder = AutoEncoder(
        1,
        config_vals["num_filters"],
        config_vals["latent_dimension"],
        config_vals["regularize"],
    ).to(DEVICE)
    auto_encoder.eval()
    for _, (shape_test, _) in enumerate(train_dataset):
        auto_encoder(shape_test.to(DEVICE))
        break
    summary(auto_encoder, (1, 32, 32))
    auto_encoder.train()
    discriminator = Discriminator(
        1,
        config_vals["num_filters"],
        config_vals["latent_dimension"],
        config_vals["regularize"],
    ).to(DEVICE)
    # Create optimizer
    ae_optimizer = getattr(torch.optim, config_vals["optimizer"])(
        auto_encoder.parameters(), lr=config_vals["ae_learning_rate"]
    )
    disc_optimizer = getattr(torch.optim, config_vals["optimizer"])(
        discriminator.parameters(), lr=config_vals["disc_learning_rate"]
    )
    generator_optimizer = getattr(torch.optim, config_vals["optimizer"])(
        auto_encoder.decoder.parameters(), lr=config_vals["gen_learning_rate"]
    )
    # Train model
    (
        _,
        auto_encoder,
        discriminator,
        ae_loss_history,
        disc_loss_history,
        gen_loss_history,
    ) = train_model(
        auto_encoder,
        discriminator,
        train_dataset,
        ae_optimizer,
        disc_optimizer,
        generator_optimizer,
        config_vals["epochs"],
        config_vals["model_type"],
        output_dir,
    )
    auto_encoder.eval()
    discriminator.eval()
    # Plot loss history
    plot_loss_history(ae_loss_history, disc_loss_history, gen_loss_history, output_dir)
    train_dataset, _ = process_into_dataset(
        train_x,
        train_y,
        batch_size=config_vals["batch_size"],
        mode=config_vals["dataset"],
        threshold=config_vals["threshold"],
        patch_size=config_vals["patch_size"],
        stride=config_vals["patch_stride"],
        filter_rfi_patches=False,
        shuffle=False,
        limit=config_vals.get("limit", None),
    )
    evaluate_chiles(
        auto_encoder,
        train_dataset,
        paddings,
        config_vals.get("neighbours"),
        config_vals.get("latent_dimension"),
        train_x[0].shape[0],
        config_vals.get("patch_size"),
        config_vals["model_name"],
        config_vals["model_type"],
        config_vals.get("anomaly_type"),
        config_vals["dataset"],
    )
    torch.save(auto_encoder.state_dict(), os.path.join(output_dir, "autoencoder.pt"))
    save_json(config_vals, output_dir, "config")


if __name__ == "__main__":
    config_vals = get_dataset_params("CHILES")
    main(config_vals)
