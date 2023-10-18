"""
Main training routine for the DAE model.
Copyright (c) 2023, Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""
import json
import os

import numpy as np
import optuna
import torch

from torchsummary import summary
from tqdm import tqdm

from config import DEVICE, get_dataset_params, get_output_dir, VERBOSE
from data import load_data, process_into_dataset
from evaluation import evaluate_model, mid_run_calculate_metrics
from loss import ae_loss, generator_loss, discriminator_loss
from models import AutoEncoder, Discriminator
from plotting import plot_intermediate_images, plot_loss_history
from utils import generate_model_name, save_json


def train_step(
    auto_encoder,
    discriminator,
    x_images,
    ae_optimizer,
    disc_optimizer,
    generator_optimizer,
):
    """
    Executes a single training step for the autoencoder, discriminator and generator.
    """
    auto_encoder.train()
    discriminator.train()
    x_hat = auto_encoder(x_images)
    real_output, _ = discriminator(x_images)
    fake_output, _ = discriminator(x_hat)

    auto_loss = ae_loss(x_images, x_hat)
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


def train_model(
    auto_encoder,
    discriminator,
    train_dataset,
    ae_optimizer,
    disc_optimizer,
    generator_optimizer,
    epochs,
    model_type,
    output_dir,
    config_vals=None,
    test_dataset=None,
    test_masks_original=None,
    train_x=None,
    trial: optuna.Trial = None,
    unshuffled_train_dataset=None,
):
    """
    Trains the auto-encoder, discriminator and generator.
    """
    ae_loss_history = []
    disc_loss_history = []
    gen_loss_history = []
    metrics = {"f1": 1.0}
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-----------")
        running_ae_loss = 0.0
        running_disc_loss = 0.0
        running_gen_loss = 0.0
        for x_images, y_masks in tqdm(train_dataset) if VERBOSE else train_dataset:
            x_images, y_masks = x_images.to(DEVICE), y_masks.to(DEVICE)

            ae_loss_val, disc_loss, gen_loss = train_step(
                auto_encoder,
                discriminator,
                x_images,
                ae_optimizer,
                disc_optimizer,
                generator_optimizer,
            )
            ae_optimizer.zero_grad()
            disc_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            running_ae_loss += ae_loss_val.item() * len(x_images)
            running_disc_loss += disc_loss.item() * len(x_images)
            running_gen_loss += gen_loss.item() * len(x_images)

        interim_ae_loss = running_ae_loss / len(train_dataset)
        interim_disc_loss = running_disc_loss / len(train_dataset)
        interim_gen_loss = running_gen_loss / len(train_dataset)

        ae_loss_history.append(interim_ae_loss)
        disc_loss_history.append(interim_disc_loss)
        gen_loss_history.append(interim_gen_loss)
        print("Autoencoder Loss: ", interim_ae_loss)
        print("Discriminator Loss: ", interim_disc_loss)
        print("Generator Loss: ", interim_gen_loss)

        if (
            config_vals is not None
            and test_dataset is not None
            and test_masks_original is not None
            and train_x is not None
            and trial is not None
            and unshuffled_train_dataset is not None
        ):
            metrics = mid_run_calculate_metrics(
                auto_encoder,
                test_masks_original,
                test_dataset,
                unshuffled_train_dataset,
                config_vals["neighbours"],
                config_vals["latent_dimension"],
                train_x[0].shape[0],
                config_vals["patch_size"],
            )
            trial.report(metrics["f1"], epoch)
            print(f"f1:\t{metrics['f1']}")
            if trial.should_prune():
                raise optuna.TrialPruned()
        else:
            plot_intermediate_images(
                auto_encoder,
                train_dataset,
                epoch + 1,
                model_type,
                output_dir,
                train_dataset.batch_size,
            )
    return (
        metrics["f1"],
        auto_encoder,
        discriminator,
        ae_loss_history,
        disc_loss_history,
        gen_loss_history,
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
    train_x, train_y, test_x, test_y, _ = load_data(config_vals)
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
    test_dataset, test_masks_original = process_into_dataset(
        test_x,
        test_y,
        batch_size=config_vals["batch_size"],
        mode=config_vals["dataset"],
        threshold=config_vals["threshold"]
        if config_vals["dataset"] == "HERA"
        else None,
        patch_size=config_vals["patch_size"],
        stride=config_vals["patch_stride"],
        shuffle=False,
        get_orig=True,
    )
    # Create model
    auto_encoder = AutoEncoder(
        1,
        config_vals["num_filters"],
        config_vals["latent_dimension"],
        config_vals["regularize"],
    ).to(DEVICE)
    auto_encoder.eval()
    for _, (shape_test, _) in enumerate(test_dataset):
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
        filter_rfi_patches=True,
        shuffle=False,
        limit=config_vals.get("limit", None),
    )
    # Test model
    evaluate_model(
        auto_encoder,
        test_masks_original,
        test_dataset,
        train_dataset,
        config_vals.get("neighbours"),
        config_vals.get("latent_dimension"),
        train_x[0].shape[0],
        config_vals.get("patch_size"),
        config_vals["model_name"],
        config_vals["model_type"],
        config_vals.get("anomaly_type"),
        config_vals.get("dataset"),
    )
    torch.save(auto_encoder.state_dict(), os.path.join(output_dir, "autoencoder.pt"))
    save_json(config_vals, output_dir, "config")


def main_sweep_threshold(num_trials: int = 10):
    """
    Runs a number of trials sweeping through the AOFlagger threshold parameter.
    :param num_trials: How many trials to execute for a single threshold
    """
    config_vals = get_dataset_params("HERA")
    threshold_range = [0.5, 1, 3, 5, 7, 9, 10, 20, 50, 100, 200]
    for threshold in threshold_range:
        config_vals["threshold"] = threshold
        for trial in range(1, num_trials + 1):
            config_vals["trial"] = trial
            main(config_vals)


def main_sweep_noise(num_trials: int = 10):
    """
    Runs a number of trials sweeping through each rfi noise type for out-of-distribution tests.
    :param num_trials: How many trials to execute for a single threshold
    """
    config_vals = get_dataset_params("HERA")
    rfi_exclusion_vals = [None, "rfi_stations", "rfi_dtv", "rfi_impulse", "rfi_scatter"]
    for rfi_excluded in rfi_exclusion_vals:
        config_vals["excluded_rfi"] = rfi_excluded
        for trial in range(1, num_trials + 1):
            config_vals["trial"] = trial
            main(config_vals)


def main_standard():
    """
    A single standard trial of the DAE model.
    If sweep is set to true, a simple hyperparameter grid search is performed.
    """
    sweep = False
    num_layers_vals = [2, 3]
    rfi_exclusion_vals = [None, "rfi_stations", "rfi_dtv", "rfi_impulse", "rfi_scatter"]
    dataset = "TABASCAL"
    config_vals = get_dataset_params(dataset)
    # config_vals["threshold"] = None
    if sweep:
        for num_layers in num_layers_vals:
            for rfi_excluded in rfi_exclusion_vals:
                config_vals["num_layers"] = num_layers
                config_vals["excluded_rfi"] = rfi_excluded
                print(config_vals)
                main(config_vals)
    else:
        main(config_vals)


def rerun_evaluation(input_dir):
    """
    Reruns the final evaluation routine for a given DAE trial.
    Useful if changes are made to the evaluation routine.
    """
    # Load config
    with open(
        os.path.join(input_dir, "config.json"), "r", encoding="utf-8"
    ) as config_file:
        config_vals = json.load(config_file)

    train_x, train_y, test_x, test_y, _ = load_data(config_vals)
    train_dataset, _ = process_into_dataset(
        train_x,
        train_y,
        batch_size=config_vals["batch_size"],
        mode=config_vals["dataset"],
        threshold=config_vals["threshold"],
        patch_size=config_vals["patch_size"],
        stride=config_vals["patch_stride"],
        filter_rfi_patches=True,
        shuffle=False,
        limit=config_vals.get("limit", None),
    )
    test_dataset, test_masks_original = process_into_dataset(
        test_x,
        test_y,
        batch_size=config_vals["batch_size"],
        mode=config_vals["dataset"],
        threshold=config_vals["threshold"]
        if config_vals["dataset"] == "HERA"
        else None,
        patch_size=config_vals["patch_size"],
        stride=config_vals["patch_stride"],
        shuffle=False,
        get_orig=True,
    )

    # Load model
    model_path = os.path.join(input_dir, "autoencoder.pt")
    model = AutoEncoder(
        1,
        config_vals["num_filters"],
        config_vals["latent_dimension"],
        config_vals["regularize"],
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(DEVICE)
    for threshold in [0.15]:
        print(f"Threshold: {threshold}")
        metrics = evaluate_model(
            model,
            test_masks_original,
            test_dataset,
            train_dataset,
            config_vals.get("neighbours"),
            config_vals.get("latent_dimension"),
            train_x[0].shape[0],
            config_vals.get("patch_size"),
            config_vals["model_name"],
            config_vals["model_type"],
            config_vals.get("anomaly_type"),
            config_vals.get("dataset"),
            threshold=threshold
        )
        print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    rerun_evaluation("outputs/DAE/MISO/DAE_MISO_TABASCAL_64_2_10_trial_1_didactic-whale")
