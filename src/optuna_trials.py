"""
Contains code for optuna hyperparameter trial runs.
Copyright (c) 2023, Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""
import json
import os

import optuna
import torch
from optuna.trial import TrialState

from config import DEVICE, get_output_dir
from data import process_into_dataset, load_data
from evaluation import evaluate_model
from plotting import plot_loss_history
from main import train_model
from models import AutoEncoder, Discriminator
from utils import generate_model_name, save_json


def run_trial(trial: optuna.Trial):
    """
    Trial for optuna hyperparameter optimization.
    Is effectively a condensed version of main.py.
    :return: MSE: float
    """
    latent_dimension = 32
    config_vals = {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "epochs": trial.suggest_int("epochs", 2, 128),
        "ae_learning_rate": trial.suggest_float("ae_learning_rate", 1e-4, 10e-4),
        "gen_learning_rate": trial.suggest_float("gen_learning_rate", 1e-4, 10e-4),
        "disc_learning_rate": trial.suggest_float("disc_learning_rate", 1e-4, 10e-4),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        "num_layers": 5,
        "latent_dimension": latent_dimension,
        "num_filters": 16,
        "neighbours": trial.suggest_int("neighbours", 1, 25),
        "patch_size": 32,
        "patch_stride": 32,
        "threshold": 10,
        "anomaly_type": "MISO",
        "dataset": "HERA",
        "model_type": "DAE",
        "excluded_rfi": None,
        "time_length": None,
        "average_n": None,
        "trial": trial._trial_id + 1,
    }
    config_vals["model_name"] = generate_model_name(config_vals)
    print(json.dumps(config_vals, indent=4))
    output_dir = (
        f'./{get_output_dir()}/{config_vals["model_type"]}/{config_vals["anomaly_type"]}/'
        f'{config_vals["model_name"]}/'
    )
    train_x, train_y, test_x, test_y, _ = load_data(
        excluded_rfi=config_vals["excluded_rfi"]
    )
    train_dataset, _ = process_into_dataset(
        train_x,
        train_y,
        batch_size=config_vals["batch_size"],
        mode="HERA",
        threshold=config_vals["threshold"],
        patch_size=config_vals["patch_size"],
        stride=config_vals["patch_stride"],
        filter_rfi_patches=True,
        shuffle=True,
    )
    unshuffled_train_dataset, _ = process_into_dataset(
        train_x,
        train_y,
        batch_size=config_vals["batch_size"],
        mode="HERA",
        threshold=config_vals["threshold"],
        patch_size=config_vals["patch_size"],
        stride=config_vals["patch_stride"],
        filter_rfi_patches=True,
        shuffle=False,
    )
    test_dataset, test_masks_original = process_into_dataset(
        test_x,
        test_y,
        batch_size=config_vals["batch_size"],
        mode="HERA",
        threshold=config_vals["threshold"],
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
    ).to(DEVICE)
    discriminator = Discriminator(
        1,
        config_vals["num_filters"],
        config_vals["latent_dimension"],
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
        f1_score,
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
        config_vals=config_vals,
        test_dataset=test_dataset,
        test_masks_original=test_masks_original,
        train_x=train_x,
        trial=trial,
        unshuffled_train_dataset=unshuffled_train_dataset,
    )
    auto_encoder.eval()
    discriminator.eval()
    # Plot loss history
    plot_loss_history(ae_loss_history, disc_loss_history, gen_loss_history, output_dir)
    # Test model
    metrics = evaluate_model(
        auto_encoder,
        test_masks_original,
        test_dataset,
        unshuffled_train_dataset,
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
    f1_score = metrics["f1"]
    return f1_score


def main_optuna():
    """
    Main function for optuna hyperparameter optimization.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(run_trial, n_trials=32)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    with open(
        f"{get_output_dir()}{os.sep}best_trial.json", "w", encoding="utf-08"
    ) as ofile:
        json.dump(trial.params, ofile, indent=4)
    with open(
        f"{get_output_dir()}{os.sep}completed_trials.json", "w", encoding="utf-08"
    ) as ofile:
        completed_trials_out = []
        for trial_params in complete_trials:
            completed_trials_out.append(trial_params.params)
        json.dump(completed_trials_out, ofile, indent=4)
    with open(
        f"{get_output_dir()}{os.sep}pruned_trials.json", "w", encoding="utf-08"
    ) as ofile:
        pruned_trials_out = []
        for trial_params in pruned_trials:
            pruned_trials_out.append(trial_params.params)
        json.dump(pruned_trials_out, ofile, indent=4)


if __name__ == "__main__":
    main_optuna()
