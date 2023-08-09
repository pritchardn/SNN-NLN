"""
Contains code to convert DAE to SNN and evaluate the SNN model.
Copyright (c) 2023, Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""
import glob
import json
import math
import os

import matplotlib.image
import numpy as np
import optuna
import torch
from matplotlib import animation
from matplotlib import pyplot as plt
from optuna.trial import TrialState
from spikingjelly.activation_based import ann2snn
from tqdm import tqdm

import config
from config import get_output_dir
from data import (
    load_data,
    process_into_dataset,
    reconstruct_latent_patches,
    reconstruct_patches,
)
from evaluation import _calculate_metrics
from models import AutoEncoder
from utils import (
    generate_model_name,
    generate_output_dir,
    load_config,
    scale_image,
    save_json,
)


def animated_plotting(out_images):
    """
    Plots inference as animation
    """
    for i in range(len(out_images[0])):
        fig, axes = plt.subplots()
        axes.axis("off")
        images = []
        for j, image in enumerate(out_images):
            temp_im = np.moveaxis(image[i], 0, -1) * 127.5 + 127.5
            image = axes.imshow(temp_im, animated=True)
            if j == 0:
                axes.imshow(temp_im)  # show an initial one first
            images.append([image])
        ani = animation.ArtistAnimation(
            fig, images, interval=50, blit=True, repeat_delay=1000
        )
        ani.save(f"output_{i}.gif", writer="pillow", fps=10)
        plt.close("all")
        print(f"Done {i}")


def plot_output_states(out_images):
    """
    Plots inference as separate images
    """
    for i in range(len(out_images[0])):
        plt.figure(figsize=(10, 10))
        sub_range = int(math.ceil(np.sqrt(len(out_images))))
        _, axarr = plt.subplots(sub_range, sub_range)
        axarr = axarr.flatten()
        for j, out_image in enumerate(out_images):
            axarr[j].axis("off")
            axarr[j].imshow(np.moveaxis(out_image[i], 0, -1) * 127.5 + 127.5)
        plt.axis("off")
        plt.savefig(f"output_{i}.png")
        plt.close("all")
        print(f"Done {i}")


def plot_input_images(in_images):
    """
    Plots input images
    """
    for i, image in enumerate(in_images):
        plt.figure(figsize=(10, 10))
        plt.imshow(np.moveaxis(image, 0, -1) * 127.5 + 127.5)
        plt.axis("off")
        plt.savefig(f"input_{i}.png")
        plt.close("all")
        print(f"Done {i}")


def plot_outputs(out_images):
    """
    Plots outputs separately and as an animated gif.
    """
    for output_states in out_images:
        plot_output_states(output_states)
        animated_plotting(output_states)


def infer_snn(model, dataloader, runtime=50, batch_limit=1, n_limit=-1):
    """
    Runs inference on an SNN model.
    Adds additional runtime and batch_limit variables
    :param model: The SNN model
    :param dataloader: Torch dataloader to run inference on
    :param runtime: The runtime to simulate the sNN for.
    :param batch_limit: The number of batches to run inference on. If -1, will run on all images.
    """
    model.eval().to("cuda")
    full_output = []
    i = 0
    bound = runtime - n_limit - 1
    with torch.no_grad():
        for img, _ in tqdm(dataloader):
            img = img.to("cuda")
            out_images = []
            if runtime is None:
                out = model(img)
                print(out.shape)
            else:
                for module in model.modules():
                    if hasattr(module, "reset"):
                        module.reset()
                for time_step in range(runtime):
                    out = model(img)
                    # out = scale_image(out)
                    # Add current state to image building
                    if time_step > bound:
                        out_images.append(out.cpu().numpy())
                full_output.append(out_images)
            i += 1
            if i == batch_limit:
                break
    return np.asarray(full_output)  # [B, T, N, C, W, H]


def convert_to_snn(model, test_data_loader, convert_threshold):
    """
    Converts an ANN model to an SNN model using SpikingJelly
    """
    model_converter = ann2snn.Converter(
        mode=convert_threshold, dataloader=test_data_loader
    )
    snn_model = model_converter(model)
    snn_model.graph.print_tabular()
    return snn_model


def test_snn_model(snn_model, test_data_loader, time_length=64, average_n=10):
    """
    Tests an SNN model, simply runs through and plots outputs
    """
    full_output = infer_snn(
        snn_model, test_data_loader, runtime=time_length, n_limit=average_n
    )
    plot_outputs(full_output)
    plot_input_images(
        test_data_loader.dataset[: test_data_loader.batch_size][0]
        .cpu()
        .detach()
        .numpy()
    )


def load_test_dataset(config_vals: dict):
    """
    Loads the test dataset.
    """
    _, _, test_x, test_y, _ = load_data()
    test_dataset, y_data_orig = process_into_dataset(
        test_x,
        test_y,
        batch_size=config_vals["batch_size"],
        mode=config_vals["dataset"],
        threshold=config_vals["threshold"],
        patch_size=config_vals["patch_size"],
        stride=config_vals["patch_stride"],
        shuffle=False,
        get_orig=True,
    )
    return test_dataset, y_data_orig


def load_ann_model(input_dir: str, config_vals: dict):
    """
    Loads an ANN model from a directory. Assumes this is a model from this project.
    """
    model_path = os.path.join(input_dir, "autoencoder.pt")
    model = AutoEncoder(
        1,
        config_vals["num_filters"],
        config_vals["latent_dimension"],
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def snln(x_hat, test_dataset, average_n, limit=None):
    """
    Spiking NLN by averaging over last n timesteps.
    """
    x_hat_trimmed = x_hat[:, -average_n:, :, :, :, :]
    sum_x_hat = np.vstack(np.sum(x_hat_trimmed, axis=1))
    sum_x_hat = scale_image(sum_x_hat)
    if limit:
        images = test_dataset.dataset[:limit][0].cpu().detach().numpy()
    else:
        images = test_dataset.dataset[:][0].cpu().detach().numpy()
    error = images - sum_x_hat
    return error


def evaluate_snn(
    model,
    test_dataset,
    test_masks_original,
    patch_size,
    original_size,
    runtime,
    average_n,
):
    """
    Evaluates an SNN model.
    """
    test_masks_original_reconstructed = reconstruct_patches(
        test_masks_original, original_size, patch_size
    )
    x_hat = infer_snn(
        model, test_dataset, runtime=runtime, batch_limit=-1, n_limit=average_n
    )
    snln_error = snln(x_hat, test_dataset, average_n)
    if patch_size:
        if snln_error.ndim == 4:
            snln_error_recon = reconstruct_patches(
                snln_error, original_size, patch_size
            )
        else:
            snln_error_recon = reconstruct_latent_patches(
                snln_error, original_size, patch_size
            )
    else:
        snln_error_recon = snln_error
    snln_metrics = _calculate_metrics(
        test_masks_original_reconstructed, snln_error_recon
    )
    return snln_metrics, snln_error_recon, x_hat


def reconstruct_snn_inference(
    inference: np.array, original_size: int, kernel_size: int
):
    """
    Reconstructs the inference from the SNN model into original image size.
    """
    transpose = np.vstack(inference.transpose(0, 2, 4, 5, 1, 3))
    n_patches = original_size // kernel_size
    reconstruction = np.empty(
        [
            transpose.shape[0] // n_patches**2,
            kernel_size * n_patches,
            kernel_size * n_patches,
            transpose.shape[-2],
            transpose.shape[-1],
        ]
    )

    start, counter, indx, batch = 0, 0, 0, []

    for i in range(n_patches, transpose.shape[0] + 1, n_patches):
        batch.append(
            np.reshape(
                np.stack(transpose[start:i, ...], axis=0),
                (
                    n_patches * kernel_size,
                    kernel_size,
                    transpose.shape[-2],
                    transpose.shape[-1],
                ),
            )
        )
        start = i
        counter += 1
        if counter == n_patches:
            reconstruction[indx, ...] = np.hstack(batch)
            indx += 1
            counter, batch = 0, []
    reconstruction = reconstruction.transpose(0, 3, 4, 2, 1)
    return reconstruction


def plot_snn_results(
    original_images, test_masks_recon, snln_error_recon, inference, output_dir
):
    """
    Plots the results of an SNN model.
    """
    plot_directory = os.path.join(output_dir, "results")
    os.makedirs(plot_directory, exist_ok=True)
    plot_images = np.moveaxis(original_images[:10, ...], 1, -1)
    plot_masks = np.moveaxis(test_masks_recon[:10, ...], 1, -1)
    snln_error_recon = np.where(snln_error_recon > 0.0, 1, 0.0)
    plot_snln = np.moveaxis(snln_error_recon[:10, ...], 1, -1)
    plot_inference = inference[:10, ...]
    fig, axs = plt.subplots(10, 4, figsize=(10, 8))
    axs[0, 0].set_title("Original", fontsize=5)
    axs[0, 1].set_title("Mask", fontsize=5)
    axs[0, 2].set_title("SNLN", fontsize=5)
    axs[0, 3].set_title("Inference", fontsize=5)
    ims = []
    for j in range(plot_inference.shape[1]):
        for i in range(10):
            axs[i, 0].imshow(
                plot_images[i, ..., 0], interpolation="nearest", aspect="auto"
            )
            axs[i, 1].imshow(
                plot_masks[i, ..., 0], interpolation="nearest", aspect="auto"
            )
            axs[i, 2].imshow(
                plot_snln[i, ..., 0], interpolation="nearest", aspect="auto"
            )
            temp_im = np.moveaxis(plot_inference[i][j], 0, -1) * 127.5 + 127.5
            axs[i, 3].imshow(temp_im, interpolation="nearest", aspect="auto")
            axs[i, 0].axis("off")
            axs[i, 1].axis("off")
            axs[i, 2].axis("off")
            axs[i, 3].axis("off")
        plt.savefig(os.path.join(plot_directory, f"results_{j}.png"), dpi=300)
    plt.close("all")
    fig = plt.figure()
    plt.axis("off")
    fig.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    for j in range(plot_inference.shape[1]):
        ims.append(
            [
                plt.imshow(
                    matplotlib.image.imread(
                        os.path.join(plot_directory, f"results_{j}.png")
                    ),
                    animated=True,
                    extent=[0, 1, 0, 1],
                    interpolation="nearest",
                    aspect="auto",
                )
            ]
        )
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(
        os.path.join(plot_directory, "results.gif"), writer="pillow", fps=10, dpi=300
    )
    plt.close("all")
    for j in range(plot_inference.shape[1] - 1):
        os.remove(os.path.join(plot_directory, f"results_{j}.png"))


def save_results(config_vals: dict, snn_metrics: dict, output_dir: str):
    """
    Saves metrics and config to files.
    """
    save_json(config_vals, output_dir, "config")
    save_json(snn_metrics, output_dir, "metrics")


def main_snn(
    input_dir: str,
    time_length=config.SNN_PARAMS["time_length"],
    average_n=config.SNN_PARAMS["average_n"],
    out_model_type="SDAE",
    skip_exists=True,
    plot=True,
    convert_threshold=config.SNN_PARAMS["convert_threshold"],
):
    """
    The main conversion, inference and evaluation function.
    """
    config_vals = load_config(input_dir)
    config_vals["time_length"] = time_length
    config_vals["average_n"] = average_n
    config_vals["model_type"] = out_model_type
    config_vals["convert_threshold"] = convert_threshold
    config_vals["model_name"] = generate_model_name(config_vals)
    output_dir = generate_output_dir(config_vals)
    print(output_dir)
    if skip_exists and os.path.exists(output_dir):
        return
    # Get dataset
    test_dataset, test_masks_original = load_test_dataset(config_vals)
    # Load model
    model = load_ann_model(input_dir, config_vals)
    # Convert to SNN
    snn_model = convert_to_snn(model, test_dataset, convert_threshold)
    # Evaluate
    sln_metrics, snln_error_recon, inference = evaluate_snn(
        snn_model,
        test_dataset,
        test_masks_original,
        config_vals["patch_size"],
        512,
        time_length,
        average_n,
    )
    # Save results to file
    os.makedirs(output_dir, exist_ok=True)
    save_results(config_vals, sln_metrics, output_dir)
    torch.save(snn_model.state_dict(), os.path.join(output_dir, "snn_autoencoder.pt"))
    # Plot results
    if plot:
        test_images = reconstruct_patches(
            test_dataset.dataset[:][0].cpu().detach().numpy(),
            512,
            config_vals["patch_size"],
        )
        test_masks_original_recon = reconstruct_patches(
            test_masks_original, 512, config_vals["patch_size"]
        )
        inference_recon = reconstruct_snn_inference(
            inference, 512, config_vals["patch_size"]
        )
        plot_snn_results(
            test_images,
            test_masks_original_recon,
            snln_error_recon,
            inference_recon,
            output_dir,
        )


def run_trial_snn(
    trial: optuna.Trial, config_vals: dict, test_dataset, test_masks_original, model
):
    """
    Version of run_trial for SNN models for optuna optimization.
    """
    config_vals["time_length"] = trial.suggest_int("time_length", 16, 64)
    config_vals["average_n"] = trial.suggest_int("average_n", 1, 64)
    config_vals["convert_threshold"] = trial.suggest_float(
        "convert_threshold", 0.0, 1.0
    )
    config_vals["model_type"] = "SDAE"
    if "trial" in config_vals:
        config_vals["trial"] = None
    config_vals["model_name"] = generate_model_name(config_vals)
    output_dir = generate_output_dir(config_vals)
    print(config_vals)
    if config_vals["average_n"] > config_vals["time_length"]:
        raise optuna.exceptions.TrialPruned
    # Convert to SNN
    snn_model = convert_to_snn(model, test_dataset, config_vals["convert_threshold"])
    # Evaluate
    sln_metrics, _, _ = evaluate_snn(
        snn_model,
        test_dataset,
        test_masks_original,
        config_vals["patch_size"],
        512,
        config_vals["time_length"],
        config_vals["average_n"],
    )
    # Save results to file
    os.makedirs(output_dir, exist_ok=True)
    save_results(config_vals, sln_metrics, output_dir)
    torch.save(snn_model.state_dict(), os.path.join(output_dir, "snn_autoencoder.pt"))
    return sln_metrics["f1"]


def main_optuna_snn(input_dir: str, n_trials=64):
    """
    Main loop for optuna hyperparameter optimization of SNN models.
    """
    study = optuna.create_study(direction="maximize")
    config_vals = load_config(input_dir)
    # Get dataset
    test_dataset, test_masks_original = load_test_dataset(config_vals)
    # Load model
    model = load_ann_model(input_dir, config_vals)
    study.optimize(
        lambda trial: run_trial_snn(
            trial, config_vals, test_dataset, test_masks_original, model
        ),
        n_trials=n_trials,
    )
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
        f"{get_output_dir()}{os.sep}best_trial_snn.json", "w", encoding="utf-8"
    ) as ofile:
        json.dump(trial.params, ofile, indent=4)
    with open(
        f"{get_output_dir()}{os.sep}completed_trials_snn.json", "w", encoding="utf-8"
    ) as ofile:
        completed_trials_out = []
        for trial_params in complete_trials:
            completed_trials_out.append(trial_params.params)
        json.dump(completed_trials_out, ofile, indent=4)
    with open(
        f"{get_output_dir()}{os.sep}pruned_trials_snn.json", "w", encoding="utf-8"
    ) as ofile:
        pruned_trials_out = []
        for trial_params in pruned_trials:
            pruned_trials_out.append(trial_params.params)
        json.dump(pruned_trials_out, ofile, indent=4)


def main_standard(
    input_dir=f"./{get_output_dir()}/DAE/MISO/DAE_MISO_HERA_32_2_10/",
    input_model_type="DAE",
    output_model_type="SDAE",
):
    """
    Standard single trial for SNN run. If sweep is set to True, a simple
    grid-search of hyperparameters is performed
    """
    sweep = False
    input_dirs = glob.glob(f"./{get_output_dir()}/{input_model_type}/MISO/*")
    time_lengths = [32, 64]
    average_n = [2, 4, 8, 16, 32]
    if sweep:
        for curr_input_dir in input_dirs:
            for time_length in time_lengths:
                for window_size in average_n:
                    plot = False
                    if time_length == 64 and average_n == 32:
                        plot = True
                    print(f"{curr_input_dir}\t{time_length}\t{window_size}")
                    main_snn(curr_input_dir, time_length, window_size, plot=plot)
    else:
        main_snn(
            input_dir,
            config.SNN_PARAMS["time_length"],
            config.SNN_PARAMS["average_n"],
            out_model_type=output_model_type,
            skip_exists=True,
            plot=False,
        )


def run_and_rename(inpur_dir: str, output_dir_name: str):
    for trial_dir in os.listdir(inpur_dir):
        print(os.path.join(inpur_dir, trial_dir))
        main_standard(os.path.join(inpur_dir, trial_dir))
    os.rename(
        os.path.join(get_output_dir(), "SDAE"),
        os.path.join(get_output_dir(), output_dir_name),
    )


if __name__ == "__main__":
    main_snn(
        "./outputs/DAE/MISO/DAE_MISO_HERA_32_5_10_trial_1_nimble-slug/",
        time_length=256,
        average_n=128,
        plot=True,
    )
