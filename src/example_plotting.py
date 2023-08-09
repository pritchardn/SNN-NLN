"""
Creates plots for presentations
Copyright (c) 2023, Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from tqdm import tqdm

from ann2snn import (
    load_ann_model,
    convert_to_snn,
    infer_snn,
    reconstruct_snn_inference,
    snln,
)
from config import DEVICE
from data import load_data, process_into_dataset, reconstruct_patches
from evaluation import infer, nln, nln_errors
from utils import load_config

CMAP = "inferno"


def plot_image_patches(images, i, output_dir, filename_prefix):
    plt.figure(figsize=(10, 10))
    plt.imshow(images[i][0, :, :], cmap=CMAP)
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, f"{filename_prefix}_{i}.png"),
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close("all")


def plot_latent_patch(latent, i, output_dir, filename_prefix):
    plt.figure(figsize=(10, 10))
    plt.imshow(latent[i][np.newaxis, :], cmap=CMAP)
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, f"{filename_prefix}_{i}.png"),
        bbox_inches="tight",
        pad_inches=0.0,
    )
    plt.close("all")


def plot_inference_patch(inference_patches, i, plot_dir, filename_prefix):
    for j in range(128):
        plt.figure(figsize=(10, 10))
        plt.imshow(inference_patches[0][j][i], cmap=CMAP)
        plt.axis("off")
        plt.savefig(
            os.path.join(plot_dir, f"{filename_prefix}_{j}.png"),
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close("all")


def plot_inference_recon(inference_recon, i, plot_dir, filename_prefix):
    for j in range(128):
        plt.figure(figsize=(10, 10))
        plt.imshow(inference_recon[i][j], cmap=CMAP)
        plt.axis("off")
        plt.savefig(
            os.path.join(plot_dir, f"{filename_prefix}_{j}.png"),
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close("all")


def make_or_load_inference(model, test_dataset, output_dir):
    if os.path.exists(os.path.join(output_dir, "snn_inference.npy")):
        inference = np.load(os.path.join(output_dir, "snn_inference.npy"))
    else:
        snn_model = convert_to_snn(model, test_dataset, "99.9%")
        snn_model.to(DEVICE)
        # Get SNN Inference
        inference = infer_snn(
            snn_model, test_dataset, runtime=256, batch_limit=40, n_limit=128
        )
        # Save SNN Inference
        np.save(os.path.join(output_dir, "snn_inference.npy"), inference)
    return inference


def animate_inference_patches(inference_patches, i, plot_dir, filename):
    fig = plt.figure(figsize=(10, 10), frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    fig.add_axes(ax)
    ims = []
    for j in range(128):
        ims.append(
            [
                ax.imshow(inference_patches[0, j, i, :, :], cmap=CMAP, animated=True),
            ]
        )
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(os.path.join(plot_dir, filename), writer="pillow", fps=10)
    plt.close("all")


def animate_inference_recon(inference_recon, i, plot_dir, filename):
    fig = plt.figure(figsize=(10, 10), frameon=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    fig.add_axes(ax)
    ims = []
    for j in range(128):
        ims.append(
            [
                ax.imshow(inference_recon[i, j, 0, :, :], cmap=CMAP, animated=True),
            ]
        )
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(os.path.join(plot_dir, filename), writer="pillow", fps=10)
    plt.close("all")


if __name__ == "__main__":
    output_dir = "./outputs/examples"
    os.makedirs(output_dir, exist_ok=True)
    # Get NLN Model
    model_dir = "./outputs/DAE/MISO/DAE_MISO_HERA_32_5_10_trial_7_impossible-wombat"
    config_vals = load_config(model_dir)
    model = load_ann_model(model_dir, config_vals)
    model.to(DEVICE)
    # Get NLN Data
    train_x, train_y, test_x, test_y, _ = load_data()
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
    train_dataset, _ = process_into_dataset(
        train_x,
        train_y,
        batch_size=config_vals["batch_size"],
        mode="HERA",
        threshold=config_vals["threshold"],
        patch_size=config_vals["patch_size"],
        stride=config_vals["patch_stride"],
        shuffle=False,
        get_orig=False,
    )
    # Run inference and get data
    image_batch, predictions, latent = None, None, None
    for image_batch, _ in test_dataset:
        image_batch = image_batch.to(DEVICE)
        predictions = model(image_batch).cpu().detach().numpy()
        latent = model.encoder(image_batch).cpu().detach().numpy()
        image_batch = image_batch.cpu().detach().numpy()
        break
    predictions = infer(model, test_dataset, 32)
    image_reconstructions = reconstruct_patches(
        test_dataset.dataset[:2560][0].cpu().detach().numpy(), 512, 32
    )
    output_recon = reconstruct_patches(predictions[:2560], 512, 32)

    # Get NLN error
    z_train = infer(model.encoder, train_dataset, 32, True)
    z_query = infer(model.encoder, test_dataset, 32, True)
    x_hat_train = infer(model, train_dataset, 32)
    neighbours_dist, neighbours_idx, neighbour_mask = nln(
        z_train, z_query, config_vals["neighbours"]
    )
    nln_error = nln_errors(
        test_dataset, predictions, x_hat_train, neighbours_idx, neighbour_mask
    )
    # Reconstruct NLN outputs
    nln_error_recon = reconstruct_patches(nln_error[:2560], 512, 32)

    # Plot inputs
    output_dir = os.path.join("outputs", "examples", "nln_inputs")
    os.makedirs(output_dir, exist_ok=True)
    for i in range(10):
        plot_image_patches(image_batch, i, output_dir, "nln_input")
        plot_image_patches(image_reconstructions, i, output_dir, "nln_input_recon")

    # Plot latent
    output_dir = os.path.join("outputs", "examples", "nln_latent")
    os.makedirs(output_dir, exist_ok=True)
    for i in range(10):
        plot_latent_patch(latent, i, output_dir, "nln_latent")

    # Plot output
    output_dir = os.path.join("outputs", "examples", "nln_output")
    os.makedirs(output_dir, exist_ok=True)
    for i in range(10):
        plot_image_patches(predictions, i, output_dir, "nln_output")
        plot_image_patches(output_recon, i, output_dir, "nln_output_recon")
    # Plot masks
    output_dir = os.path.join("outputs", "examples", "nln_masks")
    os.makedirs(output_dir, exist_ok=True)
    for i in range(10):
        plot_image_patches(test_masks_original, i, output_dir, "nln_mask")
        plot_image_patches(nln_error_recon, i, output_dir, "nln_masks_recon")

    # Get SNN Model
    output_dir = os.path.join("outputs", "examples", "snln_inference")
    os.makedirs(output_dir, exist_ok=True)
    inference = make_or_load_inference(model, test_dataset, output_dir)
    snln_error = snln(inference, test_dataset, 128, limit=2560)
    snln_error_recon = reconstruct_patches(snln_error, 512, 32)
    snln_error_recon_binarized = np.where(snln_error_recon > 0.0, 1, 0.0)

    # Plot SNLN inference patches
    inference_cum_sum = inference.cumsum(axis=1)
    inference_recon = reconstruct_snn_inference(inference, 512, 32)
    inference_cum_sum_recon = reconstruct_snn_inference(inference_cum_sum, 512, 32)
    inference = np.moveaxis(inference, 3, -1)
    inference_cum_sum = np.moveaxis(inference_cum_sum, 3, -1)
    inference_recon = np.moveaxis(inference_recon, 2, -1)

    print("Plotting SNLN Inference")
    for i in tqdm(range(10)):
        plot_dir = os.path.join(output_dir, "snn_inference_patches", f"patch_{i}")
        os.makedirs(plot_dir, exist_ok=True)
        plot_inference_patch(inference, i, plot_dir, "snn_inference_patch")
        plot_inference_patch(inference_cum_sum, i, plot_dir, "snn_inference_patch_sum")
        plot_inference_recon(inference_recon, i, plot_dir, "inference_recon")
        plot_image_patches(snln_error_recon, i, plot_dir, "snln_error_recon")
        plot_image_patches(
            snln_error_recon_binarized, i, plot_dir, "snln_error_recon_binarized"
        )
        # Plot SNLN Animated Inference
        animate_inference_patches(inference, i, plot_dir, "results.gif")
        animate_inference_patches(inference_cum_sum, i, plot_dir, "results_sum.gif")
        # Plot The same for reconstructions
        animate_inference_recon(inference_recon, i, plot_dir, "inference_recon.gif")
        animate_inference_recon(
            inference_cum_sum_recon, i, plot_dir, "inference_sum_recon.gif"
        )
