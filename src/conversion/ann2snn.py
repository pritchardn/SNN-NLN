import glob
import json
import math
import os

import matplotlib.animation as animation
import matplotlib.image
import numpy as np
import torch
from matplotlib import pyplot as plt
from spikingjelly.activation_based import ann2snn
from tqdm import tqdm

from data import load_data, process_into_dataset, reconstruct_latent_patches, reconstruct_patches
from evaluation import _calculate_metrics
from main import save_config
from models import Autoencoder
from utils import generate_model_name, generate_output_dir


def animated_plotting(out_images):
    for i in range(len(out_images[0])):
        fig, ax = plt.subplots()
        ax.axis('off')
        ims = []
        for j in range(len(out_images)):
            temp_im = np.moveaxis(out_images[j][i], 0, -1) * 127.5 + 127.5
            im = ax.imshow(temp_im, animated=True)
            if j == 0:
                ax.imshow(temp_im)  # show an initial one first
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        ani.save(f'output_{i}.gif', writer='pillow', fps=10)
        plt.close('all')
        print(f"Done {i}")


def plot_output_states(out_images):
    for i in range(len(out_images[0])):
        plt.figure(figsize=(10, 10))
        sub_range = int(math.ceil(np.sqrt(len(out_images))))
        f, axarr = plt.subplots(sub_range, sub_range)
        axarr = axarr.flatten()
        for j in range(len(out_images)):
            axarr[j].axis('off')
            axarr[j].imshow(np.moveaxis(out_images[j][i], 0, -1) * 127.5 + 127.5)
        plt.axis('off')
        plt.savefig(f'output_{i}.png')
        plt.close('all')
        print(f"Done {i}")


def plot_input_images(in_images):
    for i in range(len(in_images)):
        plt.figure(figsize=(10, 10))
        plt.imshow(np.moveaxis(in_images[i], 0, -1) * 127.5 + 127.5)
        plt.axis('off')
        plt.savefig(f'input_{i}.png')
        plt.close('all')
        print(f"Done {i}")


def plot_outputs(out_images):
    for output_states in out_images:
        plot_output_states(output_states)
        animated_plotting(output_states)


def infer_snn(model, dataloader, runtime=50, batch_limit=1):
    model.eval().to('cuda')
    full_output = []
    i = 0
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(dataloader)):
            img = img.to('cuda')
            out_images = []
            if runtime is None:
                out = model(img)
                print(out.shape)
            else:
                for m in model.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
                for t in range(runtime):
                    if t == 0:
                        out = model(img)
                    else:
                        out += model(img)
                    # Add current state to image building
                    out_images.append(out.cpu().numpy())
                full_output.append(out_images)
            i += 1
            if i == batch_limit:
                break
    return np.asarray(full_output)  # [B, T, N, C, W, H]


def convert_to_snn(model, test_data_loader):
    model_converter = ann2snn.Converter(mode='max', dataloader=test_data_loader)
    snn_model = model_converter(model)
    snn_model.graph.print_tabular()
    return snn_model


def test_snn_model(snn_model, test_data_loader):
    full_output = infer_snn(snn_model, test_data_loader, runtime=64)
    plot_outputs(full_output)


def load_config(input_dir: str):
    config_file_path = os.path.join(input_dir, "config.json")
    with open(config_file_path, "r") as f:
        config = json.load(f)
    return config


def load_test_dataset(config_vals: dict):
    _, _, test_x, test_y, rfi_models = load_data()
    test_dataset, y_data_orig = process_into_dataset(test_x, test_y,
                                                     batch_size=config_vals['batch_size'],
                                                     mode='HERA',
                                                     threshold=config_vals['threshold'],
                                                     patch_size=config_vals['patch_size'],
                                                     stride=config_vals['patch_stride'],
                                                     shuffle=False,
                                                     get_orig=True)
    return test_dataset, y_data_orig


def load_ann_model(input_dir: str, config_vals: dict, test_dataset: torch.utils.data.Dataset):
    model_path = os.path.join(input_dir, "autoencoder.pt")
    model = Autoencoder(config_vals['num_layers'], config_vals['latent_dimension'],
                        config_vals['num_filters'], test_dataset.dataset[0][0].shape)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def snln(x_hat, test_dataset, average_n):
    # x_hat: [B, T, N, C, W, H]
    x_hat_trimmed = x_hat[:, -average_n:, :, :, :, :]
    average_x_hat = np.vstack(np.mean(x_hat_trimmed, axis=1))
    images = test_dataset.dataset[:][1].cpu().detach().numpy()
    error = images - average_x_hat
    return error


def evaluate_snn(model, test_dataset, test_masks_original, patch_size, original_size, runtime,
                 average_n):
    test_masks_original_reconstructed = reconstruct_patches(test_masks_original, original_size,
                                                            patch_size)
    x_hat = infer_snn(model, test_dataset, runtime=runtime, batch_limit=-1)
    snln_error = snln(x_hat, test_dataset, average_n)
    if patch_size:
        if snln_error.ndim == 4:
            snln_error_recon = reconstruct_patches(snln_error, original_size, patch_size)
        else:
            snln_error_recon = reconstruct_latent_patches(snln_error, original_size, patch_size)
    else:
        snln_error_recon = snln_error
    snln_metrics = _calculate_metrics(test_masks_original_reconstructed, snln_error_recon)
    return snln_metrics, snln_error_recon, x_hat


def reconstruct_snn_inference(inference: np.array, original_size: int, kernel_size: int):
    t = np.vstack(inference.transpose(0, 2, 4, 5, 1, 3))
    n_patches = original_size // kernel_size
    reconstruction = np.empty(
        [t.shape[0] // n_patches ** 2,
         kernel_size * n_patches,
         kernel_size * n_patches,
         t.shape[-2],
         t.shape[-1]])

    start, counter, indx, b = 0, 0, 0, []

    for i in range(n_patches, t.shape[0] + 1, n_patches):
        b.append(np.reshape(np.stack(t[start:i, ...], axis=0),
                            (n_patches * kernel_size, kernel_size, t.shape[-2], t.shape[-1])))
        start = i
        counter += 1
        if counter == n_patches:
            reconstruction[indx, ...] = np.hstack(b)
            indx += 1
            counter, b = 0, []
    reconstruction = reconstruction.transpose(0, 3, 4, 1, 2)
    return reconstruction


def plot_snn_results(original_images, test_masks_recon, snln_error_recon, inference, output_dir):
    plot_directory = os.path.join(output_dir, "results")
    os.makedirs(plot_directory, exist_ok=True)
    plot_images = np.moveaxis(original_images[:10, ...], 1, -1)
    plot_masks = np.moveaxis(test_masks_recon[:10, ...], 1, -1)
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
            axs[i, 0].imshow(plot_images[i, ..., 0], interpolation='nearest', aspect='auto')
            axs[i, 1].imshow(plot_masks[i, ..., 0], interpolation='nearest', aspect='auto')
            axs[i, 2].imshow(plot_snln[i, ..., 0], interpolation='nearest', aspect='auto')
            temp_im = np.moveaxis(plot_inference[i][j], 0, -1) * 127.5 + 127.5
            axs[i, 3].imshow(temp_im, interpolation='nearest', aspect='auto')
            axs[i, 0].axis('off')
            axs[i, 1].axis('off')
            axs[i, 2].axis('off')
            axs[i, 3].axis('off')
        plt.savefig(os.path.join(plot_directory, "results_{}.png".format(j)), dpi=300)
    plt.close('all')
    fig = plt.figure()
    plt.axis('off')
    fig.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    for j in range(plot_inference.shape[1]):
        ims.append(
            [plt.imshow(
                matplotlib.image.imread(os.path.join(plot_directory, "results_{}.png".format(j))),
                animated=True, extent=[0, 1, 0, 1], interpolation='nearest', aspect='auto')])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(os.path.join(plot_directory, "results.gif"), writer='pillow', fps=10, dpi=300)
    plt.close('all')
    for j in range(plot_inference.shape[1] - 1):
        os.remove(os.path.join(plot_directory, "results_{}.png".format(j)))


def save_results(config_vals: dict, snn_metrics: dict, output_dir: str):
    save_config(config_vals, output_dir)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(snn_metrics, f, indent=4)


def main(input_dir: str, time_length, average_n, skip_exists=True):
    config_vals = load_config(input_dir)
    config_vals['time_length'] = time_length
    config_vals['average_n'] = average_n
    config_vals["model_type"] = "SDAE"
    config_vals['model_name'] = generate_model_name(config_vals)
    output_dir = generate_output_dir(config_vals)
    if skip_exists and os.path.exists(output_dir):
        return
    # Get dataset
    test_dataset, test_masks_original = load_test_dataset(config_vals)
    # Load model
    model = load_ann_model(input_dir, config_vals, test_dataset)
    # Convert to SNN
    snn_model = convert_to_snn(model, test_dataset)
    # Evaluate
    sln_metrics, snln_error_recon, inference = evaluate_snn(snn_model, test_dataset,
                                                            test_masks_original,
                                                            config_vals['patch_size'], 512,
                                                            time_length, average_n)
    # Plot results
    test_images = reconstruct_patches(test_dataset.dataset[:][0].cpu().detach().numpy(), 512,
                                      config_vals['patch_size'])
    test_masks_original_recon = reconstruct_patches(test_masks_original, 512,
                                                    config_vals['patch_size'])
    inference_recon = reconstruct_snn_inference(inference, 512, config_vals['patch_size'])
    plot_snn_results(test_images, test_masks_original_recon, snln_error_recon, inference_recon,
                     output_dir)
    # Save results to file
    os.makedirs(output_dir, exist_ok=True)
    save_results(config_vals, sln_metrics, output_dir)
    torch.save(snn_model.state_dict(), os.path.join(output_dir, "snn_autoencoder.pt"))


if __name__ == "__main__":
    SWEEP = False
    input_dirs = glob.glob("./outputs/DAE/MISO/*")
    time_lengths = [32, 64]
    average_n = [2, 4, 8, 16, 32]
    i = 0
    if SWEEP:
        for input_dir in input_dirs:
            for time_length in time_lengths:
                for n in average_n:
                    print(f"{input_dir}\t{time_length}\t{n}")
                    main(input_dir, time_length, n)
    else:
        input_dir = "./outputs/DAE/MISO/DAE_MISO_HERA_32_2_10/"
        main(input_dir, 32, 5, skip_exists=False)
