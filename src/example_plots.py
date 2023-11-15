import matplotlib.pyplot as plt
import numpy as np

from ann2snn import load_ann_model, convert_to_snn, evaluate_snn
from config import DEVICE
from data import reconstruct_patches, load_data, process_into_dataset
from evaluation import infer, nln, nln_errors
from utils import load_config


def plot_spectrogram(image, title, output_name):
    plt.figure(figsize=(5, 5))
    plt.imshow(image, aspect="auto", interpolation="nearest")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency Bins")
    plt.savefig(f"{output_name}_{title}.png", dpi=300)
    plt.close("all")


def plot_final_images(
    dataset: str, original, mask, nln_output, snln_output, num_examples: int = 1
):
    _, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].set_title("Original")
    axes[1].set_title("Mask")
    axes[2].set_title("NLN Output")
    axes[3].set_title("SNLN Output")
    # original = np.moveaxis(original, 0, -1)
    # mask = np.moveaxis(mask, 1, -1)
    # nln_output = np.moveaxis(nln_output, 1, -1)
    # snln_output = np.moveaxis(snln_output, 1, -1)
    for i in range(num_examples):
        temp = original[i, ...]
        temp = np.moveaxis(temp, 0, -1)
        axes[0].imshow(temp.astype(np.float32), interpolation="nearest", aspect="auto")
        plot_spectrogram(temp.astype(np.float32), "original", f"{dataset}_{i}")
        temp = mask[i, ...]
        temp = np.moveaxis(temp, 0, -1)
        axes[1].imshow(temp.astype(np.float32), interpolation="nearest", aspect="auto")
        plot_spectrogram(temp.astype(np.float32), "mask", f"{dataset}_{i}")
        temp = nln_output[i, ...]
        temp = np.moveaxis(temp, 0, -1)
        axes[2].imshow(temp.astype(np.float32), interpolation="nearest", aspect="auto")
        plot_spectrogram(temp.astype(np.float32), "nln", f"{dataset}_{i}")
        temp = snln_output[i, ...]
        temp = np.moveaxis(temp, 0, -1)
        axes[3].imshow(temp.astype(np.float32), interpolation="nearest", aspect="auto")
        plot_spectrogram(temp.astype(np.float32), "snln", f"{dataset}_{i}")
        plt.savefig(f"{dataset}_{i}.png", dpi=300)


def create_dataset_examples(input_dir: str, config_vals: dict):
    original_size = 512
    patch_size = config_vals["patch_size"]
    latent_dimension = config_vals["latent_dimension"]
    neighbours = config_vals["neighbours"]
    # Get dataset
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
        limit=config_vals.get("limit", None),
    )
    # Load model
    model = load_ann_model(input_dir, config_vals).to(DEVICE)

    test_masks_original_reconstructed = reconstruct_patches(
        test_dataset.dataset[:][1].cpu().detach().numpy(), original_size, patch_size
    )
    test_orig_reconstructed = reconstruct_patches(
        test_dataset.dataset[:][0].cpu().detach().numpy(), original_size, patch_size
    )
    z_train = infer(model.encoder, train_dataset, latent_dimension, True)
    z_query = infer(model.encoder, test_dataset, latent_dimension, True)

    x_hat_train = infer(model, train_dataset, patch_size, False)

    neighbours_dist, neighbours_idx, neighbour_mask = nln(z_train, z_query, neighbours)

    x_hat = infer(model, test_dataset, patch_size, False)

    nln_error = nln_errors(
        test_dataset, x_hat, x_hat_train, neighbours_idx, neighbour_mask
    )
    nln_error_recon = reconstruct_patches(nln_error, original_size, patch_size)

    # Convert to SNN
    snn_model = convert_to_snn(model, test_dataset, "99.9%")
    sln_metrics, snln_error_recon, inference = evaluate_snn(
        snn_model,
        test_dataset,
        test_masks_original,
        config_vals["patch_size"],
        512,
        256,
        128,
    )

    if config_vals["dataset"] == "HERA":
        combined_recon = np.where(
            nln_error_recon > np.median(nln_error_recon) + np.std(nln_error_recon), 1, 0
        )
        snln_error_recon = np.where(
            snln_error_recon > snln_error_recon.mean() + snln_error_recon.std() * 2,
            1,
            0,
        )

    elif config_vals["dataset"] == "LOFAR":
        combined_recon = np.where(
            nln_error_recon > nln_error_recon.mean() + nln_error_recon.std() * 2.5, 1, 0
        )
        snln_error_recon = np.where(
            snln_error_recon > snln_error_recon.mean() + snln_error_recon.std() * 2.5,
            1,
            0,
        )
    elif config_vals["dataset"] == "TABASCAL":
        combined_recon = np.where(
            nln_error_recon > nln_error_recon.mean() + nln_error_recon.std() * 2, 1, 0
        )
        snln_error_recon = np.where(
            snln_error_recon > snln_error_recon.mean() + snln_error_recon.std(), 1, 0
        )
    else:
        raise ValueError("Dataset not implemented")

    return (
        test_orig_reconstructed,
        test_masks_original_reconstructed,
        combined_recon,
        snln_error_recon,
    )


def main(input_dir: str):
    config_vals = load_config(input_dir)
    config_vals["limit"] = 20
    originals, masks, nln_outputs, snln_error_recon = create_dataset_examples(
        input_dir, config_vals
    )
    plot_final_images(
        config_vals["dataset"],
        originals,
        masks,
        nln_outputs,
        snln_error_recon,
        config_vals["limit"],
    )


if __name__ == "__main__":
    # Make HERA plot
    input_dir = (
        "outputs/FINAL/HERA/DAE/MISO/DAE_MISO_HERA_64_2_10_trial_1_warping-lobster"
    )
    # main(input_dir)
    # Make LOFAR plot
    input_dir = (
        "outputs/FINAL/LOFAR/DAE/MISO/DAE_MISO_LOFAR_64_2_10_trial_10_wisteria-mule"
    )
    main(input_dir)
    # Make TABASCAL plot
    input_dir = "outputs/FINAL/TABASCAL/DAE/MISO/DAE_MISO_TABASCAL_64_2_10_trial_3_jumping-fossa"
    # main(input_dir)
