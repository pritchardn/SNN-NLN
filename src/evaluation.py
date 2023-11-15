"""
Contains methods used to calculate model performance for all trials.
Copyright (c) 2023, Nicholas Pritchard  <nicholas.pritchard@icrar.org>
"""
import json
import os

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    mean_squared_error,
    balanced_accuracy_score,
)
from torch import nn

from config import DEVICE, get_output_dir
from data import reconstruct_patches, reconstruct_latent_patches
from models import AutoEncoder


def infer(
    model: nn.Module,
    dataset: torch.utils.data.DataLoader,
    edge_size: int,
    latent=False,
):
    """
    Infer the output of a model on a dataset.
    """
    if latent:
        output = np.empty([len(dataset.dataset), edge_size], dtype=np.float32)
    else:
        output = np.empty(
            [len(dataset.dataset), 1, edge_size, edge_size],
            dtype=np.float32,
        )
    start = 0
    for image_batch, _ in dataset:
        image_batch = image_batch.to(DEVICE)
        predictions = model(image_batch).cpu().detach().numpy()
        output[start : start + len(predictions), ...] = predictions
        start += len(predictions)
    return output


def save_metrics(
    ae_metrics: dict,
    nln_metrics: dict,
    dist_metrics: dict,
    combined_metrics: dict,
    model_type: str,
    anomaly_type: str,
    model_name: str,
):
    """
    Saves metrics to a json file.
    """
    output_filepath = os.path.join(
        get_output_dir(), model_type, anomaly_type, model_name
    )
    os.makedirs(output_filepath, exist_ok=True)
    with open(
        os.path.join(output_filepath, "metrics.json"), "w", encoding="utf-8"
    ) as metric_file:
        json.dump(
            {
                "ae": ae_metrics,
                "nln": nln_metrics,
                "dist": dist_metrics,
                "combined": combined_metrics,
            },
            metric_file,
            indent=4,
        )


def plot_final_images(
    metrics: dict,
    neighbour: int,
    model_type: str,
    anomaly_type: str,
    model_name: str,
    test_images_reconstructed,
    test_masks_reconstructed,
    error_reconstructed,
    nln_error_reconstructed,
    distrubtions_reconstructed,
    combined_reconstructed,
    latent_reconstructed,
):
    """
    Plots final images for a model. Plots interim inferance panes too.
    """
    _, axes = plt.subplots(10, 7, figsize=(10, 8))
    axes[0, 0].set_title("Inp", fontsize=5)
    axes[0, 1].set_title("Mask", fontsize=5)
    axes[0, 2].set_title(f'Recon {metrics.get("ae_ao_auroc", 0)}', fontsize=5)
    axes[0, 3].set_title(f'NLN {metrics.get("nln_ao_auroc", 0)}', fontsize=5)
    axes[0, 4].set_title(
        f'Dist {metrics.get("dists_ao_auroc", 0)} {neighbour}', fontsize=5
    )
    axes[0, 5].set_title(
        f'Combined {metrics.get("combined_ao_auroc", 0)} {neighbour}', fontsize=5
    )
    axes[0, 6].set_title(
        f'Recon {metrics.get("combined_ao_auroc", 0)} {neighbour}', fontsize=5
    )
    test_images_reconstructed = np.moveaxis(test_images_reconstructed, 1, -1)
    test_masks_reconstructed = np.moveaxis(test_masks_reconstructed, 1, -1)
    error_reconstructed = np.moveaxis(error_reconstructed, 1, -1)
    nln_error_reconstructed = np.moveaxis(nln_error_reconstructed, 1, -1)
    distrubtions_reconstructed = np.moveaxis(distrubtions_reconstructed, 1, -1)
    combined_reconstructed = np.moveaxis(combined_reconstructed, 1, -1)
    latent_reconstructed = np.moveaxis(latent_reconstructed, 1, -1)
    for i in range(10):
        random_index = np.random.randint(len(test_images_reconstructed))
        axes[i, 0].imshow(
            test_images_reconstructed[random_index, ..., 0].astype(np.float32),
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="auto",
        )
        axes[i, 1].imshow(
            test_masks_reconstructed[random_index, ..., 0].astype(np.float32),
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="auto",
        )
        axes[i, 2].imshow(
            error_reconstructed[random_index, ..., 0].astype(np.float32),
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="auto",
        )
        axes[i, 3].imshow(
            nln_error_reconstructed[random_index, ..., 0].astype(np.float32),
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="auto",
        )
        axes[i, 4].imshow(
            distrubtions_reconstructed[random_index, ..., 0].astype(np.float32),
            interpolation="nearest",
            aspect="auto",
        )
        axes[i, 5].imshow(
            combined_reconstructed[random_index, ..., 0].astype(np.float32),
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="auto",
        )
        axes[i, 6].imshow(
            latent_reconstructed[random_index, ..., 0].astype(np.float32),
            vmin=0,
            vmax=1,
            interpolation="nearest",
            aspect="auto",
        )

    output_filepath = os.path.join(
        get_output_dir(), model_type, anomaly_type, model_name
    )
    os.makedirs(output_filepath, exist_ok=True)
    plt.savefig(os.path.join(output_filepath, f"neighbours_{neighbour}.png"), dpi=300)


def get_error_dataset(
    images: torch.utils.data.DataLoader, x_hat: np.ndarray, image_size: int
):
    """
    Calculates the error between the original images and the
    reconstructed images for an entire dataset.
    """
    output = np.empty(
        [len(images.dataset), 1, image_size, image_size], dtype=np.float32
    )
    start = 0
    for image, _ in images:
        error = image.cpu().detach().numpy() - x_hat[start : start + len(image), ...]
        output[0 : len(error), ...] = error
        start += len(error)
    return output


def _calculate_metrics(test_masks_orig_recon: np.ndarray, error_recon: np.ndarray):
    if error_recon.shape[1] == 1 and test_masks_orig_recon.shape[1] == 1:
        error_recon = np.moveaxis(error_recon, 1, -1)
        test_masks_orig_recon = np.moveaxis(test_masks_orig_recon, 1, -1)
    false_pos_rate, true_pos_rate, _ = roc_curve(
        test_masks_orig_recon.flatten() > 0, error_recon.flatten()
    )
    acc = balanced_accuracy_score(
        test_masks_orig_recon.flatten() > 0, error_recon.flatten() > 0
    )
    mse = mean_squared_error(test_masks_orig_recon.flatten(), error_recon.flatten())
    true_auroc = auc(false_pos_rate, true_pos_rate)
    precision, recall, _ = precision_recall_curve(
        test_masks_orig_recon.flatten() > 0, error_recon.flatten()
    )
    true_auprc = auc(recall, precision)
    f1_scores = 2 * recall * precision / (recall + precision)
    true_f1 = np.max(f1_scores)
    if error_recon.shape[1] != 1 and test_masks_orig_recon.shape[1] != 1:
        # In-case the side effects are used later.
        error_recon = np.moveaxis(error_recon, -1, 1)
        test_masks_orig_recon = np.moveaxis(test_masks_orig_recon, -1, 1)
    return {
        "auroc": float(true_auroc),
        "auprc": float(true_auprc),
        "f1": float(true_f1),
        "mse": float(mse),
        "acc": float(acc),
    }


def nln(z_train, z_query, neighbours):
    """
    Calculates the nearest neighbours of a query set in the latent representation of a training set.
    :return:
    """
    index = faiss.IndexFlatL2(z_train.shape[1])
    index.add(z_train.astype(np.float32))
    neighbours_dist, neighbours_idx = index.search(
        z_query.astype(np.float32), neighbours
    )
    neighbour_mask = np.zeros([len(neighbours_idx)], dtype=bool)

    return neighbours_dist, neighbours_idx, neighbour_mask


def nln_errors(
    test_dataset: torch.utils.data.DataLoader,
    x_hat,
    x_hat_train,
    neighbours_idx,
    neighbour_mask,
):
    """
    Calculates the error between the inferred neighbours of each test image in a whole dataset.
    """
    test_images = test_dataset.dataset[:][0].cpu().detach().numpy()
    test_images_stacked = np.stack([test_images] * neighbours_idx.shape[-1], axis=1)
    neighbours = x_hat_train[neighbours_idx]

    error_nln = test_images_stacked - neighbours
    np.abs(error_nln, dtype=np.float32, out=error_nln)
    error = np.mean(error_nln, axis=1)  # nanmean for frNN

    error_recon = test_images - x_hat
    np.abs(error_recon, dtype=np.float32, out=error_recon)
    error[neighbour_mask] = error_recon[neighbour_mask]
    return error


def get_dists(neighbours_dist, original_size: int, patch_size: int = None):
    """
    Calculates the mean distance between each test image and its neighbours.
    """
    dists = np.mean(neighbours_dist, axis=tuple(range(1, neighbours_dist.ndim)))
    if patch_size is not None:
        dists = np.array([[d] * patch_size ** 2 for i, d in enumerate(dists)]).reshape(
            len(dists), patch_size, patch_size
        )
        dists_recon = reconstruct_patches(
            np.expand_dims(dists, axis=1), original_size, patch_size
        )
        return dists_recon
    return dists


def calculate_metrics(
    model: AutoEncoder,
    test_masks_original: np.ndarray,
    test_dataset: torch.utils.data.DataLoader,
    train_dataset: torch.utils.data.DataLoader,
    neighbours: int,
    model_name: str,
    model_type: str,
    anomaly_type: str,
    latent_dimension: int,
    original_size: int,
    patch_size: int = None,
    dataset="HERA",
    evaluate_run=False,
):
    """
    The function for calculating metrics for a model trial.
    """
    test_masks_original_reconstructed = reconstruct_patches(
        test_masks_original, original_size, patch_size
    )
    z_train = infer(model.encoder, train_dataset, latent_dimension, True)
    z_query = infer(model.encoder, test_dataset, latent_dimension, True)

    x_hat = infer(model, test_dataset, patch_size, False)
    x_hat_train = infer(model, train_dataset, patch_size, False)

    error = get_error_dataset(test_dataset, x_hat, patch_size)

    error_recon = reconstruct_patches(error, original_size, patch_size)

    ae_metrics = _calculate_metrics(test_masks_original_reconstructed, error_recon)
    neighbours_dist, neighbours_idx, neighbour_mask = nln(z_train, z_query, neighbours)

    x_hat = infer(model, test_dataset, patch_size, False)

    nln_error = nln_errors(
        test_dataset, x_hat, x_hat_train, neighbours_idx, neighbour_mask
    )

    if patch_size:
        if nln_error.ndim == 4:
            nln_error_recon = reconstruct_patches(nln_error, original_size, patch_size)
        else:
            nln_error_recon = reconstruct_latent_patches(
                nln_error, original_size, patch_size
            )
    else:
        nln_error_recon = nln_error

    dists_recon = get_dists(neighbours_dist, original_size, patch_size)

    nln_metrics = _calculate_metrics(test_masks_original_reconstructed, nln_error_recon)

    if dataset == "HERA":
        combined_recon = nln_error_recon * np.array(
            [d > np.percentile(d, 10) for d in dists_recon]
        )
    elif dataset == "LOFAR":
        combined_recon = np.clip(
            nln_error_recon, nln_error_recon.mean() + nln_error_recon.std() * 5, 1.0
        ) * np.array([d > np.percentile(d, 66) for d in dists_recon])
    elif dataset == "TABASCAL":
        combined_recon = np.clip(
            nln_error_recon, nln_error_recon.mean() + nln_error_recon.std() * 5, 1.0
        ) * np.array([d > np.percentile(d, 66) for d in dists_recon])
    else:
        raise ValueError("Dataset not implemented")
    combined_recon = np.nan_to_num(combined_recon)
    combined_metrics = _calculate_metrics(test_masks_original, combined_recon)

    dist_metrics = _calculate_metrics(test_masks_original_reconstructed, dists_recon)

    if not evaluate_run:
        test_images_recon = reconstruct_patches(
            test_dataset.dataset[:][0].cpu().detach().numpy(), original_size, patch_size
        )
        test_masks_reconstructed = reconstruct_patches(
            test_dataset.dataset[:][1].cpu().detach().numpy(), original_size, patch_size
        )
        smoothed_x_hat = np.ones_like(x_hat)
        for i in range(len(x_hat)):
            smoothed_x_hat[i, 0, :, :] = x_hat[i, 0, :, :]
        x_hat_recon = reconstruct_patches(smoothed_x_hat, original_size, patch_size)
        plot_final_images(
            ae_metrics,
            neighbours,
            model_type,
            anomaly_type,
            model_name,
            test_images_recon,
            test_masks_reconstructed,
            error_recon,
            nln_error_recon,
            dists_recon,
            combined_recon,
            x_hat_recon,
        )
        np.save(
            f"{get_output_dir()}/{model_type}/{anomaly_type}/{model_name}/test_query.npy",
            z_query,
        )
    return ae_metrics, nln_metrics, dist_metrics, combined_metrics


def mid_run_calculate_metrics(
    model: AutoEncoder,
    test_masks_original: np.ndarray,
    test_dataset: torch.utils.data.DataLoader,
    train_dataset: torch.utils.data.DataLoader,
    neighbours: int,
    latent_dimension: int,
    original_size: int,
    patch_size: int = None,
):
    """
    Calculates metrics for a model trial mid-run. Only calculates nln metrics to avoid useless work.
    """
    test_masks_original_reconstructed = reconstruct_patches(
        test_masks_original, original_size, patch_size
    )
    z_train = infer(model.encoder, train_dataset, latent_dimension, True)
    z_query = infer(model.encoder, test_dataset, latent_dimension, True)

    x_hat = infer(model, test_dataset, patch_size, False)
    x_hat_train = infer(model, train_dataset, patch_size, False)

    _, neighbours_idx, neighbour_mask = nln(z_train, z_query, neighbours)
    nln_error = nln_errors(
        test_dataset, x_hat, x_hat_train, neighbours_idx, neighbour_mask
    )

    if patch_size:
        if nln_error.ndim == 4:
            nln_error_recon = reconstruct_patches(nln_error, original_size, patch_size)
        else:
            nln_error_recon = reconstruct_latent_patches(
                nln_error, original_size, patch_size
            )
    else:
        nln_error_recon = nln_error

    nln_metrics = _calculate_metrics(test_masks_original_reconstructed, nln_error_recon)
    return nln_metrics


def evaluate_model(
    model,
    test_masks,
    test_dataset,
    train_dataset,
    neighbours,
    latent_dimension,
    original_size,
    patch_size,
    model_name,
    model_type,
    anomaly_type,
    dataset,
    evaluate_run=False,
):
    """
    Evaluates a model trial.
    """
    ae_metrics, nln_metrics, dist_metrics, combined_metrics = calculate_metrics(
        model,
        test_masks,
        test_dataset,
        train_dataset,
        neighbours,
        model_name,
        model_type,
        anomaly_type,
        latent_dimension,
        original_size,
        patch_size,
        dataset,
        evaluate_run,
    )
    # TODO: Move metric saving to outside
    save_metrics(
        ae_metrics,
        nln_metrics,
        dist_metrics,
        combined_metrics,
        model_type,
        anomaly_type,
        model_name,
    )
    return nln_metrics
