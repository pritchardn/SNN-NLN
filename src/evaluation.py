import os
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import faiss
import json
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from torch import nn

from config import DEVICE
from data import reconstruct_patches, reconstruct_latent_patches
from models import Autoencoder


def infer(model: nn.Module, dataset: torch.utils.data.DataLoader, batch_size: int,
          latent_dimension: int, latent=False):
    if latent:
        output = np.empty([len(dataset.dataset), latent_dimension], dtype=np.float32)
    else:
        output = np.empty([len(dataset.dataset), 1, latent_dimension, latent_dimension],
                          dtype=np.float32)
    start = 0
    for batch, (x, y) in enumerate(dataset):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        predictions = model(x).cpu().detach().numpy()
        output[start:start + len(predictions), ...] = predictions
        start += len(predictions)
    return output


def save_metrics(ae_metrics: dict, nln_metrics: dict, dist_metrics: dict, model_type: str,
                 anomaly_type: str, model_name: str):
    output_filepath = os.path.join("outputs", model_type, anomaly_type, model_name)
    os.makedirs(output_filepath, exist_ok=True)
    with open(os.path.join(output_filepath, "metrics.json"), "w") as f:
        json.dump({"ae": ae_metrics, "nln": nln_metrics, "dist": dist_metrics}, f)


def plot_final_images(metrics: dict, neighbour: int, model_type: str,
                      anomaly_type: str, model_name: str, test_images_reconstructed,
                      test_masks_reconstructed, error_reconstructed, nln_error_reconstructed,
                      distrubtions_reconstructed, combined_reconstructed,
                      latent_reconstructed):
    fig, axs = plt.subplots(10, 7, figsize=(10, 8))
    axs[0, 0].set_title('Inp', fontsize=5)
    axs[0, 1].set_title('Mask', fontsize=5)
    axs[0, 2].set_title(f'Recon {metrics.get("ae_ao_auroc", 0)}', fontsize=5)
    axs[0, 3].set_title(f'NLN {metrics.get("nln_ao_auroc", 0)}', fontsize=5)
    axs[0, 4].set_title(f'Dist {metrics.get("dists_ao_auroc", 0)} {neighbour}', fontsize=5)
    axs[0, 5].set_title(f'Combined {metrics.get("combined_ao_auroc", 0)} {neighbour}', fontsize=5)
    axs[0, 6].set_title(f'Recon {metrics.get("combined_ao_auroc", 0)} {neighbour}', fontsize=5)

    for i in range(10):
        r = np.random.randint(len(test_images_reconstructed))
        axs[i, 0].imshow(test_images_reconstructed[r, ..., 0].astype(np.float32), vmin=0, vmax=1,
                         interpolation='nearest', aspect='auto')
        axs[i, 1].imshow(test_masks_reconstructed[r, ..., 0].astype(np.float32), vmin=0, vmax=1,
                         interpolation='nearest', aspect='auto')
        axs[i, 2].imshow(error_reconstructed[r, ..., 0].astype(np.float32), vmin=0, vmax=1,
                         interpolation='nearest', aspect='auto')
        axs[i, 3].imshow(nln_error_reconstructed[r, ..., 0].astype(np.float32), vmin=0, vmax=1,
                         interpolation='nearest', aspect='auto')
        axs[i, 4].imshow(distrubtions_reconstructed[r, ..., 0].astype(np.float32),
                         interpolation='nearest',
                         aspect='auto')
        axs[i, 5].imshow(combined_reconstructed[r, ..., 0].astype(np.float32), vmin=0, vmax=1,
                         interpolation='nearest', aspect='auto')
        axs[i, 6].imshow(latent_reconstructed[r, ..., 0].astype(np.float32), vmin=0, vmax=1,
                         interpolation='nearest', aspect='auto')

    output_filepath = os.path.join("outputs", model_type, anomaly_type, model_name)
    os.makedirs(output_filepath, exist_ok=True)
    plt.savefig(os.path.join(output_filepath, f"neighbours_{neighbour}.png"), dpi=300)


def get_error_dataset(images: torch.utils.data.DataLoader, x_hat: np.ndarray, image_size: int):
    output = np.empty([len(images.dataset), 1, image_size, image_size], dtype=np.float32)
    start = 0
    for batch, (x, y) in enumerate(images):
        error = x.cpu().detach().numpy() - x_hat[start:start + len(x), ...]
        output[0:len(error), ...] = error
        start += len(error)
    return output


def _calculate_metrics(test_masks_orig_recon: np.ndarray, error_recon: np.ndarray):
    fpr, tpr, thr = roc_curve(test_masks_orig_recon.flatten() > 0, error_recon.flatten())
    true_auroc = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(test_masks_orig_recon.flatten() > 0,
                                                           error_recon.flatten())
    true_auprc = auc(recall, precision)
    f1 = 2 * (precision * recall) / (precision + recall)
    true_f1 = f1.max()
    return {'auroc': true_auroc, 'auprc': true_auprc, 'f1': true_f1}


def nln(z, z_query, neighbours):
    index = faiss.IndexFlatL2(z.shape[1])
    index.add(z)
    neighbours_dist, indices = index.search(z_query, neighbours)
    neighbour_mask = np.zeros([len(indices)], dtype=bool)
    return neighbours_dist, indices, neighbour_mask


def nln_errors(test_dataset: torch.utils.data.DataLoader, x_hat, x_hat_train, neighbours_idx,
               neighbour_mask):
    test_images = test_dataset.dataset[:][1].cpu().detach().numpy()
    test_images_stacked = np.stack([test_images] * neighbours_idx.shape[-1], axis=1)
    neighbours = x_hat_train[neighbours_idx]
    error_nln = test_images_stacked - neighbours
    error_recon = test_images - x_hat
    error = np.mean(error_nln, axis=1)  # nanmean for frNN
    error[neighbour_mask] = error_recon[neighbour_mask]
    return error


def get_dists(neighbours_dist, original_size: int, patch_size: int = None):
    dists = np.mean(neighbours_dist, axis=tuple(range(1, neighbours_dist.ndim)))
    if patch_size is not None:
        dists = np.array([[d] * patch_size ** 2 for i, d in enumerate(dists)]).reshape(len(dists),
                                                                                       patch_size,
                                                                                       patch_size)
        dists_recon = reconstruct_patches(np.expand_dims(dists, axis=1), original_size, patch_size)
        return dists_recon
    else:
        return dists


def calculate_metrics(model: Autoencoder, train_dataset: torch.utils.data.DataLoader,
                      test_masks_original: np.ndarray,
                      test_dataset: torch.utils.data.DataLoader, neighbours: int, batch_size: int,
                      model_name: str, model_type: str, anomaly_type: str,
                      latent_dimension: int, original_size: int, patch_size: int = None,
                      dataset='HERA'
                      ):
    test_images_recon = reconstruct_patches(test_dataset.dataset[:][0].cpu().detach().numpy(),
                                            original_size, patch_size)
    test_masks_reconstructed = reconstruct_patches(
        test_dataset.dataset[:][1].cpu().detach().numpy(), original_size, patch_size)
    z = infer(model.encoder, test_dataset, batch_size, latent_dimension, True)
    z_query = infer(model.encoder, test_dataset, batch_size, latent_dimension, True)

    x_hat_train = infer(model, train_dataset, batch_size, latent_dimension, False)
    x_hat = infer(model, test_dataset, batch_size, latent_dimension, False)
    x_hat_recon = reconstruct_patches(x_hat, original_size, patch_size)
    x_hat_train_recon = reconstruct_patches(x_hat_train, original_size, patch_size)

    error = get_error_dataset(test_dataset, x_hat, patch_size)

    error_recon = reconstruct_patches(error, original_size, patch_size)

    ae_metrics = _calculate_metrics(test_masks_original, error_recon)
    nln_metrics = {}
    dist_metrics = {}
    combined_metrics = {}
    for neighbour in range(1, neighbours+1):
        neighbours_dist, neighbours_idx, neighbour_mask = nln(z, z_query, neighbour)
        nln_error = nln_errors(test_dataset, x_hat, x_hat_train, neighbours_idx, neighbour_mask)

        if patch_size:
            if nln_error.ndim == 4:
                nln_error_recon = reconstruct_patches(nln_error, original_size, patch_size)
            else:
                nln_error_recon = reconstruct_latent_patches(nln_error, original_size, patch_size)
        else:
            nln_error_recon = nln_error

        dists_recon = get_dists(neighbours_dist, original_size, patch_size)

        if dataset == 'HERA':
            combined_recon = nln_error_recon * np.array([d > np.percentile(d, 10) for d in dists_recon])
        elif dataset == 'LOFAR':
            combined_recon = np.clip(nln_error_recon, nln_error_recon.mean() + nln_error_recon.std() * 5,
                                     1.0) * np.array(
                [d > np.percentile(d, 66) for d in dists_recon]
            )
        else:
            raise ValueError('Dataset not implemented')
        combined_recon = np.nan_to_num(combined_recon)
        combined_metrics[neighbour] = _calculate_metrics(test_masks_original, combined_recon)

        nln_metrics[neighbour] = _calculate_metrics(test_masks_original, nln_error_recon)

        dist_metrics[neighbour] = _calculate_metrics(test_masks_original, dists_recon)

        plot_final_images(ae_metrics, neighbour, model_type, anomaly_type, model_name,
                          test_images_recon,
                          test_masks_reconstructed, error_recon, nln_error_recon, dists_recon,
                          combined_recon, x_hat_recon)
    return ae_metrics, nln_metrics, dist_metrics


def evaluate_model(model, train_dataset, test_masks, test_dataset, neighbours, batch_size,
                   latent_dimension, original_size, patch_size, model_name, model_type,
                   anomaly_type, dataset):
    ae_metrics, nln_metrics, dist_metrics = calculate_metrics(model, train_dataset, test_masks,
                                                              test_dataset, neighbours,
                                                              batch_size, model_name, model_type,
                                                              anomaly_type, latent_dimension,
                                                              original_size, patch_size, dataset)
    save_metrics(ae_metrics, nln_metrics, dist_metrics, model_type, anomaly_type, model_name)
