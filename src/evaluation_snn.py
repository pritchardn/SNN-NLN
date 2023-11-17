from data import reconstruct_patches, reconstruct_latent_patches
from evaluation import infer, _calculate_metrics
from models_snn_direct import SDAutoEncoder


def snln(x_hat, test_dataset):
    images = test_dataset.dataset[:][0].cpu().detach().numpy()
    return images - x_hat


def evaluate_snn_rate(
    model: SDAutoEncoder,
    test_dataset,
    test_masks_original,
    patch_size,
    original_size,
    time_length,
):
    """
    Evaluates rate-based SNN.
    """
    test_masks_original_reconstructed = reconstruct_patches(
        test_masks_original, original_size, patch_size
    )
    model.change_timestep(time_length)
    x_hat = infer(model, test_dataset, patch_size, False)
    snln_error = snln(x_hat, test_dataset)
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
