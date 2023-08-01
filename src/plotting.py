"""
Contains plotting functions used throughout training.
Copyright (c) 2023, Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""
import math
import os

import numpy as np
from torch import nn
from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from config import DEVICE


def plot_intermediate_images(
    auto_encoder: nn.Module,
    dataset: TensorDataset,
    epoch: int,
    title: str,
    outputdir: str,
    batch_size: int,
):
    """
    Plots intermediate images from the autoencoder.
    Plots will be arranged in two square grids, the top containing images from the dataset.
    The bottom containing the corresponding predictions from the autoencoder.
    """
    for image_batch, _ in dataset:
        image_batch = image_batch.to(DEVICE)
        predictions = auto_encoder(image_batch).cpu().detach().numpy()
        image_batch = image_batch.cpu().detach().numpy()
        images = np.concatenate((image_batch, predictions), axis=0)
        side_length = int(math.sqrt(batch_size))
        fig = plt.figure(figsize=(side_length, side_length * 2))
        grid = ImageGrid(
            fig,
            111,  # similar to subplot(111)
            nrows_ncols=(side_length * 2, side_length),
            axes_pad=0.1,  # pad between axes in inch.
        )
        for i in range(2 * batch_size):
            # Iterating over the grid returns the Axes.
            grid[i].imshow(images[i, 0, :, :] * 127.5 + 127.5, aspect="auto")
            grid[i].axis("off")
        plt.axis("off")
        output_path = os.path.join(outputdir, "results")
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, f"{title}-{epoch}.png"))
        plt.close("all")
        break
