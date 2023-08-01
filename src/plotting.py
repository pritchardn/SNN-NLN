import math
import os

import numpy as np
import torch.nn as nn
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import TensorDataset

from config import DEVICE


def plot_intermediate_images(
    auto_encoder: nn.Module,
    dataset: TensorDataset,
    epoch: int,
    title: str,
    outputdir: str,
    batch_size: int,
):
    for batch, (x, y) in enumerate(dataset):
        x = x.to(DEVICE)
        predictions = auto_encoder(x).cpu().detach().numpy()
        x = x.cpu().detach().numpy()
        images = np.concatenate((x, predictions), axis=0)
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
        plot_filename = f"{output_path}{os.sep}{title}-{epoch}.png"
        plt.savefig(plot_filename)
        plt.close("all")
        break
