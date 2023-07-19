import math
import os
import numpy as np

import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset
from config import WANDB_ACTIVE, DEVICE


def remove_stripes(image: np.ndarray):
    temp = np.array(image, copy=True)
    # Replace rightmost column with left neighbour
    temp[-1, :] = temp[-2, :]
    # Replace bottom row with upper neighbour
    temp[:, -1] = temp[:, -2]
    return temp


def plot_intermediate_images(auto_encoder: nn.Module, dataset: TensorDataset, epoch: int,
                             title: str, outputdir: str, batch_size: int):
    for batch, (x, y) in enumerate(dataset):
        x = x.to(DEVICE)
        predictions = auto_encoder(x).cpu().detach().numpy()
        plt.figure(figsize=(10, 10))
        for i in range(min(batch_size, 24)):
            sub_range = int(math.ceil(math.sqrt(batch_size)))
            plt.subplot(sub_range, sub_range, i + 1)
            if predictions.shape[1] == 1:  # 1 channel only
                plt.imshow(remove_stripes(predictions[i, 0, :, :]) * 127.5 + 127.5)

            if predictions.shape[1] == 3:  # RGB
                plt.imshow(predictions[i, ...], vmin=0, vmax=1)
            plt.axis('off')
        x = x.cpu().detach().numpy()
        for i in range(min(batch_size, 24)): # , 48
            # TODO: Get the range back up to 24 so both panes are visible
            sub_range = int(math.ceil(math.sqrt(batch_size)))
            plt.subplot(sub_range, sub_range, i + 1)
            if x.shape[1] == 1:
                plt.imshow(x[i - min(batch_size, 24), 0, :, :] * 127.5 + 127.5)
            plt.axis('off')

        output_path = os.path.join(outputdir, "results")
        os.makedirs(output_path, exist_ok=True)
        plot_filename = f"{output_path}{os.sep}{title}-{epoch}.png"
        plt.tight_layout()
        plt.savefig(plot_filename)
        if WANDB_ACTIVE:
            wandb.log({'example-reconstruction': wandb.Image(plot_filename)})
        plt.close('all')
        break
