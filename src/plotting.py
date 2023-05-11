import math
import os

import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset
from config import WANDB_ACTIVE, DEVICE


def plot_intermediate_images(auto_encoder: nn.Module, dataset: TensorDataset, epoch: int,
                             title: str, outputdir: str, batch_size: int):
    for batch, (x, y) in enumerate(dataset):
        x = x.to(DEVICE)
        predictions = auto_encoder(x).cpu().detach().numpy()
        plt.figure(figsize=(10, 10))
        for i in range(batch_size):
            sub_range = int(math.sqrt(batch_size))
            plt.subplot(sub_range, sub_range, i + 1)
            if predictions.shape[1] == 1:  # 1 channel only
                plt.imshow(predictions[i, 0, :, :] * 127.5 + 127.5)

            if predictions.shape[1] == 3:  # RGB
                plt.imshow(predictions[i, ...], vmin=0, vmax=1)
            plt.axis('off')

        output_path = os.path.join(outputdir, "results")
        os.makedirs(output_path, exist_ok=True)
        plot_filename = f"{output_path}{os.sep}{title}-{epoch}.png"
        plt.tight_layout()
        plt.savefig(plot_filename)
        if WANDB_ACTIVE:
            wandb.log({'example-reconstruction': wandb.Image(plot_filename)})
        plt.clf()
        break
