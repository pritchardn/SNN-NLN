import os

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_intermediate_images(auto_encoder: nn.Module, dataset: TensorDataset, epoch: int,
                             title: str, outputdir: str):
    for batch, (x, y) in enumerate(dataset):
        x = x.to(device)
        predictions = auto_encoder(x).cpu().detach().numpy()
        plt.figure(figsize=(10, 10))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            if predictions.shape[1] == 1:  # 1 channel only
                plt.imshow(predictions[i, 0, :, :] * 127.5 + 127.5)

            if predictions.shape[1] == 3:  # RGB
                plt.imshow(predictions[i, ...], vmin=0, vmax=1)
            plt.axis('off')

        output_path = os.path.join(outputdir, "results")
        os.makedirs(output_path, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"{output_path}{os.sep}{title}-{epoch}.png")
        plt.clf()
        break
