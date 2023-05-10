import os
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset


def plot_intermediate_images(auto_encoder: nn.Module, dataset: TensorDataset, epoch: int, title: str,
                             outputdir: str):
    for batch, (x, y) in enumerate(dataset):
        predictions = auto_encoder(x).cpu().detach().numpy()
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
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
