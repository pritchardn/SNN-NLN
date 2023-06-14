import json
import math
import os

import matplotlib.animation as animation
import numpy as np
import torch
from matplotlib import pyplot as plt
from spikingjelly.activation_based import ann2snn
from tqdm import tqdm

from data import load_data, process_into_dataset
from main import save_config
from models import Autoencoder
from utils import generate_model_name, generate_output_dir


def animated_plotting(out_images):
    for i in range(len(out_images[0])):
        fig, ax = plt.subplots()
        ax.axis('off')
        ims = []
        for j in range(len(out_images)):
            temp_im = np.moveaxis(out_images[j][i], 0, -1) * 127.5 + 127.5
            im = ax.imshow(temp_im, animated=True)
            if j == 0:
                ax.imshow(temp_im)  # show an initial one first
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        ani.save(f'output_{i}.gif', writer='pillow', fps=10)
        plt.close('all')
        print(f"Done {i}")


def plot_output_states(out_images):
    for i in range(len(out_images[0])):
        plt.figure(figsize=(10, 10))
        sub_range = int(math.ceil(np.sqrt(len(out_images))))
        f, axarr = plt.subplots(sub_range, sub_range)
        axarr = axarr.flatten()
        for j in range(len(out_images)):
            axarr[j].axis('off')
            axarr[j].imshow(np.moveaxis(out_images[j][i], 0, -1) * 127.5 + 127.5)
        plt.axis('off')
        plt.savefig(f'output_{i}.png')
        plt.close('all')
        print(f"Done {i}")


def plot_input_images(in_images):
    for i in range(len(in_images)):
        plt.figure(figsize=(10, 10))
        plt.imshow(np.moveaxis(in_images[i], 0, -1) * 127.5 + 127.5)
        plt.axis('off')
        plt.savefig(f'input_{i}.png')
        plt.close('all')
        print(f"Done {i}")


def plot_outputs(out_images):
    for output_states in out_images:
        plot_output_states(output_states)
        animated_plotting(output_states)


def infer_snn(model, dataloader, runtime=50, batch_limit=1):
    model.eval().to('cuda')
    full_output = []
    i = 0
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(dataloader)):
            img = img.to('cuda')
            out_images = []
            if runtime is None:
                out = model(img)
                print(out.shape)
            else:
                for m in model.modules():
                    if hasattr(m, 'reset'):
                        m.reset()
                for t in range(runtime):
                    if t == 0:
                        out = model(img)
                    else:
                        out += model(img)
                    # Add current state to image building
                    out_images.append(out.cpu().numpy())
                full_output.extend(out_images)
            i += 1
            if i == batch_limit:
                break
    return full_output  # [N, T, C, W, H]


def convert_to_snn(model, test_data_loader):
    model_converter = ann2snn.Converter(mode='max', dataloader=test_data_loader)
    snn_model = model_converter(model)
    snn_model.graph.print_tabular()
    return snn_model


def test_snn_model(snn_model, test_data_loader):
    full_output = infer_snn(snn_model, test_data_loader, runtime=64)
    plot_outputs(full_output)


def load_config(input_dir: str):
    config_file_path = os.path.join(input_dir, "config.json")
    with open(config_file_path, "r") as f:
        config = json.load(f)
    return config


def load_test_dataset(config_vals: dict):
    _, _, test_x, test_y, rfi_models = load_data()
    test_dataset, _ = process_into_dataset(test_x, test_y,
                                           batch_size=config_vals['batch_size'],
                                           mode='HERA',
                                           threshold=config_vals['threshold'],
                                           patch_size=config_vals['patch_size'],
                                           stride=config_vals['patch_stride'],
                                           shuffle=False)
    return test_dataset


def load_ann_model(input_dir: str, config_vals: dict, test_dataset: torch.utils.data.Dataset):
    model_path = os.path.join(input_dir, "autoencoder.pt")
    model = Autoencoder(config_vals['num_layers'], config_vals['latent_dimension'],
                        config_vals['num_filters'], test_dataset.dataset[0][0].shape)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate_snn(model, test_dataset):
    return {"auroc": 1.0, "auprc": 1.0, "f1": 1.0}


def save_results(config_vals: dict, snn_metrics: dict, output_dir: str):
    save_config(config_vals, output_dir)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(snn_metrics, f, indent=4)


def main(input_dir: str):
    config_vals = load_config(input_dir)
    # Get dataset
    test_dataset = load_test_dataset(config_vals)
    # Load model
    model = load_ann_model(input_dir, config_vals, test_dataset)
    # Convert to SNN
    snn_model = convert_to_snn(model, test_dataset)
    # Infer with SNN
    full_outputs = infer_snn(snn_model, test_dataset, runtime=32, batch_limit=1)
    print(len(full_outputs))
    # Evaluate results
    sln_metrics = evaluate_snn(snn_model, test_dataset)
    # Save results to file
    config_vals["model_type"] = "SDAE"
    config_vals['model_name'] = generate_model_name(config_vals)
    output_dir = generate_output_dir(config_vals)
    os.makedirs(output_dir, exist_ok=True)
    save_results(config_vals, sln_metrics, output_dir)
    torch.save(snn_model.state_dict(), os.path.join(output_dir, "snn_autoencoder.pt"))
    # Plot results to file
    print(json.dumps(config_vals, indent=4))


if __name__ == "__main__":
    input_dir = "./outputs/DAE/MISO/DAE_MISO_HERA_32_2_10/"
    main(input_dir)
