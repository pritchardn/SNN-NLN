import numpy as np
import torch
from matplotlib import pyplot as plt
from spikingjelly.activation_based import ann2snn
from tqdm import tqdm

from data import load_data, process_into_dataset
from models import Autoencoder


def plot_output_states(out_images):
    plt.figure(figsize=(50, 50))
    for i in range(len(out_images[0])):
        sub_range = int(np.sqrt(len(out_images[0])))+1
        plt.subplot(sub_range, sub_range, i + 1)
        slice = out_images[0]
        little_image = slice[i, 0, :, :]
        plt.imshow(little_image * 127.5 + 127.5)
        plt.axis('off')
    plt.show()
    plt.close('all')


def run_through_data(model, dataloader, runtime=50):
    model.eval().to('cuda')
    correct = 0.0
    total = 0.0
    if runtime:
        corrects = np.zeros(runtime)
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
                plot_output_states(out_images)


def convert_to_snn(model, train_data_loader, test_data_loader):
    model_converter = ann2snn.Converter(mode='max', dataloader=train_data_loader)
    snn_model = model_converter(model)
    snn_model.graph.print_tabular()
    run_through_data(model, test_data_loader)


if __name__ == "__main__":
    train_x, train_y, test_x, test_y, rfi_models = load_data()
    config_vals = {
        'batch_size': 32,
        'threshold': 10,
        'patch_size': 32,
        'patch_stride': 32,
        'num_layers': 2,
        'latent_dimension': 32,
        'num_filters': 32
    }
    model_path = 'autoencoder.pt'
    train_dataset = process_into_dataset(train_x, train_y, batch_size=config_vals['batch_size'],
                                         mode='HERA', threshold=config_vals['threshold'],
                                         patch_size=config_vals['patch_size'],
                                         stride=config_vals['patch_stride'],
                                         filter=True)
    test_dataset = process_into_dataset(test_x, test_y, batch_size=config_vals['batch_size'],
                                        mode='HERA', threshold=config_vals['threshold'],
                                        patch_size=config_vals['patch_size'],
                                        stride=config_vals['patch_stride'])

    model = Autoencoder(config_vals['num_layers'], config_vals['latent_dimension'],
                        config_vals['num_filters'], train_dataset.dataset[0][0].shape)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    convert_to_snn(model, train_dataset, test_dataset)
