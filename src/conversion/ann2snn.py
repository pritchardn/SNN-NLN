import torch
from tqdm import tqdm
from spikingjelly.activation_based import neuron, ann2snn
from spikingjelly import visualizing
from matplotlib import pyplot as plt
import numpy as np


def plot_output_states(out_images):
    plt.figure(figsize=(25, 25))
    for i in range(len(out_images)):
        sub_range = int(np.sqrt(len(out_images)))
        plt.subplot(sub_range, sub_range, i + 1)
        plt.imshow(out_images[i, 0, :, :] * 127.5 + 127.5)
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
