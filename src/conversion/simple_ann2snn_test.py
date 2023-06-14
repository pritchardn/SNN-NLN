import torch

from conversion.ann2snn import convert_to_snn, test_snn_model
from data import load_data, process_into_dataset
from models import Autoencoder

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
    test_dataset, test_labels_orig = process_into_dataset(test_x, test_y,
                                                          batch_size=config_vals['batch_size'],
                                                          mode='HERA',
                                                          threshold=config_vals['threshold'],
                                                          patch_size=config_vals['patch_size'],
                                                          stride=config_vals['patch_stride'])

    model = Autoencoder(config_vals['num_layers'], config_vals['latent_dimension'],
                        config_vals['num_filters'], test_dataset.dataset[0][0].shape)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    snn_model = convert_to_snn(model, test_dataset)
    test_snn_model(snn_model, test_dataset)
