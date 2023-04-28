import os

import numpy as np
import torch
from torch.utils.data import TensorDataset


def load_data(excluded_rfi=None, data_path='data'):
    if excluded_rfi is None:
        rfi_models = []
        file_path = os.path.join(data_path, 'HERA_04-03-2022_all.pkl')
        train_x, train_y, test_x, test_y = np.load(file_path, allow_pickle=True)
    else:
        rfi_models = ['rfi_stations', 'rfi_dtv', 'rfi_impulse', 'rfi_scatter']
        rfi_models.remove(excluded_rfi)
        test_file_path = os.path.join(data_path, f'HERA_04-03-2022_{excluded_rfi}.pkl')
        _, _, test_x, test_y = np.load(test_file_path, allow_pickle=True)

        train_file_path = os.path.join(data_path, f'HERA_04-03-2022_{"-".join(rfi_models)}.pkl')
        train_x, train_y, _, _ = np.load(train_file_path, allow_pickle=True)
    train_x[train_x == np.inf] = np.finfo(train_x.dtype).max
    test_x[test_x == np.inf] = np.finfo(test_x.dtype).max
    test_x = test_x.astype('float32')
    train_x = train_x.astype('float32')
    return train_x, train_y, test_x, test_y, rfi_models


def process_into_dataset(x_data, y_data, batch_size, shuffle=True):
    dset = TensorDataset(torch.from_numpy(x_data), torch.from_numpy(y_data))
    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)
