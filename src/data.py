import os
import aoflagger as aof
import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


def limit_entries(image_data, masks, limit: int):
    if limit is not None:
        indx = np.random.permutation(len(image_data))[:limit]
        image_data = image_data[indx]
        masks = masks[indx]
    return image_data, masks


def clip_data(image_data, masks):
    _max = np.mean(image_data[np.invert(masks)]) + 4 * np.std(image_data[np.invert(masks)])
    _min = np.absolute(np.mean(image_data[np.invert(masks)]) - np.std(image_data[np.invert(masks)]))
    image_data = np.clip(image_data, _min, _max)
    image_data = np.log(image_data)
    image_data = (image_data - _min) / (_max - _min)
    return image_data


def flag_data(image_data, threshold: int = None, mode="HERA"):
    mask = None
    if threshold is not None:
        mask = np.empty(image_data[..., 0].shape, dtype=bool)

        aoflagger = aof.AOFlagger()
        strategy = None
        if mode == 'HERA':
            strategy = aoflagger.load_strategy_file(
                f'data{os.sep}flagging{os.sep}hera_{threshold}.lua')
        elif mode == 'LOFAR':
            strategy = aoflagger.load_strategy_file(
                f'data{os.sep}flagging{os.sep}lofar-default-{threshold}.lua')
        if not strategy:
            return None
        # LOAD data into AOFlagger structure
        for indx in tqdm(range(len(image_data))):
            _data = aoflagger.make_image_set(image_data.shape[1], image_data.shape[2], 1)
            _data.set_image_buffer(0, image_data[indx, ..., 0])  # Real values

            flags = strategy.run(_data)
            flag_mask = flags.get_buffer()
            mask[indx, ...] = flag_mask
    return mask


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


def process_into_dataset(x_data, y_data, batch_size, mode, shuffle=True, limit=None, threshold=None):
    x_data, y_data = limit_entries(x_data, y_data, limit)
    masks = flag_data(x_data, threshold, mode)
    if masks is not None:
        y_data = np.expand_dims(masks, axis=-1)
    x_data = clip_data(x_data, y_data)
    x_data = np.moveaxis(x_data, -1, 1)
    y_data = np.moveaxis(y_data, -1, 1)
    dset = TensorDataset(torch.from_numpy(x_data), torch.from_numpy(y_data))
    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)
