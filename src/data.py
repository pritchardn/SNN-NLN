"""
Contains code for loading and processing training data.
Copyright (c) 2023, Nicholas Pritchard <nicholas.pritchard@icrar.org>
"""
import copy
import os
import pickle

import aoflagger as aof
import numpy as np
import sklearn.model_selection
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from config import get_data_dir


def limit_entries(image_data, masks, limit: int):
    """
    Limits the number of entries in the dataset by random selection.
    """
    if limit is not None:
        indx = np.random.permutation(len(image_data))[:limit]
        image_data = image_data[indx]
        masks = masks[indx]
    return image_data, masks


def clip_data(image_data, masks, mode="HERA"):
    """
    Clips data to within [mean - std, mean + 4*std] and then logs and rescales.
    """
    if mode == "HERA":
        max_threshold = 4
        min_threshold = 1
    elif mode == "LOFAR":
        max_threshold = 95
        min_threshold = 3
    else:  # mode == "TABASCAL"
        max_threshold = 95
        min_threshold = 3
    _max = np.mean(image_data[np.invert(masks)]) + max_threshold * np.std(
        image_data[np.invert(masks)]
    )
    _min = np.absolute(
        np.mean(image_data[np.invert(masks)])
        - min_threshold * np.std(image_data[np.invert(masks)])
    )
    image_data = np.clip(image_data, _min, _max)
    image_data = np.log(image_data)
    # Rescale
    minimum, maximum = np.min(image_data), np.max(image_data)
    image_data = (image_data - minimum) / (maximum - minimum)
    return image_data


def flag_data(image_data, threshold: int = None, mode="HERA"):
    """
    Flags data using AOFlagger. Selects strategy file based on dataset mode.
    Supports 'HERA' and 'LOFAR' modes.
    """
    mask = None
    if threshold is not None:
        mask = np.empty(image_data[..., 0].shape, dtype=bool)

        aoflagger = aof.AOFlagger()
        strategy = None
        if mode == "HERA":
            strategy = aoflagger.load_strategy_file(
                f"{get_data_dir()}{os.sep}flagging{os.sep}hera_{threshold}.lua"
            )
        elif mode == "LOFAR":
            strategy = aoflagger.load_strategy_file(
                f"{get_data_dir()}{os.sep}flagging{os.sep}lofar-default-{threshold}.lua"
            )
        elif mode == "TABASCAL":
            strategy = aoflagger.load_strategy_file(
                f"{get_data_dir()}{os.sep}flagging{os.sep}meerkat-default.lua"
            )
        if not strategy:
            return None
        # LOAD data into AOFlagger structure
        for indx in tqdm(range(len(image_data))):
            _data = aoflagger.make_image_set(
                image_data.shape[1], image_data.shape[2], 1
            )
            _data.set_image_buffer(0, image_data[indx, ..., 0])  # Real values

            flags = strategy.run(_data)
            flag_mask = flags.get_buffer()
            mask[indx, ...] = flag_mask
    return mask


def extract_patches(data: torch.Tensor, kernel_size: int, stride: int):
    """
    Extracts patches from a tensor. Implements the same functionality as found in tensorflow.
    """
    _, channels, _, _ = data.shape
    # Extract patches
    patches = data.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
        -1, channels, kernel_size, kernel_size
    )
    return patches


def reconstruct_patches(images: np.array, original_size: int, kernel_size: int):
    """
    Reconstructs patches into images. Implements the same functionality as found in tensorflow.
    Transposes the images to match the tensorflow implementation but returns the images in the
    original format.
    """
    transposed = images.transpose(0, 3, 2, 1)
    n_patches = original_size // kernel_size
    recon = np.empty(
        [
            images.shape[0] // n_patches**2,
            kernel_size * n_patches,
            kernel_size * n_patches,
            images.shape[1],
        ]
    )

    start, counter, indx, batch = 0, 0, 0, []

    for i in range(n_patches, images.shape[0] + 1, n_patches):
        batch.append(
            np.reshape(
                np.stack(transposed[start:i, ...], axis=0),
                (n_patches * kernel_size, kernel_size, images.shape[1]),
            )
        )
        start = i
        counter += 1
        if counter == n_patches:
            recon[indx, ...] = np.hstack(batch)
            indx += 1
            counter, batch = 0, []

    return recon.transpose(0, 3, 2, 1)


def reconstruct_latent_patches(images: np.ndarray, original_size: int, patch_size: int):
    """
    Reconstructs patches into images. Assumes that the images are square and single channel.
    """
    n_patches = original_size // patch_size
    recon = np.empty([images.shape[0] // n_patches**2, n_patches**2])

    start, end = 0, n_patches**2

    for j, _ in enumerate(range(0, images.shape[0], n_patches**2)):
        recon[j, ...] = images[start:end, ...]
        start = end
        end += n_patches**2
    return recon


def load_hera_data(excluded_rfi=None, data_path=get_data_dir()):
    """
    Loads original data from pickle files.
    If excluded_rfi is None, training and test data will contain all types of RFI.
    If excluded_rfi is not None, training data will NOT include RFI of that type but test data
    will contain all types of RFI.
    """
    if excluded_rfi is None:
        rfi_models = []
        file_path = os.path.join(data_path, "HERA_04-03-2022_all.pkl")
        train_x, train_y, test_x, test_y = np.load(file_path, allow_pickle=True)
    else:
        rfi_models = ["rfi_stations", "rfi_dtv", "rfi_impulse", "rfi_scatter"]
        rfi_models.remove(excluded_rfi)
        test_file_path = os.path.join(data_path, f"HERA_04-03-2022_{excluded_rfi}.pkl")
        _, _, test_x, test_y = np.load(test_file_path, allow_pickle=True)

        train_file_path = os.path.join(
            data_path, f'HERA_04-03-2022_{"-".join(rfi_models)}.pkl'
        )
        train_x, train_y, _, _ = np.load(train_file_path, allow_pickle=True)
    train_x[train_x == np.inf] = np.finfo(train_x.dtype).max
    test_x[test_x == np.inf] = np.finfo(test_x.dtype).max
    test_x = test_x.astype("float32")
    train_x = train_x.astype("float32")
    return train_x, train_y, test_x, test_y, rfi_models


def load_lofar_data(data_path=get_data_dir()):
    filepath = os.path.join(data_path, "LOFAR_Full_RFI_dataset.pkl")
    print(f"Loading LOFAR data from {filepath}")
    with open(filepath, "rb") as f:
        train_x, train_y, test_x, test_y = pickle.load(f)
        return train_x, train_y, test_x, test_y, []


def load_tabascal_data(data_path=get_data_dir()):
    filepath = os.path.join(
        data_path,
        "ultraviolet-condor_obs_64A_512T-0440-1037_004I_512F-1.000e+09-1.000e+10_1000AST_2SAT_3GRD.pkl",
    )
    print(f"Loading Tabascal data from {filepath}")
    with open(filepath, "rb") as f:
        image_data, masks = pickle.load(f)
        image_data = image_data.astype("float32")
        masks = masks.astype("float32")
        train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(
            image_data, masks, test_size=0.2
        )
        return train_x, train_y, test_x, test_y, []


def load_data(config_vals, data_path=get_data_dir()):
    """
    Loads data from pickle files.
    """
    dataset = config_vals["dataset"]
    if dataset == "HERA":
        return load_hera_data(config_vals["excluded_rfi"], data_path=data_path)
    elif dataset == "LOFAR":
        return load_lofar_data(data_path=data_path)
    elif dataset == "TABASCAL":
        return load_tabascal_data(data_path=data_path)
    else:
        raise ValueError(f"Dataset {dataset} not supported.")


def process_into_dataset(
    x_data,
    y_data,
    batch_size,
    mode,
    shuffle=True,
    limit=None,
    threshold=None,
    patch_size=None,
    stride=None,
    filter_rfi_patches=False,
    get_orig=False,
):
    """
    Applies pre-processing steps to the data and returns a torch DataLoader.
    :param get_orig: If true, input y_data will be returned back, else None.
    """
    x_data, y_data = limit_entries(x_data, y_data, limit)
    if get_orig:
        y_data_orig = copy.deepcopy(y_data)
    else:
        y_data_orig = None
    masks = flag_data(x_data, threshold, mode)
    if masks is not None:
        y_data = np.expand_dims(masks, axis=-1)
    x_data = clip_data(x_data, y_data, mode)
    y_data = y_data.astype(bool)
    x_data = np.moveaxis(x_data, -1, 1)
    y_data = np.moveaxis(y_data, -1, 1)
    x_data = torch.from_numpy(x_data)
    y_data = torch.from_numpy(y_data)
    if get_orig:
        y_data_orig = np.moveaxis(y_data_orig, -1, 1)
        y_data_orig = torch.from_numpy(y_data_orig)
    if patch_size is not None and stride is not None:
        x_data = extract_patches(x_data, patch_size, stride)
        y_data = extract_patches(y_data, patch_size, stride)
        if get_orig:
            y_data_orig = extract_patches(y_data_orig, patch_size, stride)
            y_data_orig = y_data_orig.cpu().detach().numpy()
    if filter_rfi_patches:
        # I Hate this back and forwards, but hindsight is 20/20
        x_data = x_data.numpy()
        y_data = y_data.numpy()
        z_data = np.invert(np.any(y_data, axis=(1, 2, 3)))
        x_data = x_data[z_data]
        y_data = y_data[z_data]
        x_data = torch.from_numpy(x_data)
        y_data = torch.from_numpy(y_data)
    dset = TensorDataset(x_data, y_data)
    return (
        torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle),
        y_data_orig,
    )
