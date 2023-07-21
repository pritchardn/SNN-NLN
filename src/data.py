import copy
import os

import aoflagger as aof
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from tqdm import tqdm


def limit_entries(image_data, masks, limit: int):
    if limit is not None:
        indx = np.random.permutation(len(image_data))[:limit]
        image_data = image_data[indx]
        masks = masks[indx]
    return image_data, masks


def clip_data(image_data, masks):
    _max = np.mean(image_data[np.invert(masks)]) + 4 * np.std(
        image_data[np.invert(masks)]
    )
    _min = np.absolute(
        np.mean(image_data[np.invert(masks)]) - np.std(image_data[np.invert(masks)])
    )
    image_data = np.clip(image_data, _min, _max)
    image_data = np.log(image_data)
    mi, ma = np.min(image_data), np.max(image_data)
    image_data = (image_data - mi) / (ma - mi)
    return image_data


def flag_data(image_data, threshold: int = None, mode="HERA"):
    mask = None
    if threshold is not None:
        mask = np.empty(image_data[..., 0].shape, dtype=bool)

        aoflagger = aof.AOFlagger()
        strategy = None
        if mode == "HERA":
            strategy = aoflagger.load_strategy_file(
                f"data{os.sep}flagging{os.sep}hera_{threshold}.lua"
            )
        elif mode == "LOFAR":
            strategy = aoflagger.load_strategy_file(
                f"data{os.sep}flagging{os.sep}lofar-default-{threshold}.lua"
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


def extract_patches(x: torch.Tensor, kernel_size: int, stride: int, batch_size):
    b, c, h, w = x.shape
    scaling_factor = (h // kernel_size) ** 2
    input_start, input_end = 0, batch_size
    output_start, output_end = 0, batch_size * scaling_factor
    output = torch.Tensor(b * scaling_factor, c, kernel_size, kernel_size)
    # Extract patches
    patches = x.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, c, kernel_size, kernel_size)

    return patches
    # return output


def reconstruct_patches(images: np.array, original_size: int, kernel_size: int):
    t = images.transpose(0, 3, 2, 1)
    n_patches = original_size // kernel_size
    recon = np.empty(
        [
            images.shape[0] // n_patches ** 2,
            kernel_size * n_patches,
            kernel_size * n_patches,
            images.shape[1],
        ]
    )

    start, counter, indx, b = 0, 0, 0, []

    for i in range(n_patches, images.shape[0] + 1, n_patches):
        b.append(
            np.reshape(
                np.stack(t[start:i, ...], axis=0),
                (n_patches * kernel_size, kernel_size, images.shape[1]),
            )
        )
        start = i
        counter += 1
        if counter == n_patches:
            recon[indx, ...] = np.hstack(b)
            indx += 1
            counter, b = 0, []

    return recon.transpose(0, 3, 2, 1)


def reconstruct_latent_patches(images: np.ndarray, original_size: int, patch_size: int):
    n_patches = original_size // patch_size
    recon = np.empty([images.shape[0] // n_patches ** 2, n_patches ** 2])

    start, end, labels_recon = 0, n_patches ** 2, []

    for j, i in enumerate(range(0, images.shape[0], n_patches ** 2)):
        recon[j, ...] = images[start:end, ...]
        start = end
        end += n_patches ** 2
    return recon


def load_data(excluded_rfi=None, data_path="data"):
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
    filter=False,
    get_orig=False,
):
    x_data, y_data = limit_entries(x_data, y_data, limit)
    if get_orig:
        y_data_orig = copy.deepcopy(y_data)
    else:
        y_data_orig = None
    masks = flag_data(x_data, threshold, mode)
    if masks is not None:
        y_data = np.expand_dims(masks, axis=-1)
    x_data = clip_data(x_data, y_data)
    y_data = y_data.astype(bool)
    x_data = np.moveaxis(x_data, -1, 1)
    y_data = np.moveaxis(y_data, -1, 1)
    x_data = torch.from_numpy(x_data)
    y_data = torch.from_numpy(y_data)
    if get_orig:
        y_data_orig = np.moveaxis(y_data_orig, -1, 1)
        y_data_orig = torch.from_numpy(y_data_orig)
    if patch_size is not None and stride is not None:
        x_data = extract_patches(x_data, patch_size, stride, batch_size)
        y_data = extract_patches(y_data, patch_size, stride, batch_size)
        if get_orig:
            y_data_orig = extract_patches(y_data_orig, patch_size, stride, batch_size)
            y_data_orig = y_data_orig.cpu().detach().numpy()
    if filter:
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
