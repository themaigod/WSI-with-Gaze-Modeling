import os

import numpy as np
from torch.utils.data import Dataset
from PIL import Image

np.random.seed(0)

from torchvision import transforms  # noqa

from wsi.data.annotation import Annotation  # noqa


class GridImageDataset(Dataset):
    """
    Data producer that generate a square grid, e.g. 3x3, of patches and their
    corresponding labels from pre-sampled images.
    """

    def __init__(self, data_path, npy_path, img_size, patch_size, mix=1,
                 crop_size=224, normalize=True):
        """
        Initialize the data producer.

        Arguments:
            data_path: string, path to pre-sampled images using patch_gen.py
            json_path: string, path to the annotations in json format
            img_size: int, size of pre-sampled images, e.g. 768
            patch_size: int, size of the patch, e.g. 256
            crop_size: int, size of the final crop that is feed into a CNN,
                e.g. 224 for ResNet
            normalize: bool, if normalize the [0, 255] pixel values to [-1, 1],
                mostly False for debuging purpose
        """
        self._data_path = data_path
        self._npy_path = npy_path
        self._img_size = img_size
        self._mix = mix
        self._patch_size = patch_size
        self._crop_size = crop_size
        self._normalize = normalize
        self._color_jitter = transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04)
        self._preprocess()

    def _preprocess(self):
        if self._img_size % self._patch_size != 0:
            raise Exception('Image size / patch size != 0 : {} / {}'.
                            format(self._img_size, self._patch_size))

        self._patch_per_side = self._img_size // self._patch_size
        self._grid_size = self._patch_per_side * self._patch_per_side

        file_box = []
        for filename in os.listdir(self._data_path):
            if filename[:-4] not in file_box:
                file_box.append(filename[:-4])

        self._pids = file_box

        self._num_image = len(self._pids)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        file_name = self._pids[idx]
        img = Image.open(os.path.join(self._data_path, file_name + ".jpg"))

        # the grid of labels for each patch
        label_grid = np.zeros((self._patch_per_side, self._patch_per_side),
                              dtype=np.float32)
        label_list = np.load(os.path.join(self._data_path, file_name + ".npy"))
        length = 0
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                label = label_list[length, 2]
                # extracted images from WSI is transposed with respect to
                # the original WSI (x, y)
                label_grid[y_idx, x_idx] = label
                length = length + 1

        # color jitter
        img = self._color_jitter(img)

        # use left_right flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label_grid = np.fliplr(label_grid)

        # use rotate
        num_rotate = np.random.randint(0, 4)
        img = img.rotate(90 * num_rotate)
        label_grid = np.rot90(label_grid, num_rotate)

        # PIL image:   H x W x C
        # torch image: C X H X W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0) / 128.0

        # flatten the square grid
        img_flat = np.zeros(
            (self._grid_size, 3, self._crop_size, self._crop_size),
            dtype=np.float32)
        label_flat = np.zeros(self._grid_size, dtype=np.float32)

        idx = 0
        for x_idx in range(self._patch_per_side):
            for y_idx in range(self._patch_per_side):
                # center crop each patch
                x_start = int(
                    (x_idx + 0.5) * self._patch_size - self._crop_size / 2)
                x_end = x_start + self._crop_size
                y_start = int(
                    (y_idx + 0.5) * self._patch_size - self._crop_size / 2)
                y_end = y_start + self._crop_size
                img_flat[idx] = img[:, x_start:x_end, y_start:y_end]
                label_flat[idx] = label_grid[x_idx, y_idx]

                idx += 1

        return (img_flat, label_flat)
