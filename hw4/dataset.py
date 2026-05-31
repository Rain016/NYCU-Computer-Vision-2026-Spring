import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class TrainDataset(Dataset):
    def __init__(self, data_dir, file_list=None, patch_size=128):
        self.patch_size = patch_size
        self.degraded_dir = os.path.join(data_dir, 'train', 'degraded')
        self.clean_dir = os.path.join(data_dir, 'train', 'clean')

        if file_list is not None:
            self.degraded_files = sorted(file_list)
        else:
            self.degraded_files = sorted(
                glob.glob(os.path.join(self.degraded_dir, '*.png'))
            )

    def __len__(self):
        return len(self.degraded_files)

    def _get_clean_path(self, degraded_path):
        name = os.path.basename(degraded_path)
        if name.startswith('rain-'):
            clean_name = name.replace('rain-', 'rain_clean-')
        elif name.startswith('snow-'):
            clean_name = name.replace('snow-', 'snow_clean-')
        else:
            raise ValueError(f'Unknown degradation prefix: {name}')
        return os.path.join(self.clean_dir, clean_name)

    def _random_crop(self, deg, cln):
        w, h = deg.size
        ps = self.patch_size
        # pad if smaller than patch
        if w < ps or h < ps:
            pad_w = max(0, ps - w)
            pad_h = max(0, ps - h)
            deg = TF.pad(deg, (0, 0, pad_w, pad_h), padding_mode='reflect')
            cln = TF.pad(cln, (0, 0, pad_w, pad_h), padding_mode='reflect')
            w, h = deg.size
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        deg = TF.crop(deg, y, x, ps, ps)
        cln = TF.crop(cln, y, x, ps, ps)
        return deg, cln

    def _augment(self, deg, cln):
        if random.random() > 0.5:
            deg, cln = TF.hflip(deg), TF.hflip(cln)
        if random.random() > 0.5:
            deg, cln = TF.vflip(deg), TF.vflip(cln)
        k = random.randint(0, 3)
        if k:
            deg = TF.rotate(deg, 90 * k)
            cln = TF.rotate(cln, 90 * k)
        return deg, cln

    def __getitem__(self, idx):
        deg_path = self.degraded_files[idx]
        cln_path = self._get_clean_path(deg_path)

        deg = Image.open(deg_path).convert('RGB')
        cln = Image.open(cln_path).convert('RGB')

        deg, cln = self._random_crop(deg, cln)
        deg, cln = self._augment(deg, cln)

        return TF.to_tensor(deg), TF.to_tensor(cln)


class TestDataset(Dataset):
    def __init__(self, data_dir):
        self.degraded_dir = os.path.join(data_dir, 'test', 'degraded')
        files = glob.glob(os.path.join(self.degraded_dir, '*.png'))
        self.degraded_files = sorted(
            files, key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )

    def __len__(self):
        return len(self.degraded_files)

    def __getitem__(self, idx):
        path = self.degraded_files[idx]
        filename = os.path.basename(path)
        img = Image.open(path).convert('RGB')
        return TF.to_tensor(img), filename


def make_train_val_split(data_dir, val_ratio=0.1, seed=42):
    """Returns (train_files, val_files) balanced across rain and snow."""
    degraded_dir = os.path.join(data_dir, 'train', 'degraded')
    all_files = sorted(glob.glob(os.path.join(degraded_dir, '*.png')))

    rain = [f for f in all_files if os.path.basename(f).startswith('rain-')]
    snow = [f for f in all_files if os.path.basename(f).startswith('snow-')]

    rng = random.Random(seed)
    rng.shuffle(rain)
    rng.shuffle(snow)

    n_rain_val = max(1, int(len(rain) * val_ratio))
    n_snow_val = max(1, int(len(snow) * val_ratio))

    val_files = rain[:n_rain_val] + snow[:n_snow_val]
    train_files = rain[n_rain_val:] + snow[n_snow_val:]
    return train_files, val_files
