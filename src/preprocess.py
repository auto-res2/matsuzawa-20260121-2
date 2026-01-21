import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from omegaconf import DictConfig

#######################################################################
# --------------------  Custom augmentation  ------------------------ #
#######################################################################

class RandomCropWithParam:
    """RandomCrop that returns crop severity (fraction of area removed)."""
    def __init__(self, size: int = 32, padding: int = 4):
        self.size, self.padding = size, padding

    def __call__(self, img):
        img_padded = F.pad(img, [self.padding] * 4, fill=0)
        i, j, h, w = transforms.RandomCrop.get_params(img_padded, (self.size, self.size))
        cropped = F.crop(img_padded, i, j, h, w)
        severity = 1.0 - (h * w) / (img_padded.size[0] * img_padded.size[1])
        return cropped, severity


class RandomErasingWithParam:
    """RandomErasing variant that returns severity (fraction of area erased)."""
    def __init__(self, p: float = 0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.p, self.scale, self.ratio = p, scale, ratio

    def __call__(self, tensor: torch.Tensor):  # expects Tensor C×H×W
        if random.random() > self.p:
            return tensor, 0.0
        i, j, h, w, v = transforms.RandomErasing.get_params(tensor, scale=self.scale, ratio=self.ratio, value=[0.0])
        erased = F.erase(tensor, i, j, h, w, v=v, inplace=False)
        severity = (h * w) / (tensor.size(1) * tensor.size(2))
        return erased, severity

#######################################################################
# -------------------------  Dataset  ------------------------------- #
#######################################################################

class CIFARWithParams(torch.utils.data.Dataset):
    def __init__(self, base_dataset, augment: bool = True):
        self.base = base_dataset
        self.augment = augment
        self._trial = False  # flag toggled by build_loaders

        # Define transforms
        self.crop = RandomCropWithParam(32, 4) if augment else None
        self.hflip = transforms.RandomHorizontalFlip() if augment else None
        self.erase = RandomErasingWithParam() if augment else None
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        aug_params = [0.0, 0.0]  # [crop, erase]

        # ----------------  Augmentations on PIL Image  ----------------
        if self.augment:
            img, crop_sev = self.crop(img)
            aug_params[0] = crop_sev
            img = self.hflip(img)
        # Convert to Tensor
        img = transforms.ToTensor()(img)

        # ----------------  Erasing on Tensor  -------------------------
        if self.augment:
            img, erase_sev = self.erase(img)
            aug_params[1] = erase_sev
        img = self.normalize(img)

        return img, torch.tensor(aug_params, dtype=torch.float32), label

#######################################################################
# ---------------------  Public loader builder  ---------------------- #
#######################################################################

def build_loaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    root = Path(".cache") / cfg.dataset.name
    root.mkdir(parents=True, exist_ok=True)

    train_full = datasets.CIFAR10(root=str(root), train=True, download=True)
    test_base = datasets.CIFAR10(root=str(root), train=False, download=True)

    # Custom train/val split according to cfg
    indices = list(range(len(train_full)))
    random.Random(cfg.training.seed).shuffle(indices)
    train_count = cfg.dataset.split.train
    val_count = cfg.dataset.split.val

    train_idx = indices[:train_count]
    val_idx = indices[train_count:train_count + val_count]

    train_ds = CIFARWithParams(Subset(train_full, train_idx), augment=True)
    val_ds = CIFARWithParams(Subset(train_full, val_idx), augment=False)
    test_ds = CIFARWithParams(test_base, augment=False)

    # Trial-mode shortening
    if cfg.mode == "trial":
        for ds in (train_ds, val_ds, test_ds):
            ds._trial = True
        short_len = cfg.training.batch_size * 2
        train_ds, val_ds, test_ds = (Subset(train_ds, list(range(short_len))),
                                     Subset(val_ds, list(range(short_len))),
                                     Subset(test_ds, list(range(short_len))))

    def _build_loader(ds, shuffle):
        return DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True)

    return _build_loader(train_ds, True), _build_loader(val_ds, False), _build_loader(test_ds, False)
