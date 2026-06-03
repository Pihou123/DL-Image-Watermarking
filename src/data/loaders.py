"""数据集读取、划分和 DataLoader 构建工具。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CachedTensorDataset(Dataset):
    """读取预处理后的 .pt 张量数据集。"""

    def __init__(self, cache_dir: Path, crop_size: tuple[int, int], normalize: bool = True, is_train: bool = True):
        self.cache_dir = Path(cache_dir)
        self.crop_size = crop_size
        self.normalize = normalize
        self.is_train = is_train
        self.files = sorted(self.cache_dir.glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        tensor = torch.load(self.files[index], weights_only=True)
        # 缓存张量格式为 [C, H, W]，数值范围为 [0, 1]。
        _, h, w = tensor.shape
        crop_h, crop_w = self.crop_size

        if self.is_train:
            top = torch.randint(0, max(1, h - crop_h + 1), (1,)).item()
            left = torch.randint(0, max(1, w - crop_w + 1), (1,)).item()
        else:
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2

        tensor = tensor[:, top:top + crop_h, left:left + crop_w]
        tensor = tensor.to(torch.float32)

        if self.normalize:
            tensor = (tensor - 0.5) / 0.5

        return tensor, 0


class FlatImageDataset(Dataset):
    """读取单层图片目录的数据集。"""
    def __init__(self, image_paths: Sequence[Path], transform=None):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # 保持与 ImageFolder 相同的返回格式。
        return image, 0


def _collect_images(source_dir: Path) -> list[Path]:
    files = [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def _build_imagefolder_datasets(train_dir: Path, val_dir: Path, train_tf, val_tf):
    train_dataset = datasets.ImageFolder(str(train_dir), train_tf)
    val_dataset = datasets.ImageFolder(str(val_dir), val_tf)
    return train_dataset, val_dataset


def _build_flat_split_datasets(source_dir: Path, train_tf, val_tf, train_split: float, split_seed: int):
    images = _collect_images(source_dir)
    if len(images) < 2:
        raise FileNotFoundError(
            f"Not enough images under {source_dir}. At least 2 images are required for train/val split."
        )

    train_split = float(train_split)
    if not 0.0 < train_split < 1.0:
        raise ValueError(f"dataset.train_split must be in (0, 1), got {train_split}")

    rng = random.Random(int(split_seed))
    shuffled = images[:]
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_split)
    split_idx = max(1, min(len(shuffled) - 1, split_idx))

    train_paths = shuffled[:split_idx]
    val_paths = shuffled[split_idx:]

    train_dataset = FlatImageDataset(train_paths, transform=train_tf)
    val_dataset = FlatImageDataset(val_paths, transform=val_tf)
    return train_dataset, val_dataset


def _resolve_interpolation(name: str) -> InterpolationMode:
    key = str(name).strip().lower()
    table = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "lanczos": InterpolationMode.LANCZOS,
    }
    if key not in table:
        available = ", ".join(sorted(table.keys()))
        raise ValueError(f"Unknown interpolation '{name}'. Available: {available}")
    return table[key]


def _build_transforms(dataset_cfg: dict) -> dict[str, transforms.Compose]:
    image_h, image_w = [int(v) for v in dataset_cfg.get("image_size", [64, 64])]
    crop_size = (image_h, image_w)

    preprocess_cfg = dict(dataset_cfg.get("preprocess", {}))

    interpolation = _resolve_interpolation(preprocess_cfg.get("interpolation", "bicubic"))
    antialias = bool(preprocess_cfg.get("antialias", True))

    default_short_side = max(image_h, image_w)
    train_short_side = int(preprocess_cfg.get("train_resize_short_side", default_short_side))
    val_short_side = int(preprocess_cfg.get("val_resize_short_side", train_short_side))

    min_crop_side = min(crop_size)
    train_short_side = max(train_short_side, min_crop_side)
    val_short_side = max(val_short_side, min_crop_side)

    random_hflip_prob = float(preprocess_cfg.get("random_hflip_prob", 0.0))

    train_ops = [
        # 裁剪前保持图像宽高比。
        transforms.Resize(train_short_side, interpolation=interpolation, antialias=antialias),
        transforms.RandomCrop(crop_size),
    ]
    if random_hflip_prob > 0.0:
        train_ops.append(transforms.RandomHorizontalFlip(p=random_hflip_prob))
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    val_ops = [
        transforms.Resize(val_short_side, interpolation=interpolation, antialias=antialias),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]

    return {"train": transforms.Compose(train_ops), "val": transforms.Compose(val_ops)}


def build_dataloaders(dataset_cfg: dict, batch_size: int) -> tuple[DataLoader, DataLoader]:
    num_workers = int(dataset_cfg.get("num_workers", 4))
    pin_memory = bool(dataset_cfg.get("pin_memory", True)) and torch.cuda.is_available()
    image_size = tuple(dataset_cfg.get("image_size", [64, 64]))

    cache_dir = dataset_cfg.get("use_cache", None)
    if cache_dir:
        cache_root = Path(cache_dir)
        train_cache = cache_root / "train"
        val_cache = cache_root / "val"
        print(f"Using cached tensors from: {cache_root}")

        train_dataset = CachedTensorDataset(
            cache_dir=train_cache, crop_size=image_size, normalize=True, is_train=True
        )
        val_dataset = CachedTensorDataset(
            cache_dir=val_cache, crop_size=image_size, normalize=True, is_train=False
        )
    else:
        data_transforms = _build_transforms(dataset_cfg)

        train_dir = Path(dataset_cfg.get("train_dir", "")) if dataset_cfg.get("train_dir") else None
        val_dir = Path(dataset_cfg.get("val_dir", "")) if dataset_cfg.get("val_dir") else None

        use_imagefolder = bool(train_dir and val_dir and train_dir.exists() and val_dir.exists())

        if use_imagefolder:
            try:
                train_dataset, val_dataset = _build_imagefolder_datasets(
                    train_dir, val_dir, data_transforms["train"], data_transforms["val"]
                )
            except Exception as exc:
                raise RuntimeError(
                    "Found train_dir/val_dir, but they are not valid ImageFolder directories. "
                    "Expected structure like train/<class_name>/*.jpg and val/<class_name>/*.jpg."
                ) from exc
        else:
            source_dir = Path(dataset_cfg.get("source_dir", "")) if dataset_cfg.get("source_dir") else None
            if not source_dir or not source_dir.exists():
                raise FileNotFoundError(
                    "Dataset path is invalid. Either provide existing ImageFolder-style train_dir/val_dir, "
                    "or provide dataset.source_dir as a flat image folder (for automatic split)."
                )

            train_dataset, val_dataset = _build_flat_split_datasets(
                source_dir=source_dir,
                train_tf=data_transforms["train"],
                val_tf=data_transforms["val"],
                train_split=float(dataset_cfg.get("train_split", 0.9)),
                split_seed=int(dataset_cfg.get("split_seed", 42)),
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader
