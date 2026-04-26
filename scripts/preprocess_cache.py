"""Preprocess images into cached tensors to eliminate PIL decode + resize overhead.

Usage:
    python scripts/preprocess_cache.py --config configs/base.yaml
    python scripts/preprocess_cache.py --config configs/base.yaml --num_workers 8

This script reads the dataset config, applies resize+ToTensor to every image, and
saves the resulting tensors to a cache directory.  The cached tensors retain the
post-resize dimensions so that RandomCrop/CenterCrop can still be randomized
at training time.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.common.config import load_config
from src.data.loaders import _collect_images, _resolve_interpolation, IMAGE_EXTENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess and cache dataset tensors.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers for preprocessing.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Override cache output directory (default: data/cache/<dataset_name>).",
    )
    return parser.parse_args()


def _resize_transform(dataset_cfg: dict, is_train: bool) -> transforms.Compose:
    image_h, image_w = [int(v) for v in dataset_cfg.get("image_size", [64, 64])]
    preprocess_cfg = dict(dataset_cfg.get("preprocess", {}))
    interpolation = _resolve_interpolation(preprocess_cfg.get("interpolation", "bicubic"))
    antialias = bool(preprocess_cfg.get("antialias", True))

    default_short_side = max(image_h, image_w)
    train_short = int(preprocess_cfg.get("train_resize_short_side", default_short_side))
    val_short = int(preprocess_cfg.get("val_resize_short_side", train_short))
    min_side = min(image_h, image_w)
    short_side = max(train_short if is_train else val_short, min_side)

    return transforms.Compose(
        [
            transforms.Resize(short_side, interpolation=interpolation, antialias=antialias),
            transforms.ToTensor(),
        ]
    )


def _process_one(args_tuple: tuple[str, str, dict]) -> tuple[str, str]:
    """Process a single image: load, resize, save as .pt. Returns (name, status)."""
    img_path_str, out_dir_str, dataset_cfg = args_tuple
    img_path = Path(img_path_str)
    out_dir = Path(out_dir_str)
    out_path = out_dir / f"{img_path.stem}.pt"

    if out_path.exists():
        return (img_path.name, "skip")

    try:
        image = Image.open(img_path).convert("RGB")
        is_train = (out_dir.name == "train")
        tf = _resize_transform(dataset_cfg, is_train=is_train)
        tensor = tf(image)
        torch.save(tensor, out_path)
        return (img_path.name, "ok")
    except Exception as exc:
        return (img_path.name, f"error: {exc}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]

    # Determine image paths (same logic as loaders.py).
    train_dir = Path(dataset_cfg.get("train_dir", "")) if dataset_cfg.get("train_dir") else None
    val_dir = Path(dataset_cfg.get("val_dir", "")) if dataset_cfg.get("val_dir") else None

    use_imagefolder = bool(train_dir and val_dir and train_dir.exists() and val_dir.exists())

    if use_imagefolder:
        print(f"Mode: ImageFolder  train={train_dir}  val={val_dir}")
        train_paths = []
        val_paths = []
        for class_dir in sorted(train_dir.iterdir()):
            if class_dir.is_dir():
                for ext in IMAGE_EXTENSIONS:
                    train_paths.extend(sorted(class_dir.glob(f"*{ext}")))
        for class_dir in sorted(val_dir.iterdir()):
            if class_dir.is_dir():
                for ext in IMAGE_EXTENSIONS:
                    val_paths.extend(sorted(class_dir.glob(f"*{ext}")))
    else:
        source_dir = Path(dataset_cfg.get("source_dir", ""))
        if not source_dir or not source_dir.exists():
            raise FileNotFoundError(f"source_dir not found: {source_dir}")
        print(f"Mode: flat split  source={source_dir}")
        train_split = float(dataset_cfg.get("train_split", 0.9))
        split_seed = int(dataset_cfg.get("split_seed", 42))
        images = _collect_images(source_dir)
        rng = random.Random(split_seed)
        shuffled = images[:]
        rng.shuffle(shuffled)
        idx = max(1, min(len(shuffled) - 1, int(len(shuffled) * train_split)))
        train_paths = shuffled[:idx]
        val_paths = shuffled[idx:]

    print(f"Train images: {len(train_paths)}")
    print(f"Val images:   {len(val_paths)}")

    # Prepare cache directory.
    dataset_name = dataset_cfg.get("source_dir", dataset_cfg.get("train_dir", "dataset"))
    dataset_name = Path(dataset_name).name
    cache_root = Path(args.cache_dir) if args.cache_dir else Path("data") / f"cache_{dataset_name}"
    train_cache = cache_root / "train"
    val_cache = cache_root / "val"
    train_cache.mkdir(parents=True, exist_ok=True)
    val_cache.mkdir(parents=True, exist_ok=True)
    print(f"Cache dir: {cache_root}")

    image_size = tuple(dataset_cfg.get("image_size", [64, 64]))
    print(f"Target image size: {image_size}")
    print()

    # Build task list.
    tasks: list[tuple[str, str, dict]] = []
    for p in train_paths:
        tasks.append((str(p), str(train_cache), dataset_cfg))
    for p in val_paths:
        tasks.append((str(p), str(val_cache), dataset_cfg))

    print(f"Processing {len(tasks)} images with {args.num_workers} workers ...")
    skipped = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = list(tqdm(
            executor.map(_process_one, tasks),
            total=len(tasks),
            desc="Processing",
        ))

    for _, status in futures:
        if status == "skip":
            skipped += 1
        elif status != "ok":
            errors += 1
            if errors <= 5:
                print(f"  ERROR: {status}")

    ok = len(tasks) - skipped - errors
    print(f"\nComplete: {ok} ok, {skipped} skipped (already cached), {errors} errors")

    # Save metadata.
    meta = {
        "image_size": list(image_size),
        "train_count": len(list(train_cache.glob("*.pt"))),
        "val_count": len(list(val_cache.glob("*.pt"))),
    }
    torch.save(meta, cache_root / "meta.pt")
    print(f"Cache saved to: {cache_root}")


if __name__ == "__main__":
    main()
