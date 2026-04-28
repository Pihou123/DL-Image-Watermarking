"""Watermark embedding and extraction for arbitrary-size images.

Uses patch-based tiling: slices the image into 64x64 patches, embeds/extracts
the same message on each patch, then stitches back / votes.

Usage (embed):
    python scripts/inference.py embed \
        --image input.jpg \
        --output watermarked.png \
        --message "hello" \
        --ckpt outputs/runs/<run>/checkpoints/best.pth \
        --config configs/base.yaml

Usage (extract):
    python scripts/inference.py extract \
        --image watermarked.png \
        --message_len 30 \
        --ckpt outputs/runs/<run>/checkpoints/best.pth \
        --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.common.config import load_config
from src.common.runtime import resolve_device
from src.common.seed import set_seed
from src.engine.checkpoint import load_checkpoint
from src.models.hidden_system import HiddenSystem
from src.noise.manager import NoiseManager

PATCH_SIZE = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watermark embed/extract for arbitrary images.")
    sub = parser.add_subparsers(dest="mode", required=True)

    embed_parser = sub.add_parser("embed", help="Embed watermark into image.")
    embed_parser.add_argument("--image", required=True, help="Path to input image.")
    embed_parser.add_argument("--output", required=True, help="Path to output watermarked image.")
    embed_parser.add_argument("--message", required=True, help="Message string to embed (max 30 chars).")
    embed_parser.add_argument("--ckpt", required=True, help="Path to model checkpoint.")
    embed_parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    embed_parser.add_argument("--device", default="auto", help="Device override.")

    extract_parser = sub.add_parser("extract", help="Extract watermark from image.")
    extract_parser.add_argument("--image", required=True, help="Path to watermarked image.")
    extract_parser.add_argument("--message_len", type=int, default=30, help="Expected message bit length.")
    extract_parser.add_argument("--ckpt", required=True, help="Path to model checkpoint.")
    extract_parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config.")
    extract_parser.add_argument("--device", default="auto", help="Device override.")

    return parser.parse_args()


def _msg_to_bits(message: str, length: int = 30) -> torch.Tensor:
    bits = []
    for ch in message[:length]:
        val = ord(ch) & 0xFF
        for b in range(8):
            bits.append((val >> (7 - b)) & 1)
    bits = bits[:length]
    while len(bits) < length:
        bits.append(0)
    return torch.tensor(bits, dtype=torch.float32).unsqueeze(0)


def _bits_to_msg(bits: np.ndarray) -> str:
    chars = []
    for i in range(0, len(bits) - 7, 8):
        val = 0
        for b in range(8):
            val |= (int(bits[i + b]) << (7 - b))
        if 32 <= val < 127:
            chars.append(chr(val))
        else:
            chars.append(".")
    return "".join(chars)


def _load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def _pad_to_multiple(img: Image.Image, multiple: int) -> Image.Image:
    """Pad image to next multiple using mirror reflection to avoid black borders."""
    w, h = img.size
    new_w = ((w + multiple - 1) // multiple) * multiple
    new_h = ((h + multiple - 1) // multiple) * multiple
    if new_w != w or new_h != h:
        padded = Image.new("RGB", (new_w, new_h))
        padded.paste(img, (0, 0))
        # Mirror the edge pixels to fill the padded strip area instead of leaving black
        if new_w > w:
            edge = img.crop((w - 1, 0, w, h))
            stretched = edge.resize((new_w - w, h), Image.LANCZOS)
            padded.paste(stretched, (w, 0))
        if new_h > h:
            # Include the right-padded area so the bottom strip is continuous
            edge = padded.crop((0, h - 1, new_w, h))
            stretched = edge.resize((new_w, new_h - h), Image.LANCZOS)
            padded.paste(stretched, (0, h))
        return padded
    return img


def _image_to_patches(img: Image.Image, patch_size: int) -> tuple[list[Image.Image], int, int, int, int]:
    """Slice image into patches. Returns (patches, orig_w, orig_h, cols, rows)."""
    img = _pad_to_multiple(img, patch_size)
    orig_w, orig_h = img.size
    cols = orig_w // patch_size
    rows = orig_h // patch_size
    patches = []
    for row in range(rows):
        for col in range(cols):
            box = (col * patch_size, row * patch_size, (col + 1) * patch_size, (row + 1) * patch_size)
            patches.append(img.crop(box))
    return patches, orig_w, orig_h, cols, rows


def _patches_to_image(patches: list[Image.Image], orig_w: int, orig_h: int, cols: int, rows: int) -> Image.Image:
    out = Image.new("RGB", (orig_w, orig_h))
    idx = 0
    for row in range(rows):
        for col in range(cols):
            out.paste(patches[idx], (col * PATCH_SIZE, row * PATCH_SIZE))
            idx += 1
    return out


def _build_model(config_path: str, ckpt_path: str, device: torch.device):
    cfg = load_config(config_path)

    image_size = tuple(cfg["dataset"].get("image_size", [64, 64]))
    noise_manager = NoiseManager(cfg["noise"], device=device).to(device)
    model = HiddenSystem(
        model_cfg=cfg["model"],
        train_cfg=cfg["train"],
        image_size=(int(image_size[0]), int(image_size[1])),
        noise_manager=noise_manager,
        device=device,
    ).to(device)

    load_checkpoint(ckpt_path, model=model, device=device, scaler=None, strict=False)
    model.eval()
    return model, int(cfg["model"].get("payload_length", cfg["model"]["message_length"]))


@torch.no_grad()
def embed(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(42)

    model, msg_len = _build_model(args.config, args.ckpt, device)
    message_bits = _msg_to_bits(args.message, msg_len).to(device)

    img = _load_image(args.image).convert("RGB")
    orig_w, orig_h = img.size
    patches, pad_w, pad_h, cols, rows = _image_to_patches(img, PATCH_SIZE)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    inv_tf = transforms.Compose([
        transforms.Normalize([-1.0, -1.0, -1.0], [2.0, 2.0, 2.0]),
        transforms.ToPILImage(),
    ])

    encoded_patches = []
    for patch in patches:
        tensor = tf(patch).unsqueeze(0).to(device)
        expanded = model._expand_message(message_bits)
        encoded = model.encoder_decoder.encoder(tensor, expanded)
        encoded_patch = inv_tf(encoded[0].cpu().clamp(-1, 1))
        encoded_patches.append(encoded_patch)

    full = _patches_to_image(encoded_patches, pad_w, pad_h, cols, rows)
    full = full.crop((0, 0, orig_w, orig_h))
    full.save(args.output)
    print(f"Watermarked image saved to: {args.output}")


@torch.no_grad()
def extract(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    set_seed(42)

    model, _ = _build_model(args.config, args.ckpt, device)

    img = _load_image(args.image).convert("RGB")
    patches, _, _, _, _ = _image_to_patches(img, PATCH_SIZE)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    all_bits = []
    for patch in patches:
        tensor = tf(patch).unsqueeze(0).to(device)
        decoded = model.encoder_decoder.decoder(tensor)
        compressed, _ = model._compress_message(decoded)
        bits = (compressed.cpu().numpy().round().clip(0, 1)).astype(int).flatten()
        all_bits.append(bits)

    all_bits = np.array(all_bits)
    majority = (all_bits.mean(axis=0) >= 0.5).astype(int)

    print(f"Total patches: {all_bits.shape[0]}")
    print(f"Extracted bits (majority vote): {''.join(map(str, majority))}")
    print(f"Decoded message: {_bits_to_msg(majority)}")

    agreement = (all_bits == majority).mean(axis=0)
    print(f"Per-bit agreement: min={agreement.min():.2%}  avg={agreement.mean():.2%}")


def main() -> None:
    args = parse_args()
    if args.mode == "embed":
        embed(args)
    elif args.mode == "extract":
        extract(args)


if __name__ == "__main__":
    main()
