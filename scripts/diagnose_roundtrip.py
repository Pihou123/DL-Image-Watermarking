from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_config
from src.common.runtime import resolve_device
from src.engine.checkpoint import load_checkpoint
from src.engine.metrics import compute_psnr, compute_ssim
from src.models.hidden_system import HiddenSystem
from src.noise.manager import NoiseManager


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare tensor identity extraction with uint8 image round-trip extraction.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint.")
    parser.add_argument("--image_dir", default="data/mirflickr25k", help="Directory containing input images.")
    parser.add_argument("--num_images", type=int, default=32, help="Number of images to evaluate.")
    parser.add_argument("--bits", default="110", help="Binary payload. It is padded/truncated to payload_length.")
    parser.add_argument("--device", default="auto", help="Device, e.g. auto/cuda/cpu.")
    parser.add_argument(
        "--preprocess",
        choices=["val", "native"],
        default="val",
        help="val uses training validation preprocessing; native pads/crops no image content and matches UI patch input.",
    )
    return parser.parse_args()


def _normalize_bits(bits: str, length: int) -> str:
    clean = "".join(ch for ch in bits if ch in "01")
    if not clean:
        raise ValueError("--bits must contain at least one 0 or 1.")
    return clean[:length].ljust(length, "0")


def _image_paths(image_dir: Path, limit: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [path for path in sorted(image_dir.rglob("*")) if path.suffix.lower() in exts]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths[:limit]


def _roundtrip_uint8(image: torch.Tensor) -> torch.Tensor:
    return torch.round((image + 1.0) * 0.5 * 255.0) / 255.0 * 2.0 - 1.0


def main() -> None:
    args = _parse_args()
    ckpt_path = Path(args.ckpt)
    device = resolve_device(args.device)

    payload = torch.load(ckpt_path, map_location=device)
    cfg = payload.get("config") or load_config("configs/base.yaml")
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

    payload_length = int(cfg["model"].get("payload_length", cfg["model"]["message_length"]))
    bit_string = _normalize_bits(args.bits, payload_length)
    message = torch.tensor([[int(ch) for ch in bit_string]], dtype=torch.float32, device=device)
    expanded = model._expand_message(message)

    if args.preprocess == "val":
        transform = transforms.Compose(
            [
                transforms.Resize(96, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    rows = []
    direct_scores = []
    roundtrip_scores = []
    psnr_scores = []
    ssim_scores = []

    for image_path in _image_paths(Path(args.image_dir), args.num_images):
        image = Image.open(image_path).convert("RGB")
        x = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            encoded = model.encoder_decoder.encode(x, expanded).clamp(-1, 1)

            decoded_direct = model.encoder_decoder.decoder(encoded)
            compressed_direct, _ = model._compress_message(decoded_direct)
            direct_acc = (compressed_direct.round().clamp(0, 1) == message).float().mean().item()

            roundtrip = _roundtrip_uint8(encoded)
            decoded_roundtrip = model.encoder_decoder.decoder(roundtrip)
            compressed_roundtrip, _ = model._compress_message(decoded_roundtrip)
            roundtrip_acc = (compressed_roundtrip.round().clamp(0, 1) == message).float().mean().item()

            psnr = compute_psnr(encoded, x)
            ssim = compute_ssim(encoded, x)

        direct_scores.append(direct_acc)
        roundtrip_scores.append(roundtrip_acc)
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        rows.append((image_path.name, direct_acc, roundtrip_acc, psnr, ssim))

    direct_avg = sum(direct_scores) / len(direct_scores)
    roundtrip_avg = sum(roundtrip_scores) / len(roundtrip_scores)

    print(f"checkpoint: {ckpt_path}")
    print(f"payload_bits: {bit_string}")
    print(f"preprocess: {args.preprocess}")
    print(f"images: {len(rows)}")
    print(f"direct_tensor_acc_avg: {direct_avg:.4f}")
    print(f"uint8_roundtrip_acc_avg: {roundtrip_avg:.4f}")
    print(f"roundtrip_drop_avg: {direct_avg - roundtrip_avg:.4f}")
    print(f"psnr_avg: {sum(psnr_scores) / len(psnr_scores):.2f}")
    print(f"ssim_avg: {sum(ssim_scores) / len(ssim_scores):.4f}")
    print("worst_roundtrip:")
    for name, direct_acc, roundtrip_acc, psnr, ssim in sorted(rows, key=lambda item: item[2])[:8]:
        print(
            f"{name} direct={direct_acc:.4f} roundtrip={roundtrip_acc:.4f} "
            f"psnr={psnr:.2f} ssim={ssim:.4f}"
        )


if __name__ == "__main__":
    main()
