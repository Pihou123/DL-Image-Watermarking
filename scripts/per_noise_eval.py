"""Per-noise accuracy breakdown using a trained checkpoint.

Usage:
    python scripts/per_noise_eval.py --run_dir outputs/runs/<run_name>
    python scripts/per_noise_eval.py --run_dir outputs/runs/<run_name> --ckpt epoch_0015.pth
    python scripts/per_noise_eval.py --config configs/base.yaml --ckpt <path>
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
from tqdm import tqdm

from src.common.config import load_config
from src.common.runtime import resolve_device
from src.common.seed import set_seed
from src.data.loaders import build_dataloaders
from src.engine.checkpoint import load_checkpoint
from src.engine.metrics import compute_psnr, compute_ssim
from src.models.hidden_system import HiddenSystem
from src.noise.manager import NoiseManager
from src.noise.registry import create_noise, list_registered_noise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-noise accuracy evaluation.")
    parser.add_argument("--run_dir", default=None, help="Training run directory (reads resolved_config.yaml).")
    parser.add_argument("--config", default="configs/base.yaml", help="Fallback config if no run_dir.")
    parser.add_argument(
        "--ckpt",
        default="best.pth",
        help="Checkpoint filename inside run_dir/checkpoints/, or full path.",
    )
    parser.add_argument("--device", default="auto", help="Device override.")
    return parser.parse_args()


def _bit_accuracy(decoded: torch.Tensor, messages: torch.Tensor) -> float:
    decoded_bits = decoded.detach().cpu().numpy().round().clip(0, 1)
    targets = messages.detach().cpu().numpy()
    bit_error = float(np.mean(np.abs(decoded_bits - targets)))
    return float(1.0 - bit_error)


@torch.no_grad()
def evaluate_noise(
    model: HiddenSystem,
    val_loader,
    noise_name: str,
    noise_module,
    device: torch.device,
) -> dict[str, float]:
    """Run inference with a specific noise layer forced on every sample."""
    original_manager = model.encoder_decoder.noise_manager
    original_layers = model.encoder_decoder.noise_manager.layers
    original_specs = model.encoder_decoder.noise_manager.layer_specs

    # Patch the noise manager to always use the target noise.
    target_key = f"{noise_name}_eval"
    model.encoder_decoder.noise_manager.layers = torch.nn.ModuleDict({target_key: noise_module})
    model.encoder_decoder.noise_manager.layer_specs = [
        {"key": target_key, "name": noise_name, "probability": 1.0}
    ]
    model.encoder_decoder.noise_manager.strategy = "single_random"

    model.encoder_decoder.eval()
    model.discriminator.eval()

    bit_accs: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []

    for images, _ in tqdm(val_loader, desc=f"  {noise_name}", leave=False):
        images = images.to(device)
        message_length = int(model.model_cfg.get("payload_length", model.model_cfg["message_length"]))
        messages = torch.randint(
            0, 2, (images.shape[0], message_length), dtype=torch.float32, device=device
        )

        expanded = model._expand_message(messages)
        encoded, noised, decoded, _ = model.encoder_decoder(images, expanded)
        compressed, _ = model._compress_message(decoded)

        bit_accs.append(_bit_accuracy(compressed, messages))
        psnrs.append(compute_psnr(encoded, images))
        ssims.append(compute_ssim(encoded, images))

    # Restore original manager state.
    model.encoder_decoder.noise_manager.layers = original_layers
    model.encoder_decoder.noise_manager.layer_specs = original_specs
    model.encoder_decoder.noise_manager.strategy = original_manager.strategy

    return {
        "bit_acc": float(np.mean(bit_accs)),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
    }


def main() -> None:
    args = parse_args()

    # Resolve config.
    if args.run_dir:
        run_dir = Path(args.run_dir)
        resolved_cfg_path = run_dir / "resolved_config.yaml"
        if resolved_cfg_path.exists():
            cfg = load_config(resolved_cfg_path)
        else:
            cfg = load_config(args.config)
    else:
        run_dir = None
        cfg = load_config(args.config)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    set_seed(int(cfg["experiment"].get("seed", 42)))

    # Build dataloader.
    train_loader, val_loader = build_dataloaders(
        cfg["dataset"], batch_size=int(cfg["train"]["batch_size"])
    )

    # Build model.
    image_size = tuple(cfg["dataset"].get("image_size", [64, 64]))
    noise_manager = NoiseManager(cfg["noise"], device=device).to(device)
    model = HiddenSystem(
        model_cfg=cfg["model"],
        train_cfg=cfg["train"],
        image_size=(int(image_size[0]), int(image_size[1])),
        noise_manager=noise_manager,
        device=device,
    ).to(device)

    # Load checkpoint.
    ckpt_path = Path(args.ckpt)
    if run_dir and not ckpt_path.is_absolute():
        ckpt_path = run_dir / "checkpoints" / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = load_checkpoint(ckpt_path, model=model, device=device, scaler=None, strict=False)
    ckpt_epoch = int(payload.get("epoch", "?"))
    print(f"Checkpoint: {ckpt_path}  (epoch {ckpt_epoch})")
    print(f"Val samples: {len(val_loader.dataset)}")
    print()

    # Collect noise names from the config (preserves order and params).
    layer_configs = cfg["noise"].get("layers", [])
    noise_entries: list[tuple[str, dict]] = []
    for entry in layer_configs:
        name = str(entry["name"]).lower()
        params = dict(entry.get("params", {}))
        noise_entries.append((name, params))

    # Evaluate.
    results: dict[str, dict[str, float]] = {}
    for noise_name, params in noise_entries:
        try:
            noise_module = create_noise(noise_name, device=device, **params)
        except TypeError:
            # Some layers don't accept device.
            noise_module = create_noise(noise_name, **params)

        results[noise_name] = evaluate_noise(model, val_loader, noise_name, noise_module, device)

    # Print table.
    header = f"{'Noise':<20} {'bit_acc':>8}  {'PSNR':>7}  {'SSIM':>7}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<20} {m['bit_acc']:>8.4f}  {m['psnr']:>6.1f}  {m['ssim']:>7.4f}")

    # Average.
    avg_acc = float(np.mean([m["bit_acc"] for m in results.values()]))
    avg_psnr = float(np.mean([m["psnr"] for m in results.values()]))
    avg_ssim = float(np.mean([m["ssim"] for m in results.values()]))
    print("-" * len(header))
    print(f"{'AVERAGE':<20} {avg_acc:>8.4f}  {avg_psnr:>6.1f}  {avg_ssim:>7.4f}")

    # Highlight worst.
    worst_noise = min(results.items(), key=lambda kv: kv[1]["bit_acc"])
    best_noise = max(results.items(), key=lambda kv: kv[1]["bit_acc"])
    print()
    print(f"Best:  {best_noise[0]:<20} bit_acc={best_noise[1]['bit_acc']:.4f}")
    print(f"Worst: {worst_noise[0]:<20} bit_acc={worst_noise[1]['bit_acc']:.4f}")


if __name__ == "__main__":
    main()
