from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _unwrap_module(module):
    return module.module if hasattr(module, "module") else module


def save_checkpoint(
    checkpoint_path: str | Path,
    epoch: int,
    model,
    config: dict[str, Any],
    scaler: torch.cuda.amp.GradScaler | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "encoder_decoder": _unwrap_module(model.encoder_decoder).state_dict(),
        "discriminator": _unwrap_module(model.discriminator).state_dict(),
        "optimizer_encoder_decoder": model.optimizer_encoder_decoder.state_dict(),
        "optimizer_discriminator": model.optimizer_discriminator.state_dict(),
        "config": config,
    }

    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload.update(extra)

    torch.save(payload, path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device)

    _unwrap_module(model.encoder_decoder).load_state_dict(payload["encoder_decoder"], strict=strict)
    _unwrap_module(model.discriminator).load_state_dict(payload["discriminator"], strict=strict)

    if "optimizer_encoder_decoder" in payload:
        model.optimizer_encoder_decoder.load_state_dict(payload["optimizer_encoder_decoder"])
    if "optimizer_discriminator" in payload:
        model.optimizer_discriminator.load_state_dict(payload["optimizer_discriminator"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])

    return payload


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None

    files = sorted(ckpt_dir.glob("*.pth"))
    return files[-1] if files else None
