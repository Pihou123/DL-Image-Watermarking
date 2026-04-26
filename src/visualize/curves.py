from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def write_history_csv(history: list[dict[str, float]], output_path: str | Path) -> None:
    if not history:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict[str, float]], output_path: str | Path) -> None:
    if not history:
        return

    epochs = [row["epoch"] for row in history]

    train_loss = [row.get("train_loss", float("nan")) for row in history]
    val_loss = [row.get("val_loss", float("nan")) for row in history]
    train_bit_acc = [row.get("train_bit_acc", float("nan")) for row in history]
    val_bit_acc = [row.get("val_bit_acc", float("nan")) for row in history]
    val_psnr = [row.get("val_psnr", float("nan")) for row in history]
    val_ssim = [row.get("val_ssim", float("nan")) for row in history]

    has_psnr = any(not math.isnan(v) for v in val_psnr)

    if has_psnr:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax_loss, ax_acc, ax_psnr, ax_ssim = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax_loss, ax_acc = axes.flatten()

    ax_loss.plot(epochs, train_loss, label="train_loss")
    ax_loss.plot(epochs, val_loss, label="val_loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_acc.plot(epochs, train_bit_acc, label="train_bit_acc")
    ax_acc.plot(epochs, val_bit_acc, label="val_bit_acc")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("bit accuracy")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    if has_psnr:
        ax_psnr.plot(epochs, val_psnr, label="val_psnr", color="green")
        ax_psnr.set_xlabel("epoch")
        ax_psnr.set_ylabel("PSNR (dB)")
        ax_psnr.grid(True, alpha=0.3)
        ax_psnr.legend()

        ax_ssim.plot(epochs, val_ssim, label="val_ssim", color="orange")
        ax_ssim.set_xlabel("epoch")
        ax_ssim.set_ylabel("SSIM")
        ax_ssim.grid(True, alpha=0.3)
        ax_ssim.legend()

    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close(fig)
