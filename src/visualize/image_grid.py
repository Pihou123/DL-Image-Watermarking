from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils


def _to_01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x.detach().cpu() + 1.0) / 2.0, 0.0, 1.0)


def save_triplet_grid(
    original: torch.Tensor,
    encoded: torch.Tensor,
    noised: torch.Tensor,
    output_path: str | Path,
    max_count: int = 8,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = min(max_count, original.shape[0])
    o = _to_01(original[:count])
    e = _to_01(encoded[:count])
    n = _to_01(noised[:count])

    stacked = torch.cat([o, e, n], dim=0)
    vutils.save_image(stacked, str(output), nrow=count)


def save_difference_heatmap(
    original: torch.Tensor,
    encoded: torch.Tensor,
    output_path: str | Path,
    max_count: int = 8,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = min(max_count, original.shape[0])
    diff = (encoded[:count] - original[:count]).abs().detach().cpu().mean(dim=1)

    cols = count
    fig, axes = plt.subplots(1, cols, figsize=(2 * cols, 2))
    if cols == 1:
        axes = [axes]

    for i in range(cols):
        axes[i].imshow(diff[i].numpy(), cmap="inferno")
        axes[i].axis("off")
        axes[i].set_title(f"diff_{i}")

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close(fig)
