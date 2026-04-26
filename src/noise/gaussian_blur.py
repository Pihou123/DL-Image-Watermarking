from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .base import BaseNoiseLayer
from .registry import register_noise


def _make_gaussian_kernel_2d(sigma: float, device: torch.device) -> torch.Tensor:
    """Build a 2D Gaussian kernel sized 6*sigma+1."""
    radius = int(math.ceil(3.0 * sigma))
    size = 2 * radius + 1
    coords = torch.arange(size, dtype=torch.float32, device=device) - radius
    g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    g1d /= g1d.sum()
    k2d = g1d[:, None] * g1d[None, :]
    return k2d.expand(3, 1, size, size).contiguous()


@register_noise("gaussian_blur")
class GaussianBlurLayer(BaseNoiseLayer):
    """Applies Gaussian blur via depthwise convolution."""

    def __init__(self, sigma: float = 1.0, device: torch.device | None = None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._sigma = float(sigma)
        kernel = _make_gaussian_kernel_2d(self._sigma, device)
        self.register_buffer("kernel", kernel)

    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        padding = self.kernel.shape[-1] // 2
        return F.conv2d(encoded, self.kernel, groups=3, padding=padding)
