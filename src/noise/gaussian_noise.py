from __future__ import annotations

import torch

from .base import BaseNoiseLayer
from .registry import register_noise


@register_noise("gaussian_noise")
class GaussianNoiseLayer(BaseNoiseLayer):
    """Adds zero-mean Gaussian noise to the encoded image."""

    def __init__(self, std: float = 0.05, device: torch.device | None = None):
        super().__init__()
        self._std = float(std)

    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(encoded) * self._std
        return encoded + noise
