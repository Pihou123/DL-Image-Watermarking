from __future__ import annotations

import torch

from .base import BaseNoiseLayer
from .registry import register_noise


@register_noise("dropout")
class DropoutLayer(BaseNoiseLayer):
    """
    Randomly drops pixels from the encoded image and replaces them with
    the corresponding pixels from the cover (original) image.

    This simulates an attacker erasing portions of the watermarked image.
    Implementation follows the original HiDDeN paper (Zhu et al., 2018).
    """

    def __init__(self, keep_ratio: float = 0.7, device: torch.device | None = None):
        super().__init__()
        self.keep_ratio = float(keep_ratio)

    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        mask = torch.bernoulli(
            torch.full_like(encoded, self.keep_ratio)
        )
        return encoded * mask + cover * (1 - mask)

