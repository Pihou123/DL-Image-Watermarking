from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseNoiseLayer
from .registry import register_noise


@register_noise("resize")
class ResizeLayer(BaseNoiseLayer):
    """
    Randomly scales the image up or down, then bilinearly resizes back to
    the original resolution. This simulates common image resizing operations
    (e.g. down-sampling for web upload, then zooming back).

    ratio_range controls the scale factor: (0.5, 0.5) means exactly 50% size;
    (0.8, 1.2) randomly scales between 80% and 120% of original.
    """

    def __init__(self, ratio_min: float = 0.5, ratio_max: float = 1.0, device: torch.device | None = None):
        super().__init__()
        self.ratio_min = float(ratio_min)
        self.ratio_max = float(ratio_max)

    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        _, _, h, w = encoded.shape
        ratio = self.ratio_min + torch.rand(1).item() * (self.ratio_max - self.ratio_min)
        scaled = F.interpolate(encoded, scale_factor=ratio, mode="bilinear", align_corners=False)
        return F.interpolate(scaled, size=(h, w), mode="bilinear", align_corners=False)
