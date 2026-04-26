from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import BaseNoiseLayer
from .registry import register_noise


@register_noise("crop")
class CropLayer(BaseNoiseLayer):
    """
    Randomly crops a contiguous region of the image, then bilinearly
    resizes it back to the original resolution.
    """

    def __init__(self, keep_ratio: float = 0.7, device: torch.device | None = None):
        super().__init__()
        self.keep_ratio = float(keep_ratio)

    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        _, _, h, w = encoded.shape
        crop_h = max(1, int(h * self.keep_ratio))
        crop_w = max(1, int(w * self.keep_ratio))
        top = torch.randint(0, h - crop_h + 1, (1,)).item()
        left = torch.randint(0, w - crop_w + 1, (1,)).item()
        cropped = encoded[:, :, top : top + crop_h, left : left + crop_w]
        return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)
