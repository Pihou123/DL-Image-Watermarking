"""随机像素替换遮挡噪声层。"""

from __future__ import annotations

import torch

from .base import BaseNoiseLayer
from .registry import register_noise


@register_noise("dropout")
class DropoutLayer(BaseNoiseLayer):
    """随机用原图像素替换含水印像素。"""

    def __init__(self, keep_ratio: float = 0.7, device: torch.device | None = None):
        super().__init__()
        self.keep_ratio = float(keep_ratio)

    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        mask = torch.bernoulli(
            torch.full_like(encoded, self.keep_ratio)
        )
        return encoded * mask + cover * (1 - mask)
