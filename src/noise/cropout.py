"""保留局部含水印区域的噪声层。"""

from __future__ import annotations

import torch

from .base import BaseNoiseLayer
from .registry import register_noise


@register_noise("cropout")
class CropoutLayer(BaseNoiseLayer):
    """保留一块含水印区域，其余位置使用原图填充。"""

    def __init__(self, keep_ratio: float = 0.3, device: torch.device | None = None):
        super().__init__()
        self.keep_ratio = float(keep_ratio)

    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        _, _, h, w = encoded.shape
        keep_h = max(1, int(h * self.keep_ratio))
        keep_w = max(1, int(w * self.keep_ratio))
        top = torch.randint(0, h - keep_h + 1, (1,)).item()
        left = torch.randint(0, w - keep_w + 1, (1,)).item()
        mask = torch.zeros(1, 1, h, w, device=encoded.device)
        mask[:, :, top : top + keep_h, left : left + keep_w] = 1.0
        return encoded * mask + cover * (1 - mask)
