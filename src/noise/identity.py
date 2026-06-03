"""不改变图像的占位噪声层。"""

from __future__ import annotations

import torch

from .base import BaseNoiseLayer
from .registry import register_noise


@register_noise("identity")
class IdentityNoise(BaseNoiseLayer):
    """直接返回输入图像。"""
    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        return encoded
