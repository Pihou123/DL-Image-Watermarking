from __future__ import annotations

import torch

from .base import BaseNoiseLayer
from .registry import register_noise


@register_noise("identity")
class IdentityNoise(BaseNoiseLayer):
    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        return encoded
