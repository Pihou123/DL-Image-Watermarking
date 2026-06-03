"""噪声层基类。"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaseNoiseLayer(nn.Module):
    """所有噪声层的统一接口。"""
    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
