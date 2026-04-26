from __future__ import annotations

import torch
import torch.nn as nn


class BaseNoiseLayer(nn.Module):
    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
