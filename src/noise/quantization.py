from __future__ import annotations

import numpy as np
import torch

from .base import BaseNoiseLayer
from .registry import register_noise


def _transform(tensor: torch.Tensor, target_range: tuple[float, float]) -> torch.Tensor:
    src_min = tensor.amin()
    src_max = tensor.amax()
    if torch.isclose(src_min, src_max):
        return torch.full_like(tensor, target_range[0])

    normalized = (tensor - src_min) / (src_max - src_min)
    return normalized * (target_range[1] - target_range[0]) + target_range[0]


@register_noise("quantization")
@register_noise("quantize")
class QuantizationNoise(BaseNoiseLayer):
    def __init__(self, fourier_terms: int = 10, device: torch.device | None = None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.min_value = 0.0
        self.max_value = 255.0
        self.n_terms = int(fourier_terms)

        weights = [((-1) ** (n + 1)) / (np.pi * (n + 1)) for n in range(self.n_terms)]
        scales = [2 * np.pi * (n + 1) for n in range(self.n_terms)]

        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32, device=device))
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32, device=device))

    def _fourier_rounding(self, tensor: torch.Tensor) -> torch.Tensor:
        # Expand tensor to [1, B, C, H, W] so Fourier-term dimension is separated from batch dimension.
        tensor_expanded = tensor.unsqueeze(0)
        scales = self.scales.view(-1, 1, 1, 1, 1)
        weights = self.weights.view(-1, 1, 1, 1, 1)

        z = weights * torch.sin(tensor_expanded * scales)
        z = z.sum(dim=0)
        return tensor + z

    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        original_min = float(encoded.amin().item())
        original_max = float(encoded.amax().item())

        q = _transform(encoded, (self.min_value, self.max_value))
        q = self._fourier_rounding(torch.clamp(q, self.min_value, self.max_value))
        q = _transform(q, (original_min, original_max))
        return q

