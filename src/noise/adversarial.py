"""可训练对抗扰动噪声层。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseNoiseLayer
from .registry import register_noise


@register_noise("adversarial")
class AdversarialNoiseLayer(BaseNoiseLayer):
    """可训练的有界扰动层。"""

    is_adversarial = True

    def __init__(self, epsilon: float = 0.05, channels: int = 32, blocks: int = 3, device: torch.device | None = None):
        super().__init__()
        self.epsilon = float(epsilon)
        hidden = int(channels)
        depth = max(1, int(blocks))

        layers: list[nn.Module] = [
            nn.Conv2d(3, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(hidden, 3, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        perturbation = self.epsilon * torch.tanh(self.net(encoded))
        return torch.clamp(encoded + perturbation, -1.0, 1.0)
