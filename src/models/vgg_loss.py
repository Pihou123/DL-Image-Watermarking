from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16


class VGGLoss(nn.Module):
    """VGG feature extractor for perceptual loss."""

    def __init__(self, feature_layers: int = 16, device: torch.device | None = None):
        super().__init__()
        try:
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            model = vgg16(weights=None)

        self.features = model.features[:feature_layers].eval()
        for p in self.features.parameters():
            p.requires_grad = False

        if device is not None:
            self.features.to(device)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # VGG expects [0, 1].
        x = (image + 1.0) / 2.0
        return self.features(x)
