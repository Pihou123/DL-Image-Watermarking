from __future__ import annotations

import torch.nn as nn

from .conv_bn_relu import ConvBNRelu


class Discriminator(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        channels = int(model_cfg["discriminator_channels"])
        blocks = int(model_cfg["discriminator_blocks"])

        layers = [ConvBNRelu(3, channels)]
        for _ in range(blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.backbone = nn.Sequential(*layers)
        self.linear = nn.Linear(channels, 1)

    def forward(self, image):
        x = self.backbone(image)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear(x)
        return x
