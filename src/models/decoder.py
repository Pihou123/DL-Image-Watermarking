from __future__ import annotations

import torch.nn as nn

from .conv_bn_relu import ConvBNRelu


class Decoder(nn.Module):
    def __init__(self, model_cfg: dict):
        super().__init__()
        channels = int(model_cfg["decoder_channels"])
        blocks = int(model_cfg["decoder_blocks"])
        message_length = int(model_cfg["message_length"])

        layers = [ConvBNRelu(3, channels)]
        for _ in range(blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, message_length))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(message_length, message_length)

    def forward(self, image_with_wm):
        x = self.layers(image_with_wm)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
