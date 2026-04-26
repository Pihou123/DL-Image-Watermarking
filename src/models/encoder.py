from __future__ import annotations

import torch
import torch.nn as nn

from .conv_bn_relu import ConvBNRelu


class Encoder(nn.Module):
    def __init__(self, model_cfg: dict, image_size: tuple[int, int]):
        super().__init__()
        self.height, self.width = image_size
        self.message_length = int(model_cfg["message_length"])
        channels = int(model_cfg["encoder_channels"])
        blocks = int(model_cfg["encoder_blocks"])

        layers = [ConvBNRelu(3, channels)]
        for _ in range(blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat = ConvBNRelu(channels + 3 + self.message_length, channels)
        self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)

    def forward(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        message = message.to(image.device, dtype=torch.float32)
        expanded_message = message.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.height, self.width)

        encoded = self.conv_layers(image)
        concat = torch.cat([expanded_message, encoded, image], dim=1)
        out = self.after_concat(concat)
        out = self.final_layer(out)
        return out
