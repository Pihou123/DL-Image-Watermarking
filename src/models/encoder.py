"""水印编码器模块。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention_blocks import GaborAttention, MultiScaleDilatedBlock
from .conv_bn_relu import ConvBNRelu


class Encoder(nn.Module):
    """将图像和水印 bit 融合生成编码结果。"""
    def __init__(self, model_cfg: dict, image_size: tuple[int, int]):
        super().__init__()
        self.height, self.width = image_size
        self.message_length = int(model_cfg["message_length"])
        channels = int(model_cfg["encoder_channels"])
        blocks = int(model_cfg["encoder_blocks"])
        self.use_gabor_attention = bool(model_cfg.get("use_gabor_attention", False))

        layers = [ConvBNRelu(3, channels)]
        for _ in range(blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        self.conv_layers = nn.Sequential(*layers)

        mca_blocks = int(model_cfg.get("encoder_mca_blocks", 0)) if bool(model_cfg.get("use_encoder_mca", False)) else 0
        mca_dilations = model_cfg.get("encoder_mca_dilations", [1, 2, 5])
        self.mca_layers = nn.Sequential(
            *[MultiScaleDilatedBlock(channels, dilations=mca_dilations) for _ in range(mca_blocks)]
        )
        self.gabor_attention = (
            GaborAttention(
                channels=channels,
                kernel_size=int(model_cfg.get("gabor_kernel_size", 15)),
                orientations=int(model_cfg.get("gabor_orientations", 8)),
                sigmas=model_cfg.get("gabor_sigmas", [3.0, 5.0]),
                gamma=float(model_cfg.get("gabor_gamma", 0.5)),
                wavelength=float(model_cfg.get("gabor_wavelength", 8.0)),
                strength=float(model_cfg.get("gabor_attention_strength", 0.5)),
            )
            if self.use_gabor_attention
            else None
        )
        self.after_concat = ConvBNRelu(channels + 3 + self.message_length, channels)
        self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)

    def forward(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        message = message.to(image.device, dtype=torch.float32)
        expanded_message = message.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.height, self.width)

        encoded = self.conv_layers(image)
        encoded = self.mca_layers(encoded)
        if self.gabor_attention is not None:
            encoded = self.gabor_attention(image, encoded)
        concat = torch.cat([expanded_message, encoded, image], dim=1)
        out = self.after_concat(concat)
        out = self.final_layer(out)
        return out
