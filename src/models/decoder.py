"""水印解码器模块。"""

from __future__ import annotations

import torch.nn as nn

from .attention_blocks import SpatialTransformer
from .conv_bn_relu import ConvBNRelu, DiffConvBNRelu


class Decoder(nn.Module):
    """从含水印图像中预测水印 bit。"""
    def __init__(self, model_cfg: dict):
        super().__init__()
        channels = int(model_cfg["decoder_channels"])
        blocks = int(model_cfg["decoder_blocks"])
        message_length = int(model_cfg["message_length"])
        use_diff_conv = bool(model_cfg.get("use_decoder_diff_conv", False))
        diff_layers = int(model_cfg.get("decoder_diff_layers", 3))
        diff_scale = float(model_cfg.get("decoder_diff_scale", 1.0))
        self.stn = (
            SpatialTransformer(
                in_channels=3,
                hidden_channels=int(model_cfg.get("decoder_stn_channels", 32)),
                transform_scale=float(model_cfg.get("decoder_stn_scale", 0.1)),
            )
            if bool(model_cfg.get("use_decoder_stn", False))
            else None
        )

        layers = []
        for index in range(blocks):
            in_channels = 3 if index == 0 else channels
            if use_diff_conv and index < diff_layers:
                layers.append(DiffConvBNRelu(in_channels, channels, diff_scale=diff_scale))
            else:
                layers.append(ConvBNRelu(in_channels, channels))

        layers.append(ConvBNRelu(channels, message_length))
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))

        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(message_length, message_length)

    def forward(self, image_with_wm):
        if self.stn is not None:
            image_with_wm = self.stn(image_with_wm)
        x = self.layers(image_with_wm)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x
