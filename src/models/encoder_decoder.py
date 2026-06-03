"""编码器、噪声层和解码器的串联模块。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class EncoderDecoder(nn.Module):
    """将图像和水印 bit 融合生成编码结果。"""
    def __init__(self, model_cfg: dict, image_size: tuple[int, int], noise_manager):
        super().__init__()
        self.encoder = Encoder(model_cfg, image_size=image_size)
        self.decoder = Decoder(model_cfg)
        self.noise_manager = noise_manager
        self.use_residual_embedding = bool(model_cfg.get("residual_embedding", False))
        self.residual_scale = float(model_cfg.get("residual_scale", 0.5))
        self.residual_activation = str(model_cfg.get("residual_activation", "tanh")).lower()
        self.clamp_encoded = bool(model_cfg.get("clamp_encoded", True))

    def encode(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(image, message)
        if not self.use_residual_embedding:
            return encoder_output

        if self.residual_activation == "tanh":
            residual = torch.tanh(encoder_output)
        elif self.residual_activation in {"none", "identity"}:
            residual = encoder_output
        else:
            raise ValueError(
                "Unknown model.residual_activation: "
                f"{self.residual_activation}. Available: tanh, none"
            )

        encoded_image = image + self.residual_scale * residual
        if self.clamp_encoded:
            encoded_image = torch.clamp(encoded_image, -1.0, 1.0)
        return encoded_image

    def forward(self, image: torch.Tensor, message: torch.Tensor, epoch: int | None = None):
        encoded_image = self.encode(image, message)
        noised_image, noise_meta = self.noise_manager(encoded_image, image, epoch=epoch)
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message, noise_meta
