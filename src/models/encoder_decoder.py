from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class EncoderDecoder(nn.Module):
    def __init__(self, model_cfg: dict, image_size: tuple[int, int], noise_manager):
        super().__init__()
        self.encoder = Encoder(model_cfg, image_size=image_size)
        self.decoder = Decoder(model_cfg)
        self.noise_manager = noise_manager

    def forward(self, image: torch.Tensor, message: torch.Tensor, epoch: int | None = None):
        encoded_image = self.encoder(image, message)
        noised_image, noise_meta = self.noise_manager(encoded_image, image, epoch=epoch)
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message, noise_meta
