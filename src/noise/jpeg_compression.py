from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNoiseLayer
from .registry import register_noise


def _dct_coeff(n: int, k: int, n_size: int) -> float:
    return float(np.cos(np.pi / n_size * (n + 0.5) * k))


def _idct_coeff(n: int, k: int, n_size: int) -> float:
    return float((int(n == 0) * (-0.5) + np.cos(np.pi / n_size * (k + 0.5) * n)) * np.sqrt(1.0 / (2.0 * n_size)))


def _gen_filters(size_x: int, size_y: int, coeff_fun) -> np.ndarray:
    filters = np.zeros((size_x * size_y, size_x, size_y), dtype=np.float32)
    for k_y in range(size_y):
        for k_x in range(size_x):
            for n_y in range(size_y):
                for n_x in range(size_x):
                    filters[k_y * size_x + k_x, n_y, n_x] = coeff_fun(n_y, k_y, size_y) * coeff_fun(n_x, k_x, size_x)
    return filters


def _jpeg_mask_for_channel(height: int, width: int, window_size: int, keep_count: int) -> np.ndarray:
    mask = np.zeros((window_size, window_size), dtype=np.uint8)
    order = sorted(
        ((x, y) for x in range(window_size) for y in range(window_size)),
        key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]),
    )
    for i, j in order[:keep_count]:
        mask[i, j] = 1

    tiled = np.tile(mask, (int(np.ceil(height / window_size)), int(np.ceil(width / window_size))))
    return tiled[:height, :width]


def _rgb_to_yuv(image_rgb: torch.Tensor) -> torch.Tensor:
    yuv = torch.empty_like(image_rgb)
    yuv[:, 0, :, :] = 0.299 * image_rgb[:, 0, :, :] + 0.587 * image_rgb[:, 1, :, :] + 0.114 * image_rgb[:, 2, :, :]
    yuv[:, 1, :, :] = -0.14713 * image_rgb[:, 0, :, :] - 0.28886 * image_rgb[:, 1, :, :] + 0.436 * image_rgb[:, 2, :, :]
    yuv[:, 2, :, :] = 0.615 * image_rgb[:, 0, :, :] - 0.51499 * image_rgb[:, 1, :, :] - 0.10001 * image_rgb[:, 2, :, :]
    return yuv


def _yuv_to_rgb(image_yuv: torch.Tensor) -> torch.Tensor:
    rgb = torch.empty_like(image_yuv)
    rgb[:, 0, :, :] = image_yuv[:, 0, :, :] + 1.13983 * image_yuv[:, 2, :, :]
    rgb[:, 1, :, :] = image_yuv[:, 0, :, :] - 0.39465 * image_yuv[:, 1, :, :] - 0.58060 * image_yuv[:, 2, :, :]
    rgb[:, 2, :, :] = image_yuv[:, 0, :, :] + 2.03211 * image_yuv[:, 1, :, :]
    return rgb


@register_noise("jpeg")
class JpegCompressionNoise(BaseNoiseLayer):
    def __init__(self, device: torch.device | None = None, yuv_keep_weights: tuple[int, int, int] = (25, 9, 9)):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        dct = torch.tensor(_gen_filters(8, 8, _dct_coeff), dtype=torch.float32, device=self.device).unsqueeze(1)
        idct = torch.tensor(_gen_filters(8, 8, _idct_coeff), dtype=torch.float32, device=self.device).unsqueeze(1)
        self.register_buffer("dct_filters", dct)
        self.register_buffer("idct_filters", idct)

        self.yuv_keep_weights = tuple(yuv_keep_weights)
        self.jpeg_mask: torch.Tensor | None = None

    def _ensure_mask(self, height: int, width: int) -> None:
        if self.jpeg_mask is not None and self.jpeg_mask.shape[1] >= height and self.jpeg_mask.shape[2] >= width:
            return

        mask = torch.empty((3, height, width), device=self.device)
        for channel, keep_count in enumerate(self.yuv_keep_weights):
            mask_np = _jpeg_mask_for_channel(height, width, window_size=8, keep_count=int(keep_count))
            mask[channel] = torch.from_numpy(mask_np).to(self.device)
        self.jpeg_mask = mask

    def _apply_conv(self, image: torch.Tensor, use_dct: bool) -> torch.Tensor:
        filters = self.dct_filters if use_dct else self.idct_filters
        channels = []

        for channel in range(image.shape[1]):
            x = image[:, channel, :, :].unsqueeze(1)
            x = F.conv2d(x, filters, stride=8)
            x = x.permute(0, 2, 3, 1)
            x = x.view(x.shape[0], x.shape[1], x.shape[2], 8, 8)
            x = x.permute(0, 1, 3, 2, 4)
            x = x.contiguous().view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3] * x.shape[4])
            channels.append(x.unsqueeze(1))

        return torch.cat(channels, dim=1)

    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        noised = encoded
        pad_h = (8 - noised.shape[2] % 8) % 8
        pad_w = (8 - noised.shape[3] % 8) % 8
        noised = nn.ZeroPad2d((0, pad_w, 0, pad_h))(noised)

        yuv = _rgb_to_yuv(noised)
        dct = self._apply_conv(yuv, use_dct=True)

        self._ensure_mask(dct.shape[2], dct.shape[3])
        assert self.jpeg_mask is not None
        mask = self.jpeg_mask[:, : dct.shape[2], : dct.shape[3]].unsqueeze(0)

        dct_masked = dct * mask
        yuv_recon = self._apply_conv(dct_masked, use_dct=False)
        rgb_recon = _yuv_to_rgb(yuv_recon)

        if pad_h > 0:
            rgb_recon = rgb_recon[:, :, :-pad_h, :]
        if pad_w > 0:
            rgb_recon = rgb_recon[:, :, :, :-pad_w]
        return rgb_recon
