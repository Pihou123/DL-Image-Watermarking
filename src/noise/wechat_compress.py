"""微信压缩近似噪声层。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNoiseLayer
from .registry import register_noise


@register_noise("wechat")
class WechatCompressionLayer(BaseNoiseLayer):
    """近似模拟微信图片压缩。"""

    def __init__(
        self,
        device: torch.device | None = None,
        max_long_side: int = 1280,
        yuv_keep_weights: tuple[int, int, int] = (20, 8, 8),
    ):
        """设置压缩参数。"""
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.max_long_side = int(max_long_side)
        self.yuv_keep_weights = tuple(yuv_keep_weights)

        # 首次使用时构建 JPEG 层。
        self._jpeg_layer: nn.Module | None = None

    def _ensure_jpeg_layer(self, h: int, w: int) -> None:
        """按需创建 JPEG 层。"""
        if self._jpeg_layer is not None:
            return
        from .jpeg_compression import JpegCompressionNoise

        self._jpeg_layer = JpegCompressionNoise(
            device=self.device,
            yuv_keep_weights=self.yuv_keep_weights,
        )

    def forward(self, encoded: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
        _, _, h, w = encoded.shape
        long_side = max(h, w)

        # 缩放超出长边限制的图像。
        if long_side > self.max_long_side:
            ratio = self.max_long_side / long_side
            new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
            resized = F.interpolate(encoded, size=(new_h, new_w), mode="bilinear", align_corners=False)
            # 还原到原始尺寸。
            resized = F.interpolate(resized, size=(h, w), mode="bilinear", align_corners=False)
        else:
            resized = encoded

        # 应用近似 JPEG 压缩。
        self._ensure_jpeg_layer(h, w)
        assert self._jpeg_layer is not None
        compressed = self._jpeg_layer(resized, cover)
        return compressed
