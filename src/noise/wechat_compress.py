from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNoiseLayer
from .registry import register_noise


@register_noise("wechat")
class WechatCompressionLayer(BaseNoiseLayer):
    """
    Simulates WeChat image compression pipeline:
    1. Optional downscale when long side exceeds a threshold
    2. Chroma subsampling (YUV 4:2:0)
    3. JPEG compression using approximate WeChat quantization table

    This is an *approximation* of WeChat's real compression behavior.
    WeChat uses a proprietary quantization matrix; this implementation
    uses a plausible low-quality encoding to simulate the effect.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        max_long_side: int = 1280,
        yuv_keep_weights: tuple[int, int, int] = (20, 8, 8),
    ):
        """
        Args:
            max_long_side: If the image's longest side exceeds this, it is
                           downscaled (and then bilinearly restored).
            yuv_keep_weights: DCT coefficients retained per Y/U/V channel in
                              each 8x8 block.  Smaller than default JPEG to
                              simulate WeChat's more aggressive compression.
        """
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.max_long_side = int(max_long_side)
        self.yuv_keep_weights = tuple(yuv_keep_weights)

        # Lazy-init DCT / mask in the first forward pass.
        self._jpeg_layer: nn.Module | None = None

    def _ensure_jpeg_layer(self, h: int, w: int) -> None:
        """Lazy-build the internal JPEG compression module."""
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

        # --- step 1: resize if needed ---
        if long_side > self.max_long_side:
            ratio = self.max_long_side / long_side
            new_h, new_w = int(round(h * ratio)), int(round(w * ratio))
            resized = F.interpolate(encoded, size=(new_h, new_w), mode="bilinear", align_corners=False)
            # restore to original size (simulates user viewing a downscaled image)
            resized = F.interpolate(resized, size=(h, w), mode="bilinear", align_corners=False)
        else:
            resized = encoded

        # --- step 2-3: chroma subsampling + JPEG (approximated by DCT masking) ---
        self._ensure_jpeg_layer(h, w)
        assert self._jpeg_layer is not None
        compressed = self._jpeg_layer(resized, cover)
        return compressed
