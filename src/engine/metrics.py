from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F


def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Generate a 1D Gaussian kernel."""
    coords = torch.arange(size, dtype=torch.float32)
    mean = (size - 1) / 2.0
    kernel = torch.exp(-((coords - mean) ** 2) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def _gaussian_filter(channels: int, kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Build a 2D Gaussian filter kernel of shape [C, 1, H, W] for depthwise conv."""
    k1d = _gaussian_kernel(kernel_size, sigma).to(device)
    k2d = k1d[:, None] * k1d[None, :]
    kernel = k2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return kernel


def compute_psnr(encoded: torch.Tensor, cover: torch.Tensor) -> float:
    """
    Compute PSNR between encoded and cover images.

    Images are assumed in [-1, 1] range; normalized to [0, 1] before computation.
    Returns PSNR in dB.
    """
    encoded_01 = (encoded.detach().float().clamp(-1, 1) + 1.0) / 2.0
    cover_01 = (cover.detach().float().clamp(-1, 1) + 1.0) / 2.0
    mse = float(F.mse_loss(encoded_01, cover_01).item())
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim(
    encoded: torch.Tensor,
    cover: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """
    Compute SSIM between encoded and cover images.

    Images are assumed in [-1, 1] range; normalized to [0, 1] before computation.
    Standard SSIM parameters: window_size=11, sigma=1.5, K1=0.01, K2=0.03.
    Returns mean SSIM over the batch.
    """
    encoded_01 = (encoded.detach().float().clamp(-1, 1) + 1.0) / 2.0
    cover_01 = (cover.detach().float().clamp(-1, 1) + 1.0) / 2.0
    device = encoded.device
    data_range = 1.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    batch_size, channels = encoded_01.shape[0], encoded_01.shape[1]
    kernel = _gaussian_filter(channels, window_size, sigma, device).float()

    mu_enc = F.conv2d(encoded_01, kernel, groups=channels, padding=window_size // 2)
    mu_cov = F.conv2d(cover_01, kernel, groups=channels, padding=window_size // 2)

    mu_enc_sq = mu_enc ** 2
    mu_cov_sq = mu_cov ** 2
    mu_enc_cov = mu_enc * mu_cov

    sigma_enc_sq = F.conv2d(encoded_01 ** 2, kernel, groups=channels, padding=window_size // 2) - mu_enc_sq
    sigma_cov_sq = F.conv2d(cover_01 ** 2, kernel, groups=channels, padding=window_size // 2) - mu_cov_sq
    sigma_enc_cov = F.conv2d(encoded_01 * cover_01, kernel, groups=channels, padding=window_size // 2) - mu_enc_cov

    ssim_map = ((2.0 * mu_enc_cov + c1) * (2.0 * sigma_enc_cov + c2)) / (
        (mu_enc_sq + mu_cov_sq + c1) * (sigma_enc_sq + sigma_cov_sq + c2)
    )

    return float(ssim_map.mean().item())


@dataclass
class MetricsAverager:
    sums: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def update(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value)
            self.counts[key] = self.counts.get(key, 0) + 1

    def averages(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, total in self.sums.items():
            count = max(1, self.counts.get(key, 1))
            out[key] = total / count
        return out
