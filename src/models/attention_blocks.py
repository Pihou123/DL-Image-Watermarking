from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """Lightweight affine STN initialized as identity."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32, transform_scale: float = 0.1):
        super().__init__()
        self.transform_scale = float(transform_scale)
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(4, 4)),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels * 2 * 4 * 4, hidden_channels * 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels * 4, 6),
        )

        nn.init.zeros_(self.regressor[-1].weight)
        nn.init.zeros_(self.regressor[-1].bias)
        self.register_buffer(
            "identity_theta",
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        features = self.localization(x).view(batch_size, -1)
        delta = self.regressor(features).view(batch_size, 2, 3)
        theta = self.identity_theta.to(device=x.device, dtype=x.dtype) + self.transform_scale * delta
        grid = F.affine_grid(theta, size=x.size(), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


class MultiScaleDilatedBlock(nn.Module):
    """Parallel dilated convolutions with residual fusion."""

    def __init__(self, channels: int, dilations: list[int] | tuple[int, ...] = (1, 2, 5)):
        super().__init__()
        if not dilations:
            raise ValueError("dilations must contain at least one dilation rate.")

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=int(rate), dilation=int(rate)),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for rate in dilations
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(self.branches), channels, kernel_size=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.relu(x + self.fuse(multi_scale))


class GaborAttention(nn.Module):
    """Fixed Gabor filter bank followed by learnable channel attention."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        orientations: int = 8,
        sigmas: list[float] | tuple[float, ...] = (3.0, 5.0),
        gamma: float = 0.5,
        wavelength: float = 8.0,
        strength: float = 0.5,
    ):
        super().__init__()
        self.channels = int(channels)
        self.strength = float(strength)
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("gabor_kernel_size must be a positive odd integer.")

        kernels = []
        for sigma in sigmas:
            for index in range(int(orientations)):
                theta = math.pi * index / max(1, int(orientations))
                kernels.append(
                    _build_gabor_kernel(
                        kernel_size=kernel_size,
                        sigma=float(sigma),
                        theta=float(theta),
                        gamma=float(gamma),
                        wavelength=float(wavelength),
                    )
                )

        bank = torch.stack(kernels, dim=0).unsqueeze(1)
        self.register_buffer("gabor_kernels", bank)
        self.attention = nn.Sequential(
            nn.Conv2d(bank.shape[0], channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        gray = image.mean(dim=1, keepdim=True)
        padding = self.gabor_kernels.shape[-1] // 2
        responses = F.conv2d(gray, self.gabor_kernels.to(device=image.device, dtype=image.dtype), padding=padding)
        attention = self.attention(responses.abs())
        return features * (1.0 + self.strength * attention)


def _build_gabor_kernel(
    kernel_size: int,
    sigma: float,
    theta: float,
    gamma: float,
    wavelength: float,
) -> torch.Tensor:
    half = kernel_size // 2
    coords = torch.arange(-half, half + 1, dtype=torch.float32)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)
    envelope = torch.exp(-(x_theta.pow(2) + (gamma * y_theta).pow(2)) / (2.0 * sigma * sigma))
    carrier = torch.cos(2.0 * math.pi * x_theta / wavelength)
    kernel = envelope * carrier
    kernel = kernel - kernel.mean()
    norm = kernel.abs().sum().clamp_min(1e-6)
    return kernel / norm
