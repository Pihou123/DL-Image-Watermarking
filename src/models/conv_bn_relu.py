"""卷积、归一化和激活基础模块。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNRelu(nn.Module):
    """标准 Conv-BN-ReLU 模块。"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class DiffConvBNRelu(nn.Module):
    """差分卷积增强模块。"""

    def __init__(self, in_channels: int, out_channels: int, diff_scale: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.diff_scale = float(diff_scale)
        self.vanilla = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.diff_fuse = nn.Conv2d(in_channels * 5, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        kernels = torch.tensor(
            [
                [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                [[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]],
                [[2.0, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, -1.0, -2.0]],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("diff_kernels", kernels.view(5, 1, 3, 3))

    def forward(self, x):
        vanilla = self.vanilla(x)
        kernels = self.diff_kernels.to(device=x.device, dtype=x.dtype).repeat(self.in_channels, 1, 1, 1)
        diff = F.conv2d(x, kernels, padding=1, groups=self.in_channels)
        diff = self.diff_fuse(diff)
        return self.relu(self.bn(vanilla + self.diff_scale * diff))
