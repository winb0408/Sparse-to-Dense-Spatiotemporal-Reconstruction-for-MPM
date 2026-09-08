"""
Prior-aware fusion modules for mask + Wiener/warm-start inputs.

The modules are intentionally lightweight and conservative:
- default model behaviour is unchanged when prior_fusion='none';
- gated fusion starts from an identity mapping because the residual scale is
  initialized to zero;
- only image / Wiener channels are modulated, while mask and optional PSF
  channels are preserved for downstream layers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class GatedPriorFusion2D(nn.Module):
    """Use mask channels to gate early 2D image/Wiener channels."""

    def __init__(self, image_channels: int, mask_channels: int):
        super().__init__()
        if image_channels <= 0 or mask_channels <= 0:
            raise ValueError("image_channels and mask_channels must be positive.")
        self.image_channels = image_channels
        self.mask_channels = mask_channels
        self.gate = nn.Sequential(
            nn.Conv2d(mask_channels, image_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = x[:, :self.image_channels]
        mask = x[:, self.image_channels:self.image_channels + self.mask_channels]
        rest = x[:, self.image_channels + self.mask_channels:]
        gate = self.gate(mask)
        fused = image * (1.0 + self.residual_scale * (gate - 0.5))
        return torch.cat([fused, mask, rest], dim=1) if rest.numel() else torch.cat([fused, mask], dim=1)


class GatedPriorFusion3D(nn.Module):
    """Use mask volumes to gate early 3D image/Wiener channels."""

    def __init__(self, image_channels: int, mask_channels: int):
        super().__init__()
        if image_channels <= 0 or mask_channels <= 0:
            raise ValueError("image_channels and mask_channels must be positive.")
        self.image_channels = image_channels
        self.mask_channels = mask_channels
        self.gate = nn.Sequential(
            nn.Conv3d(mask_channels, image_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        image = x[:, :self.image_channels]
        mask = x[:, self.image_channels:self.image_channels + self.mask_channels]
        rest = x[:, self.image_channels + self.mask_channels:]
        gate = self.gate(mask)
        fused = image * (1.0 + self.residual_scale * (gate - 0.5))
        return torch.cat([fused, mask, rest], dim=1) if rest.numel() else torch.cat([fused, mask], dim=1)


def build_prior_fusion_2d(
    mode: str,
    image_channels: Optional[int],
    mask_channels: Optional[int],
) -> nn.Module:
    if mode == "none" or not image_channels or not mask_channels:
        return nn.Identity()
    if mode == "gated":
        return GatedPriorFusion2D(image_channels, mask_channels)
    raise ValueError(f"Unknown prior_fusion mode: {mode!r}")


def build_prior_fusion_3d(
    mode: str,
    image_channels: Optional[int],
    mask_channels: Optional[int],
) -> nn.Module:
    if mode == "none" or not image_channels or not mask_channels:
        return nn.Identity()
    if mode == "gated":
        return GatedPriorFusion3D(image_channels, mask_channels)
    raise ValueError(f"Unknown prior_fusion mode: {mode!r}")