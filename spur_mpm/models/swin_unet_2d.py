"""Fixed 2D Swin-UNet reconstruction carrier.

Adapted from PSRAS/models/swin_unet_2d.py. It is the shared backbone for every
trained arm (S*/SS*), so architecture is never a confound in the ablations.
For 2.5-D, neighbouring planes are concatenated as input channels and the loss
is taken on the centre plane.

Differences from the legacy implementation:
- standard SW-MSA with correct region masks (models/attention.py);
- no DINO / no GT-dependent logic anywhere;
- supports ablation-driven input channel count (B0..B6 input flags).

For the 2.5D regime, neighbouring slices are concatenated as input
channels and the model outputs one channel per slice; the query mask Q
selects which output slices enter the loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spur_mpm.models.attention import WindowAttention2D

__all__ = ["SwinUNet2D"]


class _Block(nn.Module):
    """Pre-norm window-attention + MLP residual block."""

    def __init__(self, dim: int, heads: int, window: int, shifted: bool):
        super().__init__()
        self.attn = WindowAttention2D(dim, heads, window, shifted)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]; WindowAttention handles norm internally for attn,
        # but we apply the residual structure here.
        h = self.attn(x)
        x = x + h
        B, C, H, W = x.shape
        y = x.flatten(2).transpose(1, 2)
        x = x + self.mlp(self.norm2(y)).transpose(1, 2).reshape(B, C, H, W)
        return x


class _Stage(nn.Module):
    def __init__(self, dim: int, heads: int, window: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            _Block(dim, heads, window, shifted=(i % 2 == 1)) for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class _Down(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in * 4, c_out, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 4, 2, 5)
        x = x.reshape(B, C * 4, H // 2, W // 2)
        return self.conv(x)


class _Up(nn.Module):
    """Upsample to c_out, fuse with the skip connection, project back to c_out."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(c_in, c_out, 2, stride=2)
        self.fuse = nn.Conv2d(c_out * 2, c_out, 1)

    def forward(self, x, skip):
        x = self.conv(x)
        return self.fuse(torch.cat([x, skip], dim=1))


class SwinUNet2D(nn.Module):
    """U-shaped Swin network for [B, in_ch, H, W] -> [B, out_ch, H, W].

    H, W should be divisible by 2^n_levels (attention pads internally, but
    the merging path needs divisibility).
    """

    def __init__(self, in_ch: int, out_ch: int, base: int = 48,
                 depths: tuple[int, ...] = (2, 2, 2), heads: tuple[int, ...] = (4, 8, 16),
                 window: int = 8, n_levels: int = 3):
        super().__init__()
        assert len(depths) == len(heads) == n_levels
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        dims = [base * (2 ** i) for i in range(n_levels)]
        self.encoders = nn.ModuleList(
            [_Stage(dims[i], heads[i], window, depths[i]) for i in range(n_levels)])
        self.downs = nn.ModuleList([_Down(dims[i], dims[i + 1])
                                    for i in range(n_levels - 1)])
        self.bottleneck = _Stage(dims[-1], heads[-1], window, depths[-1])
        self.ups = nn.ModuleList([_Up(dims[n_levels - 1 - i], dims[n_levels - 2 - i])
                                  for i in range(n_levels - 1)])
        self.decoders = nn.ModuleList(
            [_Stage(dims[n_levels - 2 - i], heads[n_levels - 2 - i], window,
                    depths[n_levels - 2 - i]) for i in range(n_levels - 1)])
        self.head = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x)
            if i < len(self.downs):
                skips.append(x)
                x = self.downs[i](x)
        x = self.bottleneck(x)
        for i, up in enumerate(self.ups):
            x = up(x, skips.pop())
            x = self.decoders[i](x)
        return self.head(x)
