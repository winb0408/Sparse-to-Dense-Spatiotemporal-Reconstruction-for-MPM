"""Unified axial-3D network (Phase 3, DESIGN_phase3.md).

One weight set covers 2D (T=1) and 2.5D (T=2K-1): spatial Swin window
attention operates per frame, and a zero-initialised axial attention
branch mixes information along z.  The axial branch is an exact no-op at
initialisation for ANY T (zero-init output projection), which makes the
T=1 degradation exact by construction rather than by hope.

Input layout convention follows DEC-013: channels are feature-major,
[B, F*T, H, W] with feature blocks [y(t0..t), mask(t0..t), x0(t0..t), ...].
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from models.attention import WindowAttention3D

__all__ = ["AxialAttention", "PriorFusionStem3D", "SwinUNet3D", "Unified3D"]


class AxialAttention(nn.Module):
    """Self-attention along z at every (H, W) location.

    x: [B, C, T, H, W] -> [B, C, T, H, W].  Tokens along T are attended
    globally (T <= ~16, so full attention is cheap).  A sinusoidal z
    encoding is added to the tokens; the output branch is zero-initialised
    so the block starts as an exact no-op regardless of T.
    """

    def __init__(self, dim: int, heads: int = 4, max_t: int = 32):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.norm = nn.LayerNorm(dim)
        pe = torch.zeros(max_t, dim)
        pos = torch.arange(max_t).float()[:, None]
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("z_enc", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        if T == 1:                       # exact identity shortcut
            return x
        # [B, T, H*W, C] tokens; every (H, W) position attends over T.
        v = x.permute(0, 2, 3, 4, 1) + self.z_enc[:T].view(1, T, 1, 1, C)
        v = v.reshape(B, T, H * W, C).permute(0, 2, 1, 3).reshape(B * H * W, T, C)
        h = self.norm(v)
        qkv = self.qkv(h).reshape(B * H * W, T, 3, self.heads, C // self.heads)
        q, k, w = qkv.permute(2, 0, 3, 1, 4).unbind(0)      # [BHW, heads, T, d]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = (attn.softmax(dim=-1) @ w)                    # [BHW, heads, T, d]
        out = self.proj(out.transpose(1, 2).reshape(B * H * W, T, C))
        out = out.view(B, H * W, T, C).permute(0, 3, 2, 1).reshape(B, C, T, H, W)
        return x + out


class _Block3D(nn.Module):
    """Spatial window attention + axial attention + MLP, all residual.

    use_axial=False makes the block purely per-frame 2D (DEC-017 rollback:
    axial attention only at the bottleneck, encoder/decoder stay swin2d-like
    for stable optimisation under query-sparse supervision).

    window_t > 1 switches the block to a FULL 3D shifted-window attention
    (Video-Swin style, EXP-018): one WindowAttention3D with temporal window
    extent wt covers (z, y, x) jointly -- no separate axial branch, no
    zero-initialisation, so 3D mixing is active from step 0 and the SW-MSA
    mask (incl. the z shift) is the standard Video-Swin construction.
    """

    def __init__(self, dim: int, heads: int, window: int, shifted: bool,
                 heads_axial: int = 4, use_checkpoint: bool = True,
                 use_axial: bool = True, window_t: int = 1):
        super().__init__()
        # Gradient checkpointing: the 3D attention stack at T=15 x 256^2 is
        # memory-heavy; blocks are dropout-free so recompute is deterministic.
        self.use_checkpoint = use_checkpoint
        if window_t > 1:
            self.spatial = WindowAttention3D(dim, heads, (window_t, window), shifted)
            self.axial = None
        else:
            # wt=1: per-frame spatial windows (no temporal mixing here).
            self.spatial = WindowAttention3D(dim, heads, (1, window), shifted)
            self.axial = AxialAttention(dim, heads_axial) if use_axial else None
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))

    def _core(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.spatial(x)
        if self.axial is not None:
            x = x + self.axial(x)
        B, C, T, H, W = x.shape
        y = x.flatten(3).transpose(1, 3).reshape(B * T, H * W, C)
        x = x + self.mlp(self.norm2(y)).reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training and x.requires_grad:
            return cp.checkpoint(self._core, x, use_reentrant=False)
        return self._core(x)


class _Stage3D(nn.Module):
    def __init__(self, dim: int, heads: int, window: int, depth: int,
                 use_axial: bool = True, window_t: int = 1):
        super().__init__()
        self.blocks = nn.ModuleList([
            _Block3D(dim, heads, window, shifted=(i % 2 == 1), use_axial=use_axial,
                     window_t=window_t)
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class _Down3D(nn.Module):
    """Spatial-only downsampling (z is never halved -- anisotropy-aware)."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, (1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        return self.conv(x)


class _Up3D(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.ConvTranspose3d(c_in, c_out, (1, 2, 2), stride=(1, 2, 2))

    def forward(self, x, skip):
        x = self.conv(x)
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return x + skip


class PriorFusionStem3D(nn.Module):
    """3D prior-fusion stem (Phase 3 upgrade of models/prior_fusion.py).

    Input [B, F*T, H, W] feature-major; output [B, base, T, H, W].  Same
    three-stream gating as the 2D stem, with T-shared 3D convs; gate is
    zero-initialised (starts at g = 0.5, see DEC-013 note).
    """

    def __init__(self, in_ch: int, t: int, base: int):
        super().__init__()
        c = in_ch // t
        assert c == 5, f"expected 5 features per slice, got {c}"
        self.t = t
        self.e_y = nn.Conv3d(1, base, (1, 3, 3), padding=(0, 1, 1))
        self.e_w = nn.Conv3d(1, base, (1, 3, 3), padding=(0, 1, 1))
        self.e_s = nn.Conv3d(3, base, (1, 3, 3), padding=(0, 1, 1))
        self.gate = nn.Conv3d(2 * base, base, (1, 3, 3), padding=(0, 1, 1))
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        v = x.view(B, 5, self.t, H, W)
        f_y = self.e_y(v[:, 0:1])
        f_w = self.e_w(v[:, 2:3])
        f_s = self.e_s(v[:, [1, 3, 4]])
        g = torch.sigmoid(self.gate(torch.cat([f_s, f_w], dim=1)))
        return f_y + g * f_w


class SwinUNet3D(nn.Module):
    """U-shaped 3D Swin network for [B, in_ch, T, H, W] -> [B, 1, T, H, W].

    Spatial dims are downsampled; T passes through unchanged.
    """

    def __init__(self, in_ch: int, base: int = 48,
                 depths: tuple[int, ...] = (2, 2, 2),
                 heads: tuple[int, ...] = (4, 8, 16),
                 window: int = 8, n_levels: int = 3,
                 bottleneck_depth: int = 1,
                 axial_at: str = "all", window_t: int = 1):
        super().__init__()
        assert axial_at in ("all", "bottleneck")
        assert len(depths) == len(heads) == n_levels
        # bottleneck_depth=1: the axial branch adds ~4C^2 per block; losing
        # one 192-dim bottleneck block keeps the parameter budget within
        # ~10% of the 2D B4-equivalent (fair comparison,
        # test_param_count_parity_with_2d).
        self.stem = nn.Conv3d(in_ch, base, (1, 3, 3), padding=(0, 1, 1))
        dims = [base * (2 ** i) for i in range(n_levels)]
        enc_axial = axial_at == "all"
        self.encoders = nn.ModuleList(
            [_Stage3D(dims[i], heads[i], window, depths[i], use_axial=enc_axial,
                      window_t=window_t)
             for i in range(n_levels)])
        self.downs = nn.ModuleList([_Down3D(dims[i], dims[i + 1])
                                    for i in range(n_levels - 1)])
        self.bottleneck = _Stage3D(dims[-1], heads[-1], window, bottleneck_depth,
                                   use_axial=True, window_t=window_t)
        self.ups = nn.ModuleList([_Up3D(dims[n_levels - 1 - i], dims[n_levels - 2 - i])
                                  for i in range(n_levels - 1)])
        self.decoders = nn.ModuleList(
            [_Stage3D(dims[n_levels - 2 - i], heads[n_levels - 2 - i], window,
                      depths[n_levels - 2 - i], use_axial=enc_axial,
                      window_t=window_t)
             for i in range(n_levels - 1)])
        self.head = nn.Conv3d(base, 1, (1, 3, 3), padding=(0, 1, 1))

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


class Unified3D(nn.Module):
    """End-to-end wrapper: flat [B, F*T, H, W] input -> [B, T, H, W] output.

    fusion=False concatenates all feature channels in the stem conv
    (B3-style); fusion=True routes them through PriorFusionStem3D (B4-style).
    """

    def __init__(self, features_per_slice: int, t: int, base: int = 48,
                 window: int = 8, fusion: bool = True, axial_at: str = "all",
                 output: str = "all", window_t: int = 1):
        super().__init__()
        assert output in ("all", "center")
        self.output = output
        self.t = t
        unet_in = base if fusion else features_per_slice
        self.fusion = fusion
        if fusion:
            assert features_per_slice == 5
            self.stem = PriorFusionStem3D(features_per_slice * t, t, base)
        else:
            self.stem = None
        self.unet = SwinUNet3D(unet_in, base=base, window=window, axial_at=axial_at,
                               window_t=window_t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        f = C // self.t
        v = x.view(B, f, self.t, H, W)
        if self.fusion:
            h = self.stem(x)
        else:
            # channels = features, depth = T: the (1,3,3) stem conv is
            # per-slice spatial mixing of that slice's features (B3-equivalent).
            h = v
        raw = self.unet(h)                 # [B, 1, T, H, W]
        if self.output == "center":
            # DEC-018: supervise/emit only the query (center) slice -- the
            # trunk (incl. bottleneck axial) still sees all T frames, but the
            # full head capacity is devoted to the single output, matching
            # swin2d's 2.5D interface.  Other frames are filled with the
            # center prediction so the [B, T, H, W] contract holds.
            c = raw[:, 0, self.t // 2]                     # [B, H, W]
            return c.unsqueeze(1).expand(-1, self.t, -1, -1)
        return raw.squeeze(1)              # [B, T, H, W]
