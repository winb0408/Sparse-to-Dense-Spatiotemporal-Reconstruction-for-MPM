"""Unified v4 reconstruction network (Phase 4, PLAN.md §2 Phase 4 / v4-draft §7).

Shared Spatial Backbone + Axis-specific Interaction Adapter:

    [B, T*5, H, W]  (feature-major: y, m, x0, c_s, u_p per slice)
        |
    per-slice shared fusion stem  (PriorFusion per slice, weights shared
        |                         across slices -- T-agnostic)
    [B*T, base, H, W]
        |
    A_Z: zero-init residual axial adapter (depthwise Conv3d along T)
        |   F + Conv3d_{(3,1,1)}(F), weights zero-init -> exact identity at
        |   start; learns depth interaction only where it pays (v4-draft §7.5)
    [B*T, base, H, W]
        |
    per-slice shared SwinUNet2D (in=base, out=1)
        |
    [B, T, H, W]

Why per-slice (not Conv2d(5T->C)): the encoder must be shared across Y
(T=1), YZ (T=2kz-1) and later XYZ/XYT regimes with different T.  The v3
full-3D / all-layer axial variants were preregistered-falsified (EXP-012/013/
014, DEC-019), so learned axial interaction is confined to one zero-init
adapter between stem and UNet and its increment is measured by ablation
(U vs U-noAZ, EXP-025).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from models.swin_unet_2d import SwinUNet2D

__all__ = ["UnifiedV4"]


class _PerSliceFusionStem(nn.Module):
    """PriorFusionStem applied per slice with shared weights.

    Same gating form as models/prior_fusion.py (f_y + g * f_w, gate
    zero-initialised) but slice-independent, so any T is accepted.
    Input [B, T*5, H, W] (feature-major) -> [B*T, base, H, W].
    """

    def __init__(self, out_ch: int):
        super().__init__()
        self.e_y = nn.Conv2d(1, out_ch, 3, padding=1)
        self.e_w = nn.Conv2d(1, out_ch, 3, padding=1)
        self.e_s = nn.Conv2d(3, out_ch, 3, padding=1)
        self.gate = nn.Conv2d(2 * out_ch, out_ch, 3, padding=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 5, T, H, W] after feature-major view
        B, _, T, H, W = x.shape
        y = x[:, 0].reshape(B * T, 1, H, W)
        x0 = x[:, 2].reshape(B * T, 1, H, W)
        cond = x[:, [1, 3, 4]].permute(0, 2, 1, 3, 4).reshape(B * T, 3, H, W)
        f_y, f_w, f_s = self.e_y(y), self.e_w(x0), self.e_s(cond)
        g = torch.sigmoid(self.gate(torch.cat([f_s, f_w], dim=1)))
        return f_y + g * f_w


class UnifiedV4(nn.Module):
    """T-agnostic unified network with optional zero-init axial adapter.

    Parameters
    ----------
    feats_per_slice : int (5 for the full b4f feature set)
    base, window    : SwinUNet2D width / attention window.
    use_az          : enable the A_Z adapter (False = U-noAZ ablation arm).
    """

    def __init__(self, feats_per_slice: int = 5, base: int = 48, window: int = 8,
                 use_az: bool = True):
        super().__init__()
        assert feats_per_slice == 5, "UnifiedV4 expects the full 5-feature set"
        self.use_az = use_az
        self.stem = _PerSliceFusionStem(base)
        if use_az:
            # depthwise temporal mixing, zero-init -> identity at start
            self.az = nn.Conv3d(base, base, (3, 1, 1), padding=(1, 0, 0),
                                groups=base)
            nn.init.zeros_(self.az.weight)
            nn.init.zeros_(self.az.bias)
        self.unet = SwinUNet2D(base, 1, base=base, window=window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T*5, H, W], feature-major (DEC-013)
        B, C, H, W = x.shape
        T = C // 5
        v = x.view(B, 5, T, H, W)
        f = self.stem(v)                                  # [B*T, base, H, W]
        if self.use_az and T > 1:
            # [B*T, C, H, W] -> [B, C, T, H, W] for the temporal conv
            g = f.view(B, T, -1, H, W).permute(0, 2, 1, 3, 4)
            g = g + self.az(g)
            f = g.permute(0, 2, 1, 3, 4).reshape(B * T, -1, H, W)
        # Per-slice UNet activation memory scales with B*T (up to 8*15=120
        # slices); gradient-checkpoint the UNet body in training mode so the
        # mixed Y(T=1)+YZ(T<=15) batches fit 48 GB (eval skips it).
        if self.training and f.requires_grad:
            out = torch.cat([cp.checkpoint(self.unet, f[i:i + 8])
                             for i in range(0, f.shape[0], 8)], dim=0)
        else:
            out = self.unet(f)                            # [B*T, 1, H, W]
        return out.view(B, T, H, W)
