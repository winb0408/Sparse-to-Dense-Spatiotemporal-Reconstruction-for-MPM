"""Reconstruction losses: L1 + standard MS-SSIM (PLAN.md §5 loss design).

DINO/perceptual terms are deliberately absent (DEC-001).  MS-SSIM follows
Wang et al. 2003: per-scale (luminance, contrast-structure) with the
standard 5-scale weights, gaussian windows, K1=0.01, K2=0.03.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ms_ssim", "ReconstructionLoss"]

_MSSSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _gaussian_window(k: int, sigma: float, device) -> torch.Tensor:
    ax = torch.arange(k, dtype=torch.float32, device=device) - (k - 1) / 2.0
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    return (g / g.sum()).view(1, 1, 1, -1)


def _ssim_components(x: torch.Tensor, y: torch.Tensor, window: torch.Tensor):
    """Per-pixel (luminance*contrast-structure) maps for [B, 1, H, W]."""
    C1, C2 = 0.01**2, 0.03**2
    mu_x = F.conv2d(x, window, padding=window.shape[-1] // 2)
    mu_y = F.conv2d(y, window, padding=window.shape[-1] // 2)
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
    s_x = F.conv2d(x * x, window, padding=window.shape[-1] // 2) - mu_x2
    s_y = F.conv2d(y * y, window, padding=window.shape[-1] // 2) - mu_y2
    s_xy = F.conv2d(x * y, window, padding=window.shape[-1] // 2) - mu_xy
    s_x, s_y, s_xy = s_x.clamp(min=0), s_y.clamp(min=0), s_xy
    l = (2 * mu_xy + C1) / (mu_x2 + mu_y2 + C1)
    cs = (2 * s_xy + C2) / (s_x + s_y + C2)
    return l, cs


def ms_ssim(x: torch.Tensor, y: torch.Tensor, levels: int = 5,
            window_size: int = 11) -> torch.Tensor:
    """Multi-scale SSIM for [B, 1, H, W] (or [B, T, H, W]: averaged over T).

    Standard formulation: prod_i l_i^{w_i} * cs_i^{w_i} per sample, then mean.
    """
    if x.ndim == 3:                       # [N, H, W] -> [N, 1, H, W]
        x, y = x[:, None], y[:, None]
    if x.ndim == 4 and x.shape[1] != 1:
        x = x.mean(dim=1, keepdim=True)
        y = y.mean(dim=1, keepdim=True)
    if x.ndim == 5:
        B = x.shape[0]
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, 1, *x.shape[-2:])
        y = y.permute(0, 2, 1, 3, 4).reshape(-1, 1, *y.shape[-2:])
        return ms_ssim(x, y, levels, window_size).reshape(B, -1).mean(dim=1).mean()
    win = _gaussian_window(window_size, 1.5, x.device)

    # Pad to a multiple of 2^(levels-1) so every downsampling step is clean.
    mult = 2 ** (levels - 1)
    H, W = x.shape[-2:]
    Hp = ((H + mult - 1) // mult) * mult
    Wp = ((W + mult - 1) // mult) * mult
    x = F.pad(x, (0, Wp - W, 0, Hp - H))
    y = F.pad(y, (0, Wp - W, 0, Hp - H))

    vals = []
    for i in range(levels):
        l, cs = _ssim_components(x, y, win)
        vals.append((l.mean((1, 2, 3)), cs.mean((1, 2, 3))))
        if i < levels - 1:
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
    out = None
    for w, (l, cs) in zip(_MSSSIM_WEIGHTS[:levels], vals):
        term = l.clamp(min=0) ** w * cs.clamp(min=0) ** w
        out = term if out is None else out * term
    return out.mean()


class ReconstructionLoss(nn.Module):
    """L = w_l1 * L1 + w_ms * (1 - MS-SSIM).  Default follows PLAN.md §8."""

    def __init__(self, w_l1: float = 1.0, w_ms: float = 0.5, ms_levels: int = 5):
        super().__init__()
        self.w_l1, self.w_ms, self.ms_levels = w_l1, w_ms, ms_levels
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        l1 = self.l1(pred, target)
        ms = ms_ssim(pred, target, self.ms_levels)
        total = self.w_l1 * l1 + self.w_ms * (1.0 - ms)
        return {"loss": total, "l1": l1.detach(), "ms_ssim": ms.detach()}
