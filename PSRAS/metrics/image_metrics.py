"""Image reconstruction metrics (PLAN.md §9.1).

PSNR / SSIM / MS-SSIM / MAE, computed per sample and aggregated later by
metrics/statistics.py (paired by sample id, never by row order).
"""

from __future__ import annotations

import torch

__all__ = ["psnr", "ssim", "mae", "split_errors"]


def psnr(pred: torch.Tensor, target: torch.Tensor, dim: tuple | None = None) -> torch.Tensor:
    mse = ((pred - target) ** 2).mean(dim=dim or tuple(range(pred.ndim)))
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))


def mae(pred: torch.Tensor, target: torch.Tensor, dim: tuple | None = None) -> torch.Tensor:
    return (pred - target).abs().mean(dim=dim or tuple(range(pred.ndim)))


def ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Global SSIM for [B, 1, H, W]-style tensors (broadcast over leading dims)."""
    from losses.reconstruction import _ssim_components, _gaussian_window
    x, y = pred, target
    if x.ndim == 3:
        x, y = x[None, None], y[None, None]
    elif x.ndim == 5:
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, 1, *x.shape[-2:])
        y = y.permute(0, 2, 1, 3, 4).reshape(-1, 1, *y.shape[-2:])
    win = _gaussian_window(11, 1.5, x.device)
    l, cs = _ssim_components(x, y, win)
    return (l * cs).mean(dim=(1, 2, 3)).reshape(pred.shape[0], -1).mean(dim=1).mean()


def split_errors(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Measured / missing split errors (PLAN.md §9.2).

    E_measured : error on positions with measurement support
    E_missing  : error on positions without support (pure inference)
    Tensors are [T, H, W] (or [B, T, H, W] with mask aligned).
    """
    m = mask.bool()
    e_meas = ((pred - target).abs() * m).sum() / m.sum().clamp(min=1)
    e_miss = ((pred - target).abs() * ~m).sum() / (~m).sum().clamp(min=1)
    return float(e_meas), float(e_miss)
