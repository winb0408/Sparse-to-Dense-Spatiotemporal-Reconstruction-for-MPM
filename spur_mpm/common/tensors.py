"""Small tensor helpers shared by train / evaluate / inference."""

from __future__ import annotations

import torch

__all__ = ["select_centre"]


def select_centre(pred: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    """Gather the query plane from a ``[B, D, H, W]`` model output.

    The query plane index is per-item because windows at the volume borders are
    clamped, so the query is not always ``D // 2``.
    """
    b, d = pred.shape[0], pred.shape[1]
    idx = center.to(pred.device).long().view(b, 1, 1, 1)
    if int(idx.max()) >= d or int(idx.min()) < 0:
        raise IndexError(f"centre index out of range for depth {d}")
    idx = idx.expand(b, 1, *pred.shape[2:])
    return torch.gather(pred, 1, idx).squeeze(1)
