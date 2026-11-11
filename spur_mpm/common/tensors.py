"""Small tensor helpers shared by train / evaluate / inference."""

from __future__ import annotations

import torch

__all__ = ["select_centre", "as_reconstruction"]


def as_reconstruction(out):
    """Get the predicted volume out of a model output.

    Legacy arms return a tensor; the PR2 model returns the plan's dict
    (``reconstruction_raw`` / ``reconstruction`` / ``reconstruction_uncertainty``
    / ``aux``). One helper, so training, evaluation and inference cannot
    disagree about which key is the prediction -- they did have to agree before
    the dict existed, and now they still do.
    """
    if isinstance(out, dict):
        for key in ("reconstruction", "reconstruction_raw"):
            if key in out and out[key] is not None:
                return out[key]
        raise KeyError(f"model output dict has neither 'reconstruction' nor "
                       f"'reconstruction_raw': {sorted(out)}")
    return out


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
