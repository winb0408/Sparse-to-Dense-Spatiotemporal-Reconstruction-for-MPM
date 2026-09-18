"""Masked measurement-domain data consistency loss (PLAN.md P0-5, M3).

    L_DC = mean | M . (H(X_hat) - Y) |

The comparison happens in the measurement domain: the reconstruction is
pushed through the forward physics (PSF) and must re-explain the actual
observation at measured positions.  Unmeasured positions never enter the
loss (there is no observation there to explain).

Scope (DEC-002 / M3): enable only when the forward model contains a PSF
(or for Protocol-A acquisition-domain data).  For pure masking with an
identity operator this loss is trivially satisfiable on measured pixels
and carries no information -- ``identity_ok=False`` guards against that.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from physics.psf import PSFConfig, gaussian_otf_3d

__all__ = ["DataConsistencyLoss"]


class DataConsistencyLoss(nn.Module):
    """Masked forward-model consistency for [B, T, H, W] or [T, H, W]."""

    def __init__(self, psf: PSFConfig, identity_ok: bool = False):
        super().__init__()
        self.psf = psf
        self.identity_ok = identity_ok
        if psf.is_identity and not identity_ok:
            raise ValueError(
                "DataConsistencyLoss with an identity operator is vacuous for "
                "pure-masking data; enable only with a PSF or acquisition-domain data.")

    def forward(self, pred: torch.Tensor, y: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        squeeze = pred.ndim == 3
        if squeeze:
            pred, y, mask = pred[None], y[None], mask[None]
        if self.psf.is_identity:
            h_pred = pred
        else:
            T, H, W = pred.shape[-3:]
            pf = torch.fft.fftn(pred.float(), dim=(-3, -2, -1))
            otf = gaussian_otf_3d((T, H, W), self.psf, device=pred.device)
            h_pred = torch.fft.ifftn(otf[None] * pf, dim=(-3, -2, -1)).real
        resid = (h_pred - y) * mask
        denom = mask.sum().clamp(min=1.0)
        return resid.abs().sum() / denom
