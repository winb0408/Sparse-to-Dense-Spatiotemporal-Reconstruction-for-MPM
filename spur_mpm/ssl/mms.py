"""Masked Measurement Self-Supervision (MMS).

The objective (plan §5.1) compares the prediction against the *withheld real
measurements*::

    L_MMS = (1/|Omega_hold|) * sum_{i in Omega_hold} rho(R_hat_i - Y_i)

with ``rho`` a robust penalty by default: blood vessels contain bright specks
and shot noise produces outliers, and an L2 objective spends its gradient on
those instead of on structure. L2 stays available as an explicit ablation.

Because ``Y_hold`` are genuine measurements, this is *not* a noise-free
supervision signal, and the model can legitimately learn to reproduce
measurement noise at held-out voxels. Report that limit honestly: MMS targets
structural consistency at sampled positions, not denoising of those positions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["charbonnier", "MMSLoss"]

_RHO = ("l1", "l2", "charbonnier")


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """``sqrt(x^2 + eps^2) - eps``: smooth-L1 that stays finite at the origin."""
    return torch.sqrt(x * x + eps * eps) - eps


class MMSLoss(nn.Module):
    """Robust reconstruction loss evaluated only on held-out measurements.

    Parameters
    ----------
    rho   : ``l1`` | ``l2`` | ``charbonnier``.
    eps   : Charbonnier smoothing constant.
    dc_weight : optional weight for a data-consistency term on measured input
                positions (``L_DC``). Default 0: the plan's baseline keeps MMS
                alone, and any ``L_DC > 0`` arm is a separate experiment because
                it invites the model to copy noisy inputs rather than reconstruct.
    """

    def __init__(self, rho: str = "charbonnier", eps: float = 1e-3,
                 dc_weight: float = 0.0):
        super().__init__()
        if rho not in _RHO:
            raise ValueError(f"unknown loss rho {rho!r}; expected {_RHO}")
        self.rho = rho
        self.eps = eps
        self.dc_weight = dc_weight

    def _rho(self, d: torch.Tensor) -> torch.Tensor:
        if self.rho == "l1":
            return d.abs()
        if self.rho == "l2":
            return d * d
        return charbonnier(d, self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                hold_mask: torch.Tensor,
                y_in: torch.Tensor | None = None,
                m_in: torch.Tensor | None = None) -> dict:
        """Compute the masked objective.

        pred, target, hold_mask : ``[B, H, W]`` (or ``[H, W]``).
        y_in, m_in              : optional ``[B, H, W]`` measurement and support
                                  for the data-consistency term.
        """
        squeeze = pred.ndim == 2
        if squeeze:
            pred, target, hold_mask = pred[None], target[None], hold_mask[None]
        m = hold_mask.bool()
        n = m.sum().clamp(min=1)
        per = self._rho(pred - target)
        loss = (per * m).sum() / n

        out = {"loss": loss, "mms": loss.detach(),
               "n_hold": float(m.sum().item())}

        if self.dc_weight > 0:
            if y_in is None or m_in is None:
                raise ValueError("dc_weight > 0 requires y_in and m_in")
            mi = m_in.bool()
            dc = ((pred - y_in).abs() * mi).sum() / mi.sum().clamp(min=1)
            out["dc"] = dc.detach()
            out["loss"] = loss + self.dc_weight * dc
        return out
