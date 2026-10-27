"""Leakage guards for the self-supervised path.

These are cheap assertions that run inside the training loop, not just in
tests, because the failure they catch is silent: a model that has seen the
target during training still produces a plausible-looking loss curve.

The properties checked here mirror the checklist in plan §5.2.
"""

from __future__ import annotations

import torch

__all__ = ["LeakageError", "assert_split_disjoint", "assert_holdout_is_measured",
           "assert_inputs_blind"]


class LeakageError(RuntimeError):
    """Raised when the self-supervised split violates its contract."""


def assert_split_disjoint(mask_in: torch.Tensor, mask_hold: torch.Tensor) -> None:
    """``Omega_in`` and ``Omega_hold`` must not overlap."""
    overlap = (mask_in.bool() & mask_hold.bool()).sum().item()
    if overlap:
        raise LeakageError(f"{overlap} voxels are in both the input and the "
                           "holdout set")


def assert_holdout_is_measured(hold_mask: torch.Tensor,
                               acq_mask: torch.Tensor) -> None:
    """Holdout targets must be positions that were really acquired.

    Guards against the classic bug of "holding out" unmeasured positions and
    then training against their zero fill, which teaches the network to output
    zeros in gaps and reports a beautiful loss.
    """
    bad = (hold_mask.bool() & ~acq_mask.bool()).sum().item()
    if bad:
        raise LeakageError(f"{bad} holdout voxels were never measured; the "
                           "target would be a zero fill, not a measurement")
    if hold_mask.bool().sum().item() == 0:
        raise LeakageError("holdout set is empty")


def assert_inputs_blind(y_in: torch.Tensor, hold_mask: torch.Tensor,
                        atol: float = 0.0) -> None:
    """The network input must be zero at every held-out position."""
    residual = y_in.float().abs() * hold_mask.bool().float()
    worst = float(residual.max().item()) if residual.numel() else 0.0
    if worst > atol:
        raise LeakageError(f"input carries signal at held-out positions "
                           f"(max |y_in| = {worst:.3e}); the holdout leaked into "
                           "the input")
