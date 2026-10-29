"""Dual-branch complementary reconstruction (arm ``SS1c``, E012).

Two independent copies of the same backbone, one per complementary half of the
acquisition:

    branch A : sees rows in M_a, is trained on the real measurements in M_b
    branch B : sees rows in M_b, is trained on the real measurements in M_a

so each branch is the other's teacher on the rows it was denied, and the pair
uses every measured voxel without ever reading a dense reference. This is the
mutual distillation the complementary design buys: no external label, no
withheld information.

The wrapper is deliberately thin, so that nothing downstream has to know it is
looking at two networks:

* ``forward`` averages the two branches -- the deployment prediction, and the
  one every existing call site (``select_centre``, evaluation, figures) gets;
* ``forward_both`` exposes the branches separately, which is all the training
  objective needs.

Both branches are built from the *same* arm record, so parameter count is
exactly two SS1s; that doubling is the honest price of the design and belongs in
the comparison, not hidden in a wider single net.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ComplementaryDual"]


class ComplementaryDual(nn.Module):
    """Two reconstruction networks on disjoint input channels."""

    def __init__(self, branch_a: nn.Module, branch_b: nn.Module):
        super().__init__()
        self.a = branch_a
        self.b = branch_b

    def split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cut the concatenated input back into the two branch stacks."""
        c = x.shape[1]
        if c % 2:
            raise ValueError(f"dual input needs an even channel count, got {c}")
        h = c // 2
        return x[:, :h], x[:, h:]

    def forward_both(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xa, xb = self.split(x)
        return self.a(xa), self.b(xb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ya, yb = self.forward_both(x)
        if ya.shape != yb.shape:
            raise RuntimeError(f"branch outputs disagree: {tuple(ya.shape)} vs "
                               f"{tuple(yb.shape)}")
        return 0.5 * (ya + yb)
