"""Sampling mask generation (measurement support M).

M(x) in {0,1} answers "was this position physically measured?" -- it is NOT
an uncertainty field (PLAN.md DEC-002 / §5 naming).

Modes
-----
- xy  : every slice keeps a K-spaced grid of rows/columns (pattern="grid")
        or a seeded uniform set of pixels (pattern="random").
- y   : every slice keeps every K-th full row (slow-axis line skipping --
        the physically meaningful lateral protocol for point-scanning 3PM,
        where acquisition time ~ N_y and in-line X pixels are nearly free).
- z   : every K-th slice is measured, others fully missing.
- yz  : axial plane skipping combined with Y line skipping
        (acquisition time ~ K * K relative to full sampling).
- xyz : axial gaps combined with lateral grid sparsity.
"""

from __future__ import annotations

import torch

__all__ = ["make_mask"]


def _xy_mask_2d(H: int, W: int, K: int, pattern: str,
                generator: torch.Generator | None,
                phase_y: int = 0, phase_x: int = 0) -> torch.Tensor:
    """Lateral mask with stack-frame phase: global row/col r is measured iff
    (r - phase) % K == 0, where phase is the window origin's offset (crop
    origin % K). Keeps the sampled lattice aligned across windows of a stack
    (structured patterns only; random ignores phase)."""
    rows = (torch.arange(H) + phase_y) % K == 0
    cols = (torch.arange(W) + phase_x) % K == 0
    if pattern == "grid":
        m = rows.view(H, 1) | cols.view(1, W)
        return m
    if pattern == "rows":
        return rows.view(H, 1).expand(H, W)
    if pattern == "random":
        keep = 1.0 / (K * K)
        return torch.rand(H, W, generator=generator) < keep
    raise ValueError(f"unknown xy pattern: {pattern}")


def make_mask(mode: str, K: int, T: int, H: int, W: int,
              pattern: str = "grid", seed: int = 0,
              phase_t: int = 0, phase_y: int = 0, phase_x: int = 0) -> torch.Tensor:
    """Build the measurement support mask.

    phase_t/y/x anchor the sampling lattice to the stack frame: a global
    slice/row/col g is measured iff (g - phase) % K == 0, where phase is the
    window origin's global index modulo K. With all phases 0 the mask equals
    the legacy window-aligned pattern.

    Returns
    -------
    [T, H, W] bool tensor, True = measured.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    if K == 1:
        return torch.ones(T, H, W, dtype=torch.bool)
    generator = torch.Generator().manual_seed(seed)

    if mode == "xy":
        base = _xy_mask_2d(H, W, K, pattern, generator, phase_y, phase_x)
        return base[None].expand(T, H, W).clone()

    if mode == "y":
        base = _xy_mask_2d(H, W, K, "rows", generator, phase_y, phase_x)
        return base[None].expand(T, H, W).clone()

    if mode == "z":
        planes = (torch.arange(T) + phase_t) % K == 0
        return planes.view(T, 1, 1).expand(T, H, W)

    if mode == "yz":
        rows = _xy_mask_2d(H, W, K, "rows", generator, phase_y, phase_x)
        planes = (torch.arange(T) + phase_t) % K == 0
        return planes.view(T, 1, 1) & rows[None]

    if mode == "xyz":
        base = _xy_mask_2d(H, W, K, pattern, generator, phase_y, phase_x)
        planes = (torch.arange(T) + phase_t) % K == 0
        return planes.view(T, 1, 1) & base[None]

    raise ValueError(f"unknown sampling mode: {mode}")
