"""Split a real acquisition into input and holdout measurement sets.

This is the mechanism that makes the project's self-supervision honest
(plan §5.1)::

    Omega_acq = Omega_in  (disjoint union)  Omega_hold

``Omega_hold`` holds positions that were **actually measured**, so the training
target is a real, noisy measurement -- never a dense reference and never a
zero-filled hole labelled as truth. The network receives only ``Omega_in``.

Three holdout geometries are supported because they are not equivalent:

``row``     whole measured rows. Closest to the deployment geometry (a missing
            row between measured rows), and the hardest for a network that
            could otherwise interpolate along X.
``plane``   whole measured planes. Matches depth interpolation and is the case
            where the model might cheat via neighbouring slices.
``scatter`` isolated voxels. Easiest; useful as an upper bound on performance
            and as a leakage canary -- if scatter is much better than row,
            the network is exploiting lateral continuity rather than learning
            to reconstruct a gap.

``ratio`` is the fraction of ``Omega_acq`` that is withheld. Guard rails: the
split raises rather than returning an empty input or holdout set, because a
silently empty set would turn the loss into a constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["HOLDOUT_GEOMETRIES", "HoldoutSpec", "split_holdout"]

HOLDOUT_GEOMETRIES = ("row", "plane", "scatter")


@dataclass(frozen=True)
class HoldoutSpec:
    """How to carve the holdout set out of an acquisition mask."""

    ratio: float = 0.5
    geometry: str = "row"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.geometry not in HOLDOUT_GEOMETRIES:
            raise ValueError(f"unknown holdout geometry {self.geometry!r}; "
                             f"expected {HOLDOUT_GEOMETRIES}")
        if not 0.0 < self.ratio < 1.0:
            raise ValueError(f"holdout ratio must be in (0, 1), got {self.ratio}")

    def to_dict(self) -> dict:
        return {"ratio": self.ratio, "geometry": self.geometry, "seed": self.seed}


def split_holdout(mask: torch.Tensor, spec: HoldoutSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a measurement mask into ``(mask_in, mask_hold)``.

    Both returned masks are bool and satisfy
    ``mask_in & mask_hold == 0`` and ``mask_in | mask_hold == mask``.
    """
    mask = mask.bool()
    if not bool(mask.any()):
        raise ValueError("holdout split on an empty acquisition mask")
    dev = mask.device

    if spec.geometry == "scatter":
        flat_in = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
        n = flat_in.numel()
        n_hold = int(round(spec.ratio * n))
        n_hold = min(max(n_hold, 1), n - 1) if n > 1 else 0
        if n_hold == 0:
            raise ValueError("acquisition mask has too few measured voxels to split")
        g = torch.Generator(device="cpu").manual_seed(spec.seed)
        perm = torch.randperm(n, generator=g)[:n_hold].to(dev)
        hold_flat = torch.zeros(mask.numel(), dtype=torch.bool, device=dev)
        hold_flat[flat_in[perm]] = True
        mask_hold = hold_flat.view(mask.shape)
        return mask & ~mask_hold, mask_hold

    # row / plane: withhold whole structural units that carry measurements.
    unit = 1 if spec.geometry == "row" else 0      # dim to reduce over
    dims = tuple(i for i in range(3) if i != unit)
    if spec.geometry == "row":
        present = mask.any(dim=(0, 2))             # [H]
    else:
        present = mask.any(dim=(1, 2))             # [Z]

    idx = torch.nonzero(present, as_tuple=False).flatten()
    n = idx.numel()
    if n < 2:
        raise ValueError(
            f"holdout geometry {spec.geometry!r} needs >=2 measured "
            f"{'rows' if spec.geometry == 'row' else 'planes'}, found {n}; "
            "reduce the sampling sparsity or use scatter geometry")

    n_hold = int(round(spec.ratio * n))
    n_hold = min(max(n_hold, 1), n - 1)
    g = torch.Generator(device="cpu").manual_seed(spec.seed)
    perm = torch.randperm(n, generator=g)[:n_hold].to(dev)
    chosen = torch.zeros_like(present)
    chosen[idx[perm]] = True

    if spec.geometry == "row":
        mask_hold = mask & chosen.view(1, -1, 1)
    else:
        mask_hold = mask & chosen.view(-1, 1, 1)
    return mask & ~mask_hold, mask_hold
