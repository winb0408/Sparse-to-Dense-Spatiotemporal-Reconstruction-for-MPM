"""Complementary split: two disjoint halves of the *same* acquisition.

MMS (``spur_mpm.ssl.mms``) withholds half of the measured rows and trains on the
other half, so one branch uses at most 50% of what was acquired. That is the
objection the dual-branch arm (E012) answers: run two branches on **disjoint**
halves and make each one reconstruct the other's rows. Then

* every measured voxel is simultaneously one branch's input and the other
  branch's training target;
* the pair as a whole uses 100% of the acquisition;
* and no dense reference is ever read, because the targets are real
  measurements either way.

What changes is only *whose* rows a branch may look at, which is what
``split_complement`` decides. It is the same row/plane partitioning as
``split_holdout`` with the "kept" half also returned, and the two halves are
guaranteed disjoint and jointly equal to the input mask -- so the union carries
exactly the acquisition and nothing else.
"""

from __future__ import annotations

import torch

from spur_mpm.ssl.holdout import HoldoutSpec

__all__ = ["split_complement"]


def split_complement(mask: torch.Tensor,
                     spec: HoldoutSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition ``mask`` into two disjoint, non-empty halves.

    Parameters
    ----------
    mask : ``[Z, H, W]`` boolean acquisition mask.
    spec : ``ratio`` is the share of measured units going to the first half
           (0.5 gives an even split), ``geometry`` picks the unit and ``seed``
           fixes the partition.

    Returns
    -------
    ``(mask_a, mask_b)`` with ``mask_a & mask_b`` empty and
    ``mask_a | mask_b == mask``.
    """
    mask = mask.bool()
    if not bool(mask.any()):
        raise ValueError("complement split on an empty acquisition mask")
    if not 0.0 < spec.ratio < 1.0:
        raise ValueError(f"complement ratio must be in (0, 1), got {spec.ratio}")
    if spec.geometry not in ("row", "plane"):
        raise ValueError("complement split needs geometry 'row' or 'plane', got "
                         f"{spec.geometry!r}")

    dev = mask.device
    if spec.geometry == "row":
        present = mask.any(dim=(0, 2))                  # [H]
    else:
        present = mask.any(dim=(1, 2))                  # [Z]
    idx = torch.nonzero(present, as_tuple=False).flatten()
    n = idx.numel()
    if n < 2:
        raise ValueError(f"complement split needs at least 2 measured units, "
                         f"got {n}")

    n_a = min(max(int(round(spec.ratio * n)), 1), n - 1)
    g = torch.Generator(device="cpu").manual_seed(spec.seed)
    perm = torch.randperm(n, generator=g)[:n_a].to(dev)

    take = torch.zeros_like(present)
    take[idx[perm]] = True
    if spec.geometry == "row":
        mask_a = mask & take.view(1, -1, 1)
    else:
        mask_a = mask & take.view(-1, 1, 1)
    return mask_a, mask & ~mask_a
