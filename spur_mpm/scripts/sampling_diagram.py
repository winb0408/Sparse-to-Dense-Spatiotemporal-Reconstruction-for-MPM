"""Draw what the sparse acquisition actually keeps, and what training keeps of that.

The qualitative figures show the support that was used for a *specific* volume,
which is the right thing to look at when reading a result but the wrong thing to
look at when asking "how is the sparse pattern built". This script answers the
second question directly from the sampling code, with no data and no model:

* the acquisition mask ``M`` -- every ``k_y``-th complete row of y, x untouched;
* the self-supervised training mask -- half of the measured rows withheld again,
  so the network never sees the targets it is scored on;
* the plane density, because ``kz=1`` means z is never subsampled.

Usage::

    python -m spur_mpm.scripts.sampling_diagram --out spur_mpm/figures/qualitative
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from spur_mpm.physics.sampling import SamplingSpec, make_mask
from spur_mpm.ssl.holdout import HoldoutSpec, split_holdout

ACQ = "#1f6fb4"
KEPT = "#e8e8e8"
HOLD = "#e8a33d"


def _strip(ax, mask2d, title, note):
    """Draw a y-x patch: columns are x, rows are y, kept voxels are light."""
    h, w = mask2d.shape
    img = np.ones((h, w, 3))
    img[:, :, 0] = np.where(mask2d, 0.91, 0.12)
    img[:, :, 1] = np.where(mask2d, 0.91, 0.44)
    img[:, :, 2] = np.where(mask2d, 0.91, 0.71)
    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x  (never subsampled)")
    ax.set_ylabel("y")
    ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, family="monospace",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="spur_mpm/figures/qualitative")
    ap.add_argument("--ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rows", type=int, default=32, help="y rows to draw")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    shape = (4, args.rows * 2, 8)          # z, y, x -- only the y/x slab is drawn
    densities = (2, 4)
    fig, axes = plt.subplots(len(densities), 3, figsize=(11.5, 6.4))

    for r, ky in enumerate(densities):
        spec = SamplingSpec(mode="y", ky=ky, kz=1)
        acq = make_mask(spec, shape, phase=(0, 0, 0)).bool()
        m_in, m_hold = split_holdout(
            acq, HoldoutSpec(ratio=args.ratio, geometry="row", seed=args.seed))
        slab = (slice(0, args.rows), slice(None))   # (y, x) at the first z plane

        _strip(axes[r, 0], acq[0][slab].numpy(),
               f"k_y={ky}: acquisition mask  (evaluation input)",
               f"kept {acq.float().mean():.3f} of voxels")
        _strip(axes[r, 1], m_in[0][slab].numpy(),
               f"k_y={ky}: what the self-supervised arm sees in training",
               f"kept {m_in.float().mean():.3f} of voxels")
        _strip(axes[r, 2], m_hold[0][slab].numpy(),
               f"k_y={ky}: withheld rows  (these are the training targets)",
               f"{m_hold.float().mean():.3f} of voxels, all measured")

        for c in range(3):
            ax = axes[r, c]
            ax.set_yticks([0, ky, 2 * ky, 3 * ky][:4])
            ax.set_xticks([])

    fig.suptitle(
        "Two levels of sparsity: every k_y-th complete row is acquired "
        "(x stays dense, z is never subsampled),\n"
        "then half of those rows are withheld from the network so its targets "
        "are still real measurements", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = out_dir / "sampling_diagram.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    print(f"wrote {path}")
    for ky in densities:
        spec = SamplingSpec(mode="y", ky=ky, kz=1)
        acq = make_mask(spec, shape, phase=(0, 0, 0)).bool()
        m_in, m_hold = split_holdout(
            acq, HoldoutSpec(ratio=args.ratio, geometry="row", seed=args.seed))
        print(f"k_y={ky}: acq {acq.float().mean():.3f} "
              f"-> train input {m_in.float().mean():.3f} "
              f"+ withheld {m_hold.float().mean():.3f}")


if __name__ == "__main__":
    main()
