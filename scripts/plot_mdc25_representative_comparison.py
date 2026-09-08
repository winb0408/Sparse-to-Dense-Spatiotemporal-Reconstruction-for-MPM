#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def robust_minmax(img: np.ndarray) -> Tuple[float, float]:
    # Per-subplot robust contrast stretch for better visual readability.
    vmin, vmax = np.percentile(img.astype(np.float32), [0.5, 99.5])
    if vmax <= vmin:
        vmin = float(np.min(img))
        vmax = float(np.max(img))
    return float(vmin), float(vmax)


def pick_representative_2d_name(k2_low_dir: Path, k4_low_dir: Path) -> str:
    k2 = {p.name for p in k2_low_dir.glob("*.tif")}
    k4 = {p.name for p in k4_low_dir.glob("*.tif")}
    common = sorted(k2 & k4)
    if not common:
        raise FileNotFoundError("No common 2D sample names between K2_xy and K4_xy.")
    # Prefer large_ sample for clearer structure; fallback to first common.
    large = [n for n in common if n.startswith("large_")]
    return large[0] if large else common[0]


def pick_representative_3d_id(k2_stacks: Path, k4_stacks: Path) -> str:
    def ids_from(stacks_dir: Path) -> set[str]:
        out = set()
        for p in stacks_dir.glob("*_sparse.npy"):
            out.add(p.stem.replace("_sparse", ""))
        return out

    common = sorted(ids_from(k2_stacks) & ids_from(k4_stacks))
    if not common:
        raise FileNotFoundError("No common 3D stack ids between K2_xy and K4_xy.")
    return common[0]


def plot_2d_comparison(root: Path, out_path: Path) -> None:
    k2_low = root / "2d/K2_xy/Validation/low"
    k2_high = root / "2d/K2_xy/Validation/high"
    k4_low = root / "2d/K4_xy/Validation/low"
    k4_high = root / "2d/K4_xy/Validation/high"
    name = pick_representative_2d_name(k2_low, k4_low)

    imgs = [
        ("2D K2_xy Sparse", tifffile.imread(k2_low / name)),
        ("2D K2_xy GT", tifffile.imread(k2_high / name)),
        ("2D K4_xy Sparse", tifffile.imread(k4_low / name)),
        ("2D K4_xy GT", tifffile.imread(k4_high / name)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=180)
    for ax, (title, img) in zip(axes.ravel(), imgs):
        vmin, vmax = robust_minmax(img)
        im = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\n{name}", fontsize=10)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle("MDC25_sparse Representative Comparison - 2D", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_3d_comparison(root: Path, out_path: Path) -> None:
    k2_stacks = root / "3d/K2_xy/Validation/stacks"
    k4_stacks = root / "3d/K4_xy/Validation/stacks"
    sid = pick_representative_3d_id(k2_stacks, k4_stacks)

    k2_sparse = np.load(k2_stacks / f"{sid}_sparse.npy")
    k2_gt = np.load(k2_stacks / f"{sid}_gt.npy")
    k4_sparse = np.load(k4_stacks / f"{sid}_sparse.npy")
    k4_gt = np.load(k4_stacks / f"{sid}_gt.npy")

    z2 = k2_gt.shape[0] // 2
    z4 = k4_gt.shape[0] // 2
    imgs = [
        (f"3D K2_xy Sparse (z={z2})", k2_sparse[z2]),
        (f"3D K2_xy GT (z={z2})", k2_gt[z2]),
        (f"3D K4_xy Sparse (z={z4})", k4_sparse[z4]),
        (f"3D K4_xy GT (z={z4})", k4_gt[z4]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=180)
    for ax, (title, img) in zip(axes.ravel(), imgs):
        vmin, vmax = robust_minmax(img)
        im = ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\nstack_id={sid}", fontsize=10)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle("MDC25_sparse Representative Comparison - 3D", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create representative comparison figures for MDC25_sparse 2D and 3D datasets."
        )
    )
    p.add_argument(
        "--mdc25_sparse_root",
        type=Path,
        default=Path("/data2/wjb/threePM/ThreePM_openDatasets/MDC25_sparse"),
        help="Root path of MDC25_sparse dataset.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path(
            "/data2/wjb/threePM/ThreePM_openDatasets/MDC25_sparse/representative_comparison"
        ),
        help="Output directory for generated figures.",
    )
    return p.parse_args()


def main() -> None:
    args = build_args()
    out2d = args.output_dir / "mdc25_sparse_representative_2d.png"
    out3d = args.output_dir / "mdc25_sparse_representative_3d.png"
    plot_2d_comparison(args.mdc25_sparse_root, out2d)
    plot_3d_comparison(args.mdc25_sparse_root, out3d)
    print(out2d)
    print(out3d)


if __name__ == "__main__":
    main()
