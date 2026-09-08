#!/usr/bin/env python3
"""
Prepare AI4Life-CIDC25 stacks into sparse 3D reconstruction datasets.

Generates three sparsification modes at configurable rates (K=2,4,8):
  - XY:  per-frame spatial subsampling (1/K^2 pixels retained)
  - Z:   axial subsampling (1/K frames retained, rest interpolated)
  - XYZ: combined XY + Z sparsification

Outputs two formats:
  1. Per-frame paired TIFFs (high/low) for 2D / 2.5D training
  2. Full .npy stacks for 3D volume training

Example:
  python scripts/prepare_cidc25_sparse3d.py \
    --cidc_root /data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25 \
    --output_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_sparse3d \
    --gt_stem A1 --split Training \
    --sparsity_factors 2 4 8 \
    --modes xy z xyz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import tifffile
from scipy.ndimage import zoom


def read_stack(path: Path) -> np.ndarray:
    """Load TIFF stack as float32 [F, H, W] in [0, 1]."""
    arr = tifffile.imread(str(path))
    if arr.ndim < 3:
        raise ValueError(f"Expected 3D stack, got shape={arr.shape} from {path}")
    while arr.ndim > 3:
        arr = arr[:, 0]
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32)
    vmin, vmax = arr.min(), arr.max()
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    return arr.clip(0.0, 1.0)


def sparsify_xy(vol: np.ndarray, K: int, pattern: str = "grid") -> np.ndarray:
    """Apply XY spatial mask to each frame. Unsampled pixels = 0."""
    F, H, W = vol.shape
    out = np.zeros_like(vol)
    if pattern == "grid":
        out[:, ::K, ::K] = vol[:, ::K, ::K]
    elif pattern == "random":
        n_keep = max(1, H * W // (K * K))
        rng = np.random.default_rng(42)
        for f in range(F):
            rows = rng.integers(0, H, size=n_keep)
            cols = rng.integers(0, W, size=n_keep)
            out[f, rows, cols] = vol[f, rows, cols]
    else:
        raise ValueError(f"Unknown XY pattern: {pattern}")
    return out


def sparsify_z(vol: np.ndarray, K: int) -> np.ndarray:
    """Keep every K-th frame, linearly interpolate the rest."""
    F, H, W = vol.shape
    kept_indices = list(range(0, F, K))
    kept = vol[kept_indices]  # [F//K, H, W]
    z_ratio = F / len(kept_indices)
    interpolated = zoom(kept, (z_ratio, 1.0, 1.0), order=1)
    return interpolated[:F].astype(np.float32)


def sparsify_xyz(vol: np.ndarray, K: int, pattern: str = "grid") -> np.ndarray:
    """Z-sparse first, then XY-sparse."""
    z_sparse = sparsify_z(vol, K)
    return sparsify_xy(z_sparse, K, pattern)


def warmstart_xy(sparse: np.ndarray, K: int) -> np.ndarray:
    """Bilinear down-up per frame to fill unsampled XY pixels."""
    F, H, W = sparse.shape
    out = np.empty_like(sparse)
    for f in range(F):
        small = zoom(sparse[f], 1.0 / K, order=1)
        out[f] = zoom(small, (H / small.shape[0], W / small.shape[1]), order=1)[:H, :W]
    return out.clip(0.0, 1.0).astype(np.float32)


def save_paired_frames(
    gt_vol: np.ndarray,
    sparse_vol: np.ndarray,
    out_dir: Path,
    stem: str,
) -> int:
    """Save per-frame paired TIFFs to high/ and low/ directories."""
    high_dir = out_dir / "high"
    low_dir = out_dir / "low"
    high_dir.mkdir(parents=True, exist_ok=True)
    low_dir.mkdir(parents=True, exist_ok=True)

    F = min(gt_vol.shape[0], sparse_vol.shape[0])
    for i in range(F):
        name = f"{stem}_f{i:04d}.tif"
        tifffile.imwrite(str(high_dir / name), gt_vol[i])
        tifffile.imwrite(str(low_dir / name), sparse_vol[i])
    return F


def save_stacks(
    gt_vol: np.ndarray,
    sparse_vol: np.ndarray,
    out_dir: Path,
    stem: str,
) -> None:
    """Save full stacks as .npy for 3D volume training."""
    stack_dir = out_dir / "stacks"
    stack_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(stack_dir / f"{stem}_gt.npy"), gt_vol)
    np.save(str(stack_dir / f"{stem}_sparse.npy"), sparse_vol)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate sparse 3D datasets from CIDC25 stacks.",
    )
    p.add_argument("--cidc_root", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--gt_stem", type=str, required=True,
                   help="Stack stem used as GT, e.g. A1 or F0.")
    p.add_argument("--split", type=str, default="Training",
                   choices=["Training", "Validation"])
    p.add_argument("--sparsity_factors", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--modes", type=str, nargs="+", default=["xy", "z", "xyz"],
                   choices=["xy", "z", "xyz"])
    p.add_argument("--xy_pattern", type=str, default="grid",
                   choices=["grid", "random"])
    p.add_argument("--max_frames", type=int, default=None,
                   help="Limit number of frames for faster prototyping.")
    args = p.parse_args()

    cidc_root = Path(args.cidc_root)
    output_root = Path(args.output_root)
    split_dir = cidc_root / args.split

    gt_path = split_dir / f"{args.gt_stem}.tif"
    if not gt_path.exists():
        available = sorted(p.name for p in split_dir.glob("*.tif"))
        raise FileNotFoundError(
            f"GT stack not found: {gt_path}\nAvailable: {available}"
        )

    print(f"Loading GT stack: {gt_path}")
    gt_vol = read_stack(gt_path)
    if args.max_frames is not None:
        gt_vol = gt_vol[:args.max_frames]
    print(f"  shape={gt_vol.shape}, dtype={gt_vol.dtype}, "
          f"range=[{gt_vol.min():.4f}, {gt_vol.max():.4f}]")

    for K in args.sparsity_factors:
        for mode in args.modes:
            tag = f"K{K}_{mode}"
            out_split = output_root / tag / args.split
            print(f"\n--- {tag} ---")

            if mode == "xy":
                sparse_raw = sparsify_xy(gt_vol, K, args.xy_pattern)
                sparse_warm = warmstart_xy(sparse_raw, K)
            elif mode == "z":
                sparse_warm = sparsify_z(gt_vol, K)
            elif mode == "xyz":
                sparse_raw = sparsify_xyz(gt_vol, K, args.xy_pattern)
                sparse_warm = warmstart_xy(sparse_raw, K)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            n = save_paired_frames(gt_vol, sparse_warm, out_split, args.gt_stem)
            save_stacks(gt_vol, sparse_warm, out_split, args.gt_stem)
            print(f"  saved {n} frame pairs + stacks to {out_split}")

    print("\nDone.")


if __name__ == "__main__":
    main()
