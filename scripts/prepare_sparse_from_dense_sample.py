#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile


def ensure_3d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported ndim={arr.ndim}, expected 2D or 3D TIFF.")


def sparse_mask_xy(hw: Tuple[int, int], factor: int, mode: str) -> np.ndarray:
    h, w = hw
    mask = np.zeros((h, w), dtype=np.uint8)
    row_step = factor if "Y" in mode else 1
    col_step = factor if "X" in mode else 1
    mask[0:h:row_step, 0:w:col_step] = 1
    return mask


def save_pair(
    dense_3d: np.ndarray,
    base_name: str,
    out_root: Path,
    factor: int,
    mode: str,
    overwrite: bool,
) -> None:
    t, h, w = dense_3d.shape
    # Spatial (XY) sparsity component
    if "X" in mode or "Y" in mode:
        mask_2d = sparse_mask_xy((h, w), factor, mode)
    else:
        mask_2d = np.ones((h, w), dtype=np.uint8)

    # Axial (Z) sparsity component
    if "Z" in mode:
        z_mask = np.zeros((t,), dtype=np.uint8)
        z_mask[0:t:factor] = 1
    else:
        z_mask = np.ones((t,), dtype=np.uint8)

    mask_3d = np.repeat(mask_2d[None, ...], t, axis=0)
    mask_3d = mask_3d * z_mask[:, None, None]
    sparse_3d = dense_3d * mask_3d.astype(dense_3d.dtype)

    name = f"{base_name}-{factor}x{mode}"
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    sparse_path = out_dir / f"{name}.tif"
    mask_path = out_dir / f"{name}_mask.tif"
    if (sparse_path.exists() or mask_path.exists()) and not overwrite:
        print(f"[Skip] exists: {name}")
        return

    tifffile.imwrite(sparse_path, sparse_3d, imagej=True)
    tifffile.imwrite(mask_path, mask_3d, imagej=True)
    print(
        f"[OK] {name}: sparse={sparse_3d.shape}, mask={mask_3d.shape}, "
            f"sampling_ratio={mask_3d.mean():.4f}"
    )


def save_dense_gt(dense_3d: np.ndarray, base_name: str, out_root: Path, overwrite: bool) -> None:
    t, h, w = dense_3d.shape
    name = f"{base_name}-dense"
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_path = out_dir / f"{name}.tif"
    mask_path = out_dir / f"{name}_mask.tif"
    if (gt_path.exists() or mask_path.exists()) and not overwrite:
        print(f"[Skip] exists: {name}")
        return
    mask = np.ones((t, h, w), dtype=np.uint8)
    tifffile.imwrite(gt_path, dense_3d, imagej=True)
    tifffile.imwrite(mask_path, mask, imagej=True)
    print(f"[OK] {name}: gt={dense_3d.shape}, mask={mask.shape}")


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build sparse variants (2x/4x, X/Y/XY) from one dense TIFF sample and "
            "generate matching masks."
        )
    )
    p.add_argument(
        "--dense_tif",
        type=Path,
        default=Path("/data2/wjb/threePM/ThreePM_openDatasets/200um-f10-z5.tif"),
        help="Input dense sample TIFF (2D or 3D).",
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data2/wjb/threePM/ThreePM_openDatasets/200um-f10-z5_sparse"),
        help="Output root directory.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return p.parse_args()


def main() -> None:
    args = build_args()
    dense = tifffile.imread(args.dense_tif)
    dense_3d = ensure_3d(dense)
    args.output_root.mkdir(parents=True, exist_ok=True)

    base_name = args.dense_tif.stem
    save_dense_gt(dense_3d, base_name, args.output_root, args.overwrite)
    for factor in (2, 4):
        for mode in ("X", "Y", "XY", "Z", "XYZ"):
            save_pair(dense_3d, base_name, args.output_root, factor, mode, args.overwrite)

    print(f"[Done] Sparse dataset written to: {args.output_root}")


if __name__ == "__main__":
    main()
