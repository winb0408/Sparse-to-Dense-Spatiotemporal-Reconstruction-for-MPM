#!/usr/bin/env python3
"""
Prepare AI4Life-MDC25 sparse datasets for 2D / 2.5D / 3D validation.

Outputs:
  - 2D (Nuclei): XY sparse only
      MDC25_sparse/2d/K{K}_xy/{Training|Validation}/high|low/*.tif
  - 3D (Tribolium): XY / Z / XYZ sparse
      MDC25_sparse/3d/K{K}_{mode}/{Training|Validation}/high|low/*.tif
      MDC25_sparse/3d/K{K}_{mode}/{Training|Validation}/stacks/*_{gt|sparse}.npy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tifffile
from scipy.ndimage import zoom


def read_tiff_float01(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    while arr.ndim > 3:
        arr = arr[:, 0]
    if arr.dtype == np.uint16:
        out = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        out = arr.astype(np.float32) / 255.0
    else:
        out = arr.astype(np.float32)
        vmin, vmax = float(out.min()), float(out.max())
        if vmax > vmin:
            out = (out - vmin) / (vmax - vmin + 1e-8)
    return out.clip(0.0, 1.0).astype(np.float32)


def sparsify_xy_2d(img: np.ndarray, k: int, pattern: str, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape
    sparse = np.zeros_like(img)
    if pattern == "grid":
        sparse[::k, ::k] = img[::k, ::k]
    elif pattern == "random":
        n_keep = max(1, (h * w) // (k * k))
        rows = rng.integers(0, h, size=n_keep)
        cols = rng.integers(0, w, size=n_keep)
        sparse[rows, cols] = img[rows, cols]
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    return sparse


def warmstart_xy_2d(sparse: np.ndarray, k: int) -> np.ndarray:
    h, w = sparse.shape
    small = zoom(sparse, 1.0 / k, order=1)
    up = zoom(small, (h / small.shape[0], w / small.shape[1]), order=1)
    return up[:h, :w].clip(0.0, 1.0).astype(np.float32)


def sparsify_xy_3d(vol: np.ndarray, k: int, pattern: str, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(vol)
    f, h, w = vol.shape
    if pattern == "grid":
        out[:, ::k, ::k] = vol[:, ::k, ::k]
    elif pattern == "random":
        n_keep = max(1, (h * w) // (k * k))
        for i in range(f):
            rows = rng.integers(0, h, size=n_keep)
            cols = rng.integers(0, w, size=n_keep)
            out[i, rows, cols] = vol[i, rows, cols]
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    return out


def warmstart_xy_3d(vol: np.ndarray, k: int) -> np.ndarray:
    return np.stack([warmstart_xy_2d(v, k) for v in vol], axis=0)


def sparsify_z_3d(vol: np.ndarray, k: int) -> np.ndarray:
    f, _, _ = vol.shape
    keep_idx = list(range(0, f, k))
    kept = vol[keep_idx]
    z_ratio = f / len(keep_idx)
    interp = zoom(kept, (z_ratio, 1.0, 1.0), order=1)
    return interp[:f].astype(np.float32)


def sparsify_xyz_3d(vol: np.ndarray, k: int, pattern: str, rng: np.random.Generator) -> np.ndarray:
    z_sparse = sparsify_z_3d(vol, k)
    xy_sparse = sparsify_xy_3d(z_sparse, k, pattern, rng)
    return warmstart_xy_3d(xy_sparse, k)


def split_ids(ids: List[str], val_ratio: float, seed: int) -> Dict[str, List[str]]:
    ids_sorted = sorted(ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids_sorted))
    n_val = max(1, int(round(len(ids_sorted) * val_ratio)))
    val_set = {ids_sorted[i] for i in perm[:n_val]}
    train = [x for x in ids_sorted if x not in val_set]
    val = [x for x in ids_sorted if x in val_set]
    return {"Training": train, "Validation": val}


@dataclass
class BuildStats:
    num_2d: int = 0
    num_3d: int = 0
    num_3d_stacks: int = 0


def save_2d_pair(gt: np.ndarray, low: np.ndarray, out_dir: Path, stem: str) -> None:
    (out_dir / "high").mkdir(parents=True, exist_ok=True)
    (out_dir / "low").mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_dir / "high" / f"{stem}.tif"), gt.astype(np.float32))
    tifffile.imwrite(str(out_dir / "low" / f"{stem}.tif"), low.astype(np.float32))


def save_3d_frame_pairs(gt: np.ndarray, low: np.ndarray, out_dir: Path, stem: str) -> int:
    (out_dir / "high").mkdir(parents=True, exist_ok=True)
    (out_dir / "low").mkdir(parents=True, exist_ok=True)
    n = min(gt.shape[0], low.shape[0])
    for i in range(n):
        name = f"{stem}_f{i:04d}.tif"
        tifffile.imwrite(str(out_dir / "high" / name), gt[i].astype(np.float32))
        tifffile.imwrite(str(out_dir / "low" / name), low[i].astype(np.float32))
    return n


def save_3d_stacks(gt: np.ndarray, low: np.ndarray, out_dir: Path, stem: str) -> None:
    stack_dir = out_dir / "stacks"
    stack_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(stack_dir / f"{stem}_gt.npy"), gt.astype(np.float32))
    np.save(str(stack_dir / f"{stem}_sparse.npy"), low.astype(np.float32))


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare sparse datasets for AI4Life-MDC25.")
    p.add_argument("--mdc25_root", type=str, required=True)
    p.add_argument("--output_root", type=str, required=True)
    p.add_argument("--sparsity_factors", type=int, nargs="+", default=[2, 4])
    p.add_argument("--xy_pattern", type=str, choices=["grid", "random"], default="grid")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--splits", type=str, nargs="+", default=["Training", "Validation"], choices=["Training", "Validation"])
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--write_3d_frame_pairs", action="store_true", default=True)
    p.add_argument("--max_2d", type=int, default=None)
    p.add_argument("--max_3d", type=int, default=None)
    args = p.parse_args()

    root = Path(args.mdc25_root)
    out_root = Path(args.output_root)
    rng = np.random.default_rng(args.seed)
    stats = BuildStats()

    # 2D source
    nuclei_root = root / "2D denoising (Nuclei)"
    large_gt = nuclei_root / "large_images" / "gt"
    large_noisy = nuclei_root / "large_images" / "noisy"
    small_gt = nuclei_root / "small_images" / "gt"
    small_noisy = nuclei_root / "small_images" / "noisy"

    def collect_2d_pairs(src_gt: Path, src_noisy: Path, prefix: str) -> Dict[str, Tuple[Path, Path]]:
        out: Dict[str, Tuple[Path, Path]] = {}
        for gt in sorted(src_gt.glob("*.tif")):
            noisy = src_noisy / gt.name
            if noisy.exists():
                out[f"{prefix}_{gt.stem}"] = (gt, noisy)
        return out

    pairs_2d = {}
    pairs_2d.update(collect_2d_pairs(large_gt, large_noisy, "large"))
    pairs_2d.update(collect_2d_pairs(small_gt, small_noisy, "small"))
    ids_2d = sorted(pairs_2d.keys())
    if args.max_2d is not None:
        ids_2d = ids_2d[: args.max_2d]
    splits_2d = split_ids(ids_2d, args.val_ratio, args.seed)

    # 3D source
    trib_root = root / "3D denoising (Tribolium)"
    gt3 = trib_root / "gt"
    noisy3 = trib_root / "noisy"
    pairs_3d: Dict[str, Tuple[Path, Path]] = {}
    for gt in sorted(gt3.glob("*.tif")):
        noisy = noisy3 / gt.name
        if noisy.exists():
            pairs_3d[gt.stem] = (gt, noisy)
    ids_3d = sorted(pairs_3d.keys())
    if args.max_3d is not None:
        ids_3d = ids_3d[: args.max_3d]
    splits_3d = split_ids(ids_3d, args.val_ratio, args.seed + 1)

    # Build 2D sparse (XY only)
    for split in args.splits:
        ids = splits_2d[split]
        for k in args.sparsity_factors:
            out_dir = out_root / "2d" / f"K{k}_xy" / split
            for sid in ids:
                gt_path, noisy_path = pairs_2d[sid]
                out_high = out_dir / "high" / f"{sid}.tif"
                out_low = out_dir / "low" / f"{sid}.tif"
                if args.skip_existing and out_high.exists() and out_low.exists():
                    continue
                gt_im = read_tiff_float01(gt_path)
                noisy_im = read_tiff_float01(noisy_path)
                sparse = sparsify_xy_2d(noisy_im, k, args.xy_pattern, rng)
                warm = warmstart_xy_2d(sparse, k)
                save_2d_pair(gt_im, warm, out_dir, sid)
                stats.num_2d += 1

    # Build 3D sparse (XY/Z/XYZ)
    for split in args.splits:
        ids = splits_3d[split]
        for k in args.sparsity_factors:
            for mode in ("xy", "z", "xyz"):
                out_dir = out_root / "3d" / f"K{k}_{mode}" / split
                for sid in ids:
                    gt_path, noisy_path = pairs_3d[sid]
                    stack_gt = out_dir / "stacks" / f"{sid}_gt.npy"
                    stack_low = out_dir / "stacks" / f"{sid}_sparse.npy"
                    if args.skip_existing and stack_gt.exists() and stack_low.exists():
                        continue

                    gt_vol = read_tiff_float01(gt_path)
                    noisy_vol = read_tiff_float01(noisy_path)
                    if mode == "xy":
                        sparse = sparsify_xy_3d(noisy_vol, k, args.xy_pattern, rng)
                        low_vol = warmstart_xy_3d(sparse, k)
                    elif mode == "z":
                        low_vol = sparsify_z_3d(noisy_vol, k)
                    else:
                        low_vol = sparsify_xyz_3d(noisy_vol, k, args.xy_pattern, rng)

                    if args.write_3d_frame_pairs:
                        save_3d_frame_pairs(gt_vol, low_vol, out_dir, sid)
                        stats.num_3d += 1
                    save_3d_stacks(gt_vol, low_vol, out_dir, sid)
                    stats.num_3d_stacks += 1

    summary = out_root / "BUILD_SUMMARY.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w", encoding="utf-8") as f:
        f.write("# MDC25 Sparse Build Summary\n\n")
        f.write(f"- mdc25_root: `{root}`\n")
        f.write(f"- output_root: `{out_root}`\n")
        f.write(f"- sparsity_factors: `{args.sparsity_factors}`\n")
        f.write(f"- xy_pattern: `{args.xy_pattern}`\n")
        f.write(f"- splits: `{args.splits}`\n")
        f.write(f"- val_ratio: `{args.val_ratio}`\n")
        f.write(f"- seed: `{args.seed}`\n")
        f.write(f"- skip_existing: `{args.skip_existing}`\n")
        f.write(f"- write_3d_frame_pairs: `{args.write_3d_frame_pairs}`\n\n")
        f.write("## Counts\n")
        f.write(f"- 2D samples written: **{stats.num_2d}**\n")
        f.write(f"- 3D frame-pair groups written: **{stats.num_3d}**\n")
        f.write(f"- 3D stack pairs written: **{stats.num_3d_stacks}**\n\n")
        f.write("## Input Split Stats\n")
        f.write(f"- 2D total IDs: {len(ids_2d)} | Training: {len(splits_2d['Training'])} | Validation: {len(splits_2d['Validation'])}\n")
        f.write(f"- 3D total IDs: {len(ids_3d)} | Training: {len(splits_3d['Training'])} | Validation: {len(splits_3d['Validation'])}\n")

    print(f"Done. Summary -> {summary}")


if __name__ == "__main__":
    main()
