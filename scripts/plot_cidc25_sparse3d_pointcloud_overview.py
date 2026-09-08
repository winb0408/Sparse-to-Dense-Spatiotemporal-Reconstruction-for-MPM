#!/usr/bin/env python3
"""
3D point-cloud overview for CIDC25_sparse3d (same style as visualize_cidc25_3d).

One figure per sparsity K with four columns:
  XY sparse | Z sparse | XYZ sparse | Ground Truth

Outputs (default):
  <sparse_root>/comparison_3d_overview_pointcloud_K2.png
  <sparse_root>/comparison_3d_overview_pointcloud_K3.png
  <sparse_root>/comparison_3d_overview_pointcloud_K4.png

Example:
  python scripts/plot_cidc25_sparse3d_pointcloud_overview.py \
    --sparse_root /data2/wjb/threePM/ThreePM_openDatasets/CIDC25_sparse3d \
    --gt_stem A1 --split Training --ks 2 3 4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reuse CIDC25 3D visualization helpers for consistent look.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prepare_cidc25_sparse3d import sparsify_xy, sparsify_xyz, sparsify_z
from scripts.visualize_cidc25_3d import robust_normalize, _sample_points


MODES = ["xy", "z", "xyz"]
MODE_TITLES = {
    "xy": "XY sparse",
    "z": "Z sparse",
    "xyz": "XYZ sparse",
    "gt": "Ground Truth",
}


def load_npy_stack(path: Path, max_frames: int | None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing stack: {path}")
    vol = np.load(str(path)).astype(np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected [F,H,W], got {vol.shape} from {path}")
    if max_frames is not None and vol.shape[0] > max_frames:
        vol = vol[:max_frames]
    return vol


def stack_paths(
    sparse_root: Path,
    k: int,
    mode: str,
    split: str,
    gt_stem: str,
) -> Tuple[Path, Path]:
    base = sparse_root / f"K{k}_{mode}" / split / "stacks"
    return base / f"{gt_stem}_sparse.npy", base / f"{gt_stem}_gt.npy"


def build_raw_sparse_volume(gt_vol: np.ndarray, mode: str, k: int, xy_pattern: str) -> np.ndarray:
    if mode == "xy":
        return sparsify_xy(gt_vol, k, xy_pattern).astype(np.float32, copy=False)
    if mode == "z":
        return sparsify_z(gt_vol, k).astype(np.float32, copy=False)
    if mode == "xyz":
        return sparsify_xyz(gt_vol, k, xy_pattern).astype(np.float32, copy=False)
    raise ValueError(f"Unknown sparse mode: {mode}")


def normalize_with_reference(
    x: np.ndarray,
    ref: np.ndarray,
    low_q: float = 0.5,
    high_q: float = 99.5,
) -> np.ndarray:
    lo = np.percentile(ref, low_q)
    hi = np.percentile(ref, high_q)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return y.astype(np.float32, copy=False)


def draw_pointcloud_axis(
    ax,
    vol_norm: np.ndarray,
    title: str,
    threshold: float,
    max_points: int,
    seed: int,
    elev: float,
    azim: float,
    cmap: str,
    point_size: float,
    point_alpha: float,
) -> int:
    coords, colors = _sample_points(vol_norm, threshold=threshold, max_points=max_points, seed=seed)
    if coords.size > 0:
        z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]
        ax.scatter(
            x, y, z,
            c=colors,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=point_size,
            alpha=point_alpha,
            linewidths=0,
        )
    ax.set_title(f"{title}\n(n={coords.shape[0]:,})", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((vol_norm.shape[2], vol_norm.shape[1], vol_norm.shape[0]))
    ax.view_init(elev=elev, azim=azim)
    return int(coords.shape[0])


def make_k_pointcloud_figure(
    sparse_root: Path,
    k: int,
    split: str,
    gt_stem: str,
    out_path: Path,
    threshold: float,
    max_points: int,
    max_frames: int | None,
    elev: float,
    azim: float,
    sparse_variant: str,
    xy_pattern: str,
    norm_mode: str,
    cmap: str,
    point_size: float,
    point_alpha: float,
) -> None:
    panels: List[Tuple[str, np.ndarray]] = []
    gt_path = sparse_root / f"K{k}_xy" / split / "stacks" / f"{gt_stem}_gt.npy"
    gt_vol = load_npy_stack(gt_path, max_frames=max_frames)

    for mode in MODES:
        if sparse_variant == "raw":
            sparse_vol = build_raw_sparse_volume(gt_vol, mode, k, xy_pattern)
        else:
            sp_path, _ = stack_paths(sparse_root, k, mode, split, gt_stem)
            sparse_vol = load_npy_stack(sp_path, max_frames=max_frames)
        panels.append((mode, sparse_vol))
    panels.append(("gt", gt_vol))
    gt_norm_ref = robust_normalize(gt_vol)

    fig = plt.figure(figsize=(22, 5.5), constrained_layout=True)
    for j, (key, vol_raw) in enumerate(panels):
        if norm_mode == "gt_shared":
            vol_norm = gt_norm_ref if key == "gt" else normalize_with_reference(vol_raw, gt_vol)
            thr = threshold
        else:
            vol_norm = robust_normalize(vol_raw)
            # Sparse panels have more background — use slightly lower threshold.
            thr = threshold if key == "gt" else min(threshold, 0.72)
        ax = fig.add_subplot(1, 4, j + 1, projection="3d")
        draw_pointcloud_axis(
            ax,
            vol_norm,
            MODE_TITLES[key],
            threshold=thr,
            max_points=max_points,
            seed=42 + k * 10 + j,
            elev=elev,
            azim=azim,
            cmap=cmap,
            point_size=point_size,
            point_alpha=point_alpha,
        )

    fig.suptitle(
        f"CIDC25 sparse 3D overview | K={k} | {gt_stem} ({split}) | {sparse_variant} point cloud | {norm_mode}",
        fontsize=14,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="4-column 3D point-cloud overview for CIDC25_sparse3d (XY/Z/XYZ/GT)."
    )
    p.add_argument(
        "--sparse_root",
        type=Path,
        default=Path("/data2/wjb/threePM/ThreePM_openDatasets/CIDC25_sparse3d"),
    )
    p.add_argument("--gt_stem", type=str, default="A1")
    p.add_argument("--split", type=str, default="Training", choices=["Training", "Validation"])
    p.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--threshold", type=float, default=0.85,
                   help="Intensity threshold on robust-normalized volume (0–1).")
    p.add_argument("--max_points", type=int, default=120000)
    p.add_argument("--max_frames", type=int, default=400,
                   help="Cap Z depth for faster rendering (None = full stack).")
    p.add_argument("--elev", type=float, default=28.0)
    p.add_argument("--azim", type=float, default=40.0)
    p.add_argument("--cmap", type=str, default="inferno")
    p.add_argument("--point_size", type=float, default=1.0)
    p.add_argument("--point_alpha", type=float, default=0.35)
    p.add_argument("--sparse_variant", type=str, default="warm", choices=["raw", "warm"],
                   help="Render raw zero-filled sparse volumes or saved warm-start volumes.")
    p.add_argument("--xy_pattern", type=str, default="grid", choices=["grid", "random"],
                   help="XY sparsification pattern used when regenerating raw sparse volumes.")
    p.add_argument("--norm_mode", type=str, default="per_volume", choices=["per_volume", "gt_shared"],
                   help="Normalization mode for point-cloud thresholding and coloring.")
    p.add_argument("--style_preset", type=str, default="default",
                   choices=["default", "best_pointcloud_magma"],
                   help="Optional visual preset for colormap, point size/alpha, and view.")
    args = p.parse_args()

    if args.style_preset == "best_pointcloud_magma":
        args.cmap = "magma_r"
        args.point_size = 0.8
        args.point_alpha = 0.28
        args.elev = 26.0
        args.azim = 42.0

    for k in args.ks:
        for mode in MODES:
            sp, _ = stack_paths(args.sparse_root, k, mode, args.split, args.gt_stem)
            if not sp.exists():
                raise FileNotFoundError(
                    f"Missing K{k} mode={mode}. Run prepare_cidc25_sparse3d.py first.\n  Expected: {sp}"
                )
        print(f"[K={k}] rendering point cloud overview …")
        suffix = "" if args.sparse_variant == "warm" else f"_{args.sparse_variant}"
        if args.norm_mode != "per_volume":
            suffix = f"{suffix}_{args.norm_mode}"
        out = args.sparse_root / f"comparison_3d_overview_pointcloud{suffix}_K{k}.png"
        make_k_pointcloud_figure(
            sparse_root=args.sparse_root,
            k=k,
            split=args.split,
            gt_stem=args.gt_stem,
            out_path=out,
            threshold=args.threshold,
            max_points=args.max_points,
            max_frames=args.max_frames,
            elev=args.elev,
            azim=args.azim,
            sparse_variant=args.sparse_variant,
            xy_pattern=args.xy_pattern,
            norm_mode=args.norm_mode,
            cmap=args.cmap,
            point_size=args.point_size,
            point_alpha=args.point_alpha,
        )
    print(f"Done. Outputs under {args.sparse_root}")


if __name__ == "__main__":
    main()
